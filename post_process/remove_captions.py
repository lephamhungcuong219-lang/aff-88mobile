#!/usr/bin/env python3
"""Detect small burned-in captions in scene clips and remove them (no AI).

Usage:
    python remove_captions.py <sources_dir> <output_dir> [--mode panel|fill] [--preview DIR]

Every <sources_dir>/<N>.mp4 goes through the process and ends up as
<output_dir>/<N>.mp4: re-encoded with the caption removed, or copied as is when
no caption is found. A report is written to <output_dir>/captions.json.

Finding captions (classic image processing, OpenCV + numpy only):
  1. Each clip is sampled at SAMPLE_FPS, downscaled to ANALYSIS_WIDTH.
  2. Caption glyphs are bright (white, or saturated colour for highlighted
     words) and outlined/shadowed with dark pixels, so a pixel counts as "text"
     when it is bright AND much brighter than its darkest neighbour.
  3. Text pixels are joined horizontally into lines; a line is a candidate when
     it is caption-sized (small, wide, horizontally centred) and in the lower
     part of the frame. Big overlay titles are too tall and are ignored.
  4. Captions of one video sit on ONE fixed line. Candidates of all clips are
     pooled, the dominant line is found, and only candidates on it are kept.
  5. Per clip, the kept boxes are merged into one region covering the whole
     sentence, with the time span in which it is visible.

Removing captions (only inside the region and time span):
    A stable mask is built for the caption: glyph pixels, their drop shadow /
    outline (pixels much darker than the local background, right next to the
    glyphs) and the glow of coloured words, seen in enough frames.
    panel  (default) the glyphs are removed as in "fill", then the whole
           sentence area is covered by a frosted-glass panel: a clean rounded
           rectangle with a crisp edge, centred on the frame, same height in
           every clip, as wide as the sentence (strong blur, slightly lightened).
           When the mask looks unreliable, the panel only blurs.
    fill   only the glyphs: masked pixels are replaced by a smooth estimate made
           from the unmasked background around them (normalised convolution).
           Falls back to "panel" when the mask looks unreliable.
Only the caption region is processed in Python (at analysis resolution, which
is enough for a smooth fill); ffmpeg overlays it back onto the full-size video.

--preview DIR saves, per clip with a caption, the region before/after.

Stdout: 'dir:<absolute output_dir>' on success, or exactly 'fail'.
Stderr: progress and the reason for a failure (human-readable).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Analysis
ANALYSIS_WIDTH = 540
SAMPLE_FPS = 8
BRIGHT_MIN = 185          # grey level of white glyphs
COLOUR_SAT_MIN = 110      # saturation of coloured (highlighted) glyphs
WARM_HUE_MAX, WARM_HUE_WRAP = 40, 165  # OpenCV hue (0-180): red..yellow, and red again
CONTRAST_MIN = 70         # glyph must be this much brighter than its darkest neighbour

# Caption geometry, as fractions of the frame
ZONE_TOP, ZONE_BOTTOM = 0.45, 0.95
LINE_H_MIN, LINE_H_MAX = 0.010, 0.080  # up to bigger text sitting on the caption line
CENTER_X_MIN, CENTER_X_MAX = 0.30, 0.70
MIN_ASPECT = 2.5          # width / height of a caption line
MAX_LINE_WIDTH = 0.995    # x width: long captions can run almost edge to edge
FILL_MIN, FILL_MAX = 0.06, 0.65

# Dominant caption line
LINE_BIN = 0.01           # y histogram bin
MIN_CLIPS_ON_LINE = 3     # fewer clips than this on one line -> no captions
LINE_Y_TOLERANCE = 0.6    # x median line height
LINE_H_RANGE = (0.6, 2.4) # accepted height relative to the median line height (bigger text on
                          # the caption line is covered too, with a taller panel)
EDGE_SNAP_S = 0.4         # caption found this close to the clip start/end: cover up to the cut
MIN_HITS_PER_CLIP = 2     # samples a caption must appear in

# Band fallback: for clips where no line-shaped caption was found (e.g. the
# caption touches bright background), look for text pixels inside the caption band.
BAND_REACH = 1.6          # x line height: rows searched above/below the line for the text
BAND_COLUMNS = 0.9        # x width: central part of the frame used to measure row density
BAND_ROW_SHARE = 0.35     # a row belongs to the text when its density reaches this share of the peak
BAND_COLUMN_MIN = 0.04    # a column belongs to the text when this share of its rows is text
BAND_MIN_ASPECT = 1.8     # width / height: short captions of two characters are accepted
BAND_DENSITY_MIN = 0.06   # text pixel share inside the band (over the text's width)
BAND_CONTRAST = 3.0       # band density must exceed the strips above/below this many times
BAND_EDGE = 0.3           # x line height: strips right above/below the band

# Region around the sentence
PAD_Y = 0.30              # x line height, added above and below (cover fits the text)
PAD_X = 0.35              # x line height, added left and right
EDGE_MARGIN = 0.03        # x width: the cover never comes closer than this to the frame edges

# Mask (sizes relative to the line height, in analysis pixels)
MASK_PRESENCE = 0.25      # pixel kept if masked in this share of the caption's frames
SHADOW_DARKER = 22        # shadow/outline: this much darker than the local background
SHADOW_REACH = 0.35       # shadow pixels must be this close to a glyph
MASK_GROW = 0.20          # final safety margin around the mask (anti-aliased edges, glow)
GLYPH_CLOSE = 0.35        # closes glyph outlines so their interior is masked
MASK_COVER_RANGE = (0.04, 0.60)  # auto mode: mask share of the region considered reliable

# Fill
FILL_SIGMA = 0.45         # smoothing of the background estimate
EDGE_FEATHER = 0.08       # soft blend at the mask border

# Frosted glass panel (a clean rounded rectangle, centred on the frame)
FROST_SIGMA = 0.8         # x line height: blur strength
FROST_LIGHTEN = 0.10      # share of white mixed in (a light, glassy look)
FROST_CORNER = 0.22       # x panel height: corner radius
FROST_EDGE_PX = 1.5       # anti-aliasing of the panel edge, in full-size pixels

ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]


class CaptionError(Exception):
    """A failure with a human-readable reason."""


def log(message: str) -> None:
    print(f"[remove_captions] {message}", file=sys.stderr)


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, check=False)


def odd(value: float, minimum: int = 3) -> int:
    return max(minimum, int(round(value))) | 1


def probe(path: Path) -> dict:
    result = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)])
    if result.returncode != 0:
        raise CaptionError(f"ffprobe failed for {path}: {result.stderr.decode(errors='replace').strip()}")
    data = json.loads(result.stdout or b"{}")
    stream = (data.get("streams") or [{}])[0]
    if "width" not in stream:
        raise CaptionError(f"No video stream in {path}")
    num, _, den = stream.get("r_frame_rate", "30/1").partition("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "rate": stream.get("r_frame_rate", "30/1"),
        "fps": float(num) / float(den or 1),
        "duration": float(data.get("format", {}).get("duration", 0) or 0),
    }


def decode(path: Path, vf: str, width: int, height: int) -> list[np.ndarray]:
    """Decode all frames after the filter vf as BGR arrays of width x height."""
    result = run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf,
                  "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    if result.returncode != 0:
        raise CaptionError(f"ffmpeg could not decode {path}: {result.stderr.decode(errors='replace').strip()}")
    size = width * height * 3
    count = len(result.stdout) // size
    return list(np.frombuffer(result.stdout[: count * size], dtype=np.uint8).reshape(count, height, width, 3))


def text_mask(frame: np.ndarray) -> np.ndarray:
    """Bright (white or saturated) pixels that sit next to much darker pixels (analysis scale)."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = np.maximum(grey, hsv[:, :, 2])
    hue = hsv[:, :, 0]
    # Highlighted caption words are yellow/orange/red; other bright saturated colours
    # (a blue sweater, a green screen) must not count as text.
    warm = (hue <= WARM_HUE_MAX) | (hue >= WARM_HUE_WRAP)
    coloured = (hsv[:, :, 2] >= BRIGHT_MIN) & (hsv[:, :, 1] >= COLOUR_SAT_MIN) & warm
    bright = (grey >= BRIGHT_MIN) | coloured
    darkest = cv2.erode(grey, np.ones((5, 5), np.uint8))
    contrast = cv2.subtract(value, darkest) >= CONTRAST_MIN
    return (bright & contrast).astype(np.uint8) * 255


def find_lines(frame: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Return caption-like line boxes (x, y, w, h) as fractions of the frame."""
    h, w = frame.shape[:2]
    mask = text_mask(frame)
    mask[: int(ZONE_TOP * h)] = 0
    mask[int(ZONE_BOTTOM * h):] = 0

    # Join glyphs of a line horizontally, and the strokes of a glyph vertically.
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 30), 3)))
    joined = cv2.dilate(joined, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    lines = []
    for i in range(1, count):
        x, y, bw, bh, _ = stats[i]
        if not (LINE_H_MIN * h <= bh <= LINE_H_MAX * h):
            continue
        if bw < MIN_ASPECT * bh or bw > MAX_LINE_WIDTH * w:
            continue
        if not (CENTER_X_MIN <= (x + bw / 2) / w <= CENTER_X_MAX):
            continue
        fill = cv2.countNonZero(mask[y:y + bh, x:x + bw]) / float(bw * bh)
        if not (FILL_MIN <= fill <= FILL_MAX):
            continue
        lines.append((x / w, y / h, bw / w, bh / h))
    return lines


def dominant_line(candidates: dict[str, list]) -> tuple[float, float] | None:
    """(center y, height) of the line most clips put captions on, or None."""
    clips_per_bin: dict[int, set] = {}
    for name, items in candidates.items():
        for _, (_, y, _, h) in items:
            clips_per_bin.setdefault(int((y + h / 2) / LINE_BIN), set()).add(name)
    if not clips_per_bin:
        return None
    scores = {b: len(set().union(*(clips_per_bin.get(b + d, set()) for d in (-1, 0, 1))))
              for b in clips_per_bin}
    best = max(scores, key=lambda b: scores[b])
    if scores[best] < MIN_CLIPS_ON_LINE:
        return None
    center = (best + 0.5) * LINE_BIN
    heights = [h for items in candidates.values() for _, (_, y, _, h) in items
               if abs(y + h / 2 - center) <= 1.5 * LINE_BIN]
    return center, float(np.median(heights))


def clip_region(items: list, line: tuple[float, float], duration: float) -> dict | None:
    """Merge one clip's on-line boxes into a single region (whole sentence) with a time span."""
    center, height = line
    kept = [(t, box) for t, box in items
            if abs(box[1] + box[3] / 2 - center) <= LINE_Y_TOLERANCE * height
            and LINE_H_RANGE[0] * height <= box[3] <= LINE_H_RANGE[1] * height]
    if len({t for t, _ in kept}) < MIN_HITS_PER_CLIP:
        return None
    # Captions are centred on the frame. A box that also caught something bright on ONE
    # side (an icon, a light) is longer on that side only, so per frame the half width is
    # the side closer to the centre; the sentence is the widest of those over the clip.
    halves = [min(0.5 - b[0], b[0] + b[2] - 0.5) for _, b in kept]
    halves = [h for h in halves if h > 0]
    if halves:
        half = max(halves)
        x0, x1 = 0.5 - half, 0.5 + half
    else:  # boxes not across the centre (not a centred caption): keep them as found
        x0 = min(b[0] for _, b in kept)
        x1 = max(b[0] + b[2] for _, b in kept)
    # Vertically: at least the caption band, more when the text on the line is taller.
    y0 = min([center - height / 2] + [b[1] for _, b in kept])
    y1 = max([center + height / 2] + [b[1] + b[3] for _, b in kept])
    step = 1.0 / SAMPLE_FPS
    start = max(0.0, min(t for t, _ in kept) - step)
    end = min(duration, max(t for t, _ in kept) + 2 * step)
    if start <= EDGE_SNAP_S:
        start = 0.0          # caption already on screen at the cut
    if end >= duration - EDGE_SNAP_S:
        end = duration       # caption still on screen at the end
    # Never reach the frame edges: keep EDGE_MARGIN on both sides.
    rx0 = max(EDGE_MARGIN, x0 - PAD_X * height)
    rx1 = min(1 - EDGE_MARGIN, x1 + PAD_X * height)
    return {
        "x": rx0, "w": rx1 - rx0,
        "y": y0 - PAD_Y * height, "h": (y1 - y0) + 2 * PAD_Y * height,
        "line_h": height, "start": round(start, 3), "end": round(end, 3), "hits": len(kept),
    }


def band_boxes(frame: np.ndarray, line: tuple[float, float]) -> tuple[float, float, float, float] | None:
    """Caption box found by text density around the caption line, or None.

    Works where line detection fails (text touching other bright edges, big text on the
    caption line). The text height is measured from the rows dense with text pixels around
    the line, so text up to LINE_H_RANGE[1] x the caption height is accepted; thin long
    edges (an arm, a table) have a low row density and do not count.
    """
    h, w = frame.shape[:2]
    center, height = line[0] * h, line[1] * h
    reach = int(BAND_REACH * height)
    lo, hi = max(0, int(center - reach)), min(h, int(center + reach))
    mask = text_mask(frame) > 0
    x0c, x1c = int((1 - BAND_COLUMNS) / 2 * w), int((1 + BAND_COLUMNS) / 2 * w)
    rows = mask[lo:hi, x0c:x1c].mean(axis=1)
    if rows.size == 0 or rows.max() < BAND_DENSITY_MIN:
        return None
    # Rows of the text: the dense run that contains (or is closest to) the line centre.
    dense = rows >= max(BAND_DENSITY_MIN / 2, BAND_ROW_SHARE * rows.max())
    dense = cv2.morphologyEx(dense.astype(np.uint8)[:, None], cv2.MORPH_CLOSE,
                             np.ones((max(3, int(0.25 * height)), 1), np.uint8))[:, 0] > 0
    runs, start = [], None
    for y, v in enumerate(list(dense) + [False]):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append((start, y))
            start = None
    if not runs:
        return None
    c = center - lo
    top, bottom = min(runs, key=lambda r: 0 if r[0] <= c <= r[1] else min(abs(r[0] - c), abs(r[1] - c)))
    text_h = bottom - top
    if not (LINE_H_RANGE[0] * height <= text_h <= LINE_H_RANGE[1] * height):
        return None
    if abs((top + bottom) / 2 - c) > LINE_Y_TOLERANCE * height:
        return None
    top, bottom = top + lo, bottom + lo

    band = mask[top:bottom]
    # Columns with text, joined across the gaps between glyphs.
    cols = band.mean(axis=0) >= BAND_COLUMN_MIN
    cols = cv2.morphologyEx(cols.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
                            np.ones((1, max(3, int(1.2 * height))), np.uint8))[0]
    col_runs, start = [], None
    for x, v in enumerate(list(cols) + [0]):
        if v and start is None:
            start = x
        elif not v and start is not None:
            col_runs.append((start, x))
            start = None
    col_runs = [(a, b) for a, b in col_runs if CENTER_X_MIN * w <= (a + b) / 2 <= CENTER_X_MAX * w
                and b - a >= BAND_MIN_ASPECT * text_h]
    if not col_runs:
        return None
    x0, x1 = max(col_runs, key=lambda r: r[1] - r[0])
    density = band[:, x0:x1].mean()
    # Thin strips right outside the text rows: glyphs of a taller text would cross them.
    edge = max(2, int(BAND_EDGE * height))
    above = mask[max(0, top - edge):top, x0:x1].mean() if top > 0 else 0.0
    below = mask[bottom:bottom + edge, x0:x1].mean() if bottom < h else 0.0
    if density < BAND_DENSITY_MIN or density < BAND_CONTRAST * max(above, below, 1e-6):
        return None
    return x0 / w, top / h, (x1 - x0) / w, (bottom - top) / h


def to_pixels(region: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """Region fractions -> pixel box inside the frame, sides divisible by 4."""
    x = max(0, int(region["x"] * width) // 4 * 4)
    y = max(0, int(region["y"] * height) // 4 * 4)
    w = min(width - x, int(region["w"] * width + 3) // 4 * 4) // 4 * 4
    h = min(height - y, int(region["h"] * height + 3) // 4 * 4) // 4 * 4
    return x, y, w, h


def in_window(index: int, fps: float, region: dict) -> bool:
    return region["start"] <= index / fps <= region["end"]


def caption_mask(crop: np.ndarray, line_px: float) -> np.ndarray:
    """Glyphs + drop shadow/outline + glow of one region crop (analysis scale)."""
    glyphs = text_mask(crop)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(grey, odd(line_px * 1.6))
    darker = cv2.subtract(background, grey) >= SHADOW_DARKER
    near = cv2.dilate(glyphs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(2 * SHADOW_REACH * line_px),) * 2)) > 0
    return ((glyphs > 0) | (darker & near)).astype(np.uint8)


def build_mask(clip: Path, info: dict, box: tuple, region: dict, scale: float) -> np.ndarray:
    """Stable 0/1 mask (analysis-scale region) for the caption's time span."""
    x, y, w, h = box
    sw, sh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    crops = decode(clip, f"crop={w}:{h}:{x}:{y},scale={sw}:{sh}", sw, sh)
    frames = [c for i, c in enumerate(crops) if in_window(i, info["fps"], region)] or crops
    line_px = region["line_h"] * info["height"] * scale
    seen = np.zeros((sh, sw), np.float32)
    for crop in frames:
        seen += caption_mask(crop, line_px)
    mask = ((seen / len(frames)) >= MASK_PRESENCE).astype(np.uint8)
    # Thick glyphs are only detected along their outline: close the strokes so their
    # bright interior is masked too (else it shows through the glass as a ghost).
    close = odd(GLYPH_CLOSE * line_px)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    grow = odd(MASK_GROW * line_px)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))


def fill_region(roi: np.ndarray, mask: np.ndarray, soft: np.ndarray, line_px: float) -> np.ndarray:
    """Replace masked pixels by a smooth estimate from the unmasked background around them."""
    keep = (1 - mask).astype(np.float32)
    data = roi.astype(np.float32)
    if not keep.any():
        return roi
    # One smoothing scale for every pixel (mixing scales leaves visible seams), with a
    # wide second pass blended in smoothly where the first has too little background.
    sigma = max(1.5, FILL_SIGMA * line_px)
    weight = cv2.GaussianBlur(keep, (0, 0), sigma)
    total = cv2.GaussianBlur(data * keep[:, :, None], (0, 0), sigma)
    wide_weight = cv2.GaussianBlur(keep, (0, 0), sigma * 4)
    wide_total = cv2.GaussianBlur(data * keep[:, :, None], (0, 0), sigma * 4)
    near = total / np.maximum(weight, 1e-4)[:, :, None]
    far = wide_total / np.maximum(wide_weight, 1e-4)[:, :, None]
    trust = np.clip(weight / 0.15, 0, 1)[:, :, None]  # 1 = enough background nearby
    estimate = near * trust + far * (1 - trust)
    return np.clip(estimate * soft + data * (1 - soft), 0, 255).astype(np.uint8)


def panel_alpha(w: int, h: int) -> np.ndarray:
    """Rounded-rectangle panel shape (0..1, HxWx1) with a crisp, anti-aliased edge."""
    inset = int(np.ceil(FROST_EDGE_PX * 2))
    radius = max(2, int(FROST_CORNER * h))
    x0, y0, x1, y1 = inset, inset, w - 1 - inset, h - 1 - inset
    radius = max(1, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    alpha = np.zeros((h, w), np.uint8)
    cv2.rectangle(alpha, (x0 + radius, y0), (x1 - radius, y1), 255, -1)
    cv2.rectangle(alpha, (x0, y0 + radius), (x1, y1 - radius), 255, -1)
    for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
        cv2.circle(alpha, (cx, cy), radius, 255, -1, lineType=cv2.LINE_AA)
    alpha = cv2.GaussianBlur(alpha, (0, 0), FROST_EDGE_PX / 2)
    return (alpha.astype(np.float32) / 255.0)[:, :, None]


def frost(image: np.ndarray, line_px: float) -> np.ndarray:
    """Frosted-glass look: strong blur, slightly lightened (float32 result)."""
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), max(2.0, FROST_SIGMA * line_px))
    return blurred * (1 - FROST_LIGHTEN) + 255 * FROST_LIGHTEN


def process_clip(clip: Path, target: Path, info: dict, box: tuple, region: dict, method: str,
                 mask: np.ndarray | None, scale: float, preview: Path | None) -> None:
    """Fix the caption region in Python and let ffmpeg overlay it on the full-size clip.

    method "panel": remove the glyphs (when a mask is given), then cover the region with
    a frosted-glass rounded rectangle. method "fill": only remove the glyphs.
    """
    x, y, w, h = box
    sw, sh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    line_px = region["line_h"] * info["height"] * scale
    roi_size = w * h * 3

    if mask is not None:
        border = odd(EDGE_FEATHER * line_px * 2)
        soft_small = cv2.GaussianBlur(mask.astype(np.float32), (border, border), 0)
        soft_small = np.maximum(soft_small, mask.astype(np.float32))[:, :, None]
        soft_full = cv2.resize(soft_small, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
        ones = np.ones_like(soft_small)
    if method == "panel":
        alpha_full = panel_alpha(w, h)

    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-vf", f"crop={w}:{h}:{x}:{y}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    tmp = target.with_name(target.stem + ".tmp.mp4")
    enable = f"between(t,{region['start']},{region['end']})"
    with tempfile.TemporaryFile() as encoder_log:
        encoder = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-i", str(clip),
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-framerate", info["rate"], "-i", "-",
             "-filter_complex", f"[0:v][1:v]overlay={x}:{y}:eof_action=pass:enable='{enable}'[v]",
             "-map", "[v]", "-map", "0:a?", *ENCODE, "-c:a", "copy", "-movflags", "+faststart", str(tmp)],
            stdin=subprocess.PIPE, stderr=encoder_log)
        index, preview_pending = 0, preview is not None
        try:
            while True:
                raw = decoder.stdout.read(roi_size)
                if len(raw) < roi_size:
                    break
                roi = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                if in_window(index, info["fps"], region):
                    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
                    if method == "fill":
                        filled = fill_region(small, mask, ones, line_px)
                        filled_full = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                        fixed = (filled_full * soft_full + roi.astype(np.float32) * (1 - soft_full)).astype(np.uint8)
                    else:
                        if mask is not None:  # remove the glyphs first so no ghost shows through the glass
                            small = fill_region(small, mask, soft_small, line_px)
                        glass = cv2.resize(frost(small, line_px), (w, h), interpolation=cv2.INTER_CUBIC)
                        fixed = np.clip(glass * alpha_full + roi.astype(np.float32) * (1 - alpha_full),
                                        0, 255).astype(np.uint8)
                    if preview_pending:
                        save_preview(roi, fixed, preview)
                        preview_pending = False
                    roi = fixed
                encoder.stdin.write(roi.tobytes())
                index += 1
        except BrokenPipeError:
            pass
        finally:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
            decoder.stdout.close()
            decoder.wait()
            code = encoder.wait()
        if code != 0 or index == 0:
            encoder_log.seek(0)
            tmp.unlink(missing_ok=True)
            raise CaptionError(f"Encoding failed for {clip.name}: {encoder_log.read().decode(errors='replace')[-2000:]}")
    tmp.replace(target)


def save_preview(before: np.ndarray, after: np.ndarray, path: Path) -> None:
    """Region before (top) and after (bottom), scaled to 960 px wide."""
    pair = np.vstack([before, np.full((8, before.shape[1], 3), 255, np.uint8), after])
    scale = 960 / pair.shape[1]
    cv2.imwrite(str(path), cv2.resize(pair, (960, max(1, int(pair.shape[0] * scale))), interpolation=cv2.INTER_AREA))


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and remove small burned-in captions in scene clips.")
    parser.add_argument("sources_dir", type=Path, help="Directory with <N>.mp4 scene clips")
    parser.add_argument("output_dir", type=Path, help="Directory for the cleaned clips")
    parser.add_argument("--mode", choices=("panel", "fill"), default="panel",
                        help="panel (default): frosted-glass rectangle over the removed caption; "
                             "fill: only remove the glyphs")
    parser.add_argument("--preview", type=Path, help="Directory for before/after preview images")
    args = parser.parse_args()

    sources = args.sources_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    preview_dir = args.preview.expanduser().resolve() if args.preview else None
    try:
        if not sources.is_dir():
            raise CaptionError(f"Sources directory not found: {sources}")
        if out_dir == sources:
            raise CaptionError("output_dir must differ from sources_dir (originals must be kept)")
        clips = sorted((p for p in sources.iterdir()
                        if p.is_file() and p.suffix.lower() == ".mp4" and p.stem.isdigit()),
                       key=lambda p: int(p.stem))
        if not clips:
            raise CaptionError(f"No <N>.mp4 clips in {sources}")

        # Pass 1: candidate caption lines in sampled, downscaled frames of every clip.
        infos, candidates = {}, {}
        for clip in clips:
            info = probe(clip)
            a_h = int(round(info["height"] * ANALYSIS_WIDTH / info["width"] / 2)) * 2
            sampled = decode(clip, f"fps={SAMPLE_FPS},scale={ANALYSIS_WIDTH}:{a_h}", ANALYSIS_WIDTH, a_h)
            infos[clip.name] = info
            candidates[clip.name] = [(i / SAMPLE_FPS, box) for i, f in enumerate(sampled) for box in find_lines(f)]

        # Pass 2: keep candidates on the video's caption line.
        line = dominant_line(candidates)
        if line is None:
            log("No caption line found; clips are copied unchanged")
        else:
            log(f"Caption line at {line[0] * 100:.1f}% of the height, line height {line[1] * 100:.1f}%")

        out_dir.mkdir(parents=True, exist_ok=True)
        if preview_dir:
            preview_dir.mkdir(parents=True, exist_ok=True)
        report = {"mode": args.mode, "caption_line": None, "clips": {}}
        if line:
            report["caption_line"] = {"center_y": round(line[0], 4), "height": round(line[1], 4)}

        counts = {"panel": 0, "fill": 0}
        for clip in clips:
            info = infos[clip.name]
            target = out_dir / clip.name
            region = clip_region(candidates[clip.name], line, info["duration"]) if line else None
            detection = "line"
            if region is None and line:
                # Fallback: text density inside the known caption band.
                a_h = int(round(info["height"] * ANALYSIS_WIDTH / info["width"] / 2)) * 2
                sampled = decode(clip, f"fps={SAMPLE_FPS},scale={ANALYSIS_WIDTH}:{a_h}", ANALYSIS_WIDTH, a_h)
                found = [(i / SAMPLE_FPS, b) for i, f in enumerate(sampled) if (b := band_boxes(f, line))]
                region = clip_region(found, line, info["duration"])
                detection = "band"
            if region is None:
                shutil.copyfile(clip, target)
                report["clips"][clip.name] = {"caption": False}
                continue

            # The region is already centred on the frame (see clip_region) and inside the margins.
            box = to_pixels(region, info["width"], info["height"])
            scale = ANALYSIS_WIDTH / info["width"]
            method = args.mode
            mask = build_mask(clip, info, box, region, scale)
            share = float(mask.mean())
            low, high = MASK_COVER_RANGE
            if not (low <= share <= high):
                if method == "fill":
                    method = "panel"  # unreliable mask: never leave a smeared shape, use the panel
                mask = None  # panel only blurs, without removing glyphs first
            preview = preview_dir / f"{clip.stem}.jpg" if preview_dir else None
            process_clip(clip, target, info, box, region, method, mask, scale, preview)
            counts[method] += 1
            report["clips"][clip.name] = {
                "caption": True, "method": method, "glyphs_removed": mask is not None, "detection": detection,
                "box": dict(zip(("x", "y", "w", "h"), box)),
                "start": region["start"], "end": region["end"], "hits": region["hits"],
                "mask_share": None if share is None else round(share, 3),
            }
            log(f"{clip.name}: {method} ({detection}) box={box} {region['start']:.2f}-{region['end']:.2f}s"
                + ("" if share is None else f" mask={share:.2f}"))

        (out_dir / "captions.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"{len(clips)} clips: {counts['panel']} panel, {counts['fill']} fill, "
            f"{len(clips) - counts['panel'] - counts['fill']} without caption")
    except CaptionError as exc:
        log(str(exc))
        print("fail")
        return 1
    except Exception as exc:  # noqa: BLE001 - always answer dir/fail
        log(f"Unexpected error: {exc!r}")
        print("fail")
        return 1

    print(f"dir:{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
