#!/usr/bin/env python3
"""Detect small burned-in captions in scene clips, and optionally remove them (no AI).

Usage:
    python remove_captions.py <sources_dir> <output_dir> [--mode detect|panel|fill] [--preview DIR]

detect (default) only writes the report <output_dir>/captions.json: where and when
each clip shows a caption ("text_box": the text itself, no padding). No clip is
written; compose_videos.py covers the captions with its subtitle glass panel, so
the final videos are rendered straight from <sources_dir> and encoded once less.

panel / fill also process every <sources_dir>/<N>.mp4 into <output_dir>/<N>.mp4:
re-encoded with the caption removed, or copied as is when no caption is found.

Finding captions (classic image processing, OpenCV + numpy only):
  1. Each clip is sampled at SAMPLE_FPS, downscaled to ANALYSIS_WIDTH.
  2. Caption glyphs are bright (white, or saturated colour for highlighted
     words) and outlined/shadowed with dark pixels, so a pixel counts as "text"
     when it is bright AND much brighter than its darkest neighbour. A second, stricter
     mask only takes the pure white core of the glyphs (and coloured words): on bright
     backgrounds it keeps shadows and edges from joining two rows of text. Both are
     tried; the one that finds more caption rows in the clips is used for the video.
  3. Text pixels are joined horizontally into lines; a line is a candidate when
     it is caption-sized (small, wide, horizontally centred), anywhere in the frame
     (captions can be at the top or at the bottom). Big overlay titles are too tall
     and are ignored. Lines stacked in one frame are one caption of 2-3 rows.
  4. Captions of one video sit at ONE fixed place. Candidates of all clips are
     pooled and the place with the most clips and the widest text wins. Text that
     never changes (a fixed title, a logo) and small scattered marks are not captions.
     The layouts of that place (1, 2 or 3 rows) are measured, and only candidates
     fitting one of them are kept.
  5. Per clip, the kept boxes are merged into one region covering the whole
     sentence, with the time span in which it is visible. Its height is always
     the height of the clip's layout (1, 2 or 3 rows of normal text). Big text on
     the caption line is not a caption: it never changes the region.

Removing captions (panel / fill only; only inside the region and time span):
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

--preview DIR saves, per clip with a caption, the region before/after (panel / fill),
or one frame with the detected text box drawn in green (detect).

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
CORE_MIN = 245            # strict mask: every colour channel of a white glyph's core
CORE_REACH = 7            # ... with its dark outline/shadow within this many pixels (window size)
CONTRAST_MIN = 70         # glyph must be this much brighter than its darkest neighbour

# Caption geometry, as fractions of the frame
ZONE_TOP, ZONE_BOTTOM = 0.03, 0.95
LINE_H_MIN, LINE_H_MAX = 0.010, 0.060  # a spoken-line caption; bigger text is not a caption
CENTER_X_MIN, CENTER_X_MAX = 0.30, 0.70
MIN_ASPECT = 2.5          # width / height of a caption line
ROW_GAP = 1.2             # x line height: pieces of one row (letter-spaced text) at most this far apart
MAX_LINE_WIDTH = 0.995    # x width: long captions can run almost edge to edge
FILL_MIN, FILL_MAX = 0.06, 0.65

# Dominant caption line
LINE_BIN = 0.01           # y histogram bin
MIN_CLIPS_ON_LINE = 3     # fewer clips than this on one line -> no captions
MIN_CLIP_SHARE = 0.15     # ... or fewer than this share of the clips
MIN_LINE_WIDTH = 0.15     # x width: median text width on the line (marks on objects are narrower)
MAX_LINE_OFFSET = 0.06    # x width: the line's text is centred on the frame, within this
STATIC_MIN_CLIPS = 5      # with at least this many clips on the line, look for text that never changes:
STATIC_IOU = 0.5          # ... median overlap of the text of two clips above this -> fixed title / logo
LINE_Y_TOLERANCE = 0.6    # x median line height
LINE_H_RANGE = (0.6, 1.35) # accepted height relative to the median line height
EDGE_SNAP_S = 0.4         # caption found this close to the clip start/end: cover up to the cut
MIN_HITS_PER_CLIP = 2     # samples a caption must appear in

# Captions of several rows
MAX_ROWS = 3
ROW_PITCH = (0.7, 1.6)    # distance of two rows' centres, x line height
SHORT_ROW_ASPECT = 0.5    # a row next to a normal row may be as short as one character
ROW_CENTER = 0.05         # x width: every row of a block is centred on the frame, within this
LAYOUT_MIN_CLIPS = 2      # a layout other than the dominant one must be seen in this many clips
LAYOUT_REACH = 1.2        # ... with its centre this close to the dominant one's, x line height
LAYOUT_Y_TOLERANCE = 0.3  # a block belongs to its layout within this, x line height (several layouts)
ROW_GAP_DENSITY = 0.5     # band fallback: text density between two rows, relative to the rows

# Band fallback: for frames where no line-shaped caption was found (e.g. the caption touches
# bright background), look for text pixels inside the rows of the caption line.
BAND_COLUMN_MIN = 0.04    # a column belongs to the text when this share of its rows is text
BAND_MIN_ASPECT = 1.8     # width / height: short captions of two characters are accepted
BAND_DENSITY_MIN = 0.06   # text pixel share inside the band (over the text's width)
BAND_CONTRAST = 2.0       # band density must exceed the strips above/below this many times
                          # (normal captions: 2.3 and more; big text crossing the strips: below 1.4)
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
FROST_CORNER = 0.22       # x panel height of ONE row: corner radius (bigger corners would show text)
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


def text_mask(frame: np.ndarray, core: bool = False) -> np.ndarray:
    """Bright (white or saturated) pixels that sit next to much darker pixels (analysis scale).

    core: only the pure white core of the glyphs and coloured words (see the module text).
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = np.maximum(grey, hsv[:, :, 2])
    hue = hsv[:, :, 0]
    # Highlighted caption words are yellow/orange/red; other bright saturated colours
    # (a blue sweater, a green screen) must not count as text.
    warm = (hue <= WARM_HUE_MAX) | (hue >= WARM_HUE_WRAP)
    coloured = (hsv[:, :, 2] >= BRIGHT_MIN) & (hsv[:, :, 1] >= COLOUR_SAT_MIN) & warm
    bright = (frame.min(axis=2) >= CORE_MIN if core else grey >= BRIGHT_MIN) | coloured
    reach = CORE_REACH if core else 5
    darkest = cv2.erode(grey, np.ones((reach, reach), np.uint8))
    contrast = cv2.subtract(value, darkest) >= CONTRAST_MIN
    return (bright & contrast).astype(np.uint8) * 255


def center_of(box: tuple) -> float:
    return box[1] + box[3] / 2


def stacked(a: tuple, b: tuple) -> bool:
    """True for two rows of one caption: centred, as tall, right above each other."""
    if any(abs(box[0] + box[2] / 2 - 0.5) > ROW_CENTER for box in (a, b)):
        return False
    low, high = sorted((a[3], b[3]))
    return (low >= LINE_H_RANGE[0] * high
            and ROW_PITCH[0] * high <= abs(center_of(a) - center_of(b)) <= ROW_PITCH[1] * high)


def join_pieces(pieces: list[tuple]) -> list[tuple]:
    """Join pieces of one row (pixel boxes x, y, w, h): same height and place, small gap."""
    rows: list[list[int]] = []  # x0, y0, x1, y1
    for x, y, w, h in sorted(pieces):
        for row in rows:
            row_h = row[3] - row[1]
            low, high = sorted((h, row_h))
            if (low >= LINE_H_RANGE[0] * high and abs((y + h / 2) - (row[1] + row[3]) / 2) <= 0.25 * high
                    and x - row[2] <= ROW_GAP * high):
                row[1], row[2], row[3] = min(row[1], y), max(row[2], x + w), max(row[3], y + h)
                break
        else:
            rows.append([x, y, x + w, y + h])
    return [(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in rows]


def find_lines(text: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Caption-like line boxes (x, y, w, h), as fractions of the frame, in a text_mask."""
    h, w = text.shape
    mask = text.copy()
    mask[: int(ZONE_TOP * h)] = 0
    mask[int(ZONE_BOTTOM * h):] = 0

    # Join glyphs of a line horizontally, and the strokes of a glyph vertically.
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 30), 3)))
    joined = cv2.dilate(joined, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    pieces = [tuple(int(v) for v in stats[i][:4]) for i in range(1, count)
              if LINE_H_MIN * h <= stats[i][3] <= LINE_H_MAX * h and stats[i][2] >= SHORT_ROW_ASPECT * stats[i][3]]
    lines, short = [], []
    for x, y, bw, bh in join_pieces(pieces):
        if bh > LINE_H_MAX * h or bw > MAX_LINE_WIDTH * w:
            continue
        if not (CENTER_X_MIN <= (x + bw / 2) / w <= CENTER_X_MAX):
            continue
        fill = cv2.countNonZero(mask[y:y + bh, x:x + bw]) / float(bw * bh)
        if not (FILL_MIN <= fill <= FILL_MAX):
            continue
        (lines if bw >= MIN_ASPECT * bh else short).append((x / w, y / h, bw / w, bh / h))
    # A last row of one or two characters: only right above/below a normal row.
    return lines + [box for box in short if any(stacked(box, other) for other in lines)]


def group_rows(lines: list[tuple]) -> list[tuple[tuple, ...]]:
    """One frame's line boxes -> blocks: rows stacked right above each other are one caption."""
    others, centred = [], []
    for box in sorted(lines, key=center_of):
        if abs(box[0] + box[2] / 2 - 0.5) > ROW_CENTER:
            others.append([box])
            continue
        if centred and len(centred[-1]) < MAX_ROWS and stacked(centred[-1][-1], box):
            centred[-1].append(box)
        else:
            centred.append([box])
    return [tuple(b) for b in others + centred]


def block_center(block: tuple) -> float:
    return (block[0][1] + block[-1][1] + block[-1][3]) / 2


def block_box(block: tuple) -> tuple[float, float, float, float]:
    x0, x1 = min(b[0] for b in block), max(b[0] + b[2] for b in block)
    return x0, block[0][1], x1 - x0, block[-1][1] + block[-1][3] - block[0][1]


def normal_rows(block: tuple, height: float) -> bool:
    """True when every row is as tall as a normal sentence."""
    return all(LINE_H_RANGE[0] * height <= r[3] <= LINE_H_RANGE[1] * height for r in block)


def is_static(times: dict[str, set], masks: dict[str, list], center: float, height: float) -> bool:
    """True when the text in these rows is the same in most clips (a fixed title, a logo).

    times: per clip, the samples with text there.
    """
    shapes = []
    for name, seen in times.items():
        if len(seen) < MIN_HITS_PER_CLIP:
            continue
        rows = masks[name][0].shape[0]
        top, bottom = max(0, int((center - height / 2) * rows)), int(np.ceil((center + height / 2) * rows))
        share = np.mean([np.unpackbits(masks[name][int(round(t * SAMPLE_FPS))][top:bottom], axis=1,
                                       count=ANALYSIS_WIDTH) for t in seen], axis=0)
        shapes.append(share >= 0.6)
    if len(shapes) < STATIC_MIN_CLIPS:
        return False
    overlaps = [np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum())
                for i, a in enumerate(shapes) for b in shapes[i + 1:]]
    return float(np.median(overlaps)) > STATIC_IOU


def dominant_line(blocks: dict[str, list], masks: dict[str, list]) -> tuple[float, float, int] | None:
    """(center y, row height, rows) of the place the captions of this video are at, or None.

    blocks: per clip, (time, block) of group_rows. The place with the most clips wins, each
    clip counting as much as its text is wide: captions are in many clips and wide, marks on
    objects are in few and narrow. Centre and height are medians over the normal sentences:
    boxes much taller than the rest (a caption joined to bright background, bigger text) are
    left out.
    """
    per_bin: dict[tuple, dict[str, list]] = {}  # (rows, y bin) -> clip -> text widths
    for name, items in blocks.items():
        for _, b in items:
            key = (len(b), int(block_center(b) / LINE_BIN))
            per_bin.setdefault(key, {}).setdefault(name, []).append(block_box(b)[2])

    def near(key: tuple) -> dict[str, list]:
        clips: dict[str, list] = {}
        for d in (-1, 0, 1):
            for name, widths in per_bin.get((key[0], key[1] + d), {}).items():
                clips.setdefault(name, []).extend(widths)
        return clips

    scores = {key: sum(float(np.median(w)) for w in near(key).values()) for key in per_bin}
    fixed: list[tuple[float, float]] = []  # (top, bottom) of text that never changes
    for key in sorted(scores, key=lambda k: -scores[k]):
        rows, middle = key[0], (key[1] + 0.5) * LINE_BIN
        if (len(near(key)) < max(MIN_CLIPS_ON_LINE, MIN_CLIP_SHARE * len(blocks))
                or any(top <= middle <= bottom for top, bottom in fixed)):
            continue
        found = [(name, t, b) for name, items in blocks.items() for t, b in items
                 if len(b) == rows and abs(block_center(b) - middle) <= 1.5 * LINE_BIN]
        typical = float(np.median([r[3] for _, _, b in found for r in b]))
        normal = [(name, t, b) for name, t, b in found if normal_rows(b, typical)]
        if not normal:
            continue
        center = float(np.median([block_center(b) for _, _, b in normal]))
        height = float(np.median([r[3] for _, _, b in normal for r in b]))
        boxes = [block_box(b) for _, _, b in normal]
        if (float(np.median([b[2] for b in boxes])) < MIN_LINE_WIDTH
                or abs(float(np.median([b[0] + b[2] / 2 for b in boxes])) - 0.5) > MAX_LINE_OFFSET):
            continue
        times: dict[str, set] = {}
        for name, t, _ in normal:
            times.setdefault(name, set()).add(t)
        tall = float(np.median([b[3] for b in boxes]))
        if is_static(times, masks, center, tall):
            fixed.append((center - tall, center + tall))
            log(f"Text at {center * 100:.1f}% of the height never changes: not a caption")
            continue
        return center, height, rows
    return None


def caption_layouts(blocks: dict[str, list], line: tuple[float, float, int]) -> list[dict]:
    """Layouts of this video's captions, fewest rows first: rows, top, bottom, row centres."""
    center, height, main_rows = line
    layouts: dict[int, dict] = {}

    def measure(rows: int, kept: list[tuple]) -> dict:
        centers = [float(np.median([center_of(b[i]) for b in kept])) for i in range(rows)]
        return {"rows": rows, "centers": centers, "center": (centers[0] + centers[-1]) / 2,
                "top": centers[0] - height / 2, "bottom": centers[-1] + height / 2}

    def agreed(found: dict[str, list]) -> list[tuple]:
        """Blocks at the place most of these clips agree on; [] when too few clips have one."""
        found = {n: v for n, v in found.items() if len(v) >= MIN_HITS_PER_CLIP}
        if len(found) < LAYOUT_MIN_CLIPS:
            return []
        middle = float(np.median([np.median([block_center(b) for b in v]) for v in found.values()]))
        found = {n: kept for n, v in found.items()
                 if (kept := [b for b in v if abs(block_center(b) - middle) <= LAYOUT_Y_TOLERANCE * height])}
        return [b for v in found.values() for b in v] if len(found) >= LAYOUT_MIN_CLIPS else []

    for rows in sorted(range(1, MAX_ROWS + 1), key=lambda r: r != main_rows):  # the dominant one first
        reach = (LAYOUT_Y_TOLERANCE if rows == main_rows else LAYOUT_REACH) * height
        found: dict[str, list] = {}
        for name, items in blocks.items():
            for _, b in items:
                if len(b) != rows or not normal_rows(b, height) or abs(block_center(b) - center) > reach:
                    continue
                if rows == 1 and main_rows > 1 and any(abs(block_center(b) - c) <= LAYOUT_Y_TOLERANCE * height
                                                       for c in layouts[main_rows]["centers"]):
                    continue  # one row of the dominant layout, its other rows were missed
                found.setdefault(name, []).append(b)
        kept = [b for v in found.values() for b in v] if rows == main_rows else agreed(found)
        if kept:
            layouts[rows] = measure(rows, kept)
        elif rows == main_rows:
            layouts[rows] = {"rows": rows, "centers": [center], "center": center,
                             "top": center - height / 2, "bottom": center + height / 2}

    # The dominant "one row" can be a row of captions of several rows whose other rows are
    # often missed. It is, when one-row captions are also found at another place.
    others = [c for r, m in layouts.items() if r > 1 for c in m["centers"]]
    if main_rows == 1 and any(abs(center - c) <= LAYOUT_Y_TOLERANCE * height for c in others):
        found = {}
        for name, items in blocks.items():
            for _, b in items:
                if (len(b) == 1 and normal_rows(b, height) and abs(block_center(b) - center) <= LAYOUT_REACH * height
                        and all(abs(block_center(b) - c) > LAYOUT_Y_TOLERANCE * height for c in others)):
                    found.setdefault(name, []).append(b)
        kept = agreed(found)
        if kept:
            layouts[1] = measure(1, kept)
    return [layouts[r] for r in sorted(layouts)]


def match_layout(block: tuple, layouts: list[dict], height: float) -> dict | None:
    """The layout a block of normal text belongs to, or None."""
    if not normal_rows(block, height):
        return None
    center = block_center(block)
    if len(block) > 1:
        return next((m for m in layouts if m["rows"] == len(block)
                     and abs(center - m["center"]) <= LAYOUT_Y_TOLERANCE * height), None)
    # One row: a one-row caption, or one row of a bigger caption whose other rows were missed.
    # The nearest place wins; when (almost) equally near, the layout with fewer rows.
    places = [(abs(center - c), m["rows"], m) for m in layouts for c in m["centers"]]
    distance, _, layout = min(places, key=lambda p: (round(p[0] / (0.1 * height)), p[1]))
    tolerance = LINE_Y_TOLERANCE if len(places) == 1 else LAYOUT_Y_TOLERANCE
    return layout if distance <= tolerance * height else None


def clip_region(blocks: list, bands: list, layouts: list[dict], height: float, duration: float) -> dict | None:
    """Merge one clip's boxes into a single region (whole sentence) with a time span.

    blocks: (time, block) of group_rows; bands: (time, box, layout) of band_box. The width
    comes from the lines (tighter) when enough samples have one, otherwise from the bands as
    well; the time span from both. The height is that of the layouts (1-3 rows) the clip shows.
    """
    lines = [(t, block_box(b), m) for t, b in blocks if (m := match_layout(b, layouts, height))]
    seen = lines + bands
    if len({t for t, _, _ in seen}) < MIN_HITS_PER_CLIP:
        return None
    enough = len({t for t, _, _ in lines}) >= MIN_HITS_PER_CLIP
    kept = lines if enough else seen
    # Captions are centred on the frame. A box that also caught something bright on ONE
    # side (an icon, a light) is longer on that side only, so per frame the half width is
    # the side closer to the centre; the sentence is the widest of those over the clip.
    halves = [min(0.5 - b[0], b[0] + b[2] - 0.5) for _, b, _ in kept]
    halves = [h for h in halves if h > 0]
    if halves:
        half = max(halves)
        x0, x1 = 0.5 - half, 0.5 + half
    else:  # boxes not across the centre (not a centred caption): keep them as found
        x0 = min(b[0] for _, b, _ in kept)
        x1 = max(b[0] + b[2] for _, b, _ in kept)
    # Layouts the clip shows (it can show a caption of one row, then one of two rows).
    hits = {m["rows"]: len({t for t, _, k in seen if k is m}) for m in layouts}
    shown = ([m for m in layouts if hits[m["rows"]] >= MIN_HITS_PER_CLIP]
             or [max(layouts, key=lambda m: (hits[m["rows"]], m["rows"]))])
    y0, y1 = min(m["top"] for m in shown), max(m["bottom"] for m in shown)
    step = 1.0 / SAMPLE_FPS
    start = max(0.0, min(t for t, _, _ in seen) - step)
    end = min(duration, max(t for t, _, _ in seen) + 2 * step)
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
        "text": (x0, y0, x1 - x0, y1 - y0),  # the text itself: no padding, no edge margin
        "line_h": height, "start": round(start, 3), "end": round(end, 3), "hits": len({t for t, _, _ in seen}),
        "detection": "line" if enough else "band", "rows": max(m["rows"] for m in shown),
    }


def band_box(text: np.ndarray, layouts: list[dict], height: float) -> tuple[tuple, dict] | None:
    """(box, layout) of a caption found by text density inside the rows of a layout, or None.

    Works where line detection fails (text touching other bright edges). Only the rows of
    normal sentences are looked at, so the box never gets taller than the layout. Big text is
    rejected: its glyphs go on above/below those rows, or fill the gap between two rows.
    """
    for layout in layouts:
        box = layout_band(text, layout, height)
        if box:
            return box, layout
    return None


def layout_band(text: np.ndarray, layout: dict, height: float) -> tuple[float, float, float, float] | None:
    h, w = text.shape
    line_px = height * h
    top, bottom = max(0, int(round(layout["top"] * h))), min(h, int(round(layout["bottom"] * h)))
    if bottom <= top:
        return None
    mask = text > 0

    band = mask[top:bottom]
    # Columns with text, joined across the gaps between glyphs.
    cols = band.mean(axis=0) >= BAND_COLUMN_MIN
    cols = cv2.morphologyEx(cols.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
                            np.ones((1, max(3, int(1.2 * line_px))), np.uint8))[0]
    col_runs, start = [], None
    for x, v in enumerate(list(cols) + [0]):
        if v and start is None:
            start = x
        elif not v and start is not None:
            col_runs.append((start, x))
            start = None
    col_runs = [(a, b) for a, b in col_runs if CENTER_X_MIN * w <= (a + b) / 2 <= CENTER_X_MAX * w
                and b - a >= BAND_MIN_ASPECT * min(bottom - top, int(np.ceil(line_px)))]
    if not col_runs:
        return None
    x0, x1 = max(col_runs, key=lambda r: r[1] - r[0])
    density = band[:, x0:x1].mean()
    # Thin strips right outside the rows: the glyphs of a bigger text cross them.
    edge = max(2, int(BAND_EDGE * line_px))
    above = mask[max(0, top - edge):top, x0:x1].mean() if top > 0 else 0.0
    below = mask[bottom:bottom + edge, x0:x1].mean() if bottom < h else 0.0
    if density < BAND_DENSITY_MIN or density < BAND_CONTRAST * max(above, below, 1e-6):
        return None
    by_row = band[:, x0:x1].mean(axis=1)
    centers = [c * h - top for c in layout["centers"]]
    for a, b in zip(centers, centers[1:]):  # several rows: there is a gap between them
        middle, reach = (a + b) / 2, max(1.0, 0.25 * (b - a))
        if by_row[int(middle - reach):int(middle + reach) + 1].min() > ROW_GAP_DENSITY * density:
            return None
    return x0 / w, top / h, (x1 - x0) / w, (bottom - top) / h


def to_pixels(region: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """Region fractions -> pixel box inside the frame, sides divisible by 4."""
    x = max(0, int(region["x"] * width) // 4 * 4)
    y = max(0, int(region["y"] * height) // 4 * 4)
    w = min(width - x, int(region["w"] * width + 3) // 4 * 4) // 4 * 4
    h = min(height - y, int(region["h"] * height + 3) // 4 * 4) // 4 * 4
    return x, y, w, h


def text_pixels(region: dict, width: int, height: int) -> dict:
    """The text box (fractions) -> pixels, rounded outwards, inside the frame."""
    x, y, w, h = region["text"]
    x0, y0 = max(0, int(np.floor(x * width))), max(0, int(np.floor(y * height)))
    x1, y1 = min(width, int(np.ceil((x + w) * width))), min(height, int(np.ceil((y + h) * height)))
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def save_detect_preview(clip: Path, region: dict, text: dict, path: Path) -> None:
    """One frame in the middle of the caption's time span, text box drawn in green (960 px wide)."""
    t = (region["start"] + region["end"]) / 2
    result = run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(clip), "-frames:v", "1",
                  "-f", "image2pipe", "-vcodec", "png", "-"])
    frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR) if result.stdout else None
    if frame is None:
        return
    cv2.rectangle(frame, (text["x"], text["y"]), (text["x"] + text["w"], text["y"] + text["h"]),
                  (0, 255, 0), max(2, frame.shape[1] // 400))
    scale = 960 / frame.shape[1]
    cv2.imwrite(str(path), cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA))


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


def panel_alpha(w: int, h: int, line_px: float) -> np.ndarray:
    """Rounded-rectangle panel shape (0..1, HxWx1) with a crisp, anti-aliased edge."""
    inset = int(np.ceil(FROST_EDGE_PX * 2))
    radius = max(2, int(FROST_CORNER * min(h, (1 + 2 * PAD_Y) * line_px)))
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
        alpha_full = panel_alpha(w, h, region["line_h"] * info["height"])

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
    parser.add_argument("output_dir", type=Path, help="Directory for captions.json (and the cleaned clips: panel/fill)")
    parser.add_argument("--mode", choices=("detect", "panel", "fill"), default="detect",
                        help="detect (default): only write captions.json (compose_videos covers the captions); "
                             "panel: frosted-glass rectangle over the removed caption; fill: only remove the glyphs")
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

        # Pass 1: candidate caption lines in sampled, downscaled frames of every clip, with
        # the normal and with the strict (core) text mask.
        infos: dict[str, dict] = {}
        tries = [{"core": core, "blocks": {}, "masks": {}} for core in (False, True)]
        for clip in clips:
            info = probe(clip)
            a_h = int(round(info["height"] * ANALYSIS_WIDTH / info["width"] / 2)) * 2
            sampled = decode(clip, f"fps={SAMPLE_FPS},scale={ANALYSIS_WIDTH}:{a_h}", ANALYSIS_WIDTH, a_h)
            infos[clip.name] = info
            for attempt in tries:
                texts = [text_mask(f, attempt["core"]) for f in sampled]
                attempt["blocks"][clip.name] = [(i / SAMPLE_FPS, b) for i, m in enumerate(texts)
                                                for b in group_rows(find_lines(m))]
                attempt["masks"][clip.name] = [np.packbits(m > 0, axis=1) for m in texts]  # 1 bit per pixel

        # Pass 2: keep candidates on the video's caption line; the mask finding more rows wins.
        for attempt in tries:
            attempt["line"] = dominant_line(attempt["blocks"], attempt["masks"])
            attempt["layouts"] = caption_layouts(attempt["blocks"], attempt["line"]) if attempt["line"] else []
            attempt["regions"] = {}
            for name in infos if attempt["line"] else ():
                texts = (np.unpackbits(m, axis=1, count=ANALYSIS_WIDTH) for m in attempt["masks"][name])
                bands = [(i / SAMPLE_FPS, *b) for i, m in enumerate(texts)
                         if (b := band_box(m, attempt["layouts"], attempt["line"][1]))]
                region = clip_region(attempt["blocks"][name], bands, attempt["layouts"], attempt["line"][1],
                                     infos[name]["duration"])
                if region:
                    attempt["regions"][name] = region
            attempt["score"] = sum(r["rows"] for r in attempt["regions"].values())
        best = max(tries, key=lambda a: a["score"])  # equal: the normal mask
        line, layouts, regions = best["line"], best["layouts"], best["regions"]
        if line is None:
            log("No caption line found; clips are copied unchanged")
        else:
            # Where one row of text sits (the Vietnamese subtitle goes there too).
            line = (next((m["center"] for m in layouts if m["rows"] == 1), line[0]), line[1], line[2])
            log(f"Caption line at {line[0] * 100:.1f}% of the height, line height {line[1] * 100:.1f}%, "
                f"rows: {', '.join(str(m['rows']) for m in layouts)}, mask: {'core' if best['core'] else 'normal'} "
                f"(rows found: normal {tries[0]['score']}, core {tries[1]['score']})")

        out_dir.mkdir(parents=True, exist_ok=True)
        if preview_dir:
            preview_dir.mkdir(parents=True, exist_ok=True)
        report = {"mode": args.mode, "caption_line": None, "clips": {}}
        if line:
            report["caption_line"] = {"center_y": round(line[0], 4), "height": round(line[1], 4)}
        # Pixel boxes below are in this frame size (compose_videos scales them when it upscales).
        first = infos[clips[0].name]
        report["frame"] = {"width": first["width"], "height": first["height"]}

        counts = {"panel": 0, "fill": 0, "detect": 0}
        for clip in clips:
            info = infos[clip.name]
            target = out_dir / clip.name
            region = regions.get(clip.name)
            if region is None:
                if args.mode != "detect":
                    shutil.copyfile(clip, target)
                report["clips"][clip.name] = {"caption": False}
                continue

            # The region is already centred on the frame (see clip_region) and inside the margins.
            box = to_pixels(region, info["width"], info["height"])
            text = text_pixels(region, info["width"], info["height"])
            if args.mode == "detect":
                counts["detect"] += 1
                report["clips"][clip.name] = {
                    "caption": True, "method": "detect", "glyphs_removed": False, "detection": region["detection"],
                    "box": dict(zip(("x", "y", "w", "h"), box)), "text_box": text,
                    "start": region["start"], "end": region["end"], "hits": region["hits"],
                    "rows": region["rows"],
                }
                if preview_dir:
                    save_detect_preview(clip, region, text, preview_dir / f"{clip.stem}.jpg")
                log(f"{clip.name}: detect ({region['detection']}) text={tuple(text.values())} "
                    f"{region['start']:.2f}-{region['end']:.2f}s")
                continue
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
                "caption": True, "method": method, "glyphs_removed": mask is not None, "detection": region["detection"],
                "box": dict(zip(("x", "y", "w", "h"), box)), "text_box": text,
                "start": region["start"], "end": region["end"], "hits": region["hits"],
                "rows": region["rows"],
                "mask_share": None if share is None else round(share, 3),
            }
            log(f"{clip.name}: {method} ({region['detection']}) box={box} {region['start']:.2f}-{region['end']:.2f}s"
                + ("" if share is None else f" mask={share:.2f}"))

        (out_dir / "captions.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        found = sum(counts.values())
        if args.mode == "detect":
            log(f"{len(clips)} clips: {found} with caption, {len(clips) - found} without (report only)")
        else:
            log(f"{len(clips)} clips: {counts['panel']} panel, {counts['fill']} fill, "
                f"{len(clips) - found} without caption")
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
