#!/usr/bin/env python3
"""Put the Vietnamese voice and subtitles onto the final videos.

Usage:
    python compose_videos.py <tts_json> <videos_dir> <output_dir>
        [--reorder FILE] [--captions FILE]
        [--music FILE] [--music-volume 0.12] [--original-volume 0]
        [--tint dark|light] [--no-subtitles] [--video N ...] [--height 3840]

Inputs:
    tts_json    tts.json from gemini_tts.py (voice files + start/end of every line)
    videos_dir  final videos from render_videos.py (<prefix>video_<N>.mp4)
    --reorder   reorder.json (default: next to tts.json): which scenes each video uses
    --captions  captions.json from remove_captions.py (default: <tts dir>/clean/captions.json):
                where and when each scene shows a Chinese caption, so the panel covers it

Two kinds of captions.json (its "mode"):
  detect        (default pipeline) the videos still show the Chinese captions. The panel
                covers each caption's text box plus COVER_MARGIN only (no padding), and
                its glass is blurred as strongly as remove_captions' own panel used to
                be (COVER_BLUR), so no glyph shows through. This is the only cover.
  panel / fill  the videos were rendered from clips already covered by remove_captions;
                the panel grows over those old covers (OLD_COVER_OUTSET) to hide their edges.

For every video in tts.json, writes <output_dir>/<same file name> with:
  - size: upscaled (lanczos) to --height px high (default 3840, i.e. 4K 2160x3840 for 9:16)
    when the video is smaller; never downscaled. The panel and the text are drawn after
    the upscale, at full resolution, so they stay sharp. --height 0 keeps the video size.
  - audio: the voice track only (original sound removed). Optional background music
    (--music, looped, faded out) and/or the original sound at --original-volume.
  - subtitles: every line of tts.json shown from its start to its end, white bold
    text centred on a frosted-glass panel on the caption line.
The glass panel has ONE fixed size for the whole video (it never grows or shrinks): the
  widest and the tallest of (a) every Vietnamese line with its padding and (b) every
  Chinese caption cover from remove_captions.py in this video's scenes (captions.json,
  mapped to this video's time with reorder.json), centred on the frame. It is shown from
  the first to the last frame; between two lines only the text disappears.
  The text is centred in the panel on the body of the letters (cap height), not on the
  accents and descenders, so it looks vertically centred.
  Text width is measured by rendering it with the real font; when the longest line
  does not fit MAX_PANEL_WIDTH, the video uses a smaller font.

When all videos are done, <output_dir>/stats.txt is written for the person who uploads them
(final_report.py). A problem with that file is only logged: the videos are fine.

Stdout: 'dir:<absolute output_dir>' on success, or exactly 'fail'.
Stderr: progress and the reason for a failure (human-readable).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

try:  # python -m post_process.compose_videos
    from post_process import final_report
except ImportError:  # python post_process/compose_videos.py
    import final_report

FONT_FILE = Path("/usr/share/fonts/noto/NotoSans-Bold.ttf")

# Sizes relative to the video height / width
FONT_SIZE = 0.026         # x height: subtitle font size
MIN_FONT_SIZE = 0.018     # never smaller than this; longer lines fail instead
MAX_PANEL_WIDTH = 0.90    # x width: a subtitle panel is never wider (longer lines get a smaller font)
OLD_COVER_OUTSET = 0.003  # x height: grow old caption covers a little so their edges are hidden (panel/fill)
COVER_MARGIN = 0.15       # x caption line height: safety margin around a Chinese text box (detect):
                          # outline/shadow of the glyphs, samples the detection missed
COVER_BLUR = 0.85         # x caption line height: glass blur over raw Chinese captions (detect)
COVER_EDGE_MARGIN = 0.03  # x width: a Chinese cover never comes closer to the frame edges (as remove_captions'
                          # own panel): a text box reaching the edge is bright background, not text (detect)
PAD_X = 0.5               # x font size: space left/right of the longest line
PAD_Y = 0.25              # x font size: space above/below the text
DEFAULT_LINE_Y = 0.725    # caption line centre when captions.json is missing

# Glass look (same family as remove_captions.py)
BLUR_SIGMA = 0.012        # x height
TINTS = {"dark": 0.72, "light": 1.10}  # brightness factor of the glass
GLASS_SATURATION = 0.4    # colour kept in the glass (0 = grey): hides coloured caption leftovers
CORNER = 0.22             # x panel height
SHADOW = 0.06             # x font size: text shadow offset

TARGET_HEIGHT = 3840      # default --height: 4K for vertical (9:16) videos

ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]


class ComposeError(Exception):
    """A failure with a human-readable reason."""


def log(message: str) -> None:
    print(f"[compose_videos] {message}", file=sys.stderr)


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, check=False)


def probe(path: Path) -> dict:
    result = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)])
    if result.returncode != 0:
        raise ComposeError(f"ffprobe failed for {path}: {result.stderr.decode(errors='replace').strip()}")
    data = json.loads(result.stdout or b"{}")
    stream = (data.get("streams") or [{}])[0]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "duration": float(data.get("format", {}).get("duration", 0) or 0)}


def ffmpeg_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument."""
    return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def measure_text(text_file: Path, font_size: int) -> tuple[int, int, int]:
    """(width, top, bottom) in px of a text rendered with y_align=font at y=0."""
    canvas_w, canvas_h = font_size * 60, font_size * 3
    result = run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=black:s={canvas_w}x{canvas_h}:d=0.1",
                  "-vf", f"drawtext=fontfile={ffmpeg_path(FONT_FILE)}:textfile={ffmpeg_path(text_file)}:"
                         f"fontsize={font_size}:fontcolor=white:x=10:y={font_size // 2}:y_align=font",
                  "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    if result.returncode != 0:
        raise ComposeError(f"Could not measure text: {result.stderr.decode(errors='replace').strip()}")
    image = np.frombuffer(result.stdout[: canvas_w * canvas_h], np.uint8).reshape(canvas_h, canvas_w)
    cols = np.nonzero(image.max(axis=0) > 40)[0]
    rows = np.nonzero(image.max(axis=1) > 40)[0]
    if cols.size == 0:
        return 0, 0, 0
    return int(cols[-1] - cols[0] + 1), int(rows[0] - font_size // 2), int(rows[-1] - font_size // 2)


Rect = tuple[int, int, int, int]  # x, y, w, h in pixels


def union(a: Rect | None, b: Rect | None) -> Rect | None:
    if a is None or b is None:
        return a or b
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return x0, y0, x1 - x0, y1 - y0


def even(rect: Rect, width: int, height: int) -> Rect:
    """Even position/size, inside the frame."""
    x, y = max(0, rect[0] // 2 * 2), max(0, rect[1] // 2 * 2)
    w = min(width - x, (rect[0] + rect[2] - x + 1) // 2 * 2)
    h = min(height - y, (rect[1] + rect[3] - y + 1) // 2 * 2)
    return x, y, w, h


def detect_mode(captions: dict | None) -> bool:
    """True when captions.json only reports the captions: the videos still show them."""
    return (captions or {}).get("mode") == "detect"


def old_covers(plan_video: dict | None, captions: dict | None, width: int, height: int,
               scale: float = 1.0) -> list[tuple[float, float, Rect]]:
    """When and where a Chinese caption must be covered, in this video's time.

    Boxes in captions.json are in pixels of the clips; scale maps them to this video
    (the clips' size -> the composed size, e.g. 2.0 for 1080x1920 upscaled to 4K).

    detect: the text box plus COVER_MARGIN, kept COVER_EDGE_MARGIN off the frame edges.
    panel/fill: remove_captions' own cover plus OLD_COVER_OUTSET, so its edge is hidden.
    """
    if not plan_video or not captions:
        return []
    clips = captions.get("clips", {})
    if detect_mode(captions):
        line_h = float((captions.get("caption_line") or {}).get("height", 0.05))
        outset, key = int(round(COVER_MARGIN * line_h * height)), "text_box"
    else:
        outset, key = int(OLD_COVER_OUTSET * height), "box"
    covers = []
    for scene in plan_video.get("scenes", []):
        info = clips.get(scene.get("source_file", ""))
        if not (info and info.get("caption") and (info.get(key) or info.get("box"))):
            continue
        b = {k: int(round(v * scale)) for k, v in (info.get(key) or info["box"]).items()}
        start = float(scene["out_start"]) + float(info.get("start", 0))
        end = min(float(scene["out_end"]), float(scene["out_start"]) + float(info.get("end", scene["duration"])))
        if end > start:
            x0, x1 = b["x"] - outset, b["x"] + b["w"] + outset
            if detect_mode(captions):
                x0, x1 = max(x0, int(COVER_EDGE_MARGIN * width)), min(x1, int((1 - COVER_EDGE_MARGIN) * width))
            covers.append((start, end, (x0, b["y"] - outset, x1 - x0, b["h"] + 2 * outset)))
    return covers


def plan_panels(items: list[tuple[float, float, Rect]], duration: float, width: int,
                height: int) -> list[tuple[Rect, list[tuple[float, float]]]]:
    """ONE glass panel for the whole video, shown from the first to the last frame.

    Its size never changes: the widest and the tallest of everything it must hold or hide
    (every Vietnamese line with padding, every Chinese caption cover), centred on the frame.
    Between two lines only the text disappears; the glass stays.
    """
    if not items:
        return []
    size = None
    for _, _, r in items:
        size = union(size, r)
    half = max(width / 2 - size[0], size[0] + size[2] - width / 2)
    fixed = (int(width / 2 - half), size[1], int(2 * half), size[3])
    return [(even(fixed, width, height), [(0.0, round(duration, 3))])]


def panel_mask(path: Path, w: int, h: int, row_h: float) -> None:
    """Rounded-rectangle alpha mask (white = glass) with anti-aliased edges.

    The corners are those of a panel of one row (row_h): a taller panel, over a caption of
    several rows, must not show text in its corners.
    """
    scale = 4  # draw big, shrink: smooth corners
    big = np.zeros((h * scale, w * scale), np.uint8)
    r = max(2, int(CORNER * min(h, row_h) * scale))
    cv2.rectangle(big, (r, 0), (w * scale - 1 - r, h * scale - 1), 255, -1)
    cv2.rectangle(big, (0, r), (w * scale - 1, h * scale - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w * scale - 1 - r, r), (r, h * scale - 1 - r), (w * scale - 1 - r, h * scale - 1 - r)):
        cv2.circle(big, (cx, cy), r, 255, -1)
    cv2.imwrite(str(path), cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA))


def compose(video: dict, source: Path, target: Path, voice: Path, plan_video: dict | None,
            captions: dict | None, args: argparse.Namespace, work: Path) -> dict:
    info = probe(source)
    width, height, duration = info["width"], info["height"], info["duration"]
    index = video["video_index"]
    segments = video.get("segments") or []

    filters, inputs = [], ["-i", str(source), "-i", str(voice)]
    layout = None
    # Upscale first (never down): everything below is drawn at the final size.
    base = "[0:v]"
    if args.height and height < args.height:
        out_h = args.height // 2 * 2
        out_w = max(2, round(width * out_h / height / 2) * 2)
        filters.append(f"[0:v]scale={out_w}:{out_h}:flags=lanczos,setsar=1[up]")
        base, width, height = "[up]", out_w, out_h
    # captions.json boxes are in pixels of the clips (its "frame"; older reports have none:
    # the clips are as big as the rendered video).
    frame_h = float((captions or {}).get("frame", {}).get("height") or info["height"])
    # Chinese captions to cover: in detect mode they are still in the video, so the panel
    # is drawn even with --no-subtitles.
    covers = old_covers(plan_video, captions, width, height, height / frame_h)
    with_text = not args.no_subtitles and bool(segments)
    with_panel = with_text or bool(covers and detect_mode(captions))
    if with_panel:
        items, files, font = [], [], 0
        line_y = ((captions or {}).get("caption_line") or {}).get("center_y", DEFAULT_LINE_Y) * height
    if with_text:
        # Text files (no escaping problems) and measurements.
        for i, seg in enumerate(segments):
            path = work / f"v{index}_line{i}.txt"
            path.write_text(seg["text"], encoding="utf-8")
            files.append(path)
        font = int(round(FONT_SIZE * height))
        max_text = MAX_PANEL_WIDTH * width - 2 * PAD_X * font
        widths = [measure_text(p, font)[0] for p in files]
        if max(widths) > max_text:
            font = int(font * max_text / max(widths))
            if font < MIN_FONT_SIZE * height:
                longest = segments[widths.index(max(widths))]["text"]
                raise ComposeError(f"Video {index}: line too long for one subtitle line: {longest!r}")
            widths = [measure_text(p, font)[0] for p in files]
        reference = work / "ref.txt"
        reference.write_text("ẮẾỸÁgjpqy", encoding="utf-8")
        _, top, bottom = measure_text(reference, font)
        text_h = bottom - top

        # What the panel must hold: each Vietnamese line with its padding.
        for seg, text_w in zip(segments, widths):
            w = text_w + 2 * PAD_X * font
            h = text_h + 2 * PAD_Y * font
            items.append((float(seg["start"]), float(seg["end"]),
                          (int(width / 2 - w / 2), int(line_y - h / 2), int(w), int(h))))
    if with_panel:
        # ... and what it must hide: every Chinese caption of this video's scenes.
        # One fixed panel, as big as the biggest of them (see plan_panels).
        items += covers
        panels = plan_panels(items, duration, width, height)

        cap_h = float(((captions or {}).get("caption_line") or {}).get("height", 0.05)) * height
        row_h = max((1 + 2 * COVER_MARGIN) * cap_h, text_h + 2 * PAD_Y * font if with_text else 0)
        tint = TINTS[args.tint]
        blur = BLUR_SIGMA * height
        if covers and detect_mode(captions):
            line_h = float((captions.get("caption_line") or {}).get("height", 0.05))
            blur = max(blur, COVER_BLUR * line_h * height)  # the raw glyphs must not show through
        if panels:
            labels = "".join(f"[cut{k}]" for k in range(len(panels)))
            filters.append(f"{base}split={len(panels) + 1}[base]{labels}")
        chain = "[base]" if panels else base
        first_mask_input = 2
        for k, ((px, py, pw, ph), spans) in enumerate(panels):
            mask = work / f"v{index}_mask{k}.png"
            panel_mask(mask, pw, ph, row_h)
            inputs += ["-loop", "1", "-t", f"{duration:.3f}", "-i", str(mask)]
            # Desaturated glass: the colour of a removed caption (e.g. yellow) does not show through.
            filters.append(f"[cut{k}]crop={pw}:{ph}:{px}:{py},gblur=sigma={blur:.1f},"
                           f"hue=s={GLASS_SATURATION},"
                           f"colorchannelmixer=rr={tint}:gg={tint}:bb={tint},format=rgba[glass{k}]")
            filters.append(f"[{first_mask_input + k}:v]format=gray,scale={pw}:{ph}[mask{k}]")
            filters.append(f"[glass{k}][mask{k}]alphamerge[panel{k}]")
            enable = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in spans)
            out = f"[step{k}]"
            filters.append(f"{chain}[panel{k}]overlay=x={px}:y={py}:shortest=1:enable='{enable}'{out}")
            chain = out
        if panels:
            filters[-1] = filters[-1][: -len(chain)] + "[panelled]"
            chain = "[panelled]"
        chain += "null"

        # Text: one drawtext per line, centred in the panel. The optical centre of a line is
        # the body of the letters (cap height: top of "H" to the baseline); accents above and
        # descenders below are left out, otherwise every line sits visibly too low.
        if with_text:
            centre = panels[0][0][1] + panels[0][0][3] / 2 if panels else line_y
            cap_file = work / "cap.txt"
            cap_file.write_text("H", encoding="utf-8")
            _, cap_top, cap_bottom = measure_text(cap_file, font)
            text_y = int(round(centre - (cap_top + cap_bottom) / 2))
            shadow = max(1, int(SHADOW * font))
            for seg, path in zip(segments, files):
                chain += (f",drawtext=fontfile={ffmpeg_path(FONT_FILE)}:textfile={ffmpeg_path(path)}:"
                          f"fontsize={font}:fontcolor=white:shadowcolor=black@0.45:shadowx={shadow}:shadowy={shadow}:"
                          f"x=(w-text_w)/2:y={text_y}:y_align=font:"
                          f"enable='between(t,{seg['start']:.3f},{seg['end']:.3f})'")
        filters.append(chain + "[v]")
        layout = {"size": f"{width}x{height}", "font_size": font or None, "tint": args.tint, "blur": round(blur, 1),
                  "panels": [{"x": r[0], "y": r[1], "w": r[2], "h": r[3], "times": s} for r, s in panels]}
    else:
        filters.append(f"{base}null[v]")
        layout = {"size": f"{width}x{height}"}

    # Audio: voice (+ music) (+ original), exactly as long as the video.
    mix = ["[1:a]aresample=48000,apad[voice]"]
    sources = ["[voice]"]
    if args.music:
        inputs += ["-stream_loop", "-1", "-i", str(args.music)]
        music_input = len([x for x in inputs if x == "-i"]) - 1
        fade_start = max(0.0, duration - 1.0)
        mix.append(f"[{music_input}:a]aresample=48000,volume={args.music_volume},"
                   f"afade=t=out:st={fade_start:.3f}:d=1[music]")
        sources.append("[music]")
    if args.original_volume > 0:
        mix.append(f"[0:a]aresample=48000,volume={args.original_volume}[orig]")
        sources.append("[orig]")
    if len(sources) > 1:
        mix.append(f"{''.join(sources)}amix=inputs={len(sources)}:duration=first:normalize=0[a]")
    else:
        mix.append("[voice]anull[a]")
    filters += mix

    tmp = target.with_name(target.stem + ".composing.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
               "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
               *ENCODE, "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.3f}",
               "-movflags", "+faststart", str(tmp)]
    result = run(command)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise ComposeError(f"ffmpeg failed for {target.name}: {result.stderr.decode(errors='replace').strip()[-3000:]}")
    tmp.replace(target)
    out = probe(target)
    if abs(out["duration"] - duration) > 0.3:
        raise ComposeError(f"{target.name} lasts {out['duration']:.2f}s, source {duration:.2f}s")
    return {"video_index": index, "file": target.name, "duration": round(out["duration"], 3), "layout": layout}


def find_video(videos_dir: Path, index: int) -> Path:
    matches = sorted(p for p in videos_dir.glob(f"*video_{index}.mp4") if p.is_file())
    if len(matches) != 1:
        found = ", ".join(p.name for p in matches) or "none"
        raise ComposeError(f"Expected one video for index {index} in {videos_dir} (*video_{index}.mp4), found: {found}")
    return matches[0]


def load_json(path: Path | None, required: bool, label: str) -> dict | None:
    if path is None or not path.is_file():
        if required:
            raise ComposeError(f"{label} not found: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComposeError(f"Could not read {label} {path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the Vietnamese voice and subtitles to the final videos.")
    parser.add_argument("tts_json", type=Path, help="tts.json from gemini_tts.py")
    parser.add_argument("videos_dir", type=Path, help="Final videos from render_videos.py")
    parser.add_argument("output_dir", type=Path, help="Directory for the finished videos")
    parser.add_argument("--reorder", type=Path, help="reorder.json (default: next to tts.json)")
    parser.add_argument("--captions", type=Path, help="captions.json (default: <tts dir>/clean/captions.json)")
    parser.add_argument("--music", type=Path, help="Background music file (looped, faded out at the end)")
    parser.add_argument("--music-volume", type=float, default=0.12, help="Music volume (default 0.12)")
    parser.add_argument("--original-volume", type=float, default=0.0, help="Keep the original sound at this volume (default 0 = off)")
    parser.add_argument("--tint", choices=sorted(TINTS), default="dark", help="Glass tint (default dark: white text reads best)")
    parser.add_argument("--no-subtitles", action="store_true", help="Only replace the audio")
    parser.add_argument("--video", type=int, nargs="+", help="Only these video indexes")
    parser.add_argument("--height", type=int, default=TARGET_HEIGHT,
                        help=f"Upscale smaller videos to this height in px (default {TARGET_HEIGHT} = 4K); 0 keeps the size")
    args = parser.parse_args()

    tts_file = args.tts_json.expanduser().resolve()
    videos_dir = args.videos_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    try:
        if not FONT_FILE.is_file():
            raise ComposeError(f"Font not found: {FONT_FILE}")
        if out_dir == videos_dir:
            raise ComposeError("output_dir must differ from videos_dir (the source videos are kept)")
        if not videos_dir.is_dir():
            raise ComposeError(f"Videos directory not found: {videos_dir}")
        if args.music:
            args.music = args.music.expanduser().resolve()
            if not args.music.is_file():
                raise ComposeError(f"Music file not found: {args.music}")
        if args.height and args.height < 16:
            raise ComposeError(f"--height must be 0 or at least 16 px: {args.height}")
        for name in ("music_volume", "original_volume"):
            if not 0 <= getattr(args, name) <= 2:
                raise ComposeError(f"--{name.replace('_', '-')} must be between 0 and 2")

        tts = load_json(tts_file, True, "tts.json")
        plan = load_json(args.reorder.expanduser().resolve() if args.reorder else tts_file.with_name("reorder.json"),
                         False, "reorder.json")
        captions = load_json(args.captions.expanduser().resolve() if args.captions
                             else tts_file.parent / "clean" / "captions.json", False, "captions.json")
        if detect_mode(captions) and plan is None:
            raise ComposeError("captions.json says the videos still show the Chinese captions (mode detect), "
                               "but reorder.json was not found: cannot tell where to cover them")
        if plan is None or captions is None:
            log("WARNING: reorder.json or captions.json not found: the panel is sized for the subtitles only; "
                "Chinese captions still in the videos would show")
        plan_videos = {v.get("video_index"): v for v in (plan or {}).get("videos", [])}
        audio_dir = Path(tts.get("audio_dir") or tts_file.parent / "voice")

        videos = tts.get("videos") or []
        if args.video:
            videos = [v for v in videos if v.get("video_index") in args.video]
        if not videos:
            raise ComposeError(f"No videos to compose in {tts_file}")

        out_dir.mkdir(parents=True, exist_ok=True)
        # A report of a previous run must not describe videos that are being replaced.
        (out_dir / final_report.FILE_NAME).unlink(missing_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            for video in videos:
                index = video["video_index"]
                source = find_video(videos_dir, index)
                voice = audio_dir / video["audio_file"]
                if not voice.is_file():
                    raise ComposeError(f"Voice file not found: {voice}")
                log(f"Composing {source.name} ({len(video.get('segments') or [])} subtitle lines)")
                result = compose(video, source, out_dir / source.name, voice, plan_videos.get(index),
                                 captions, args, Path(tmp))
                log(f"{result['file']}: {result['duration']:.2f}s, layout {result['layout']}")
    except ComposeError as exc:
        log(str(exc))
        print("fail")
        return 1
    except Exception as exc:  # noqa: BLE001 - always answer dir/fail
        log(f"Unexpected error: {exc!r}")
        print("fail")
        return 1

    try:
        log(f"Report: {final_report.write_report(tts_file, out_dir)}")
    except Exception as exc:  # noqa: BLE001 - the videos are done: never fail because of the report
        log(f"WARNING: could not write {final_report.FILE_NAME}: {exc!r}")

    print(f"dir:{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
