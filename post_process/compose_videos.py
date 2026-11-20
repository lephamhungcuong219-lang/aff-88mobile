#!/usr/bin/env python3
"""Put the Vietnamese voice and subtitles onto the final videos.

Usage:
    python compose_videos.py <tts_json> <videos_dir> <output_dir>
        [--reorder FILE] [--captions FILE]
        [--music FILE] [--music-volume 0.12] [--original-volume 0]
        [--tint dark|light] [--no-subtitles] [--video N ...]

Inputs:
    tts_json    tts.json from gemini_tts.py (voice files + start/end of every line)
    videos_dir  final videos from render_videos.py (<prefix>video_<N>.mp4, captions removed)
    --reorder   reorder.json (default: next to tts.json): which scenes each video uses
    --captions  captions.json from remove_captions.py (default: <tts dir>/clean/captions.json),
                used so the new panel also covers the panels that hid the Chinese captions

For every video in tts.json, writes <output_dir>/<same file name> with:
  - audio: the voice track only (original sound removed). Optional background music
    (--music, looped, faded out) and/or the original sound at --original-volume.
  - subtitles: every line of tts.json shown from its start to its end, white bold
    text centred on a frosted-glass panel on the caption line.
The panel is ONE fixed rounded rectangle for the whole video, centred on the frame:
  wide enough for the longest Vietnamese line and for every caption panel left by
  remove_captions.py in the scenes of this video, so the bar never jumps size at a
  cut and no old panel edge shows. Text width is measured by rendering it with the
  real font; when the longest line does not fit, the video uses a smaller font.

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

FONT_FILE = Path("/usr/share/fonts/noto/NotoSans-Bold.ttf")

# Sizes relative to the video height / width
FONT_SIZE = 0.026         # x height: subtitle font size
MIN_FONT_SIZE = 0.018     # never smaller than this; longer lines fail instead
MAX_FONT_SIZE = 0.032     # grown up to this when the panel has room to spare
MAX_PANEL_WIDTH = 0.90    # x width
PAD_X = 0.9               # x font size: space left/right of the longest line
PAD_Y = 0.45              # x font size: space above/below the text
DEFAULT_LINE_Y = 0.725    # caption line centre when captions.json is missing

# Glass look (same family as remove_captions.py)
BLUR_SIGMA = 0.012        # x height
TINTS = {"dark": 0.72, "light": 1.10}  # brightness factor of the glass
CORNER = 0.22             # x panel height
SHADOW = 0.06             # x font size: text shadow offset

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


def caption_boxes(video: dict, plan_video: dict | None, captions: dict | None) -> list[dict]:
    """Panels left by remove_captions.py in the scenes of this video."""
    if not plan_video or not captions:
        return []
    clips = captions.get("clips", {})
    boxes = []
    for scene in plan_video.get("scenes", []):
        info = clips.get(scene.get("source_file", ""))
        if info and info.get("caption") and info.get("box"):
            boxes.append(info["box"])
    return boxes


def panel_mask(path: Path, w: int, h: int) -> None:
    """Rounded-rectangle alpha mask (white = glass) with anti-aliased edges."""
    scale = 4  # draw big, shrink: smooth corners
    big = np.zeros((h * scale, w * scale), np.uint8)
    r = max(2, int(CORNER * h * scale))
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
    if not args.no_subtitles and segments:
        # Text files (no escaping problems) and measurements.
        files = []
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

        # Panel: centred horizontally, fits the longest line and every old caption panel of
        # these scenes. Vertically it only grows where an old panel sticks out.
        line_y = (captions or {}).get("caption_line", {}).get("center_y", DEFAULT_LINE_Y) * height
        old = caption_boxes(video, plan_video, captions)
        half_w = max(widths) / 2 + PAD_X * font
        for box in old:
            half_w = max(half_w, width / 2 - box["x"], box["x"] + box["w"] - width / 2)
        half_w = min(half_w + 2, width / 2 - 2)

        # A panel made wide by old panels leaves room: grow the font (up to MAX_FONT_SIZE) to use it.
        room = 2 * half_w - 2 * PAD_X * font
        if max(widths) < room:
            grown = min(int(font * room / max(widths)), int(MAX_FONT_SIZE * height))
            if grown > font:
                font = grown
                widths = [measure_text(p, font)[0] for p in files]
        _, top, bottom = measure_text(reference, font)

        text_h = bottom - top
        y0 = line_y - text_h / 2 - PAD_Y * font
        y1 = line_y + text_h / 2 + PAD_Y * font
        for box in old:
            y0, y1 = min(y0, box["y"]), max(y1, box["y"] + box["h"])
        px, pw = int(width / 2 - half_w) // 2 * 2, int(2 * half_w) // 2 * 2
        py, ph = int(y0 - 2) // 2 * 2, int(y1 - y0 + 4) // 2 * 2
        line_y = py + ph / 2  # the text sits in the middle of the panel
        mask = work / f"v{index}_mask.png"
        panel_mask(mask, pw, ph)
        inputs += ["-loop", "1", "-t", f"{duration:.3f}", "-i", str(mask)]
        tint = TINTS[args.tint]

        # Glass: crop, blur, tint, round, overlay.
        filters.append(f"[0:v]split[base][cut]")
        filters.append(f"[cut]crop={pw}:{ph}:{px}:{py},gblur=sigma={BLUR_SIGMA * height:.1f},"
                       f"colorchannelmixer=rr={tint}:gg={tint}:bb={tint},format=rgba[glass]")
        filters.append(f"[2:v]format=gray,scale={pw}:{ph}[mask]")
        filters.append(f"[glass][mask]alphamerge[panel]")
        chain = "[base][panel]overlay=x={px}:y={py}:shortest=1".format(px=px, py=py)

        # Text: one drawtext per line, vertically centred on the panel.
        text_y = int(line_y - (top + bottom) / 2)
        shadow = max(1, int(SHADOW * font))
        for seg, path in zip(segments, files):
            chain += (f",drawtext=fontfile={ffmpeg_path(FONT_FILE)}:textfile={ffmpeg_path(path)}:"
                      f"fontsize={font}:fontcolor=white:shadowcolor=black@0.45:shadowx={shadow}:shadowy={shadow}:"
                      f"x=(w-text_w)/2:y={text_y}:y_align=font:"
                      f"enable='between(t,{seg['start']:.3f},{seg['end']:.3f})'")
        filters.append(chain + "[v]")
        layout = {"font_size": font, "panel": {"x": px, "y": py, "w": pw, "h": ph}, "tint": args.tint}
    else:
        filters.append("[0:v]null[v]")

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
        for name in ("music_volume", "original_volume"):
            if not 0 <= getattr(args, name) <= 2:
                raise ComposeError(f"--{name.replace('_', '-')} must be between 0 and 2")

        tts = load_json(tts_file, True, "tts.json")
        plan = load_json(args.reorder.expanduser().resolve() if args.reorder else tts_file.with_name("reorder.json"),
                         False, "reorder.json")
        captions = load_json(args.captions.expanduser().resolve() if args.captions
                             else tts_file.parent / "clean" / "captions.json", False, "captions.json")
        if plan is None or captions is None:
            log("reorder.json or captions.json not found: the panel is sized for the subtitles only")
        plan_videos = {v.get("video_index"): v for v in (plan or {}).get("videos", [])}
        audio_dir = Path(tts.get("audio_dir") or tts_file.parent / "voice")

        videos = tts.get("videos") or []
        if args.video:
            videos = [v for v in videos if v.get("video_index") in args.video]
        if not videos:
            raise ComposeError(f"No videos to compose in {tts_file}")

        out_dir.mkdir(parents=True, exist_ok=True)
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

    print(f"dir:{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
