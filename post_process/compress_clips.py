#!/usr/bin/env python3
"""Make small copies of scene clips for sending to Gemini.

Usage:
    python compress_clips.py <clips_dir> <output_dir> [--short-side 480] [--fps 30] [--no-label]

Reads <clips_dir>/<N>.mp4 (the clips made by split_video.py) and writes
<output_dir>/<N>.mp4 with:
    - the short side scaled to --short-side px (480 -> 480x854 for 9:16 video)
    - frame rate capped at --fps
    - H.264 CRF 30 + mono AAC 64k audio (Gemini also listens to the audio)
    - "SCENE <N>" drawn in the top-left corner (unless --no-label), so the
      model can still tell scenes apart if the clips get merged into one video

Originals are never modified.

Stdout: 'dir:<absolute output_dir>' on success, or exactly 'fail'.
Stderr: progress and the reason for a failure (human-readable).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import subprocess
import sys
from pathlib import Path

FONT_FILE = Path("/usr/share/fonts/noto/NotoSans-Bold.ttf")


class CompressError(Exception):
    """A failure with a human-readable reason."""


def log(message: str) -> None:
    print(f"[compress_clips] {message}", file=sys.stderr)


def build_filter(scene: int, short_side: int, fps: int, label: bool) -> str:
    # Scale whichever side is shorter to short_side; -2 keeps the aspect ratio with an even size.
    scale = (
        f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
    )
    filters = [scale, f"fps='min(source_fps,{fps})'"]
    if label:
        font = f"fontfile='{FONT_FILE}':" if FONT_FILE.is_file() else ""
        filters.append(
            f"drawtext={font}text='SCENE {scene}':x=16:y=16:fontsize=h/18:"
            "fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8"
        )
    return ",".join(filters)


def compress(source: Path, target: Path, scene: int, short_side: int, fps: int, label: bool) -> None:
    tmp = target.with_name(target.stem + ".tmp.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-vf", build_filter(scene, short_side, fps, label),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ac", "1", "-b:a", "64k",
        "-movflags", "+faststart",
        str(tmp),
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise CompressError(f"ffmpeg failed for {source}:\n{result.stderr.strip()}")
    tmp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Make small, labelled copies of scene clips for Gemini.")
    parser.add_argument("clips_dir", type=Path, help="Directory with <N>.mp4 scene clips")
    parser.add_argument("output_dir", type=Path, help="Directory for the compressed copies")
    parser.add_argument("--short-side", type=int, default=480, help="Short side in px (default 480)")
    parser.add_argument("--fps", type=int, default=30, help="Maximum frame rate (default 30)")
    parser.add_argument("--no-label", action="store_true", help="Do not draw 'SCENE N' on the video")
    args = parser.parse_args()

    clips_dir = args.clips_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    try:
        if not clips_dir.is_dir():
            raise CompressError(f"Clips directory not found: {clips_dir}")
        if out_dir == clips_dir:
            raise CompressError("output_dir must differ from clips_dir (originals must be kept)")

        clips = sorted(
            (p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4" and p.stem.isdigit()),
            key=lambda p: int(p.stem),
        )
        if not clips:
            raise CompressError(f"No <N>.mp4 clips in {clips_dir}")

        out_dir.mkdir(parents=True, exist_ok=True)
        before = after = 0
        for clip in clips:
            scene = int(clip.stem)
            target = out_dir / f"{scene}.mp4"
            compress(clip, target, scene, args.short_side, args.fps, not args.no_label)
            before += clip.stat().st_size
            after += target.stat().st_size
        log(f"{len(clips)} clips: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    except CompressError as exc:
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
