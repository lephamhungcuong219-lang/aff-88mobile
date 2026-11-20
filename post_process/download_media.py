#!/usr/bin/env python3
"""Download a video and its music, re-encoding HEVC video to H.264.

Usage:
    python download_media.py <aweme_id> <video_url> <music_url> <output_dir>

Files:
    <output_dir>/<aweme_id>.mp4
    <output_dir>/<aweme_id>.mp3

If the downloaded video is HEVC (H.265), it is re-encoded to H.264 + AAC MP4,
which plays everywhere, and the HEVC download is deleted. Other codecs are
kept as downloaded.

Stdout: 'mp4:<absolute path of the final mp4>' on success, or exactly 'fail'.
Stderr: progress and the reason for a failure (human-readable).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.error import URLError

# Douyin's CDN rejects requests without a browser-like User-Agent and Referer.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Referer": "https://www.douyin.com/",
}
TIMEOUT_S = 60
CHUNK = 1 << 16



class MediaError(Exception):
    """A failure with a human-readable reason."""


def log(message: str) -> None:
    print(f"[download_media] {message}", file=sys.stderr)


def download(url: str, target: Path) -> None:
    """Stream url to target via a .part file so failures leave no half file."""
    part = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, part.open("wb") as f:
            shutil.copyfileobj(response, f, CHUNK)
        if part.stat().st_size == 0:
            raise MediaError(f"Empty response from {url}")
        part.replace(target)
    except (URLError, OSError, TimeoutError, ValueError) as exc:
        part.unlink(missing_ok=True)
        raise MediaError(f"Download failed ({exc}) for {url}") from exc
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def video_codec(path: Path) -> str:
    """Return the codec name of the first video stream (e.g. 'hevc', 'h264')."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, errors="replace", check=False,
    )
    codec = result.stdout.strip()
    if result.returncode != 0 or not codec:
        raise MediaError(f"ffprobe could not read a video stream in {path}: {result.stderr.strip()}")
    return codec


def reencode_to_h264(source: Path, target: Path) -> None:
    """Re-encode to H.264/AAC MP4 with faststart (plays in browsers and players)."""
    tmp = target.with_name(target.stem + ".encoding.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(tmp),
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise MediaError(f"ffmpeg re-encode failed for {source}:\n{result.stderr.strip()}")
    tmp.replace(target)


def safe_name(value: str) -> str:
    """Use the id as a filename; reject anything that could escape the folder."""
    name = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise MediaError(f"awemeId must contain only letters, digits, '_' or '-': {value!r}")
    return name


def fetch_video(url: str, video: Path) -> None:
    # Downloaded file is temporary: deleted after a HEVC re-encode (or on error).
    raw = video.with_name(video.stem + ".download.mp4")
    try:
        download(url, raw)
        codec = video_codec(raw)
        if codec == "hevc":
            log("video is HEVC, re-encoding to H.264")
            reencode_to_h264(raw, video)
        else:
            log(f"video codec is {codec}, keeping as is")
            raw.replace(video)
    finally:
        raw.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a video and its music; re-encode HEVC to H.264.")
    parser.add_argument("aweme_id", help="awemeId, used as the file name")
    parser.add_argument("video_url", help="URL of the video (e.g. videoUrlNoWatermark)")
    parser.add_argument("music_url", help="URL of the music file (e.g. musicUrl)")
    parser.add_argument("output_dir", type=Path, help="Directory for all downloaded files")
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    try:
        name = safe_name(args.aweme_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        video = out_dir / f"{name}.mp4"
        fetch_video(args.video_url, video)
        download(args.music_url, out_dir / f"{name}.mp3")
    except MediaError as exc:
        log(str(exc))
        print("fail")
        return 1
    except Exception as exc:  # noqa: BLE001 - always answer done/fail
        log(f"Unexpected error: {exc!r}")
        print("fail")
        return 1

    print(f"mp4:{video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
