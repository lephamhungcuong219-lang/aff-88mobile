#!/usr/bin/env python3
"""Render the videos planned in reorder.json by joining scene clips with ffmpeg.

Usage:
    python render_videos.py <reorder_json> <clips_dir> <output_dir>
        [--proxy-dir DIR] [--proxy-clips DIR] [--proxy-height PX]
        [--prefix TEXT] [--no-audio] [--height PX]

For every entry in reorder.json "videos", ALL its clips <clips_dir>/<source_file>
are joined, whole, in "position" order into
    <output_dir>/<prefix>video_<video_index>.mp4           final video

With --proxy-dir, the same run also renders a light copy of every video:
    <proxy_dir>/proxy_<prefix>video_<video_index>.mp4      proxy video
from --proxy-clips (default: clips_dir; the original clips, captions kept, so
Gemini can read them), --proxy-height px high (default 854, i.e. 480 px
wide for 9:16). Same scenes, same order, same length: only smaller.

Typical call (captions are covered later by compose_videos.py, see remove_captions.py
--mode detect, so the final videos and the proxies both come from the original clips):
    render_videos.py reorder.json sources/ output/ --proxy-dir proxy/ --proxy-clips sources/

Each rendered video (final and proxy) is checked against the plan: its length
must match "total_duration" within DURATION_TOLERANCE_S, otherwise the run fails.

The clips are re-encoded (H.264 + AAC, faststart) rather than stream-copied:
joining with -c copy can leave small audio gaps or glitches at every cut, which
is noticeable with ~1 s scenes. All clips are normalised to the size, frame rate
and audio format of the first clip; a clip without audio gets silence.

--no-audio drops the original sound (e.g. to add music or a voice-over later).

Stdout on success: 'dir:<absolute output_dir>', plus a second line
        'proxy:<absolute proxy_dir>' with --proxy-dir. On failure exactly 'fail'.
Stderr: progress and the reason for a failure (human-readable).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DURATION_TOLERANCE_S = 0.5  # fail when the rendered length differs more than this from the plan


class RenderError(Exception):
    """A failure with a human-readable reason."""


def log(message: str) -> None:
    print(f"[render_videos] {message}", file=sys.stderr)


def probe(path: Path) -> dict:
    """Return video size/fps and audio presence/format of a clip."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate,sample_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RenderError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise RenderError(f"No video stream in {path}")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": video.get("r_frame_rate", "30/1"),
        "has_audio": audio is not None,
        "sample_rate": int(audio["sample_rate"]) if audio else None,
        "duration": float(data.get("format", {}).get("duration", 0) or 0),
    }


def output_size(ref: dict, out_height: int | None) -> tuple[int, int]:
    """Size of the first clip, or scaled to out_height keeping the aspect ratio (even sides)."""
    if not out_height or out_height >= ref["height"]:
        return ref["width"], ref["height"]
    width = round(ref["width"] * out_height / ref["height"] / 2) * 2
    return max(2, width), out_height // 2 * 2


def build_command(clips: list[Path], infos: list[dict], target: Path, audio: bool,
                  out_height: int | None = None) -> list[str]:
    ref = infos[0]
    (width, height), fps = output_size(ref, out_height), ref["fps"]
    rate = next((i["sample_rate"] for i in infos if i["sample_rate"]), 44100)

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        command += ["-i", str(clip)]

    filters, labels = [], []
    silence_inputs = 0
    for index, info in enumerate(infos):
        # Only resample the frame rate when it differs: the fps filter rounds each
        # clip's end up to a frame boundary, adding a little length per cut.
        fps_filter = f"fps={fps}," if info["fps"] != fps else ""
        filters.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,{fps_filter}format=yuv420p,"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        labels.append(f"[v{index}]")
        if not audio:
            continue
        if info["has_audio"]:
            source = f"[{index}:a]"
        else:
            # Extra silent input, exactly as long as this clip.
            silence = len(clips) + silence_inputs
            silence_inputs += 1
            command += ["-f", "lavfi", "-t", f"{info['duration']:.3f}",
                        "-i", f"anullsrc=channel_layout=stereo:sample_rate={rate}"]
            source = f"[{silence}:a]"
        filters.append(
            f"{source}aresample={rate},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        labels.append(f"[a{index}]")

    streams = "a=1" if audio else "a=0"
    outputs = "[outv][outa]" if audio else "[outv]"
    filters.append(f"{''.join(labels)}concat=n={len(clips)}:v=1:{streams}{outputs}")

    command += ["-filter_complex", ";".join(filters), "-map", "[outv]"]
    if audio:
        command += ["-map", "[outa]", "-c:a", "aac", "-b:a", "192k"]
    command += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(target),
    ]
    return command


def render_video(video: dict, sources_dir: Path, target: Path, audio: bool, out_height: int | None) -> float:
    scenes = sorted(video.get("scenes") or [], key=lambda s: s.get("position", 0))
    if not scenes:
        raise RenderError(f"Video {video.get('video_index')} has no scenes")
    planned_count = video.get("scene_count")
    if planned_count is not None and int(planned_count) != len(scenes):
        raise RenderError(f"Video {video.get('video_index')}: scene_count {planned_count} "
                          f"but {len(scenes)} scenes listed in reorder.json")

    clips = []
    for scene in scenes:
        name = Path(str(scene.get("source_file", ""))).name  # never leave sources_dir
        clip = sources_dir / name
        if not name or not clip.is_file():
            raise RenderError(f"Missing clip for scene {scene.get('scene_number')}: {clip}")
        clips.append(clip)
    infos = [probe(clip) for clip in clips]

    tmp = target.with_name(target.stem + ".rendering.mp4")
    command = build_command(clips, infos, tmp, audio, out_height)
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RenderError(f"ffmpeg failed for {target.name}:\n{result.stderr.strip()[-3000:]}")
    tmp.replace(target)
    return probe(target)["duration"]


def render_checked(video: dict, clips_dir: Path, target: Path, audio: bool, height: int | None) -> None:
    """Render one video and fail if its length does not match the plan."""
    count = len(video.get("scenes") or [])
    log(f"Rendering {target.name} ({count} scenes)")
    duration = render_video(video, clips_dir, target, audio, height)
    expected = float(video.get("total_duration") or 0)
    if expected and abs(duration - expected) > DURATION_TOLERANCE_S:
        raise RenderError(f"{target.name} lasts {duration:.2f}s but the plan says {expected:.2f}s "
                          f"(tolerance {DURATION_TOLERANCE_S}s): scenes may be missing or wrong clips used")
    log(f"{target.name}: {duration:.2f}s, {count} scenes (planned {expected:.2f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Join scene clips into the videos planned in reorder.json.")
    parser.add_argument("reorder_json", type=Path, help="reorder.json from gemini_reorder.py")
    parser.add_argument("clips_dir", type=Path, help="Clips for the final videos (normally sources)")
    parser.add_argument("output_dir", type=Path, help="Directory for the final videos")
    parser.add_argument("--proxy-dir", type=Path, help="Also render light proxy videos into this directory")
    parser.add_argument("--proxy-clips", type=Path, help="Clips for the proxies (default: clips_dir; normally sources)")
    parser.add_argument("--proxy-height", type=int, default=854, help="Proxy height in px (default 854)")
    parser.add_argument("--prefix", default="", help="File name prefix, e.g. the awemeId and '_'")
    parser.add_argument("--no-audio", action="store_true", help="Drop the original audio")
    parser.add_argument("--height", type=int, help="Final video height in px (default: source size)")
    args = parser.parse_args()

    reorder = args.reorder_json.expanduser().resolve()
    clips_dir = args.clips_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    proxy_dir = args.proxy_dir.expanduser().resolve() if args.proxy_dir else None
    proxy_clips = args.proxy_clips.expanduser().resolve() if args.proxy_clips else clips_dir

    try:
        for name, value in (("--height", args.height), ("--proxy-height", args.proxy_height)):
            if value is not None and value < 16:
                raise RenderError(f"{name} must be at least 16 px: {value}")
        if args.prefix and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.prefix):
            raise RenderError(f"--prefix may contain only letters, digits, '_', '-', '.': {args.prefix!r}")
        for name, folder in (("Clips", clips_dir), ("Proxy clips", proxy_clips if proxy_dir else clips_dir)):
            if not folder.is_dir():
                raise RenderError(f"{name} directory not found: {folder}")
        if not reorder.is_file():
            raise RenderError(f"reorder_json must be the reorder.json file, got: {reorder}")
        if out_dir in (clips_dir, proxy_clips):
            raise RenderError(f"output_dir must not be a clips directory: {out_dir}")
        if proxy_dir and proxy_dir in (out_dir, clips_dir, proxy_clips):
            raise RenderError(f"--proxy-dir must be its own directory (not output_dir or a clips directory): {proxy_dir}")
        try:
            plan = json.loads(reorder.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RenderError(f"Could not read {reorder}: {exc}") from exc
        videos = plan.get("videos") if isinstance(plan, dict) else None
        if not videos:
            raise RenderError(f"No 'videos' in {reorder}")

        # Check every clip of every video up front, so a missing folder fails before any rendering.
        needed = sorted({Path(str(s.get("source_file", ""))).name
                         for v in videos for s in (v.get("scenes") or [])}, key=lambda n: (len(n), n))
        for label, folder in (("clips_dir", clips_dir), ("--proxy-clips", proxy_clips if proxy_dir else None)):
            if folder is None:
                continue
            missing = [n for n in needed if not n or not (folder / n).is_file()]
            if missing:
                hint = (" (remove_captions.py --mode detect writes no clips: use sources/)"
                        if folder.name == "clean" else "")
                raise RenderError(f"{len(missing)}/{len(needed)} clips missing in {label} {folder}{hint}: "
                                  f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")

        out_dir.mkdir(parents=True, exist_ok=True)
        if proxy_dir:
            proxy_dir.mkdir(parents=True, exist_ok=True)
        audio = not args.no_audio
        for position, video in enumerate(videos, start=1):
            name = f"{args.prefix}video_{video.get('video_index', position)}.mp4"
            render_checked(video, clips_dir, out_dir / name, audio, args.height)
            if proxy_dir:
                render_checked(video, proxy_clips, proxy_dir / f"proxy_{name}", audio, args.proxy_height)
    except RenderError as exc:
        log(str(exc))
        print("fail")
        return 1
    except Exception as exc:  # noqa: BLE001 - always answer dir/fail
        log(f"Unexpected error: {exc!r}")
        print("fail")
        return 1

    print(f"dir:{out_dir}")
    if proxy_dir:
        print(f"proxy:{proxy_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
