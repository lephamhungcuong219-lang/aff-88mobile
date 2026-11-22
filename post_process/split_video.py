#!/usr/bin/env python3
"""Detect video scenes, split clips, and export a scene CSV + JSON.

Usage:
    python split_scenes.py <input_video> <output_dir>

Scenes come from two detectors together: PySceneDetect (fast cuts) and scene_cuts.py
(OpenCV: cuts PySceneDetect leaves out because they are close to another one, and
dissolves). Every place one of them finds is used; the mixed frames of a dissolve belong
to no scene. A scene still longer than SCENE_MAX_S is one long shot: it is cut into equal
parts of at most SCENE_PART_S.

Environment (.env):
    SCENE_MAX_S      optional: longest scene in seconds, default 2.9; 0 = never cut a shot
    SCENE_PART_S     optional: longest part of a cut shot in seconds, default 1.45 (two parts
                     next to each other last at most 2.9 s)
    SCENE_TRIM_S     optional: seconds that may be dropped from the end of a long shot when
                     that saves a part, default 0.3

Stdout: 'dir:<absolute output_dir>' (the folder with the clips) on success,
        followed by one line 'long_scenes:<long>_<total>[-<n>_<n>...]' (scenes lasting
        at least LONG_SCENE_S, ascending; see long_scene_line), or exactly 'fail'.
        Read the path with /^dir:(.*)$/m.
Stderr: the reason for a failure (human-readable).
Exit code: 0 for success, 1 for failure.

Files:
    <output_dir>/            scene clips
    <output_dir>/../runlog.txt
    <output_dir>/../scenelog.csv
    <output_dir>/../scenelog.json
    <output_dir>/../jump_cuts.json  pairs of scenes that must not follow each other directly
                                  (jump_cuts.py; read by gemini_reorder.py)
    <output_dir>/../stats.json    "long_scenes" and "scene_detection" (job_stats.py); written
                                  fresh on every run
"""

import argparse
import csv as csvlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

from scenedetect import AdaptiveDetector, FrameTimecode, detect
from scenedetect.output import write_scene_list

try:  # python -m post_process.split_video
    from post_process import jump_cuts, scene_cuts
    from post_process.job_stats import stats_path, update_stats
except ImportError:  # python post_process/split_video.py
    import jump_cuts
    import scene_cuts
    from job_stats import stats_path, update_stats

MIN_SCENE_LEN = "0.6s"  # PySceneDetect: fast cuts closer than this are left out (its CLI default)
# Same encoding as PySceneDetect's split-video.
ENCODE = ["-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "aac", "-sn"]


def _convert_value(value: str):
    """Turn numeric strings into int/float; keep everything else as str."""
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


LONG_SCENE_S = 3.0  # scenes lasting at least this long are counted on stdout


def long_scene_stats(scenes: list[dict]) -> dict:
    """Scenes lasting at least LONG_SCENE_S (stats.json -> "long_scenes")."""
    lengths = {int(s["Scene Number"]): float(s.get("Length (seconds)") or 0) for s in scenes}
    long = sorted(n for n, length in lengths.items() if length >= LONG_SCENE_S)
    return {
        "threshold_s": LONG_SCENE_S,
        "long_count": len(long),
        "scene_count": len(lengths),
        "scenes": long,
        "durations": {str(n): round(lengths[n], 3) for n in long},
        "long_seconds": round(sum(lengths[n] for n in long), 3),
        "total_seconds": round(sum(lengths.values()), 3),
    }


def long_scene_line(stats: dict) -> str:
    """Stdout line 'long_scenes:<long count>_<scene count>[-<long scene numbers joined by _>]'.

    Scene numbers ascending; no '-' part when none is long,
    e.g. 'long_scenes:4_47-4_6_7_18', 'long_scenes:0_47'.
    """
    line = f"long_scenes:{stats['long_count']}_{stats['scene_count']}"
    return f"{line}-{'_'.join(map(str, stats['scenes']))}" if stats["scenes"] else line


def csv_to_json(csv_path: Path, json_path: Path) -> list[dict]:
    """Convert a PySceneDetect scene CSV into JSON.

    PySceneDetect's CSV may start with a 'Timecode List:' row (cut points)
    before the real header row. That row is kept separately as 'timecode_list'.
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csvlib.reader(f) if any(cell.strip() for cell in row)]

    timecode_list = []
    if rows and rows[0] and rows[0][0].strip().lower().startswith("timecode list"):
        timecode_list = [cell.strip() for cell in rows[0][1:] if cell.strip()]
        rows = rows[1:]

    scenes = []
    if rows:
        header = [h.strip() for h in rows[0]]
        for row in rows[1:]:
            scenes.append({
                key: _convert_value(row[i]) if i < len(row) else None
                for i, key in enumerate(header)
            })

    data = {"timecode_list": timecode_list, "scenes": scenes}

    # Write to a temp file first so a crash never leaves a half-written JSON.
    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(json_path)
    return scenes


def seconds_setting(name: str, default: float) -> float:
    """A number of seconds (0 or more) from the environment; empty = default."""
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds: {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a number of seconds, 0 or more: {raw!r}")
    return value


def scene_limits() -> tuple[float, float, float]:
    """(SCENE_MAX_S, SCENE_PART_S, SCENE_TRIM_S) from the environment (.env)."""
    max_s = seconds_setting("SCENE_MAX_S", 2.9)
    part_s = seconds_setting("SCENE_PART_S", 1.45)
    trim_s = seconds_setting("SCENE_TRIM_S", 0.3)
    if 0 < max_s < 2 * scene_cuts.MIN_SCENE_S:
        raise ValueError(f"SCENE_MAX_S must be 0 or at least {2 * scene_cuts.MIN_SCENE_S:g}: {max_s:g}")
    # A part is never above the maximum, and never so small that scenes get too short.
    return max_s, max(2 * scene_cuts.MIN_SCENE_S, min(part_s or max_s, max_s)), trim_s


def find_scenes(video: Path) -> tuple[list[tuple[FrameTimecode, FrameTimecode]], dict]:
    """Scenes of both detectors together, and what each one found (stats.json -> "scene_detection")."""
    max_s, part_s, trim_s = scene_limits()
    base = detect(str(video), AdaptiveDetector(min_scene_len=MIN_SCENE_LEN), start_in_scene=True)
    if not base:
        raise RuntimeError("PySceneDetect found no frames")
    clock = base[0][0]
    frames, times = scene_cuts.read_frames(video)
    cut_frames = scene_cuts.find_cuts(frames)
    dissolve_frames = scene_cuts.find_dissolves(frames, float(clock.frame_rate), cut_frames)

    def time_of(index: int) -> float:
        return float(times[index]) if index < len(times) else base[-1][1].seconds

    cuts = [time_of(t) for t in cut_frames]
    dissolves = [(time_of(a), time_of(b + 1)) for a, b in dissolve_frames]
    scenes, added = scene_cuts.combine([(a.seconds, b.seconds) for a, b in base], cuts, dissolves)
    scenes, long_split = scene_cuts.split_long(scenes, times, max_s, part_s, trim_s)
    report = {
        "scenedetect_scenes": len(base),
        "opencv_cuts": len(cuts),
        "opencv_cuts_added": [round(t, 3) for t in added],
        "dissolves": [[round(a, 3), round(b, 3)] for a, b in dissolves],
        "max_scene_s": max_s,
        # With the numbers of the scenes each long shot was cut into.
        "long_shots_split": [{**shot, "scenes": [n for n, (a, b) in enumerate(scenes, 1)
                                                 if shot["start"] - 2e-3 <= a and b <= shot["end"] + 2e-3]}
                             for shot in long_split],
        "scene_count": len(scenes),
    }
    # PySceneDetect's own timecodes are kept where nothing changed (no rounding).
    known = {t.seconds: t for pair in base for t in pair}

    def timecode(seconds: float) -> FrameTimecode:
        return known.get(seconds) or FrameTimecode(seconds, clock)

    return [(timecode(a), timecode(b)) for a, b in scenes], report


def split(video: Path, scenes: list[tuple[FrameTimecode, FrameTimecode]], output: Path) -> None:
    """Write every scene to <output>/<N>.mp4."""
    for number, (start, end) in enumerate(scenes, 1):
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", str(start.seconds), "-i", str(video),
             "-t", str((end - start).seconds), *ENCODE, str(output / f"{number}.mp4")],
            capture_output=True, text=True, errors="replace", check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for scene {number}: {result.stderr.strip()[-2000:]}")


def fail(reason: str, log: Path | None = None, details: str = "") -> int:
    """Report a failure: reason to stderr (+ log file), 'fail' to stdout."""
    message = f"[split_scenes] {reason}"
    print(message, file=sys.stderr)
    if details:
        print(details, file=sys.stderr)
    if log is not None:
        try:
            with log.open("a", encoding="utf-8") as f:
                f.write(f"\n--- Script failure ---\n{message}\n")
                if details:
                    f.write(details + "\n")
        except OSError:
            pass
    print("fail")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Split a video into scenes using PySceneDetect.")
    parser.add_argument("input_video", type=Path, help="Path to the input video")
    parser.add_argument("output_dir", type=Path, help="Directory for clips")
    args = parser.parse_args()

    video = args.input_video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()

    if not video.is_file():
        return fail(f"Input video not found: {video} (cwd: {Path.cwd()})")

    meta_dir = output.parent  # CSV, JSON and log live next to output_dir
    log = meta_dir / "runlog.txt"
    csv = meta_dir / "scenelog.csv"
    json_file = csv.with_suffix(".json")

    try:
        output.mkdir(parents=True, exist_ok=True)

        # Ensure a successful run creates fresh scene CSV/JSON files.
        csv.unlink(missing_ok=True)
        json_file.unlink(missing_ok=True)
        jump_file = meta_dir / jump_cuts.FILE_NAME
        jump_file.unlink(missing_ok=True)  # pairs of other scenes must never be used

        # Remove clips from a previous run (1.mp4, 002.mp4, ...) so stale
        # scenes don't linger when the new run yields fewer scenes.
        for old_clip in output.iterdir():
            if old_clip.is_file() and old_clip.stem.isdigit():
                old_clip.unlink()

        try:
            found, report = find_scenes(video)
            split(video, found, output)
            with csv.open("w", encoding="utf-8", newline="") as f:
                write_scene_list(f, found)
        except Exception:
            return fail(f"Could not split {video}", log, traceback.format_exc())
        log.write_text("\n".join([
            f"PySceneDetect: {report['scenedetect_scenes']} scenes",
            f"OpenCV: {report['opencv_cuts']} cuts, {len(report['opencv_cuts_added'])} added: "
            f"{report['opencv_cuts_added']}",
            f"OpenCV dissolves (left out): {report['dissolves']}",
            f"Long shots cut into equal parts (over {report['max_scene_s']:g}s): {report['long_shots_split']}",
            f"Result: {report['scene_count']} scenes",
            *(f"  {n}: {a.seconds:.3f} - {b.seconds:.3f}" for n, (a, b) in enumerate(found, 1)),
        ]) + "\n", encoding="utf-8")

        try:
            scenes = csv_to_json(csv, json_file)
        except Exception:
            return fail(f"Could not convert {csv} to JSON", log, traceback.format_exc())

        # Only a help for gemini_reorder: without the file that rule is not applied.
        try:
            pairs = jump_cuts.find_jump_cuts(output, [s["scenes"] for s in report["long_shots_split"]])
            jump_file.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
            report["jump_cut_pairs"] = len(pairs)
        except Exception:
            print(f"[split_scenes] WARNING: could not look for jump cuts\n{traceback.format_exc()}", file=sys.stderr)

        try:
            stats = long_scene_stats(scenes)
            # New scenes: start stats.json over (overlaps of an older plan no longer apply).
            update_stats(stats_path(meta_dir), "long_scenes", stats, reset=True)
            update_stats(stats_path(meta_dir), "scene_detection", report)
        except Exception:
            return fail(f"Could not write {stats_path(meta_dir)}", log, traceback.format_exc())

        print(f"dir:{output}")  # always the first line: n8n reads it with /^dir:(.*)$/m
        print(long_scene_line(stats))
        return 0
    except Exception:
        return fail("Unexpected error", log, traceback.format_exc())


if __name__ == "__main__":
    sys.exit(main())