#!/usr/bin/env python3
"""Detect video scenes, split clips, and export a scene CSV + JSON.

Usage:
    python split_scenes.py <input_video> <output_dir>

Stdout: 'dir:<absolute output_dir>' (the folder with the clips) on success,
        or exactly 'fail'.
Stderr: the reason for a failure (human-readable).
Exit code: 0 for success, 1 for failure.

Files:
    <output_dir>/            scene clips
    <output_dir>/../runlog.txt
    <output_dir>/../scenelog.csv
    <output_dir>/../scenelog.json
"""

import argparse
import csv as csvlib
import json
import subprocess
import sys
import traceback
from pathlib import Path


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


def csv_to_json(csv_path: Path, json_path: Path) -> None:
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

        # Remove clips from a previous run (1.mp4, 002.mp4, ...) so stale
        # scenes don't linger when the new run yields fewer scenes.
        for old_clip in output.iterdir():
            if old_clip.is_file() and old_clip.stem.isdigit():
                old_clip.unlink()

        command = [
            sys.executable, "-m", "scenedetect",
            "-l", str(log),
            "-i", str(video),
            "-o", str(output),
            # Name clips by scene number only (001.mp4); renamed to 1.mp4 below.
            "split-video", "-f", "$SCENE_NUMBER",
            # list-scenes writes to the global -o dir by default; send the CSV
            # to meta_dir explicitly so it lands where we look for it.
            "list-scenes", "-o", str(meta_dir), "--filename", csv.name,
        ]
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
        output_text = (
            f"Command: {' '.join(command)}\n"
            f"Return code: {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

        if result.returncode != 0:
            return fail(f"scenedetect exited with code {result.returncode}", log, output_text)

        if not csv.is_file():
            return fail(f"scenedetect finished but CSV not found at {csv}", log, output_text)

        # Strip zero padding: 001.mp4 -> 1.mp4
        try:
            for clip in sorted(output.iterdir()):
                if clip.is_file() and clip.stem.isdigit():
                    target = clip.with_name(f"{int(clip.stem)}{clip.suffix}")
                    if target != clip:
                        clip.replace(target)
        except OSError:
            return fail(f"Could not rename clips in {output}", log, traceback.format_exc())

        try:
            csv_to_json(csv, json_file)
        except Exception:
            return fail(f"Could not convert {csv} to JSON", log, traceback.format_exc())

        print(f"dir:{output}")
        return 0
    except Exception:
        return fail("Unexpected error", log, traceback.format_exc())


if __name__ == "__main__":
    sys.exit(main())