"""stats.json of a job: statistics the pipeline steps print on stdout, kept on disk.

One file per source video, next to scenelog.json / reorder.json
(<job>/temp/<awemeId>/stats.json). Each step owns one top-level key:
    "long_scenes"  split_video.py     scenes lasting at least LONG_SCENE_S
    "scene_detection"  split_video.py  what each scene detector found
    "overlaps"     gemini_reorder.py  runs of consecutive scenes shared by videos
    "source_overlaps"  gemini_reorder.py  passages of the source inside each video
    "coherence"    gemini_reorder.py  topic blocks of each video
A step replaces only its own key, so the file keeps what the other steps wrote.
"""

import json
from pathlib import Path

FILE_NAME = "stats.json"


def stats_path(folder: Path) -> Path:
    return folder / FILE_NAME


def read_stats(path: Path) -> dict:
    """Current content, or {} when the file is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def update_stats(path: Path, key: str, value, *, reset: bool = False) -> None:
    """Set data[key] = value (value None removes the key); reset=True drops every other key."""
    data = {} if reset else read_stats(path)
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
