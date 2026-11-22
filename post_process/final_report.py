#!/usr/bin/env python3
"""Write stats.txt next to the final videos: what the person who uploads them should know.

Usage:
    python final_report.py <tts_json> <final_dir>

compose_videos.py calls write_report() when all videos are done. Run this script to write the
file again for a job that already has its final videos.

Reads (next to tts_json): reorder.json, stats.json, clean/captions.json. A missing file only
leaves its part out of the report.

Stdout: 'txt:<absolute final_dir>/stats.txt' on success, or exactly 'fail'.
Stderr: the reason for a failure (human-readable).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:  # python -m post_process.final_report
    from post_process.gemini_reorder import shared_runs
except ImportError:  # python post_process/final_report.py
    from gemini_reorder import shared_runs

FILE_NAME = "stats.txt"
LINE = "=" * 60


def read_json(path: Path) -> dict:
    """The JSON object in the file, or {} when it is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def number(value: float, digits: int = 1) -> str:
    """Vietnamese decimal comma: 2.9 -> '2,9'."""
    return f"{value:.{digits}f}".replace(".", ",")


def seconds(value: float) -> str:
    return f"{number(value)} giây"


def limit(value: float) -> str:
    return f"{value:g}".replace(".", ",") + " giây"


def span(start: float, end: float) -> str:
    """Position in the final video: '0:03 - 0:06' (rounded outwards)."""
    a, b = int(start), int(-(-end // 1))
    return f"{a // 60}:{a % 60:02d} - {b // 60}:{b % 60:02d}"


def length_of(path: Path) -> float | None:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                            capture_output=True, text=True, check=False)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def merged(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Spans that touch, joined."""
    result: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1] + 0.05:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def longest_source_passage(scenes: list[dict]) -> tuple[float, float, float]:
    """(seconds, start, end) of the longest passage of the source in a video.

    A passage is scenes n, n+1, ... next to each other; a scene on its own is one too.
    """
    best = (0.0, 0.0, 0.0)
    first = 0
    for k in range(len(scenes)):
        if k and scenes[k]["scene_number"] != scenes[k - 1]["scene_number"] + 1:
            first = k
        length = sum(s["duration"] for s in scenes[first:k + 1])
        if length > best[0] + 1e-6:
            best = (length, scenes[first]["out_start"], scenes[k]["out_end"])
    return best


def longest_shared_passage(scenes: list[dict], others: list[dict]) -> float:
    """Seconds of the longest run of scenes (same scenes, same order) two videos share; a
    scene on its own is a run too. 0 = nothing in common."""
    durations = {s["scene_number"]: s["duration"] for s in scenes}
    runs = shared_runs([s["scene_number"] for s in scenes], [s["scene_number"] for s in others])
    return max((sum(durations[n] for n in run) for run in runs), default=0.0)


def overlap_part(index: int, plans: dict[int, dict], rules: dict) -> tuple[list[str], list[str]]:
    """(lines, warnings): the longest passages shared with the source and with each other video."""
    scenes = plans.get(index, {}).get("scenes")
    if not scenes:
        return [], []
    length, start, end = longest_source_passage(scenes)
    lines, warnings = [f"Đoạn trùng dài nhất với video gốc: {number(length, 2)} giây"], []
    max_s = rules.get("source_run_max_s") or 0
    if 0 < max_s < length - 1e-6:
        warnings.append(f"Đoạn giống hệt video gốc dài {number(length, 2)} giây ở {span(start, end)}, "
                        f"quá mức cho phép {limit(max_s)}.")
    others = [k for k in sorted(plans) if k != index]
    if others:
        lines.append("Đoạn trùng dài nhất với các video khác:")
        lines += [f"  Video {k}: {number(longest_shared_passage(scenes, plans[k].get('scenes', [])), 2)} giây"
                  for k in others]
    return lines, warnings


def caption_part(scenes: list[dict], captions: dict) -> list[str]:
    """Warnings about places where Chinese text may still be visible."""
    clips = captions.get("clips")
    if not clips:
        return []
    if not captions.get("caption_line"):
        return ["Không tìm thấy dòng phụ đề tiếng Trung trong video gốc: xem cả video để chắc không còn chữ Trung."]
    missing = merged([(s["out_start"], s["out_end"]) for s in scenes
                      if not clips.get(s.get("source_file", ""), {}).get("caption")])
    if not missing:
        return []
    return ["Không dò thấy phụ đề tiếng Trung ở " + ", ".join(span(a, b) for a, b in missing)
            + ": xem lại đoạn này có còn chữ Trung không."]


def build_report(tts_file: Path, final_dir: Path) -> str:
    folder = tts_file.parent
    tts, plan, stats = read_json(tts_file), read_json(folder / "reorder.json"), read_json(folder / "stats.json")
    captions = read_json(folder / "clean" / "captions.json")
    plans = {v["video_index"]: v for v in plan.get("videos", [])}
    sections, summary = [], []

    for video in tts.get("videos") or []:
        index = video["video_index"]
        planned = plans.get(index, {})
        scenes = planned.get("scenes", [])
        files = sorted(final_dir.glob(f"*video_{index}.mp4"))
        warnings: list[str] = []
        duration = (length_of(files[0]) if files else None) or video.get("video_duration") or 0

        body = [LINE, f"VIDEO {index}", LINE]
        body.append(f"File: {files[0].name if files else '(chưa có file)'}")
        if not files:
            warnings.append("Chưa có file video này trong thư mục final.")
        if planned.get("title"):
            body.append(f"Tiêu đề gợi ý (do AI viết): {planned['title']}")
        if planned.get("angle"):
            body.append(f"Nội dung: {planned['angle']}")
        body.append(f"Thời lượng: {seconds(duration)}" + (f", {len(scenes)} cảnh" if scenes else ""))
        if plan:
            lines, found = overlap_part(index, plans, plan.get("rules") or {})
            body += lines
            warnings += found
            warnings += caption_part(scenes, captions)
        text = " ".join(s["text"].strip() for s in video.get("segments") or [])
        if text:
            body += ["", "Lời thoại (do AI viết, kiểm tra số liệu và tên sản phẩm trước khi đăng):", f"  {text}"]
        body += ["", "CẦN XEM LẠI:", *(f"  - {w}" for w in warnings)] if warnings else ["", "Không có cảnh báo."]
        sections.append("\n".join(body))
        summary.append(f"  Video {index} ({seconds(duration)}): "
                       + (f"CẦN XEM LẠI, {len(warnings)} điểm" if warnings else "ổn"))

    found = stats.get("scene_detection", {})
    total = plan.get("source_total_duration")
    head = ["BÁO CÁO VIDEO", f"Sản phẩm: {plan.get('product') or 'không rõ'}", f"Mã video gốc: {folder.name}",
            f"Tạo lúc: {datetime.now():%d/%m/%Y %H:%M}"]
    if total:
        head.append(f"Video gốc: {seconds(total)} dùng được, {plan.get('source_scene_count')} cảnh"
                    + (f" (đã bỏ {len(found['dissolves'])} đoạn chuyển mờ)" if found.get("dissolves") else ""))
    head += ["", f"TÓM TẮT ({len(summary)} video)", *summary,
             "", "Báo cáo do máy tạo từ kế hoạch dựng; không thay cho việc xem lại video trước khi đăng."]
    return "\n".join(head) + "\n\n" + "\n\n".join(sections) + "\n"


def write_report(tts_file: Path, final_dir: Path) -> Path:
    """Write final_dir/stats.txt (UTF-8 with BOM and CRLF: opens correctly in any Notepad)."""
    path = final_dir / FILE_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(build_report(tts_file, final_dir).replace("\n", "\r\n").encode("utf-8-sig"))
    tmp.replace(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write stats.txt for the person who uploads the final videos.")
    parser.add_argument("tts_json", type=Path, help="tts.json from gemini_tts.py")
    parser.add_argument("final_dir", type=Path, help="Directory with the final videos")
    args = parser.parse_args()
    try:
        tts_file, final_dir = args.tts_json.expanduser().resolve(), args.final_dir.expanduser().resolve()
        if not tts_file.is_file() or not final_dir.is_dir():
            raise FileNotFoundError(f"Not found: {tts_file if not tts_file.is_file() else final_dir}")
        path = write_report(tts_file, final_dir)
    except Exception as exc:  # noqa: BLE001 - always answer txt/fail
        print(f"[final_report] {exc!r}", file=sys.stderr)
        print("fail")
        return 1
    print(f"txt:{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
