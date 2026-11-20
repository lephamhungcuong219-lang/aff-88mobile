#!/usr/bin/env python3
"""Ask Gemini to turn the scenes of a Douyin product video into several short videos.

Usage:
    python gemini_reorder.py <scenelog_json> <clips_dir> <output_json> [--dry-run]

Sends every <clips_dir>/<N>.mp4 (small copies from compress_clips.py) to Gemini
inline as base64, in ONE generateContent request. Each clip is preceded by a
text label "Scene N", where N is the "Scene Number" in scenelog.json.

Gemini plans several output videos, each with its own angle on the product
(chosen from what the source video actually shows), and returns only the
ORDER of scene numbers for each one. This script then checks the plan:
    - each video lasts at most MAX_VIDEO_S (target TARGET_VIDEO_S) and at least MIN_VIDEO_S
    - a scene appears at most once inside a video
    - every scene is used in at least one video
    - a scene is used in at most MAX_SCENE_USES videos
    - two videos share at most MAX_SHARED_RATIO of the smaller one's scenes
If a rule is broken, the exact problems are sent back to Gemini (same
conversation, clips included) to fix, up to ANSWER_ATTEMPTS times.
All timestamps are computed from scenelog.json, never taken from Gemini.

Environment:
    GEMINI_API_KEY   required (put it in .env)
    GEMINI_MODEL     optional, default below

Files (next to <output_json>):
    request.json        the request body sent to Gemini (first call; retries
                        add the model's answer + the fix prompt, which are
                        recorded in gem_response.json)
    gem_response.json   raw Gemini response of every attempt, with the rule
                        problems found and the fix prompt sent back; written
                        after each attempt, so it also exists after a failure
    <output_json>       the final plan (Gemini's choices + timestamps computed
                        from scenelog.json); only written on success

--dry-run only writes request.json, without calling Gemini (useful to inspect
size and prompt).

Stdout: 'json:<absolute output_json>' on success ('json:<request.json>' with
        --dry-run), or exactly 'fail'.
Stderr: progress; a failure is one line "[gemini_reorder] <CODE>: <message>" with the codes
        of gemini_errors.py (E_QUOTA_DAILY is reported at once, without retrying).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import base64
import json
import math
import os
import sys
import time
import urllib.request
from itertools import combinations
from pathlib import Path
from urllib.error import HTTPError, URLError

try:  # python -m post_process.gemini_reorder
    from post_process.gemini_errors import (E_AUTH, E_INVALID_ANSWER, E_NETWORK, E_NO_OUTPUT,
                                            E_UNEXPECTED, GeminiError, format_error, http_error,
                                            is_daily_quota, parse_retry_after)
except ImportError:  # python post_process/gemini_reorder.py
    from gemini_errors import (E_AUTH, E_INVALID_ANSWER, E_NETWORK, E_NO_OUTPUT, E_UNEXPECTED,
                               GeminiError, format_error, http_error, is_daily_quota, parse_retry_after)

DEFAULT_MODEL = "gemini-3.8-flash"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
KEY_PLACEHOLDER = "your-gemini-api-key-here"

# Output video rules
TARGET_VIDEO_S = 15.0
MAX_VIDEO_S = 20.0     # hard limit (target + allowed tolerance)
MIN_VIDEO_S = 8.0      # shorter videos are not worth publishing
MAX_SCENE_USES = 2     # a scene may appear in at most this many videos
MAX_SHARED_RATIO = 0.5 # shared scenes / scene count of the smaller video

MAX_INLINE_BYTES = 20 * 1024 * 1024  # documented safe limit for a whole inline request
CLIP_FPS = 6           # clips are ~0.6-2.5 s; the default 1 FPS would see only 1-2 frames
REQUEST_TIMEOUT_S = 600
HTTP_ATTEMPTS = 4      # retries on 429 / 5xx / network errors
ANSWER_ATTEMPTS = 3    # Gemini calls allowed to produce a valid plan

PROMPT = """\
Bạn là biên tập viên video ngắn chuyên làm nội dung bán hàng trên Douyin/TikTok.

Bạn nhận được {count} phân đoạn (scene) liên tiếp được cắt ra từ MỘT video giới thiệu sản phẩm
(tổng {total:.1f} giây). Sản phẩm có thể là bất kỳ thứ gì (đồ điện tử, mỹ phẩm, thời trang, đồ gia dụng,
thực phẩm...): hãy tự nhận biết sản phẩm và những khía cạnh mà video đang trình bày, từ hình ảnh,
chữ trên màn hình và âm thanh.
Mỗi phân đoạn có nhãn "Scene N" ngay trước nó (và chữ "SCENE N" ở góc hình). N chính là scene_number.
Các scene đang theo thứ tự gốc: Scene 1 là đầu video, Scene {count} là cuối video.

Thời lượng từng scene (giây):
{durations}

NHIỆM VỤ: từ các scene này, lên kế hoạch cho NHIỀU video ngắn mới để đăng riêng.
- Mỗi video có MỘT góc tiếp cận (angle) riêng, dựa trên những khía cạnh mà video gốc THỰC SỰ thể hiện.
  Ví dụ với tai nghe: một video về độ bền/chất lượng hoàn thiện, một video về tính năng/công dụng,
  một video về trải nghiệm âm thanh/chống ồn. Đây chỉ là ví dụ: hãy chọn góc phù hợp với nội dung thật,
  đừng ép một góc mà video không có đủ cảnh để thể hiện.
- Nếu một khía cạnh không đủ cảnh để làm thành một video đủ thời lượng, được phép kết hợp nhiều khía cạnh
  vào cùng một video (khi đó ghi rõ các khía cạnh trong "angle"). Điều quan trọng nhất là thời lượng
  của từng video đúng quy tắc bên dưới.
- Số lượng video do bạn quyết định theo nội dung, tối thiểu {min_videos} video.
- Mỗi video nên có cấu trúc riêng phù hợp với góc của nó: hook mạnh ở 1-2 giây đầu, phần chính, kết
  (cảnh sản phẩm tổng thể / kêu gọi hành động). Các video phải khác nhau rõ rệt về cảnh mở đầu và thứ tự,
  để người xem không nhận ra chúng được dựng từ cùng một video.

QUY TẮC BẮT BUỘC (sẽ được kiểm tra tự động):
- Thời lượng mỗi video (tổng thời lượng các scene của nó) khoảng {target:.0f} giây,
  KHÔNG dưới {min_s:.0f} giây và KHÔNG quá {max_s:.0f} giây. Hãy cộng thời lượng ở bảng trên cẩn thận.
- Trong một video, mỗi scene xuất hiện tối đa một lần.
- MỌI scene từ 1 đến {count} phải được dùng trong ít nhất một video.
- Một scene được dùng trong tối đa {max_uses} video.
- Hai video bất kỳ dùng chung không quá {shared_pct}% số scene của video ngắn hơn.
- Chỉ trả về scene_number theo thứ tự trong từng video. KHÔNG tự tính timestamp.
- Giữ cạnh nhau những scene thuộc cùng một hành động/câu nói liên tục nếu tách ra sẽ vô nghĩa.
- "key_points": 2-3 ý chính cụ thể (thông số, lợi ích, điểm khác biệt) mà lời thoại của video đó nên nói,
  lấy từ những gì video gốc thực sự thể hiện (hình, chữ trên màn hình, lời nói). Không bịa thông số.
- Viết "angle", "title", "summary", "key_points", "description", "reason" bằng tiếng Việt.
"""

FIX_PROMPT = """\
Kế hoạch vừa rồi vi phạm quy tắc:
{problems}

Hãy sửa lại và trả về TOÀN BỘ kế hoạch theo đúng schema, giữ các góc tiếp cận còn hợp lệ.
Nếu một video quá ngắn vì góc của nó thiếu cảnh, hãy kết hợp thêm khía cạnh khác vào video đó
hoặc gộp nó với video khác.
Nhắc lại: mỗi video {min_s:.0f}-{max_s:.0f} giây (mục tiêu {target:.0f} giây), mọi scene 1-{count} được dùng
ít nhất một lần, mỗi scene tối đa {max_uses} video, không lặp scene trong cùng một video.
"""

SCENE_ITEM = {
    "type": "OBJECT",
    "properties": {
        "scene_number": {"type": "INTEGER"},
        "description": {"type": "STRING", "description": "Nội dung scene"},
        "reason": {"type": "STRING", "description": "Vì sao đặt scene ở vị trí này"},
    },
    "required": ["scene_number", "description", "reason"],
    "propertyOrdering": ["scene_number", "description", "reason"],
}

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "product": {"type": "STRING", "description": "Sản phẩm được giới thiệu"},
        "aspects": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Các khía cạnh của sản phẩm mà video gốc thể hiện",
        },
        "videos": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "angle": {"type": "STRING", "description": "Góc tiếp cận (một hoặc nhiều khía cạnh)"},
                    "title": {"type": "STRING", "description": "Tiêu đề gợi ý"},
                    "summary": {"type": "STRING", "description": "Ý tưởng dựng video"},
                    "key_points": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "2-3 ý chính (thông số/lợi ích cụ thể) mà lời thoại của video này nên nói",
                    },
                    "scenes": {"type": "ARRAY", "items": SCENE_ITEM},
                },
                "required": ["angle", "title", "summary", "key_points", "scenes"],
                "propertyOrdering": ["angle", "title", "summary", "key_points", "scenes"],
            },
        },
    },
    "required": ["product", "aspects", "videos"],
    "propertyOrdering": ["product", "aspects", "videos"],
}


ReorderError = GeminiError  # message + error code (see gemini_errors.py)
MAX_RETRY_WAIT_S = 120      # a 429 asking to wait longer than this is not retried


def log(message: str) -> None:
    print(f"[gemini_reorder] {message}", file=sys.stderr)


def log_error(error: Exception, script: str = "gemini_reorder") -> None:
    print(format_error(script, error), file=sys.stderr)


def min_videos(scenes: list[dict]) -> int:
    """Fewest videos that can cover every scene within MAX_VIDEO_S each."""
    return max(1, math.ceil(sum(s["duration"] for s in scenes) / MAX_VIDEO_S))


def load_scenes(scenelog: Path) -> list[dict]:
    try:
        data = json.loads(scenelog.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReorderError(f"Could not read {scenelog}: {exc}") from exc
    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not scenes:
        raise ReorderError(f"No 'scenes' in {scenelog}")
    try:
        scenes = sorted(
            (
                {
                    "scene_number": int(s["Scene Number"]),
                    "start": float(s["Start Time (seconds)"]),
                    "end": float(s["End Time (seconds)"]),
                    "duration": float(s["Length (seconds)"]),
                }
                for s in scenes
            ),
            key=lambda s: s["scene_number"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReorderError(f"Unexpected scene format in {scenelog}: {exc!r}") from exc
    numbers = [s["scene_number"] for s in scenes]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ReorderError(f"Scene numbers in {scenelog} are not 1..{len(numbers)}")
    longest = max(s["duration"] for s in scenes)
    if longest > MAX_VIDEO_S:
        raise ReorderError(f"A scene lasts {longest:.1f}s, longer than one output video ({MAX_VIDEO_S:.0f}s)")
    return scenes


def rule_values(scenes: list[dict]) -> dict:
    return {
        "count": len(scenes),
        "target": TARGET_VIDEO_S,
        "min_s": MIN_VIDEO_S,
        "max_s": MAX_VIDEO_S,
        "max_uses": MAX_SCENE_USES,
        "shared_pct": round(MAX_SHARED_RATIO * 100),
    }


def build_request(scenes: list[dict], clips_dir: Path) -> dict:
    durations = "\n".join(f"  Scene {s['scene_number']}: {s['duration']:.3f}" for s in scenes)
    prompt = PROMPT.format(
        durations=durations,
        total=sum(s["duration"] for s in scenes),
        min_videos=min_videos(scenes),
        **rule_values(scenes),
    )
    parts = [{"text": prompt}]

    total = 0
    for s in scenes:
        clip = clips_dir / f"{s['scene_number']}.mp4"
        if not clip.is_file():
            raise ReorderError(f"Missing clip for scene {s['scene_number']}: {clip}")
        raw = clip.read_bytes()
        total += len(raw)
        parts.append({"text": f"Scene {s['scene_number']} ({s['duration']:.3f}s):"})
        parts.append({
            "inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(raw).decode("ascii")},
            "videoMetadata": {"fps": CLIP_FPS},
        })
    parts.append({"text": f"Hết {len(scenes)} scene. Trả về kế hoạch theo đúng schema."})

    encoded = total * 4 // 3
    log(f"{len(scenes)} clips, {total / 1e6:.1f} MB raw, ~{encoded / 1e6:.1f} MB as base64")
    if encoded > MAX_INLINE_BYTES:
        raise ReorderError(
            f"Inline request too large (~{encoded / 1e6:.1f} MB > 20 MB). "
            "Compress clips smaller (compress_clips.py --short-side 360)."
        )

    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "temperature": 0.8,
        },
    }


def call_gemini(body: dict, model: str, api_key: str) -> dict:
    request = urllib.request.Request(
        API_URL.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        delay = 5 * 2 ** (attempt - 1)
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            if exc.code == 429 and is_daily_quota(detail):
                raise http_error(exc.code, detail, "Gemini") from exc  # E_QUOTA_DAILY: no retry
            hint = parse_retry_after(detail)
            delay = hint + 1 if hint is not None else delay
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == HTTP_ATTEMPTS or delay > MAX_RETRY_WAIT_S:
                raise http_error(exc.code, detail, "Gemini") from exc
            log(f"Gemini HTTP {exc.code}, retrying in {delay:.0f}s ({attempt}/{HTTP_ATTEMPTS})")
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == HTTP_ATTEMPTS:
                raise ReorderError(f"Could not reach Gemini: {exc}", E_NETWORK) from exc
            log(f"Network error ({exc}), retrying ({attempt}/{HTTP_ATTEMPTS})")
        time.sleep(delay)
    raise ReorderError("unreachable", E_UNEXPECTED)


def extract_answer(response: dict) -> tuple[dict, dict]:
    """Return (parsed JSON answer, the model's content to send back on a retry)."""
    candidates = response.get("candidates") or []
    if not candidates:
        raise ReorderError(f"Gemini returned no candidates (promptFeedback: {response.get('promptFeedback')})",
                           E_NO_OUTPUT)
    candidate = candidates[0]
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise ReorderError(f"Empty answer (finishReason: {candidate.get('finishReason')})", E_NO_OUTPUT)
    try:
        answer = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReorderError(f"Answer is not valid JSON: {exc}; text starts with: {text[:300]!r}",
                           E_NO_OUTPUT) from exc
    if not isinstance(answer, dict):
        raise ReorderError("Answer JSON is not an object", E_NO_OUTPUT)
    return answer, {"role": "model", "parts": parts}


def find_problems(answer: dict, scenes: list[dict]) -> list[str]:
    """Check the plan against the rules; return human-readable problems (empty = valid)."""
    durations = {s["scene_number"]: s["duration"] for s in scenes}
    videos = answer.get("videos")
    if not isinstance(videos, list) or not videos:
        return ["Không có danh sách 'videos'."]

    problems, plans = [], []
    for index, video in enumerate(videos, start=1):
        items = video.get("scenes") if isinstance(video, dict) else None
        if not isinstance(items, list) or not items:
            problems.append(f"Video {index}: không có scene nào.")
            plans.append([])
            continue
        numbers = [i.get("scene_number") if isinstance(i, dict) else None for i in items]
        unknown = sorted({str(n) for n in numbers if n not in durations})
        if unknown:
            problems.append(f"Video {index}: scene_number không tồn tại: {', '.join(unknown)}.")
        repeated = sorted({n for n in numbers if n in durations and numbers.count(n) > 1})
        if repeated:
            problems.append(f"Video {index}: scene lặp lại trong cùng video: {repeated}.")
        valid = [n for n in numbers if n in durations]
        plans.append(valid)
        length = sum(durations[n] for n in dict.fromkeys(valid))
        if length > MAX_VIDEO_S:
            problems.append(f"Video {index}: dài {length:.2f}s, vượt {MAX_VIDEO_S:.0f}s.")
        elif length < MIN_VIDEO_S:
            problems.append(f"Video {index}: chỉ {length:.2f}s, ngắn hơn {MIN_VIDEO_S:.0f}s.")

    uses = {n: [i for i, plan in enumerate(plans, start=1) if n in plan] for n in durations}
    unused = [n for n, where in uses.items() if not where]
    if unused:
        problems.append(f"Scene chưa được dùng ở video nào: {unused}.")
    overused = {n: where for n, where in uses.items() if len(where) > MAX_SCENE_USES}
    for n, where in overused.items():
        problems.append(f"Scene {n} dùng trong {len(where)} video {where}, tối đa {MAX_SCENE_USES}.")

    for (a, plan_a), (b, plan_b) in combinations(enumerate(plans, start=1), 2):
        if not plan_a or not plan_b:
            continue
        shared = set(plan_a) & set(plan_b)
        smaller = min(len(set(plan_a)), len(set(plan_b)))
        if plan_a == plan_b:
            problems.append(f"Video {a} và {b} giống hệt nhau.")
        elif len(shared) > MAX_SHARED_RATIO * smaller:
            problems.append(
                f"Video {a} và {b} dùng chung {len(shared)}/{smaller} scene "
                f"(tối đa {round(MAX_SHARED_RATIO * 100)}%): {sorted(shared)}."
            )
    return problems


def build_output(answer: dict, scenes: list[dict], model: str) -> dict:
    by_number = {s["scene_number"]: s for s in scenes}
    videos = []
    for index, video in enumerate(answer["videos"], start=1):
        timeline, cursor = [], 0.0
        for position, item in enumerate(video["scenes"], start=1):
            scene = by_number[item["scene_number"]]
            timeline.append({
                "position": position,
                "scene_number": scene["scene_number"],
                "source_file": f"{scene['scene_number']}.mp4",
                "src_start": scene["start"],
                "src_end": scene["end"],
                "duration": scene["duration"],
                "out_start": round(cursor, 3),
                "out_end": round(cursor + scene["duration"], 3),
                "description": item.get("description", ""),
                "reason": item.get("reason", ""),
            })
            cursor += scene["duration"]
        videos.append({
            "video_index": index,
            "angle": video.get("angle", ""),
            "title": video.get("title", ""),
            "summary": video.get("summary", ""),
            "key_points": [str(p) for p in (video.get("key_points") or [])],
            "scene_count": len(timeline),
            "total_duration": round(cursor, 3),
            "scenes": timeline,
        })

    usage = {s["scene_number"]: [] for s in scenes}
    for video in videos:
        for item in video["scenes"]:
            usage[item["scene_number"]].append(video["video_index"])
    return {
        "model": model,
        "product": answer.get("product", ""),
        "aspects": answer.get("aspects", []),
        "rules": {
            "target_video_s": TARGET_VIDEO_S,
            "max_video_s": MAX_VIDEO_S,
            "min_video_s": MIN_VIDEO_S,
            "max_scene_uses": MAX_SCENE_USES,
            "max_shared_ratio": MAX_SHARED_RATIO,
        },
        "source_scene_count": len(scenes),
        "source_total_duration": round(sum(s["duration"] for s in scenes), 3),
        "video_count": len(videos),
        "videos": videos,
        "scene_usage": {str(n): where for n, where in usage.items()},
    }


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def plan_videos(body: dict, scenes: list[dict], model: str, api_key: str, response_file: Path) -> dict:
    """Ask Gemini, feeding rule violations back into the same conversation.

    Every attempt (raw response, problems, fix prompt) is saved to response_file.
    """
    log_data = {"model": model, "attempts": []}
    problems: list[str] = []

    def record(entry: dict) -> None:
        log_data["attempts"].append(entry)
        write_json(response_file, log_data)

    for attempt in range(1, ANSWER_ATTEMPTS + 1):
        log(f"Calling {model} (attempt {attempt}/{ANSWER_ATTEMPTS})")
        try:
            response = call_gemini(body, model, api_key)
        except ReorderError as exc:
            record({"attempt": attempt, "accepted": False, "error": str(exc)})
            raise
        usage = response.get("usageMetadata") or {}
        if usage:
            log(f"Tokens: prompt {usage.get('promptTokenCount')}, total {usage.get('totalTokenCount')}")

        entry = {"attempt": attempt, "accepted": False, "response": response}
        try:
            answer, model_turn = extract_answer(response)
        except ReorderError as exc:
            problems = [str(exc)]
            log(problems[0])
            record({**entry, "problems": problems})
            continue  # nothing usable to correct; ask again from scratch

        problems = find_problems(answer, scenes)
        if not problems:
            record({**entry, "accepted": True, "problems": []})
            return answer
        log("Plan breaks the rules:\n  " + "\n  ".join(problems))
        fix = FIX_PROMPT.format(problems="\n".join(f"- {p}" for p in problems), **rule_values(scenes))
        record({**entry, "problems": problems, "fix_prompt": fix})
        body["contents"] = body["contents"][:1] + [model_turn, {"role": "user", "parts": [{"text": fix}]}]
    raise ReorderError(f"No valid plan after {ANSWER_ATTEMPTS} attempts: {'; '.join(problems)}", E_INVALID_ANSWER)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan several short videos from Douyin product video scenes.")
    parser.add_argument("scenelog_json", type=Path, help="scenelog.json from split_video.py")
    parser.add_argument("clips_dir", type=Path, help="Directory with compressed <N>.mp4 clips")
    parser.add_argument("output_json", type=Path, help="Where to write the video plans")
    parser.add_argument("--dry-run", action="store_true", help="Only write the request body, do not call Gemini")
    args = parser.parse_args()

    scenelog = args.scenelog_json.expanduser().resolve()
    clips_dir = args.clips_dir.expanduser().resolve()
    output = args.output_json.expanduser().resolve()
    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    request_file = output.with_name("request.json")
    response_file = output.with_name("gem_response.json")

    try:
        if output in (request_file, response_file):
            raise ReorderError(f"output_json must not be named {output.name}")
        scenes = load_scenes(scenelog)
        body = build_request(scenes, clips_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(request_file, body)

        if args.dry_run:
            print(f"json:{request_file}")
            return 0

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == KEY_PLACEHOLDER:
            raise ReorderError("GEMINI_API_KEY is not set (fill it in .env, then recreate the container)", E_AUTH)

        # Drop results of a previous run so the three files always belong together.
        output.unlink(missing_ok=True)
        response_file.unlink(missing_ok=True)

        answer = plan_videos(body, scenes, model, api_key, response_file)
        result = build_output(answer, scenes, model)
        log(", ".join(f"video {v['video_index']}: {v['total_duration']:.1f}s" for v in result["videos"]))
        write_json(output, result)
    except ReorderError as exc:
        log_error(exc)
        print("fail")
        return 1
    except Exception as exc:  # noqa: BLE001 - always answer json/fail
        log_error(GeminiError(f"{exc!r}", E_UNEXPECTED))
        print("fail")
        return 1

    print(f"json:{output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
