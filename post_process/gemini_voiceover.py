#!/usr/bin/env python3
"""Ask Gemini to write the Vietnamese voice-over of every planned video.

Usage:
    python gemini_voiceover.py <reorder_json> <proxy_dir> [--output FILE] [--video N ...]
                               [--rate SYLLABLES_PER_S] [--dry-run]

For every video in reorder.json, its proxy <proxy_dir>/*video_<N>.mp4 (the light
copy with the original captions, from render_videos.py --proxy-dir) is sent to
Gemini inline, together with what the planning step already knows: product,
aspects, the video's angle/title/summary/key_points and its scene list with
timings. One request per video.

Gemini returns the voice-over as short subtitle-sized segments (no timestamps
yet), each in two forms:
    text    what is shown as subtitle ("chỉ 4,7g")
    spoken  what the TTS reads, numbers and units in words ("chỉ bốn phẩy bảy gam")
plus a speaking-style instruction for Gemini TTS.

The script checks every answer and sends the problems back to Gemini (same
conversation) to fix, up to ANSWER_ATTEMPTS times:
    - spoken length fits the video: TARGET_FILL of duration x --rate syllables,
      between MIN_FILL and MAX_FILL
    - every segment's text fits a subtitle line (MAX_SEGMENT_CHARS)
    - no empty segment

Style: energetic sales voice, "mình" (speaker) - "bạn" (viewer).

Files (next to reorder.json unless --output):
    voiceover.json               the voice-over of every video (with --video, only
                                 those videos are replaced, the others are kept)
    gem_voiceover_response.json  raw Gemini responses of every attempt

Environment: GEMINI_API_KEY (required), GEMINI_MODEL (optional).

Stdout: 'json:<absolute voiceover.json>' on success, or exactly 'fail'.
Stderr: progress; a failure is one line "[gemini_voiceover] <CODE>: <message>" with the codes
        of gemini_errors.py (E_QUOTA_DAILY is reported at once, without retrying).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

try:  # python -m post_process.gemini_voiceover
    from post_process.gemini_errors import E_AUTH, E_INVALID_ANSWER, E_UNEXPECTED, GeminiError, format_error
    from post_process.gemini_reorder import (DEFAULT_MODEL, KEY_PLACEHOLDER, MAX_INLINE_BYTES,
                                             ReorderError, call_gemini, extract_answer, write_json)
except ImportError:  # python post_process/gemini_voiceover.py
    from gemini_errors import E_AUTH, E_INVALID_ANSWER, E_UNEXPECTED, GeminiError, format_error
    from gemini_reorder import (DEFAULT_MODEL, KEY_PLACEHOLDER, MAX_INLINE_BYTES,
                                ReorderError, call_gemini, extract_answer, write_json)

DEFAULT_RATE = 3.9        # Vietnamese syllables per second, measured with Gemini TTS voice "Puck"
                          # (Fenrir ~4.4, Laomedeia/Sadachbia/Zephyr ~3.75); change with --rate
TARGET_FILL = 0.88        # aim for this share of the video filled with speech (room to breathe)
MIN_FILL, MAX_FILL = 0.70, 1.00
MAX_SEGMENT_CHARS = 36    # one subtitle line
CLIP_FPS = 4              # proxy frames per second Gemini looks at
ANSWER_ATTEMPTS = 3

PROMPT = """\
Bạn là copywriter viết lời thoại cho video bán hàng ngắn trên TikTok/Douyin, thị trường Việt Nam.

Video đính kèm dài {duration:.1f} giây, được dựng lại từ một video gốc tiếng Trung giới thiệu sản phẩm.
Chữ tiếng Trung trên màn hình và giọng tiếng Trung trong video là của bản gốc: dùng chúng để hiểu
sản phẩm, KHÔNG dịch từng câu. Video mới sẽ được lồng giọng tiếng Việt và phụ đề tiếng Việt.

Thông tin đã phân tích từ video gốc:
- Sản phẩm: {product}
- Các khía cạnh video gốc thể hiện: {aspects}
- Góc tiếp cận của video này: {angle}
- Tiêu đề gợi ý: {title}
- Ý tưởng dựng: {summary}
- Ý chính cần nói: {key_points}

Các cảnh theo thứ tự (giây trong video này):
{scenes}

NHIỆM VỤ: viết lời thoại tiếng Việt cho video này.
Phong cách: năng động, hào hứng, kiểu review bán hàng TikTok Shop. Người nói xưng "mình", gọi người xem là "bạn".
Cấu trúc: câu mở đầu thật cuốn hút trong 2 giây đầu, phần chính bám theo góc tiếp cận và các ý chính,
kết thúc bằng một câu kêu gọi ngắn (ví dụ rủ bạn bấm vào giỏ hàng / xem thêm).
Lời thoại nên đi cùng nhịp với hình: nói về chi tiết nào thì cố gắng rơi vào lúc cảnh đó đang chiếu.

QUY TẮC BẮT BUỘC (sẽ được kiểm tra tự động):
- Độ dài: phần "spoken" của tất cả các đoạn cộng lại khoảng {target} âm tiết
  (tối thiểu {min_syl}, tối đa {max_syl}); mỗi chữ cách nhau bởi dấu cách là một âm tiết.
- Chia lời thoại thành các đoạn ngắn để làm phụ đề: mỗi "text" tối đa {max_chars} ký tự,
  ngắt ở chỗ ngắt hơi tự nhiên, không cắt giữa một cụm từ.
- "text": dạng hiển thị phụ đề, viết số và đơn vị gọn (4,7g, -56dB, LDAC).
- "spoken": đúng nội dung của "text" nhưng viết ra đúng như sẽ đọc: số, ký hiệu, đơn vị và chữ viết tắt
  viết thành chữ tiếng Việt (ví dụ "4,7g" -> "bốn phẩy bảy gam", "-56dB" -> "âm năm mươi sáu đề xi ben",
  "ANC" -> "a en xê"). Tên thương hiệu đọc theo cách người Việt hay đọc.
- Chỉ dùng thông tin có trong video (hình, chữ, lời nói). Không bịa thông số, không hứa hẹn quá mức.
- Không nêu giá bằng tệ/nhân dân tệ hay quy đổi tiền; có thể nói chung kiểu "giá cực mềm".
- "tts_style": một câu tiếng Anh mô tả cách đọc cho máy đọc (Gemini TTS), ví dụ
  "Speak Vietnamese in an upbeat, energetic, friendly sales tone, fairly fast."
"""

FIX_PROMPT = """\
Lời thoại vừa rồi chưa đạt:
{problems}

Hãy sửa lại và trả về TOÀN BỘ lời thoại theo đúng schema. Nhắc lại: tổng "spoken" khoảng {target} âm tiết
(từ {min_syl} đến {max_syl}), mỗi "text" tối đa {max_chars} ký tự.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tts_style": {"type": "STRING", "description": "English speaking-style instruction for Gemini TTS"},
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING", "description": "Phụ đề hiển thị"},
                    "spoken": {"type": "STRING", "description": "Cách đọc, số và ký hiệu viết thành chữ"},
                },
                "required": ["text", "spoken"],
                "propertyOrdering": ["text", "spoken"],
            },
        },
    },
    "required": ["tts_style", "segments"],
    "propertyOrdering": ["tts_style", "segments"],
}


def log(message: str) -> None:
    print(f"[gemini_voiceover] {message}", file=sys.stderr)


def syllables(text: str) -> int:
    """Vietnamese syllables ~ space-separated words that contain a letter or digit."""
    return sum(1 for word in text.split() if re.search(r"\w", word))


def budget(duration: float, rate: float) -> dict:
    full = duration * rate
    return {
        "target": int(round(full * TARGET_FILL)),
        "min_syl": int(full * MIN_FILL),
        "max_syl": int(full * MAX_FILL),
        "max_chars": MAX_SEGMENT_CHARS,
    }


def find_proxy(proxy_dir: Path, index: int) -> Path:
    matches = sorted(p for p in proxy_dir.glob(f"*video_{index}.mp4") if p.is_file())
    if len(matches) != 1:
        found = ", ".join(p.name for p in matches) or "none"
        raise ReorderError(f"Expected one proxy for video {index} in {proxy_dir} (*video_{index}.mp4), found: {found}")
    return matches[0]


def build_request(plan: dict, video: dict, proxy: Path, limits: dict) -> dict:
    scenes = "\n".join(
        f"  {s['out_start']:.1f}-{s['out_end']:.1f}s: {s.get('description') or '(không mô tả)'}"
        for s in video.get("scenes", [])
    )
    prompt = PROMPT.format(
        duration=float(video.get("total_duration") or 0),
        product=plan.get("product") or "(chưa rõ)",
        aspects="; ".join(plan.get("aspects") or []) or "(chưa rõ)",
        angle=video.get("angle") or "(chưa rõ)",
        title=video.get("title") or "(chưa rõ)",
        summary=video.get("summary") or "(chưa rõ)",
        key_points="; ".join(video.get("key_points") or []) or "(tự chọn từ nội dung video)",
        scenes=scenes or "  (không có)",
        **limits,
    )
    raw = proxy.read_bytes()
    if len(raw) * 4 // 3 > MAX_INLINE_BYTES:
        raise ReorderError(f"Proxy too large to send inline: {proxy} ({len(raw) / 1e6:.1f} MB)")
    return {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(raw).decode("ascii")},
             "videoMetadata": {"fps": CLIP_FPS}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "temperature": 0.9,
        },
    }


def find_problems(answer: dict, limits: dict) -> list[str]:
    segments = answer.get("segments")
    if not isinstance(segments, list) or not segments:
        return ["Không có danh sách 'segments'."]
    problems = []
    for i, seg in enumerate(segments, start=1):
        text = str(seg.get("text", "")).strip() if isinstance(seg, dict) else ""
        spoken = str(seg.get("spoken", "")).strip() if isinstance(seg, dict) else ""
        if not text or not spoken:
            problems.append(f"Đoạn {i}: thiếu 'text' hoặc 'spoken'.")
        elif len(text) > limits["max_chars"]:
            problems.append(f"Đoạn {i}: \"{text}\" dài {len(text)} ký tự, tối đa {limits['max_chars']}.")
    total = sum(syllables(str(s.get("spoken", ""))) for s in segments if isinstance(s, dict))
    if total < limits["min_syl"]:
        problems.append(f"Lời thoại chỉ có {total} âm tiết, quá ngắn (tối thiểu {limits['min_syl']}).")
    elif total > limits["max_syl"]:
        problems.append(f"Lời thoại có {total} âm tiết, quá dài để đọc kịp (tối đa {limits['max_syl']}).")
    if not str(answer.get("tts_style", "")).strip():
        problems.append("Thiếu 'tts_style'.")
    return problems


def write_voiceover(model: str, video: dict, proxy: Path, rate: float, limits: dict,
                    api_key: str, body: dict, attempts_log: list) -> dict:
    """Ask Gemini for one video, feeding problems back; return the checked voice-over."""
    index = video.get("video_index")
    problems: list[str] = []
    for attempt in range(1, ANSWER_ATTEMPTS + 1):
        log(f"Video {index}: calling {model} (attempt {attempt}/{ANSWER_ATTEMPTS})")
        entry = {"video_index": index, "attempt": attempt, "accepted": False}
        try:
            response = call_gemini(body, model, api_key)
        except ReorderError as exc:
            attempts_log.append({**entry, "error": str(exc)})
            raise
        entry["response"] = response
        try:
            answer, model_turn = extract_answer(response)
        except ReorderError as exc:
            problems = [str(exc)]
            attempts_log.append({**entry, "problems": problems})
            log(problems[0])
            continue
        problems = find_problems(answer, limits)
        if not problems:
            attempts_log.append({**entry, "accepted": True, "problems": []})
            segments = [{"text": str(s["text"]).strip(), "spoken": str(s["spoken"]).strip()}
                        for s in answer["segments"]]
            total = sum(syllables(s["spoken"]) for s in segments)
            duration = float(video.get("total_duration") or 0)
            return {
                "video_index": index,
                "proxy_file": proxy.name,
                "title": video.get("title", ""),
                "angle": video.get("angle", ""),
                "duration": duration,
                "tts_style": str(answer["tts_style"]).strip(),
                "script": " ".join(s["spoken"] for s in segments),
                "subtitle_text": " ".join(s["text"] for s in segments),
                "syllables": total,
                "rate": rate,
                "estimated_speech_s": round(total / rate, 2),
                "segments": segments,
            }
        log(f"Video {index}: " + "; ".join(problems))
        fix = FIX_PROMPT.format(problems="\n".join(f"- {p}" for p in problems), **limits)
        attempts_log.append({**entry, "problems": problems, "fix_prompt": fix})
        body["contents"] = body["contents"][:1] + [model_turn, {"role": "user", "parts": [{"text": fix}]}]
    raise ReorderError(f"Video {index}: no valid voice-over after {ANSWER_ATTEMPTS} attempts: "
                       f"{'; '.join(problems)}", E_INVALID_ANSWER)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the Vietnamese voice-over of the planned videos with Gemini.")
    parser.add_argument("reorder_json", type=Path, help="reorder.json from gemini_reorder.py")
    parser.add_argument("proxy_dir", type=Path, help="Directory with the proxy videos (render_videos.py --proxy-dir)")
    parser.add_argument("--output", type=Path, help="Output file (default: voiceover.json next to reorder.json)")
    parser.add_argument("--video", type=int, nargs="+", help="Only (re)write these video indexes")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help=f"Speaking rate in syllables per second (default {DEFAULT_RATE})")
    parser.add_argument("--dry-run", action="store_true", help="Write the requests (without video data), do not call Gemini")
    args = parser.parse_args()

    reorder = args.reorder_json.expanduser().resolve()
    proxy_dir = args.proxy_dir.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else reorder.with_name("voiceover.json"))
    response_file = output.with_name("gem_voiceover_response.json")
    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL

    try:
        if not reorder.is_file():
            raise ReorderError(f"reorder_json must be the reorder.json file, got: {reorder}")
        if not proxy_dir.is_dir():
            raise ReorderError(f"Proxy directory not found: {proxy_dir}")
        if not 2.0 <= args.rate <= 8.0:
            raise ReorderError(f"--rate looks wrong (expected 2-8 syllables per second): {args.rate}")
        try:
            plan = json.loads(reorder.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReorderError(f"Could not read {reorder}: {exc}") from exc
        videos = plan.get("videos") if isinstance(plan, dict) else None
        if not videos:
            raise ReorderError(f"No 'videos' in {reorder}")
        if args.video:
            known = {v.get("video_index") for v in videos}
            unknown = sorted(set(args.video) - known)
            if unknown:
                raise ReorderError(f"--video {unknown} not in reorder.json (videos: {sorted(known)})")
            videos = [v for v in videos if v.get("video_index") in args.video]

        output.parent.mkdir(parents=True, exist_ok=True)
        jobs = []
        for video in videos:
            proxy = find_proxy(proxy_dir, video["video_index"])
            limits = budget(float(video.get("total_duration") or 0), args.rate)
            jobs.append((video, proxy, limits, build_request(plan, video, proxy, limits)))

        if args.dry_run:
            preview = [{"video_index": v["video_index"], "proxy": p.name, "limits": lim,
                        "prompt": b["contents"][0]["parts"][1]["text"]} for v, p, lim, b in jobs]
            dry = output.with_name("voiceover_request.json")
            write_json(dry, {"model": model, "requests": preview})
            print(f"json:{dry}")
            return 0

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == KEY_PLACEHOLDER:
            raise ReorderError("GEMINI_API_KEY is not set (fill it in .env, then recreate the container)", E_AUTH)

        # Keep the voice-overs of videos not regenerated in this run.
        existing = {}
        if args.video and output.is_file():
            try:
                existing = {v["video_index"]: v for v in json.loads(output.read_text(encoding="utf-8")).get("videos", [])}
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                existing = {}

        attempts_log: list = []
        results = dict(existing)
        try:
            for video, proxy, limits, body in jobs:
                result = write_voiceover(model, video, proxy, args.rate, limits, api_key, body, attempts_log)
                results[result["video_index"]] = result
                log(f"Video {result['video_index']}: {len(result['segments'])} segments, {result['syllables']} "
                    f"syllables, ~{result['estimated_speech_s']}s of speech for {result['duration']:.1f}s of video")
        finally:
            write_json(response_file, {"model": model, "attempts": attempts_log})

        write_json(output, {
            "model": model,
            "product": plan.get("product", ""),
            "style": {"tone": "năng động, bán hàng", "speaker": "mình", "viewer": "bạn"},
            "rate": args.rate,
            "videos": [results[k] for k in sorted(results)],
        })
    except ReorderError as exc:
        print(format_error("gemini_voiceover", exc), file=sys.stderr)
        print("fail")
        return 1
    except Exception as exc:  # noqa: BLE001 - always answer json/fail
        print(format_error("gemini_voiceover", GeminiError(f"{exc!r}", E_UNEXPECTED)), file=sys.stderr)
        print("fail")
        return 1

    print(f"json:{output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
