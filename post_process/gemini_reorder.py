#!/usr/bin/env python3
"""Ask Gemini to turn the scenes of a Douyin product video into several short videos.

Usage:
    python gemini_reorder.py <scenelog_json> <clips_dir> <output_json> [--dry-run]

Sends every <clips_dir>/<N>.mp4 (small copies from compress_clips.py) to Gemini
inline as base64, in ONE generateContent request. Each clip is preceded by a
text label "Scene N", where N is the "Scene Number" in scenelog.json. Before the clips
comes the whole, uncut sound of the source video (<job>/input/<awemeId>.mp4, small mono
MP3), so Gemini hears complete sentences and the story; without that file or without sound
the request simply has none. Gemini first writes down the speech ("source_script", with a
Vietnamese translation) and the story ("source_story"), then plans. Both are saved in
<output_json> and used again by gemini_voiceover.py.

Gemini plans several output videos, each with its own angle on the product
(chosen from what the source video actually shows), and returns only the
ORDER of scene numbers for each one. This script then checks the plan:
    - each video lasts at most MAX_VIDEO_S (target TARGET_VIDEO_S) and at least MIN_VIDEO_S
    - a scene appears at most once inside a video
    - every scene is used in at least one video (only with REORDER_USE_ALL_SCENES=1)
    - a scene is used in at most MAX_SCENE_USES videos
    - two videos share at most MAX_SHARED_RATIO of the smaller one's scenes
      (variety), and every run of consecutive scenes they share lasts at most
      MAX_SHARED_RUN_S (no recognisable passage in both)
    - a video never plays the source: scenes that follow each other in the source (n, n+1,
      ...) and are next to each other in a video last at most SOURCE_RUN_MAX_S together
      (Gemini is asked for SOURCE_RUN_TARGET_S)
    - no jump cut: two scenes of the same shot with almost the same picture are never
      played one right after the other (pairs in jump_cuts.json of split_video.py)
    - coherence: Gemini first sorts the scenes into topics (what is shown); a video is made
      of blocks of one topic. At most MAX_SWITCH_RATIO of its joints change the topic, and a
      topic that was left does not come back (short opening and closing blocks excepted)
If a rule is broken, the exact problems are sent back to Gemini (same
conversation, clips included) to fix, up to ANSWER_ATTEMPTS times.
All timestamps are computed from scenelog.json, never taken from Gemini.

Environment:
    GEMINI_API_KEY   required (put it in .env)
    GEMINI_MODEL     optional, default below
    REORDER_TARGET_VIDEOS  optional: make at most this many videos (see video_policy);
                     empty = Gemini decides
    REORDER_MIN_VIDEOS     optional: ... and at least this many; empty = exactly the target
    REORDER_USE_ALL_SCENES optional: 1 = every scene must be used, 0 (default) = the best ones
    REORDER_MAX_SWITCH_PERCENT   optional: MAX_SWITCH_RATIO in percent, default 45; 100 = no limit
    REORDER_KEEP_TOPICS_TOGETHER optional: 1 (default) = a topic never comes back, 0 = rule off
    REORDER_MAX_ATTEMPTS         optional: ANSWER_ATTEMPTS, Gemini calls for a valid plan, default 4
    TARGET_VIDEO_S   optional: target length of each video in seconds, default 15
                     (allowed range follows it, see video_length); 0 = the source length
                     is the target (same range formula), and scenes may be shared
                     by any number of videos
    REORDER_MAX_SCENE_USES      optional: MAX_SCENE_USES, default 2
    REORDER_MAX_SHARED_PERCENT  optional: MAX_SHARED_RATIO in percent, default 50
    REORDER_MAX_SHARED_RUN_S    optional: MAX_SHARED_RUN_S in seconds, default 3
                     (see reuse_limits; not used with TARGET_VIDEO_S=0)
    REORDER_SOURCE_RUN_TARGET_S optional: SOURCE_RUN_TARGET_S in seconds, default 2.7
    REORDER_SOURCE_RUN_MAX_S    optional: SOURCE_RUN_MAX_S in seconds, default 2.9
                     (see reuse_limits; also used with TARGET_VIDEO_S=0)
    REORDER_AVOID_JUMP_CUTS     optional: 1 (default) = no jump cuts, 0 = rule off

Files (next to <output_json>):
    request.json        the request body sent to Gemini (first call; retries
                        add the model's answer + the fix prompt, which are
                        recorded in gem_response.json)
    gem_response.json   raw Gemini response of every attempt, with the rule
                        problems found and the fix prompt sent back; written
                        after each attempt, so it also exists after a failure
    <output_json>       the final plan (Gemini's choices + timestamps computed
                        from scenelog.json); only written on success
    stats.json          "overlaps" (find_overlaps) and "source_overlaps" (find_source_overlaps) are
                        set on success, removed when a run starts; the keys of
                        split_video.py are kept (job_stats.py)

--dry-run only writes request.json, without calling Gemini (useful to inspect
size and prompt).

Stdout: 'json:<absolute output_json>' on success ('json:<request.json>' with
        --dry-run), or exactly 'fail'. On success the 'json:' line is followed by
        'overlap:<videos>-<scenes>-<seconds>' lines (runs of 2+ consecutive scenes that
        two or more videos share, one line per run) and
        'overlap_video:<video>-<shared s>_<video s>' lines (see overlap_lines; also saved
        in stats.json -> "overlaps"), then 'source_overlap:<video>-<scenes>-<seconds>' and
        'source_overlap_video:<video>-<seconds in passages>_<video s>_<longest s>' lines
        (output videos compared with the source, see source_overlap_lines; also saved in
        stats.json -> "source_overlaps"). Read the path with /^json:(.*)$/m.
Stderr: progress; a failure is one line "[gemini_reorder] <CODE>: <message>" with the codes
        of gemini_errors.py (E_QUOTA_DAILY is reported at once, without retrying).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import base64
import json
import math
import os
import subprocess
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
    from post_process.job_stats import stats_path, update_stats
except ImportError:  # python post_process/gemini_reorder.py
    from gemini_errors import (E_AUTH, E_INVALID_ANSWER, E_NETWORK, E_NO_OUTPUT, E_UNEXPECTED,
                               GeminiError, format_error, http_error, is_daily_quota, parse_retry_after)
    from job_stats import stats_path, update_stats

DEFAULT_MODEL = "gemini-3.8-flash"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
KEY_PLACEHOLDER = "your-gemini-api-key-here"

# Output video rules. TARGET_VIDEO_S comes from .env (see video_length); MAX/MIN follow it.
DEFAULT_TARGET_VIDEO_S = 15.0
VIDEO_S_TOLERANCE = 5.0  # allowed distance from the target, both ways
TARGET_VIDEO_S = DEFAULT_TARGET_VIDEO_S
MAX_VIDEO_S = TARGET_VIDEO_S + VIDEO_S_TOLERANCE    # 20 s by default
MIN_VIDEO_S = TARGET_VIDEO_S - VIDEO_S_TOLERANCE    # 10 s by default
# TARGET_VIDEO_S=0: the target is the source length (min/max follow it as for any target);
# a video need not use every scene. The reuse limits below do not apply: several
# videos about as long as the source cannot help sharing most scenes.
FULL_SOURCE = False
# Reuse limits, from .env (see reuse_limits); these are the defaults.
MAX_SCENE_USES = 2       # a scene may appear in at most this many videos
MAX_SHARED_RATIO = 0.5   # shared scenes / scene count of the smaller video
MAX_SHARED_RUN_S = 3.0   # longest run of consecutive scenes (same scenes, same order) two videos
                         # may share, in seconds; 0 = no limit
# Scenes that follow each other in the source, next to each other in a video: how long
# such a run may last. The target is asked for, the maximum is checked; 0 = no limit.
# Coherence, from .env (see coherence_limits)
USE_ALL_SCENES = False
MAX_SWITCH_RATIO = 0.45     # joints of a video that may change the topic; 1 = no limit
KEEP_TOPICS_TOGETHER = True
EDGE_BLOCK_SCENES = 2       # an opening/closing block of at most this many scenes may show any topic
MIN_TOPICS, MIN_TOPICS_SCENES = 3, 10  # with at least this many scenes: at least this many topics
MAX_TOPIC_SHARE = 0.6       # ... and no topic with more than this share of the scenes
JUMP_CUTS: set[tuple[int, int]] = set()  # (a, b): scene b must not come right after scene a
SOURCE_RUN_TARGET_S = 2.7
SOURCE_RUN_MAX_S = 2.9

MAX_INLINE_BYTES = 20 * 1024 * 1024  # documented safe limit for a whole inline request
AUDIO_BITRATE = "48k"  # source sound sent to Gemini: mono, speech quality
CLIP_FPS = 6           # clips are ~0.6-2.5 s; the default 1 FPS would see only 1-2 frames
REQUEST_TIMEOUT_S = 600
HTTP_ATTEMPTS = 4      # retries on 429 / 5xx / network errors
ANSWER_ATTEMPTS = 4    # Gemini calls allowed to produce a valid plan (REORDER_MAX_ATTEMPTS)

PROMPT = """\
Bạn là biên tập viên video ngắn chuyên làm nội dung bán hàng trên Douyin/TikTok.

Bạn nhận được {count} phân đoạn (scene) liên tiếp được cắt ra từ MỘT video giới thiệu sản phẩm
(tổng {total:.1f} giây). Sản phẩm có thể là bất kỳ thứ gì (đồ điện tử, mỹ phẩm, thời trang, đồ gia dụng,
thực phẩm...): hãy tự nhận biết sản phẩm và những khía cạnh mà video đang trình bày, từ hình ảnh,
chữ trên màn hình và âm thanh.
Mỗi phân đoạn có nhãn "Scene N" ngay trước nó (và chữ "SCENE N" ở góc hình). N chính là scene_number.
Các scene đang theo thứ tự gốc: Scene 1 là đầu video, Scene {count} là cuối video.
{audio_note}

Thời lượng từng scene (giây):
{durations}

VIỆC ĐẦU TIÊN: hiểu trọn video gốc trước khi lên kế hoạch. Nghe lời nói, đọc phụ đề trên hình, rồi viết:
- "source_script": nguyên văn TOÀN BỘ lời nói của video gốc, đúng thứ tự. Số liệu và tên riêng phải đúng như nghe được và đúng
  với phụ đề trên hình; hai bên khác nhau thì theo phụ đề. Không thêm bớt số 0 hay dời dấu thập phân.
- "source_script_vi": bản dịch tiếng Việt đầy đủ, giữ nguyên mọi số liệu.
- "source_story": mạch kể của video gốc trong 3-6 câu (mở đầu, vấn đề, sản phẩm, chứng minh từng tính năng, kêu gọi).
Video không có lời nói thì để "source_script" và "source_script_vi" là chuỗi rỗng.
Dùng mạch kể này để chia chủ đề, chọn góc, gom cụm và sắp thứ tự cụm: cảnh nào minh hoạ cho ý nào thì đi cùng nhau.

NHIỆM VỤ: từ các scene này, lên kế hoạch cho NHIỀU video ngắn mới để đăng riêng.
- Mỗi video có MỘT góc tiếp cận (angle) riêng, dựa trên những khía cạnh mà video gốc THỰC SỰ thể hiện.
  Ví dụ với tai nghe: một video về độ bền/chất lượng hoàn thiện, một video về tính năng/công dụng,
  một video về trải nghiệm âm thanh/chống ồn. Đây chỉ là ví dụ: hãy chọn góc phù hợp với nội dung thật,
  đừng ép một góc mà video không có đủ cảnh để thể hiện.
- Nếu một khía cạnh không đủ cảnh để làm thành một video đủ thời lượng, được phép kết hợp nhiều khía cạnh
  vào cùng một video (khi đó ghi rõ các khía cạnh trong "angle"). Điều quan trọng nhất là thời lượng
  của từng video đúng quy tắc bên dưới.
{count_rule}
- TRƯỚC HẾT, chia các scene thành 4-8 chủ đề theo NỘI DUNG HÌNH (vật gì, đang làm gì; ví dụ "mở hộp", "đeo lên tai",
  "màn hình ứng dụng", "cận cảnh hộp sạc"), ghi vào "topics". Mỗi scene thuộc đúng một chủ đề. Các scene của cùng một hành
  động (bấm nút rồi ra kết quả, mở nắp rồi lấy sản phẩm) phải cùng chủ đề.
- Mỗi video nên có cấu trúc riêng phù hợp với góc của nó: hook mạnh ở 1-2 giây đầu, phần chính, kết
  (cảnh sản phẩm tổng thể / kêu gọi hành động). Các video phải khác nhau rõ rệt về cảnh mở đầu và thứ tự,
  để người xem không nhận ra chúng được dựng từ cùng một video.

QUY TẮC BẮT BUỘC (sẽ được kiểm tra tự động):
{length_rule}
{coherence_rule}- Trong một video, mỗi scene xuất hiện tối đa một lần.
{coverage_rule}
{reuse_rules}
- Chỉ trả về scene_number theo thứ tự trong từng video. KHÔNG tự tính timestamp.
{source_rule}{jump_rule}- Giữ cạnh nhau những scene thuộc cùng một hành động/câu nói liên tục nếu tách ra sẽ vô nghĩa,
  miễn là không vi phạm các quy tắc ở trên.
- "key_points": 2-3 ý chính cụ thể (thông số, lợi ích, điểm khác biệt) mà lời thoại của video đó nên nói,
  lấy từ những gì video gốc thực sự thể hiện (hình, chữ trên màn hình, lời nói). Không bịa thông số;
  số liệu trong "title", "summary", "key_points" phải đúng như trong "source_script" hoặc chữ trên hình.
- Viết "angle", "title", "summary", "key_points", "description", "reason" bằng tiếng Việt.
"""

FIX_PROMPT = """\
Kế hoạch vừa rồi vi phạm quy tắc:
{problems}

Hãy sửa lại và trả về TOÀN BỘ kế hoạch theo đúng schema, giữ các góc tiếp cận còn hợp lệ.
Không cần viết lại "source_script", "source_script_vi", "source_story": để chuỗi rỗng (đã có từ lần trước).
Nếu một video quá ngắn vì góc của nó thiếu cảnh, hãy kết hợp thêm khía cạnh khác vào video đó
hoặc gộp nó với video khác. Khi phải tách hai scene, ưu tiên xen scene CÙNG chủ đề để không làm đứt mạch.
Nhắc lại: {length_fix}, {count_fix}{coverage_fix}
{coherence_fix}{reuse_fix}{source_fix}{jump_fix}không lặp scene trong cùng một video.
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
        "source_script": {"type": "STRING", "description": "Nguyên văn toàn bộ lời nói của video gốc"},
        "source_script_vi": {"type": "STRING", "description": "Bản dịch tiếng Việt của source_script"},
        "source_story": {"type": "STRING", "description": "Mạch kể của video gốc, tiếng Việt"},
        "aspects": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Các khía cạnh của sản phẩm mà video gốc thể hiện",
        },
        "topics": {
            "type": "ARRAY",
            "description": "Chủ đề theo nội dung hình; mỗi scene thuộc đúng một chủ đề",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING", "description": "Tên chủ đề, ngắn gọn"},
                    "scenes": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                },
                "required": ["name", "scenes"],
                "propertyOrdering": ["name", "scenes"],
            },
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
    "required": ["product", "source_script", "source_script_vi", "source_story", "aspects", "topics", "videos"],
    "propertyOrdering": ["product", "source_script", "source_script_vi", "source_story", "aspects", "topics", "videos"],
}


ReorderError = GeminiError  # message + error code (see gemini_errors.py)
MAX_RETRY_WAIT_S = 120      # a 429 asking to wait longer than this is not retried


def log(message: str) -> None:
    print(f"[gemini_reorder] {message}", file=sys.stderr)


def whole_number(name: str) -> int | None:
    """A whole number of at least 1 from the environment, or None when empty."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ReorderError(f"{name} must be a whole number: {raw!r}") from exc
    if value < 1:
        raise ReorderError(f"{name} must be at least 1: {value}")
    return value


def target_videos() -> int | None:
    """REORDER_TARGET_VIDEOS from the environment (.env): most videos to make, or None."""
    return whole_number("REORDER_TARGET_VIDEOS")


def min_video_count(target: int | None) -> int | None:
    """REORDER_MIN_VIDEOS: fewest videos to make; empty = exactly the target."""
    value = whole_number("REORDER_MIN_VIDEOS")
    return target if value is None or target is None else min(value, target)


def switch(name: str, default: bool) -> bool:
    """A 0/1 setting from the environment."""
    raw = os.environ.get(name, "").strip()
    if raw not in ("", "0", "1"):
        raise ReorderError(f"{name} must be 0 or 1: {raw!r}")
    return default if raw == "" else raw == "1"


def coherence_limits() -> None:
    """USE_ALL_SCENES, MAX_SWITCH_RATIO and KEEP_TOPICS_TOGETHER from the environment (.env)."""
    global USE_ALL_SCENES, MAX_SWITCH_RATIO, KEEP_TOPICS_TOGETHER
    USE_ALL_SCENES = switch("REORDER_USE_ALL_SCENES", False)
    KEEP_TOPICS_TOGETHER = switch("REORDER_KEEP_TOPICS_TOGETHER", True)
    raw = os.environ.get("REORDER_MAX_SWITCH_PERCENT", "").strip()
    try:
        percent = float(raw) if raw else 45.0
    except ValueError:
        percent = -1.0
    if not 0 <= percent <= 100:
        raise ReorderError(f"REORDER_MAX_SWITCH_PERCENT must be a number from 0 to 100: {raw!r}")
    MAX_SWITCH_RATIO = percent / 100


def video_length() -> None:
    """TARGET_VIDEO_S from the environment (.env): target length of each video, in seconds.

    Empty = DEFAULT_TARGET_VIDEO_S. Each video must then last between
    target - VIDEO_S_TOLERANCE and target + VIDEO_S_TOLERANCE
    (15 -> 10..20 s, 10 -> 5..15 s, 30 -> 25..35 s).
    0 = the source length (FULL_SOURCE): set by use_full_source once the scenes are loaded.
    """
    global TARGET_VIDEO_S, MAX_VIDEO_S, MIN_VIDEO_S, FULL_SOURCE
    raw = os.environ.get("TARGET_VIDEO_S", "").strip()
    try:
        target = float(raw) if raw else DEFAULT_TARGET_VIDEO_S
    except ValueError as exc:
        raise ReorderError(f"TARGET_VIDEO_S must be a number of seconds: {raw!r}") from exc
    if target == 0:
        FULL_SOURCE = True
        MAX_VIDEO_S = math.inf  # no limit until the source length is known
        return
    if not math.isfinite(target) or target - VIDEO_S_TOLERANCE < 1:
        raise ReorderError(f"TARGET_VIDEO_S must be 0 (full source) or at least "
                           f"{VIDEO_S_TOLERANCE + 1:.0f}: {raw!r}")
    TARGET_VIDEO_S = target
    MAX_VIDEO_S = target + VIDEO_S_TOLERANCE
    MIN_VIDEO_S = target - VIDEO_S_TOLERANCE


def reuse_limits() -> None:
    """How much the videos may share, from the environment (.env). Empty = the default.

    REORDER_MAX_SCENE_USES      whole number, 1 or more: videos a scene may appear in.
    REORDER_MAX_SHARED_PERCENT  0-100: share of common scenes two videos may have (keeps the
                                videos varied). 100 = no limit.
    REORDER_MAX_SHARED_RUN_S    seconds, 0 or more: longest shared run two videos may have,
                                whatever their share of common scenes. 0 = no limit.
    REORDER_SOURCE_RUN_TARGET_S seconds, 0 or more: what Gemini is asked to keep the passages
                                of the source inside a video to.
    REORDER_SOURCE_RUN_MAX_S    seconds, 0 or more: longest such passage accepted. 0 = no limit.
    """
    global MAX_SCENE_USES, MAX_SHARED_RATIO, MAX_SHARED_RUN_S, SOURCE_RUN_TARGET_S, SOURCE_RUN_MAX_S

    def number(name: str, default: float, low: float, high: float, what: str) -> float:
        raw = os.environ.get(name, "").strip()
        try:
            value = float(raw) if raw else default
        except ValueError as exc:
            raise ReorderError(f"{name} must be {what}: {raw!r}") from exc
        if not low <= value <= high or (isinstance(default, int) and value != int(value)):
            raise ReorderError(f"{name} must be {what}: {raw!r}")
        return value

    MAX_SCENE_USES = int(number("REORDER_MAX_SCENE_USES", 2, 1, 1000, "a whole number of at least 1"))
    MAX_SHARED_RATIO = number("REORDER_MAX_SHARED_PERCENT", 50.0, 0, 100, "a number from 0 to 100") / 100
    MAX_SHARED_RUN_S = number("REORDER_MAX_SHARED_RUN_S", 3.0, 0, 3600, "a number of seconds from 0 to 3600")
    SOURCE_RUN_MAX_S = number("REORDER_SOURCE_RUN_MAX_S", 2.9, 0, 3600, "a number of seconds from 0 to 3600")
    SOURCE_RUN_TARGET_S = number("REORDER_SOURCE_RUN_TARGET_S", 2.7, 0, 3600, "a number of seconds from 0 to 3600")
    if SOURCE_RUN_MAX_S > 0:  # the target is never above the maximum; 0 = same as the maximum
        SOURCE_RUN_TARGET_S = min(SOURCE_RUN_TARGET_S or SOURCE_RUN_MAX_S, SOURCE_RUN_MAX_S)


def load_jump_cuts(path: Path, scenes: list[dict]) -> None:
    """Pairs of scenes that must not follow each other (split_video.py); no file = no rule."""
    global JUMP_CUTS
    JUMP_CUTS = set()
    if not switch("REORDER_AVOID_JUMP_CUTS", True):
        return
    try:
        pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
        known = {s["scene_number"] for s in scenes}
        JUMP_CUTS = {(int(a), int(b)) for a, b in pairs if int(a) in known and int(b) in known}
    except FileNotFoundError:
        log(f"{path.name} not found (older job): jump cuts are not checked")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log(f"WARNING: could not read {path}: {exc!r}; jump cuts are not checked")


def use_full_source(scenes: list[dict]) -> None:
    """FULL_SOURCE: the target is the source length; minimum and maximum follow it as in
    video_length (a source of 49 s -> 44..54 s; the maximum cannot be reached without
    repeating a scene, so in practice a video lasts at most the source)."""
    global TARGET_VIDEO_S, MAX_VIDEO_S, MIN_VIDEO_S
    TARGET_VIDEO_S = sum(s["duration"] for s in scenes)
    MAX_VIDEO_S = TARGET_VIDEO_S + VIDEO_S_TOLERANCE
    MIN_VIDEO_S = max(1.0, TARGET_VIDEO_S - VIDEO_S_TOLERANCE)


def log_error(error: Exception, script: str = "gemini_reorder") -> None:
    print(format_error(script, error), file=sys.stderr)


def min_videos(scenes: list[dict]) -> int:
    """Fewest videos that can cover every scene within MAX_VIDEO_S each."""
    return max(1, math.ceil(sum(s["duration"] for s in scenes) / MAX_VIDEO_S))


def video_policy(scenes: list[dict], target: int | None) -> dict:
    """How many videos to make, and whether every scene must be used.

    target None: Gemini decides (at least min_videos when every scene must be used).
    target N: at most N videos and at least min_video_count (default: exactly N).
    N is lowered when the source is too short for N videos of
    MIN_VIDEO_S (each scene may be reused MAX_SCENE_USES times, except a scene longer than
    MAX_SHARED_RUN_S, which fits in one video only). When N videos of at most
    MAX_VIDEO_S cannot hold every scene (long source), N is kept and the "every scene"
    rule is dropped: Gemini picks the best scenes. Without USE_ALL_SCENES it always does.
    """
    total = sum(s["duration"] for s in scenes)
    least = min_video_count(target)

    def count_texts(most: int) -> dict:
        fewest = min(least, most)
        if fewest == most:
            return {"min_count": most, "count_rule": f"- Tạo ĐÚNG {most} video (không hơn, không kém).",
                    "count_fix": f"đúng {most} video, "}
        return {"min_count": fewest, "count_fix": f"từ {fewest} đến {most} video, ", "count_rule": (
            f"- Tạo từ {fewest} đến {most} video. Ưu tiên {most}, nhưng mỗi video phải có góc riêng với đủ cảnh hợp góc đó;\n"
            "  thà ít video mà hay, KHÔNG gom cảnh thừa thành một video.")}

    if FULL_SOURCE:  # scenes may be reused freely: any count fits, nothing to lower or drop
        texts = (count_texts(target) if target is not None else
                 {"count_rule": "- Số lượng video do bạn quyết định theo nội dung, tối thiểu 1 video.", "count_fix": ""})
        return {"target": target, "count": target, "cover_all": USE_ALL_SCENES, "notes": [], **texts}
    needed = min_videos(scenes) if USE_ALL_SCENES else 1
    if target is None:
        return {"target": None, "count": None, "cover_all": USE_ALL_SCENES, "notes": [],
                "count_rule": f"- Số lượng video do bạn quyết định theo nội dung, tối thiểu {needed} video.",
                "count_fix": ""}
    notes, count = [], target
    footage = sum(s["duration"] * (1 if 0 < MAX_SHARED_RUN_S < s["duration"] else MAX_SCENE_USES) for s in scenes)
    possible = max(1, int(footage // MIN_VIDEO_S))
    if count > possible:
        notes.append(f"{target} videos need more footage than {total:.1f}s; making {possible}")
        count = possible
    cover_all = USE_ALL_SCENES and min(least, count) >= needed
    if USE_ALL_SCENES and not cover_all:
        notes.append(f"{count} videos of at most {MAX_VIDEO_S:.0f}s cannot use all {total:.1f}s: "
                     "the best scenes are picked, not every scene")
    return {"target": target, "count": count, "cover_all": cover_all, "notes": notes, **count_texts(count)}


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


def rule_values(scenes: list[dict], policy: dict) -> dict:
    n = len(scenes)
    if policy["cover_all"]:
        coverage_rule = f"- MỌI scene từ 1 đến {n} phải được dùng trong ít nhất một video."
        coverage_fix = f"mọi scene 1-{n} được dùng ít nhất một lần, "
    else:
        coverage_rule = ("- KHÔNG bắt buộc dùng hết các scene: chọn những scene hay nhất và hợp nhất cho từng góc, "
                         "bỏ scene không ăn nhập.")
        coverage_fix = ""
    values = {
        "count": n,
        "count_rule": policy["count_rule"],
        "count_fix": policy["count_fix"],
        "length_rule": (f"- Thời lượng mỗi video (tổng thời lượng các scene của nó) khoảng {TARGET_VIDEO_S:.0f} giây,\n"
                        f"  KHÔNG dưới {MIN_VIDEO_S:.0f} giây và KHÔNG quá {MAX_VIDEO_S:.0f} giây. "
                        "Hãy cộng thời lượng ở bảng trên cẩn thận."),
        "length_fix": f"mỗi video {MIN_VIDEO_S:.0f}-{MAX_VIDEO_S:.0f} giây (mục tiêu {TARGET_VIDEO_S:.0f} giây)",
        "coverage_rule": coverage_rule,
        "coverage_fix": coverage_fix,
        "reuse_rules": (f"- Một scene được dùng trong tối đa {MAX_SCENE_USES} video.\n"
                        f"- Hai video bất kỳ dùng chung không quá {MAX_SHARED_RATIO * 100:g}% "
                        "số scene của video ngắn hơn."),
        "reuse_fix": f"mỗi scene tối đa {MAX_SCENE_USES} video, ",
    }
    if MAX_SHARED_RUN_S > 0:
        values["reuse_rules"] += (
            f"\n- Được dùng lại scene giữa các video để đủ thời lượng, nhưng MỌI đoạn trùng giữa hai video bất kỳ "
            f"dài không quá {MAX_SHARED_RUN_S:g} giây.\n"
            "  Đoạn trùng = các scene liên tiếp giống nhau, cùng thứ tự, ở cả hai video (một scene đứng riêng cũng là\n"
            "  một đoạn); độ dài là tổng thời lượng các scene trong đoạn. Muốn cắt ngắn đoạn trùng thì đổi thứ tự\n"
            "  hoặc xen scene khác vào giữa. Ví dụ video A: 1, 2, 3, 7, 4, 5 và video B: 4, 5, 9, 1, 2, 3\n"
            "  có hai đoạn trùng: 1-2-3 và 4-5.")
        values["reuse_fix"] += f"mỗi đoạn trùng giữa hai video không quá {MAX_SHARED_RUN_S:g} giây, "
    if FULL_SOURCE:  # videos about as long as the source cannot help sharing most scenes
        values["reuse_rules"] = ("- Một scene được dùng trong bao nhiêu video cũng được, nhưng các video phải khác nhau "
                                 "rõ rệt về lựa chọn và thứ tự scene, nhất là các scene mở đầu "
                                 "(không có hai video giống hệt nhau).")
        values["reuse_fix"] = "các video khác nhau về scene và thứ tự, "
    values["source_rule"] = values["source_fix"] = ""
    if SOURCE_RUN_MAX_S > 0:
        values["source_rule"] = (
            "- KHÔNG phát lại video gốc: trong một video, các scene có số tăng dần liền nhau (như 1, 2, 3 hoặc 4, 5, 6)\n"
            f"  mà đặt cạnh nhau thì tổng thời lượng của chúng nên dưới {SOURCE_RUN_TARGET_S:g} giây, "
            f"và KHÔNG ĐƯỢC quá {SOURCE_RUN_MAX_S:g} giây\n"
            f"  (chỉ dùng khoảng {SOURCE_RUN_TARGET_S:g}-{SOURCE_RUN_MAX_S:g} giây khi nhờ vậy video hay hơn hẳn). "
            "Áp dụng cho mọi đoạn từ 2 scene trở lên.\n"
            "  Đoạn sắp vượt thì nối sang scene khác cùng chủ đề nhưng không liền số (như 1, 2, 5, 6), không chen chủ đề khác.\n"
            "  Trước khi trả lời, hãy rà từng video và cộng thời lượng từng đoạn như vậy.\n")
        values["source_fix"] = (f"mỗi đoạn scene có số liền nhau tăng dần trong một video không quá "
                                f"{SOURCE_RUN_MAX_S:g} giây (nên dưới {SOURCE_RUN_TARGET_S:g} giây), ")
    values["coherence_rule"] = (
        "- MẠCH LẠC (quan trọng nhất với người xem): dựng mỗi video thành các CỤM, mỗi cụm là 2-5 scene CÙNG chủ đề đặt cạnh nhau,\n"
        "  kể trọn một ý rồi mới sang cụm khác. Trong cụm, giữ các cặp số liền nhau (n rồi n+1) khi còn trong giới hạn thời lượng;\n"
        "  khi chạm giới hạn thì nhảy sang scene khác CÙNG chủ đề, đừng chen chủ đề khác vào. Hành động phải đi trọn: đã bấm nút\n"
        "  hay mở nắp thì scene sau phải là kết quả của nó. Thứ tự các cụm theo một mạch kể: mở đầu gây chú ý, vấn đề, sản phẩm,\n"
        "  chứng minh, kêu gọi.\n")
    values["coherence_fix"] = "dựng theo cụm cùng chủ đề, hành động đi trọn, "
    if KEEP_TOPICS_TOGETHER:
        values["coherence_rule"] += (
            "- Trong một video, mỗi chủ đề chỉ xuất hiện thành MỘT cụm: đã rời chủ đề thì KHÔNG quay lại (không giới thiệu tính năng\n"
            f"  một chút, chuyển cảnh khác, rồi quay lại tính năng đó). Chỉ miễn cho cụm mở đầu và cụm kết dài không quá {EDGE_BLOCK_SCENES} scene.\n")
        values["coherence_fix"] += "đã rời chủ đề thì không quay lại (trừ cụm mở đầu và cụm kết ngắn), "
    if MAX_SWITCH_RATIO < 1:
        values["coherence_rule"] += (
            f"- Trong một video, số chỗ nối đổi chủ đề không quá {MAX_SWITCH_RATIO * 100:g}% số chỗ nối "
            f"(ví dụ 13 scene có 12 chỗ nối: tối đa {int(MAX_SWITCH_RATIO * 12 + 1e-9)} lần đổi).\n")
        values["coherence_fix"] += f"đổi chủ đề không quá {MAX_SWITCH_RATIO * 100:g}% số chỗ nối, "
    values["jump_rule"] = values["jump_fix"] = ""
    if JUMP_CUTS:
        after: dict[int, list[int]] = {}
        for a, b in sorted(JUMP_CUTS):
            after.setdefault(a, []).append(b)
        pairs = "\n".join(f"  ngay sau scene {a} KHÔNG được là: {', '.join(map(str, found))}"
                          for a, found in after.items())
        values["jump_rule"] = (
            "- KHÔNG để giật hình. Hai scene cùng góc quay chỉ mượt khi nối tiếp nhau thật trong video gốc: scene n rồi ngay sau\n"
            "  là n+1 (như 20 rồi 21) thì ĐƯỢC nối thẳng, miễn là đúng giới hạn thời lượng ở trên. Cùng góc quay nhưng thiếu một\n"
            "  đoạn ở giữa (như 20 rồi 22) hoặc ngược thứ tự (như 21 rồi 20) thì hình nhảy, trông như bị lỗi.\n"
            f"  Các cặp KHÔNG được nối thẳng trong video này (xen scene khác vào giữa, ưu tiên cùng chủ đề):\n{pairs}\n")
        values["jump_fix"] = "không nối thẳng hai scene nằm trong danh sách gây giật hình, "
    return values


def source_audio(scenelog: Path) -> bytes | None:
    """The whole sound of the source video as a small mono MP3, or None (not found / no sound)."""
    video = scenelog.parent.parent.parent / "input" / f"{scenelog.parent.name}.mp4"
    if not video.is_file():
        log(f"Source video not found, planning without its whole sound: {video}")
        return None
    try:
        result = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-vn", "-ac", "1", "-ar", "24000",
                                 "-c:a", "libmp3lame", "-b:a", AUDIO_BITRATE, "-f", "mp3", "-"],
                                capture_output=True, check=False)
    except OSError as exc:
        log(f"WARNING: could not run ffmpeg, planning without the whole sound: {exc}")
        return None
    if result.returncode != 0 or not result.stdout:
        reason = (result.stderr.decode(errors="replace").strip().splitlines() or ["no sound"])[0][:200]
        log(f"No sound from {video.name}, planning without it: {reason}")
        return None
    return result.stdout


def build_request(scenes: list[dict], clips_dir: Path, policy: dict, audio: bytes | None = None) -> dict:
    durations = "\n".join(f"  Scene {s['scene_number']}: {s['duration']:.3f}" for s in scenes)
    prompt = PROMPT.format(
        durations=durations,
        total=sum(s["duration"] for s in scenes),
        audio_note=("Trước các scene là TOÀN BỘ ÂM THANH của video gốc, liền mạch từ đầu đến cuối: dùng nó để hiểu trọn câu nói và\n"
                    "mạch kể (âm thanh trong từng scene bị cắt ngang câu)." if audio else ""),
        **rule_values(scenes, policy),
    )
    parts = [{"text": prompt}]

    total = 0
    if audio:
        total += len(audio)
        parts += [{"text": "Âm thanh gốc của cả video:"},
                  {"inlineData": {"mimeType": "audio/mp3", "data": base64.b64encode(audio).decode("ascii")}}]
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
    log(f"{len(scenes)} clips{f' + whole sound ({len(audio) / 1e3:.0f} kB)' if audio else ''}, {total / 1e6:.1f} MB raw, ~{encoded / 1e6:.1f} MB as base64")
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


def shared_runs(plan_a: list[int], plan_b: list[int]) -> list[list[int]]:
    """Maximal runs of consecutive scenes (same scenes, same order) two plans share.

    Runs of a single scene included. A scene appears once per plan.
    """
    position_b = {n: j for j, n in enumerate(plan_b)}
    runs, i = [], 0
    while i < len(plan_a):
        j = position_b.get(plan_a[i])
        length = 0
        while j is not None and i + length < len(plan_a) and j + length < len(plan_b) \
                and plan_a[i + length] == plan_b[j + length]:
            length += 1
        if length:
            runs.append(plan_a[i:i + length])
        i += max(length, 1)
    return runs


def source_runs(plan: list[int]) -> list[list[int]]:
    """Passages of the source in a plan: 2 or more scenes n, n+1, ... next to each other."""
    runs: list[list[int]] = []
    for number in plan:
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return [run for run in runs if len(run) > 1]


def read_topics(answer: dict, durations: dict[int, float]) -> tuple[dict[int, str], list[str]]:
    """({scene: topic name}, problems) of the answer's "topics"."""
    if MAX_SWITCH_RATIO >= 1 and not KEEP_TOPICS_TOGETHER:
        return {}, []  # nothing depends on them
    topics: dict[int, str] = {}
    twice, sizes = set(), {}
    for item in answer.get("topics") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip() or f"chủ đề {len(sizes) + 1}"
        for number in item.get("scenes") or []:
            if number not in durations:
                continue
            if number in topics and topics[number] != name:
                twice.add(number)
            topics.setdefault(number, name)
        sizes[name] = sum(t == name for t in topics.values())
    problems = []
    missing = [n for n in durations if n not in topics]
    if missing:
        problems.append(f"Scene chưa thuộc chủ đề nào trong 'topics': {missing}.")
    if twice:
        problems.append(f"Scene thuộc nhiều chủ đề (chỉ được một): {sorted(twice)}.")
    if len(durations) >= MIN_TOPICS_SCENES and topics:
        biggest = max(sizes, key=lambda t: sizes[t])
        if len(sizes) < MIN_TOPICS:
            problems.append(f"Chỉ có {len(sizes)} chủ đề, cần ít nhất {MIN_TOPICS} (chia theo nội dung hình).")
        elif sizes[biggest] > MAX_TOPIC_SHARE * len(durations):
            problems.append(f"Chủ đề '{biggest}' có {sizes[biggest]}/{len(durations)} scene, quá rộng: hãy chia nhỏ theo nội dung hình.")
    return topics, problems


def topic_blocks(plan: list[int], topics: dict[int, str]) -> list[tuple[str, list[int]]]:
    """A plan as blocks of neighbouring scenes of one topic."""
    blocks: list[tuple[str, list[int]]] = []
    for number in plan:
        topic = topics.get(number, f"scene {number}")
        if blocks and blocks[-1][0] == topic:
            blocks[-1][1].append(number)
        else:
            blocks.append((topic, [number]))
    return blocks


def coherence_problems(index: int, plan: list[int], topics: dict[int, str]) -> list[str]:
    """Too many topic changes, or a topic that was left and comes back."""
    blocks = topic_blocks(plan, topics)
    if not topics or len(blocks) < 3:
        return []
    shown = " | ".join(f"{topic}: {', '.join(map(str, scenes))}" for topic, scenes in blocks)
    problems = []
    joints, changes = len(plan) - 1, len(blocks) - 1
    allowed = int(MAX_SWITCH_RATIO * joints + 1e-9)
    if changes > allowed:
        problems.append(f"Video {index}: rời rạc, đổi chủ đề {changes}/{joints} chỗ nối (tối đa {allowed}). "
                        f"Các cụm hiện tại: {shown}. Hãy gom scene cùng chủ đề lại cạnh nhau.")
    if KEEP_TOPICS_TOGETHER:
        first = 1 if len(blocks[0][1]) <= EDGE_BLOCK_SCENES else 0
        last = len(blocks) - 1 if len(blocks[-1][1]) <= EDGE_BLOCK_SCENES else len(blocks)
        names = [topic for topic, _ in blocks[first:last]]
        back = sorted({t for t in names if names.count(t) > 1})
        if back:
            problems.append(f"Video {index}: rời chủ đề rồi quay lại: {', '.join(back)}. Các cụm hiện tại: {shown}. "
                            "Hãy gom mỗi chủ đề thành một cụm liền.")
    return problems


def coherence_stats(videos: list[dict], topics: dict[int, str]) -> dict:
    """How coherent the final plan is (stats.json -> "coherence")."""
    per_video = []
    for video in videos:
        plan = [s["scene_number"] for s in video["scenes"]]
        blocks = topic_blocks(plan, topics)
        per_video.append({
            "video_index": video["video_index"], "joints": len(plan) - 1,
            "topic_changes": len(blocks) - 1 if topics else None,
            "source_neighbours": sum(b == a + 1 for a, b in zip(plan, plan[1:])),
            "blocks": [{"topic": topic, "scenes": scenes} for topic, scenes in blocks] if topics else [],
        })
    return {"max_switch_percent": round(MAX_SWITCH_RATIO * 100, 1), "per_video": per_video}


def find_problems(answer: dict, scenes: list[dict], policy: dict) -> list[str]:
    """Check the plan against the rules; return human-readable problems (empty = valid)."""
    durations = {s["scene_number"]: s["duration"] for s in scenes}
    videos = answer.get("videos")
    if not isinstance(videos, list) or not videos:
        return ["Không có danh sách 'videos'."]

    plans = []
    topics, problems = read_topics(answer, durations)
    if policy["count"] is not None:
        least = policy.get("min_count", policy["count"])
        if not least <= len(videos) <= policy["count"]:
            wanted = f"đúng {least}" if least == policy["count"] else f"từ {least} đến {policy['count']}"
            problems.append(f"Có {len(videos)} video, yêu cầu {wanted} video.")
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
        if SOURCE_RUN_MAX_S > 0:
            passages = [(run, sum(durations[n] for n in run)) for run in source_runs(valid)]
            long_passages = [f"{'-'.join(map(str, run))} ({seconds:.2f}s)" for run, seconds in passages
                             if seconds > SOURCE_RUN_MAX_S + 1e-6]
            if long_passages:
                problems.append(
                    f"Video {index}: đoạn giống hệt video gốc dài quá {SOURCE_RUN_MAX_S:g}s: {', '.join(long_passages)}. "
                    "Hãy cắt đoạn bằng scene khác, ưu tiên scene cùng chủ đề."
                )

        problems += coherence_problems(index, valid, topics)
        jumps = [f"{a}-{b}" for a, b in zip(valid, valid[1:]) if (a, b) in JUMP_CUTS]
        if jumps:
            problems.append(f"Video {index}: nối thẳng hai scene cùng góc quay nhưng không nối tiếp nhau trong video gốc "
                            f"nên bị giật hình: {', '.join(jumps)}. Hãy xen scene khác vào giữa, ưu tiên cùng chủ đề.")

    uses = {n: [i for i, plan in enumerate(plans, start=1) if n in plan] for n in durations}
    unused = [n for n, where in uses.items() if not where]
    if unused and policy["cover_all"]:
        problems.append(f"Scene chưa được dùng ở video nào: {unused}.")
    overused = {} if FULL_SOURCE else {n: where for n, where in uses.items() if len(where) > MAX_SCENE_USES}
    for n, where in overused.items():
        problems.append(f"Scene {n} dùng trong {len(where)} video {where}, tối đa {MAX_SCENE_USES}.")

    for (a, plan_a), (b, plan_b) in combinations(enumerate(plans, start=1), 2):
        if not plan_a or not plan_b:
            continue
        shared = set(plan_a) & set(plan_b)
        smaller = min(len(set(plan_a)), len(set(plan_b)))
        if plan_a == plan_b:
            problems.append(f"Video {a} và {b} giống hệt nhau (cùng scene, cùng thứ tự).")
        elif not FULL_SOURCE:
            if len(shared) > MAX_SHARED_RATIO * smaller:
                problems.append(
                    f"Video {a} và {b} dùng chung {len(shared)}/{smaller} scene "
                    f"(tối đa {MAX_SHARED_RATIO * 100:g}%): {sorted(shared)}."
                )
            runs = [(run, sum(durations[n] for n in run)) for run in shared_runs(plan_a, plan_b)]
            long_runs = [f"{'-'.join(map(str, run))} ({seconds:.2f}s)" for run, seconds in runs
                         if seconds > MAX_SHARED_RUN_S + 1e-6]
            if long_runs and MAX_SHARED_RUN_S > 0:
                problems.append(
                    f"Video {a} và {b} có đoạn trùng dài quá {MAX_SHARED_RUN_S:g}s: {', '.join(long_runs)}. "
                    "Hãy đổi thứ tự hoặc xen scene khác để cắt ngắn các đoạn này."
                )
    return problems


SOURCE_KEYS = ("source_script", "source_script_vi", "source_story")  # what the source says; context only


def build_output(answer: dict, scenes: list[dict], model: str, policy: dict) -> dict:
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
        **{key: str(answer.get(key) or "").strip() for key in SOURCE_KEYS},
        "aspects": answer.get("aspects", []),
        "topics": answer.get("topics", []),
        "rules": {
            "full_source": FULL_SOURCE,
            "target_video_s": round(TARGET_VIDEO_S, 3),
            "max_video_s": round(MAX_VIDEO_S, 3),
            "min_video_s": round(MIN_VIDEO_S, 3),
            "max_scene_uses": None if FULL_SOURCE else MAX_SCENE_USES,
            "max_shared_ratio": None if FULL_SOURCE else MAX_SHARED_RATIO,
            "max_shared_run_s": None if FULL_SOURCE else MAX_SHARED_RUN_S,
            "source_run_target_s": SOURCE_RUN_TARGET_S,
            "source_run_max_s": SOURCE_RUN_MAX_S,
            "jump_cut_pairs": len(JUMP_CUTS),
            "max_switch_ratio": MAX_SWITCH_RATIO,
            "keep_topics_together": KEEP_TOPICS_TOGETHER,
            "target_videos": policy["target"],
            "videos_required": policy["count"],
            "videos_min": policy.get("min_count", policy["count"]),
            "all_scenes_used_required": policy["cover_all"],
            "notes": policy["notes"],
        },
        "source_scene_count": len(scenes),
        "source_total_duration": round(sum(s["duration"] for s in scenes), 3),
        "video_count": len(videos),
        "videos": videos,
        "scene_usage": {str(n): where for n, where in usage.items()},
    }


MIN_RUN_SCENES = 2  # shortest run of consecutive scenes worth reporting


def find_overlaps(videos: list[dict]) -> dict:
    """Runs of consecutive scenes (same scenes, same order) shared by two or more videos.

    For every pair of videos, each maximal common run of at least MIN_RUN_SCENES scenes,
    e.g. [1, 2, 3, 4, 9] and [7, 2, 3, 4, 19] share the run 2-3-4. The same run found in
    several pairs is reported once with every video that has it (videos 1, 2, 3 sharing
    2-3-4 -> one run, videos [1, 2, 3]). Seconds are the sum of the run's scene durations.
    Per video: seconds covered by any shared run, each scene counted once.
    """
    plans = {v["video_index"]: [s["scene_number"] for s in v["scenes"]] for v in videos}
    durations = {s["scene_number"]: s["duration"] for v in videos for s in v["scenes"]}
    runs: dict[tuple, set] = {}  # scenes of a run -> videos that have it (first found first)
    shared = {index: set() for index in plans}
    for (a, plan_a), (b, plan_b) in combinations(plans.items(), 2):
        for scenes in shared_runs(plan_a, plan_b):
            if len(scenes) >= MIN_RUN_SCENES:
                runs.setdefault(tuple(scenes), set()).update((a, b))
                shared[a].update(scenes)
                shared[b].update(scenes)
    per_video = []
    for v in videos:
        seconds = sum(durations[n] for n in shared[v["video_index"]])
        per_video.append({"video_index": v["video_index"], "shared_seconds": round(seconds, 3),
                          "total_duration": v["total_duration"],
                          "shared_ratio": round(seconds / v["total_duration"], 3) if v["total_duration"] else 0.0})
    return {"min_run_scenes": MIN_RUN_SCENES,
            "runs": [{"videos": sorted(where), "scenes": list(scenes),
                      "seconds": round(sum(durations[n] for n in scenes), 3)}
                     for scenes, where in runs.items()],
            "per_video": per_video}


def find_source_overlaps(videos: list[dict], source_duration: float) -> dict:
    """Passages of the source inside each output video (stats.json -> "source_overlaps").

    Scenes numbered n, n+1, ... follow each other in the source; next to each other in a
    video they are a passage of it. Listed: every run of at least MIN_RUN_SCENES scenes, and
    single scenes longer than SOURCE_RUN_MAX_S (no order of the scenes can shorten those).
    "level": "ok", "above_target" (over SOURCE_RUN_TARGET_S) or "too_long" (over
    SOURCE_RUN_MAX_S; only a single scene, a longer run is refused by find_problems).
    Per video: seconds in the listed passages, and the longest passage (single scenes too).
    """
    def level(seconds: float) -> str:
        if SOURCE_RUN_MAX_S <= 0:  # no limit
            return "ok"
        if seconds > SOURCE_RUN_MAX_S + 1e-6:
            return "too_long"
        return "above_target" if seconds > SOURCE_RUN_TARGET_S + 1e-6 else "ok"

    runs, per_video = [], []
    for v in videos:
        plan = [s["scene_number"] for s in v["scenes"]]
        durations = {s["scene_number"]: s["duration"] for s in v["scenes"]}
        in_runs = source_runs(plan)
        covered = {n for run in in_runs for n in run}
        passages = sorted([*in_runs, *([n] for n in plan if n not in covered)], key=lambda run: plan.index(run[0]))
        seconds = [round(sum(durations[n] for n in run), 3) for run in passages]
        listed = [{"video": v["video_index"], "scenes": run, "seconds": t, "level": level(t)}
                  for run, t in zip(passages, seconds) if len(run) >= MIN_RUN_SCENES or level(t) == "too_long"]
        runs += listed
        shared, longest = sum(r["seconds"] for r in listed), max(range(len(passages)), key=lambda k: seconds[k])
        per_video.append({
            "video_index": v["video_index"], "run_count": len(listed), "shared_seconds": round(shared, 3),
            "total_duration": v["total_duration"],
            "shared_ratio": round(shared / v["total_duration"], 3) if v["total_duration"] else 0.0,
            "longest_seconds": seconds[longest], "longest_scenes": passages[longest],
            # How much of the source this video uses, in any order.
            "source_used_ratio": round(v["total_duration"] / source_duration, 3) if source_duration else 0.0,
        })
    return {"min_run_scenes": MIN_RUN_SCENES, "target_s": SOURCE_RUN_TARGET_S, "max_s": SOURCE_RUN_MAX_S,
            "source_duration": round(source_duration, 3), "runs": runs, "per_video": per_video}


def source_overlap_lines(overlaps: dict) -> list[str]:
    """Stdout lines describing find_source_overlaps:

        source_overlap:<video>-<scene numbers joined by _>-<seconds>
            one line per passage, e.g. 'source_overlap:1-4_5_6-2.800'
        source_overlap_video:<video>-<seconds in passages>_<video seconds>_<longest passage seconds>
            one line per video, always printed, e.g. 'source_overlap_video:1-5.200_17.400_2.800'
    """
    lines = [f"source_overlap:{r['video']}-{'_'.join(map(str, r['scenes']))}-{r['seconds']:.3f}"
             for r in overlaps["runs"]]
    lines += [f"source_overlap_video:{p['video_index']}-{p['shared_seconds']:.3f}_{p['total_duration']:.3f}"
              f"_{p['longest_seconds']:.3f}" for p in overlaps["per_video"]]
    return lines


def overlap_lines(overlaps: dict) -> list[str]:
    """Stdout lines describing find_overlaps, one fact per line, easy to parse:

        overlap:<videos joined by _>-<scene numbers joined by _>-<seconds>
            one line per distinct shared run, with every video (2 or more, ascending) that
            has it, e.g. 'overlap:1_2-6_7_8_9-3.670', 'overlap:1_2_3_4-2_3_4-3.000'
        overlap_video:<video>-<shared seconds>_<video seconds>
            one line per video, always printed (0.000 = shares nothing), e.g. 'overlap_video:1-28.580_45.100'
    """
    # 3 decimals, like stats.json, so a parser gets the same numbers as the file.
    lines = [f"overlap:{'_'.join(map(str, r['videos']))}-{'_'.join(map(str, r['scenes']))}-{r['seconds']:.3f}"
             for r in overlaps["runs"]]
    lines += [f"overlap_video:{p['video_index']}-{p['shared_seconds']:.3f}_{p['total_duration']:.3f}"
              for p in overlaps["per_video"]]
    return lines


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def plan_videos(body: dict, scenes: list[dict], model: str, api_key: str, response_file: Path,
                policy: dict) -> dict:
    """Ask Gemini, feeding rule violations back into the same conversation.

    Every attempt (raw response, problems, fix prompt) is saved to response_file.
    """
    log_data = {"model": model, "attempts": []}
    problems: list[str] = []
    source: dict[str, str] = {}  # SOURCE_KEYS of the first answer that has them (fixes leave them empty)

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

        for key in SOURCE_KEYS:
            source.setdefault(key, "")
            source[key] = source[key] or str(answer.get(key) or "").strip()
        problems = find_problems(answer, scenes, policy)
        if not problems:
            record({**entry, "accepted": True, "problems": []})
            return {**answer, **source}
        log("Plan breaks the rules:\n  " + "\n  ".join(problems))
        fix = FIX_PROMPT.format(problems="\n".join(f"- {p}" for p in problems), **rule_values(scenes, policy))
        record({**entry, "problems": problems, "fix_prompt": fix})
        body["contents"] = body["contents"][:1] + [model_turn, {"role": "user", "parts": [{"text": fix}]}]
    raise ReorderError(f"No valid plan after {ANSWER_ATTEMPTS} attempts: {'; '.join(problems)}", E_INVALID_ANSWER)


def main() -> int:
    global ANSWER_ATTEMPTS
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
        video_length()  # before load_scenes: it checks scenes against MAX_VIDEO_S
        reuse_limits()
        coherence_limits()
        ANSWER_ATTEMPTS = whole_number("REORDER_MAX_ATTEMPTS") or ANSWER_ATTEMPTS
        scenes = load_scenes(scenelog)
        if FULL_SOURCE:
            use_full_source(scenes)
            log(f"Video length: target = source {TARGET_VIDEO_S:.1f}s ({MIN_VIDEO_S:g}-{MAX_VIDEO_S:.1f}s), "
                "scenes may be reused freely")
        else:
            log(f"Video length: target {TARGET_VIDEO_S:g}s ({MIN_VIDEO_S:g}-{MAX_VIDEO_S:g}s)")
            log(f"Reuse: a scene in at most {MAX_SCENE_USES} videos, {MAX_SHARED_RATIO * 100:g}% shared scenes, "
                f"shared runs of at most {MAX_SHARED_RUN_S:g}s (0 = no limit)")
        log(f"Source passages: target {SOURCE_RUN_TARGET_S:g}s, at most {SOURCE_RUN_MAX_S:g}s (0 = no limit)")
        load_jump_cuts(scenelog.with_name("jump_cuts.json"), scenes)
        log(f"Jump cuts: {len(JUMP_CUTS)} pairs of scenes must not follow each other")
        log(f"Coherence: at most {MAX_SWITCH_RATIO * 100:g}% of the joints change the topic, "
            f"topics kept together: {'yes' if KEEP_TOPICS_TOGETHER else 'no'}, "
            f"every scene used: {'yes' if USE_ALL_SCENES else 'no'}")
        policy = video_policy(scenes, target_videos())
        for note in policy["notes"]:
            log(f"REORDER_TARGET_VIDEOS: {note}")
        body = build_request(scenes, clips_dir, policy, source_audio(scenelog))
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(request_file, body)

        if args.dry_run:
            print(f"json:{request_file}")
            return 0

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == KEY_PLACEHOLDER:
            raise ReorderError("GEMINI_API_KEY is not set (fill it in .env, then recreate the container)", E_AUTH)

        # Drop results of a previous run so the files always belong together.
        output.unlink(missing_ok=True)
        response_file.unlink(missing_ok=True)
        stats_file = stats_path(output.parent)
        update_stats(stats_file, "overlaps", None)
        update_stats(stats_file, "source_overlaps", None)
        update_stats(stats_file, "coherence", None)

        answer = plan_videos(body, scenes, model, api_key, response_file, policy)
        result = build_output(answer, scenes, model, policy)
        log(", ".join(f"video {v['video_index']}: {v['total_duration']:.1f}s" for v in result["videos"]))
        overlaps = find_overlaps(result["videos"])
        source_overlaps = find_source_overlaps(result["videos"], sum(s["duration"] for s in scenes))
        write_json(output, result)
        update_stats(stats_file, "overlaps", overlaps)
        update_stats(stats_file, "source_overlaps", source_overlaps)
        topics, _ = read_topics(answer, {s["scene_number"]: s["duration"] for s in scenes})
        update_stats(stats_file, "coherence", coherence_stats(result["videos"], topics))
    except ReorderError as exc:
        log_error(exc)
        print("fail")
        return 1
    except Exception as exc:  # noqa: BLE001 - always answer json/fail
        log_error(GeminiError(f"{exc!r}", E_UNEXPECTED))
        print("fail")
        return 1

    print(f"json:{output}")  # always the first line: n8n reads it with /^json:(.*)$/m
    for line in [*overlap_lines(overlaps), *source_overlap_lines(source_overlaps)]:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
