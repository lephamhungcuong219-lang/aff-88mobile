#!/usr/bin/env python3
"""Turn voiceover.json into Vietnamese voice tracks with Gemini TTS, with timings per line.

Usage:
    python gemini_tts.py <voiceover_json> <output_dir> [--voice Puck] [--model MODEL]
                         [--video N ...] [--gap 0.12] [--lead 0.15] [--max-speedup 1.15]

Gemini TTS returns no timestamps, so every segment (subtitle line) is synthesised
on its own with the same voice and the video's "tts_style". The length of each
returned clip is then the exact length of that line. Per video:
  1. synthesise every segments[].spoken (a few requests in parallel)
  2. trim the silence Gemini adds around each clip
  3. join them: LEAD seconds of silence, then the lines with GAP seconds between
  4. if the track is longer than the video, speed it up (pitch kept, ffmpeg atempo)
     by at most --max-speedup; if that is still not enough, the video fails and the
     voice-over should be rewritten shorter (gemini_voiceover.py --video N)

Files:
    <output_dir>/voice_video_<N>.wav   voice track, 24 kHz mono, starts at video time 0
    tts.json (next to voiceover.json)  timings of every line, in video seconds:
        {"model", "voice", "videos": [{"video_index", "audio_file", "video_duration",
          "speech_duration", "speed", "measured_rate", "segments": [{"index", "text",
          "spoken", "start", "end"}]}]}
    With --video, only those videos are replaced in tts.json; the others are kept.

Environment: GEMINI_API_KEY (required), GEMINI_TTS_MODEL (optional, default below),
GEMINI_TTS_RPM (optional, requests per minute allowed for your tier, default 10).
Requests are spaced to stay under that limit, and a per-minute 429 waits for the time
the server asks ("retry in 41s") before retrying. A video of 7 lines takes ~45 s at 10 RPM.
The DAILY limit (e.g. 100 requests per day on Tier 1) is not retried: the run stops at
once, keeps the videos already done in tts.json and reports E_QUOTA_DAILY.

Stdout: 'json:<absolute tts.json>' on success, or exactly 'fail' (if any video failed;
        the videos that succeeded are still written).
Stderr: progress; every failure is one line "[gemini_tts] <CODE>: <message>" with the
        codes of gemini_errors.py (E_QUOTA_DAILY, E_RATE_LIMIT, E_AUTH, E_TOO_LONG, ...),
        then a summary line with the videos to re-run (--video ...).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np

try:  # python -m post_process.gemini_tts
    from post_process.gemini_errors import (E_AUTH, E_INPUT, E_NETWORK, E_NO_OUTPUT, E_QUOTA_DAILY,
                                            E_TOO_LONG, E_UNEXPECTED, GeminiError, format_error,
                                            http_error, is_daily_quota, parse_retry_after)
except ImportError:  # python post_process/gemini_tts.py
    from gemini_errors import (E_AUTH, E_INPUT, E_NETWORK, E_NO_OUTPUT, E_QUOTA_DAILY, E_TOO_LONG,
                               E_UNEXPECTED, GeminiError, format_error, http_error, is_daily_quota,
                               parse_retry_after)

API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_MODEL = "gemini-3.8-flash-tts"
DEFAULT_VOICE = "Puck"
KEY_PLACEHOLDER = "your-gemini-api-key-here"
SAMPLE_RATE = 24000

PARALLEL_REQUESTS = 4
REQUEST_TIMEOUT_S = 300
HTTP_ATTEMPTS = 6
DEFAULT_RPM = 10          # requests per minute allowed for the TTS model (Tier 1); env GEMINI_TTS_RPM
SILENCE_DB = -42.0        # trim threshold around each line
KEEP_EDGE_S = 0.04        # silence kept at both ends of a trimmed line
END_MARGIN_S = 0.10       # the voice must end this long before the video ends
MAX_RETRY_WAIT_S = 120    # a 429 asking to wait longer than this is not retried

TTSError = GeminiError
QUOTA_HIT = threading.Event()  # set as soon as the daily limit is reached: stop every request


def log(message: str) -> None:
    print(f"[gemini_tts] {message}", file=sys.stderr)


def log_error(error: Exception) -> None:
    print(format_error("gemini_tts", error), file=sys.stderr)


def syllables(text: str) -> int:
    return sum(1 for word in text.split() if re.search(r"\w", word))


class RateLimiter:
    """Spaces request starts so that at most `rpm` start per minute (thread-safe)."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm
        self.next_slot = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        time.sleep(max(0.0, slot - now))


LIMITER = RateLimiter(DEFAULT_RPM)


def retry_delay(detail: str, attempt: int) -> float:
    """Seconds to wait before retrying: the server's 'retry in ...' hint, else exponential."""
    hint = parse_retry_after(detail)
    return hint + 1.0 if hint is not None else 4.0 * 2 ** (attempt - 1)


def synthesize(text: str, style: str, voice: str, model: str, api_key: str) -> np.ndarray:
    """One TTS request -> mono int16 samples at SAMPLE_RATE."""
    body = {
        "model": model,
        "input": [{"type": "user_input", "content": [{
            "type": "text", "text": text,
            "annotations": [{"type": "speech_metadata", "style": style}] if style else [],
        }]}],
        "response_format": {"type": "audio", "mime_type": "audio/wav", "sample_rate": SAMPLE_RATE},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }
    request = urllib.request.Request(API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        if QUOTA_HIT.is_set():
            raise TTSError("Gemini TTS daily limit reached earlier in this run", E_QUOTA_DAILY)
        LIMITER.wait()
        if QUOTA_HIT.is_set():  # reached by another request while this one waited its turn
            raise TTSError("Gemini TTS daily limit reached earlier in this run", E_QUOTA_DAILY)
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1500]
            if exc.code == 429 and is_daily_quota(detail):
                QUOTA_HIT.set()  # no point retrying today: fail now, and stop the other requests
                raise http_error(exc.code, detail, "Gemini TTS") from exc
            retryable = exc.code == 429 or exc.code >= 500
            delay = retry_delay(detail, attempt)
            if not retryable or attempt == HTTP_ATTEMPTS or delay > MAX_RETRY_WAIT_S:
                raise http_error(exc.code, detail, "Gemini TTS") from exc
            log(f"HTTP {exc.code}, retrying in {delay:.0f}s ({attempt}/{HTTP_ATTEMPTS})")
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == HTTP_ATTEMPTS:
                raise TTSError(f"Could not reach Gemini TTS: {exc}", E_NETWORK) from exc
            delay = retry_delay("", attempt)
        time.sleep(delay)

    audio = None
    for step in data.get("steps", []) or []:
        for part in step.get("content", []) or []:
            if isinstance(part, dict) and part.get("data"):
                audio = part["data"]
    if not audio:
        raise TTSError(f"Gemini TTS returned no audio for: {text[:80]!r}", E_NO_OUTPUT)
    return decode_wav(base64.b64decode(audio))


def decode_wav(raw: bytes) -> np.ndarray:
    """WAV (or headerless 16-bit PCM) -> mono int16 at SAMPLE_RATE."""
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw)) as wav:
            if wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
                raise TTSError(f"Unexpected TTS audio format: {wav.getsampwidth() * 8} bit, "
                               f"{wav.getframerate()} Hz", E_NO_OUTPUT)
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            if wav.getnchannels() > 1:
                samples = samples.reshape(-1, wav.getnchannels()).mean(axis=1).astype("<i2")
            return samples
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2")


def trim(samples: np.ndarray) -> np.ndarray:
    """Cut leading/trailing silence, keeping KEEP_EDGE_S on both sides."""
    if samples.size == 0:
        return samples
    window = int(0.01 * SAMPLE_RATE)
    frames = samples[: samples.size // window * window].astype(np.float32).reshape(-1, window)
    rms = np.sqrt((frames ** 2).mean(axis=1)) / 32768.0
    loud = np.nonzero(20 * np.log10(np.maximum(rms, 1e-9)) > SILENCE_DB)[0]
    if loud.size == 0:
        return samples
    keep = int(KEEP_EDGE_S * SAMPLE_RATE)
    start = max(0, loud[0] * window - keep)
    end = min(samples.size, (loud[-1] + 1) * window + keep)
    return samples[start:end]


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.astype("<i2").tobytes())


def speed_up(samples: np.ndarray, factor: float) -> np.ndarray:
    """Faster playback without pitch change (ffmpeg atempo)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        write_wav(src, samples)
        result = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", f"atempo={factor:.4f}",
                                 "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
                                capture_output=True, text=True, errors="replace", check=False)
        if result.returncode != 0:
            raise TTSError(f"ffmpeg atempo failed: {result.stderr.strip()}", E_UNEXPECTED)
        return decode_wav(dst.read_bytes())


def render_video(video: dict, out_dir: Path, voice: str, model: str, api_key: str,
                 gap: float, lead: float, max_speedup: float) -> dict:
    index = video["video_index"]
    segments = video.get("segments") or []
    if not segments:
        raise TTSError(f"Video {index}: no segments in voiceover.json")
    duration = float(video.get("duration") or 0)
    style = str(video.get("tts_style") or "")

    log(f"Video {index}: synthesising {len(segments)} lines with {voice}")
    with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as pool:
        clips = list(pool.map(lambda s: trim(synthesize(s["spoken"], style, voice, model, api_key)), segments))

    # Join: lead silence, lines separated by gaps.
    gap_samples = np.zeros(int(gap * SAMPLE_RATE), dtype="<i2")
    parts, timings, cursor = [np.zeros(int(lead * SAMPLE_RATE), dtype="<i2")], [], lead
    for i, (segment, clip) in enumerate(zip(segments, clips), start=1):
        if i > 1:
            parts.append(gap_samples)
            cursor += gap
        length = clip.size / SAMPLE_RATE
        timings.append({"index": i, "text": segment["text"], "spoken": segment["spoken"],
                        "start": cursor, "end": cursor + length})
        parts.append(clip)
        cursor += length
    track = np.concatenate(parts)
    natural = track.size / SAMPLE_RATE
    speech_only = sum(c.size for c in clips) / SAMPLE_RATE
    measured_rate = sum(syllables(s["spoken"]) for s in segments) / max(speech_only, 1e-6)

    # Fit into the video.
    speed, budget = 1.0, duration - END_MARGIN_S
    if duration and natural > budget:
        speed = natural / budget
        if speed > max_speedup:
            raise TTSError(
                f"Video {index}: voice lasts {natural:.2f}s for a {duration:.2f}s video; needs {speed:.2f}x "
                f"(max {max_speedup}x). Rewrite it shorter: gemini_voiceover.py --video {index} "
                f"--rate {measured_rate:.2f}", E_TOO_LONG)
        track = speed_up(track, speed)
        for t in timings:
            t["start"] /= speed
            t["end"] /= speed
        log(f"Video {index}: sped up {speed:.3f}x to fit")

    audio_file = out_dir / f"voice_video_{index}.wav"
    write_wav(audio_file, track)
    for t in timings:
        t["start"], t["end"] = round(t["start"], 3), round(t["end"], 3)
    result = {
        "video_index": index,
        "audio_file": audio_file.name,
        "video_duration": duration,
        "speech_duration": round(track.size / SAMPLE_RATE, 3),
        "speed": round(speed, 4),
        "measured_rate": round(measured_rate, 2),
        "segments": timings,
    }
    log(f"Video {index}: {result['speech_duration']:.2f}s of voice for {duration:.2f}s of video "
        f"({measured_rate:.2f} syllables/s)")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Make Vietnamese voice tracks from voiceover.json with Gemini TTS.")
    parser.add_argument("voiceover_json", type=Path, help="voiceover.json from gemini_voiceover.py")
    parser.add_argument("output_dir", type=Path, help="Directory for voice_video_<N>.wav")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"Gemini TTS voice (default {DEFAULT_VOICE})")
    parser.add_argument("--model", help=f"TTS model (default: GEMINI_TTS_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--video", type=int, nargs="+", help="Only these video indexes")
    parser.add_argument("--gap", type=float, default=0.12, help="Silence between lines, seconds (default 0.12)")
    parser.add_argument("--lead", type=float, default=0.15, help="Silence before the first line (default 0.15)")
    parser.add_argument("--max-speedup", type=float, default=1.15, help="Max speed-up to fit the video (default 1.15)")
    args = parser.parse_args()

    source = args.voiceover_json.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    tts_file = source.with_name("tts.json")
    model = args.model or os.environ.get("GEMINI_TTS_MODEL", "").strip() or DEFAULT_MODEL

    try:
        try:
            rpm = float(os.environ.get("GEMINI_TTS_RPM", "").strip() or DEFAULT_RPM)
        except ValueError as exc:
            raise TTSError(f"GEMINI_TTS_RPM must be a number: {os.environ.get('GEMINI_TTS_RPM')!r}") from exc
        if rpm <= 0:
            raise TTSError(f"GEMINI_TTS_RPM must be positive: {rpm}")
        LIMITER.interval = 60.0 / rpm
        if not source.is_file():
            raise TTSError(f"voiceover_json must be the voiceover.json file, got: {source}")
        if not 1.0 <= args.max_speedup <= 1.5:
            raise TTSError(f"--max-speedup must be between 1.0 and 1.5: {args.max_speedup}")
        try:
            plan = json.loads(source.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TTSError(f"Could not read {source}: {exc}") from exc
        videos = plan.get("videos") if isinstance(plan, dict) else None
        if not videos:
            raise TTSError(f"No 'videos' in {source}")
        if args.video:
            unknown = sorted(set(args.video) - {v.get("video_index") for v in videos})
            if unknown:
                raise TTSError(f"--video {unknown} not in {source.name}")
            videos = [v for v in videos if v.get("video_index") in args.video]

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == KEY_PLACEHOLDER:
            raise TTSError("GEMINI_API_KEY is not set (fill it in .env, then recreate the container)", E_AUTH)

        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        if args.video and tts_file.is_file():
            try:
                results = {v["video_index"]: v for v in json.loads(tts_file.read_text(encoding="utf-8"))["videos"]}
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                results = {}

        failed, skipped = [], []
        for position, video in enumerate(videos):
            index = video["video_index"]
            try:
                results[index] = render_video(video, out_dir, args.voice, model, api_key,
                                              args.gap, args.lead, args.max_speedup)
            except TTSError as exc:
                log_error(exc)  # one "[gemini_tts] E_...: ..." line per failed video
                failed.append(index)
                results.pop(index, None)
                if exc.code == E_QUOTA_DAILY:
                    # Every further request would fail too: stop now.
                    skipped = [v["video_index"] for v in videos[position + 1:]]
                    break

        tmp = tts_file.with_name(tts_file.name + ".tmp")
        tmp.write_text(json.dumps({
            "model": model, "voice": args.voice, "sample_rate": SAMPLE_RATE,
            "audio_dir": str(out_dir), "videos": [results[k] for k in sorted(results)],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(tts_file)
        if failed:
            log(f"Failed videos: {failed}; not attempted: {skipped}; done: {sorted(results)} "
                f"(in {tts_file.name}). Re-run with --video {' '.join(map(str, failed + skipped))}")
            print("fail")
            return 1
    except TTSError as exc:
        log_error(exc)
        print("fail")
        return 1
    except Exception as exc:  # noqa: BLE001 - always answer json/fail
        log_error(GeminiError(f"{exc!r}", E_UNEXPECTED))
        print("fail")
        return 1

    print(f"json:{tts_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
