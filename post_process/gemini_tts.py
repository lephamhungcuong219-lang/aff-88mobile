#!/usr/bin/env python3
"""Turn voiceover.json into Vietnamese voice tracks with Gemini TTS, with timings per line.

Usage:
    python gemini_tts.py <voiceover_json> <output_dir> [--voice Puck] [--model MODEL]
                         [--fallback-model MODEL] [--accent TEXT] [--video N ...]
                         [--lead 0.15] [--max-speedup 1.15]

ONE request per video: all lines of the video are read in a single TTS call, so the
intonation flows naturally and the daily request limit goes ~7x further. Per video:
  1. send every segments[].spoken joined with a pause between lines, plus the
     video's "tts_style" and a fixed accent instruction (Southern Vietnamese)
  2. find the line boundaries: the pauses between lines are the longest silences
     close to where they are expected (expected = position by syllable count);
     if that does not give believable lines, split by syllable count instead
     ("split" in tts.json says which was used)
  3. trim the silence around the speech, start after LEAD seconds
  4. fit the video so there is no silent ending: a track that ends early gets longer
     pauses between lines (at most MAX_EXTRA_PAUSE_S each) and then a slightly slower
     delivery (not below MIN_SPEED); a track longer than the video is sped up (pitch
     kept, ffmpeg atempo) by at most --max-speedup. If even that is not enough (a slow
     voice), the video is read ONCE more by the same voice asked to speak faster
     (FASTER_STYLE, 1 extra request) and the shorter take is kept; still too long ->
     E_TOO_LONG

Models: --model (default gemini-3.8-flash-tts), then --fallback-model (default
gemini-3.1-flash-tts-preview; "" disables it). When a model reports its DAILY
limit, it is dropped for the rest of the run and the same video is retried at once
with the next model. When every model is exhausted the run stops with E_QUOTA_DAILY.
gemini-3.8 takes the style in a separate field and pause tags between lines; older
models (3.1 / 2.5) get the style written before the text and lines on new lines.

Files:
    <output_dir>/voice_video_<N>.wav   voice track, 24 kHz mono, starts at video time 0
    tts.json (next to voiceover.json)  {"model", "fallback_model", "voices", "accent",
        "audio_dir", "videos": [{"video_index", "voice", "model", "split", "audio_file",
        "video_duration", "speech_duration", "speed", "measured_rate",
        "segments": [{"index", "text", "spoken", "start", "end"}]}]}
    With --video, only those videos are replaced in tts.json; the others are kept.

Environment (set in .env; a command-line option wins over it):
    GEMINI_API_KEY              required
    GEMINI_TTS_MODEL            default gemini-3.8-flash-tts
    GEMINI_TTS_FALLBACK_MODEL   default gemini-3.1-flash-tts-preview; empty = no fallback
    GEMINI_TTS_RPM              requests per minute per model, default 10
    GEMINI_TTS_VOICES           voice rotation pool, comma-separated (default Puck,Laomedeia,
                                Sadachbia,Fenrir). Every video gets a random voice not yet
                                used in the current round; a new round starts when all are used.
                                The voice is fixed per video, also when falling back to the
                                other model. --voice X uses X for every video instead.
    GEMINI_TTS_ACCENT           accent instruction, default Southern Vietnamese; empty = none
A per-minute 429 waits the time the server asks, then retries.

Stdout: 'json:<absolute tts.json>' on success, or exactly 'fail' (if any video failed;
        the videos that succeeded are still written).
Stderr: progress; every failure is one line "[gemini_tts] <CODE>: <message>" with the
        codes of gemini_errors.py (E_QUOTA_DAILY when every model is exhausted,
        E_RATE_LIMIT, E_AUTH, E_NO_OUTPUT, E_TOO_LONG, ...), then a summary line with
        the videos to re-run (--video ...).
Exit code: 0 for success, 1 for failure.
"""

import argparse
import base64
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
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
DEFAULT_FALLBACK_MODEL = "gemini-3.1-flash-tts-preview"
LEGACY_MODEL_PREFIXES = ("gemini-3.1-", "gemini-2.5-")  # style in the text, raw PCM answer
DEFAULT_VOICE = "Puck"
DEFAULT_VOICES = "Puck,Laomedeia,Sadachbia,Fenrir"  # rotation pool (names valid for 3.8 and 3.1);
# Zephyr left out: the slowest reader in every test (~3.6 syllables/s), often too long to fit
DEFAULT_ACCENT = "Use a natural Southern Vietnamese accent (Saigon / Ho Chi Minh City)."
KEY_PLACEHOLDER = "your-gemini-api-key-here"
SAMPLE_RATE = 24000

REQUEST_TIMEOUT_S = 300
HTTP_ATTEMPTS = 6
MAX_RETRY_WAIT_S = 120    # a 429 asking to wait longer than this is not retried
DEFAULT_RPM = 10          # requests per minute allowed per TTS model (Tier 1); env GEMINI_TTS_RPM
BAD_AUDIO_RETRIES = 1     # re-ask the same model when the audio looks wrong (skipped/repeated text)

SILENCE_DB = -40.0        # below this a 10 ms frame counts as silence
MIN_PAUSE_S = 0.06        # shorter silences are not line breaks
MIN_LINE_S = 0.25         # a split line shorter than this means the split is wrong
LINE_RATE_RANGE = (2.0, 9.5)   # syllables per second of a believable line
TRACK_RATE_RANGE = (2.0, 8.5)  # of the whole track (outside: text skipped or repeated)
KEEP_EDGE_S = 0.04        # silence kept around the speech when trimming
END_MARGIN_S = 0.10       # the voice must end this long before the video ends
FILL_END_MARGIN_S = 0.25  # a short voice is stretched to end about this long before the video ends
FILL_TOLERANCE_S = 0.15   # shorter than the target by less than this: left as is
MAX_EXTRA_PAUSE_S = 0.35  # at most this much silence added to each pause between lines
MIN_SPEED = 0.92          # never slower than this when stretching (sounds natural)
FASTER_STYLE = ("Speak noticeably faster than usual, at a brisk energetic pace, "
                "with only very short pauses between sentences.")  # retry when a voice reads too slowly

TTSError = GeminiError


def log(message: str) -> None:
    print(f"[gemini_tts] {message}", file=sys.stderr)


def log_error(error: Exception) -> None:
    print(format_error("gemini_tts", error), file=sys.stderr)


def syllables(text: str) -> int:
    return sum(1 for word in text.split() if re.search(r"\w", word))


def is_legacy(model: str) -> bool:
    return model.startswith(LEGACY_MODEL_PREFIXES)


class RateLimiter:
    """Spaces request starts so that at most `rpm` start per minute."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm
        self.next_slot = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        slot = max(now, self.next_slot)
        self.next_slot = slot + self.interval
        time.sleep(max(0.0, slot - now))


class ModelPool:
    """Models in order of preference; a model hitting its daily limit is dropped."""

    def __init__(self, models: list[str], rpm: float) -> None:
        self.models = models
        self.limiters = {m: RateLimiter(rpm) for m in models}
        self.exhausted: dict[str, str] = {}  # model -> daily-limit message

    def available(self) -> list[str]:
        return [m for m in self.models if m not in self.exhausted]


def request_body(model: str, lines: list[str], style: str, voice: str) -> dict:
    if is_legacy(model):
        # Older models: no style field and pause tags are not documented for them.
        text = (f"{style}: " if style else "") + "\n".join(lines)
        content = {"type": "text", "text": text}
        response_format = {"type": "audio"}
    else:
        content = {"type": "text", "text": " <short pause> ".join(lines)}
        if style:
            content["annotations"] = [{"type": "speech_metadata", "style": style}]
        response_format = {"type": "audio", "mime_type": "audio/wav", "sample_rate": SAMPLE_RATE}
    return {
        "model": model,
        "input": [{"type": "user_input", "content": [content]}],
        "response_format": response_format,
        "generation_config": {"speech_config": [{"voice": voice}]},
    }


def synthesize(model: str, lines: list[str], style: str, voice: str, api_key: str, pool: ModelPool) -> np.ndarray:
    """One TTS request for all lines -> mono int16 samples at SAMPLE_RATE."""
    body = request_body(model, lines, style, voice)
    request = urllib.request.Request(API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        pool.limiters[model].wait()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1500]
            if exc.code == 429 and is_daily_quota(detail):
                raise http_error(exc.code, detail, f"Gemini TTS {model}") from exc  # E_QUOTA_DAILY
            hint = parse_retry_after(detail)
            delay = hint + 1.0 if hint is not None else 4.0 * 2 ** (attempt - 1)
            # 429/5xx, and the 400 "audio stream could not be completed ... please retry" of TTS models.
            retryable = exc.code == 429 or exc.code >= 500 or (exc.code == 400 and "retry" in detail.lower())
            if not retryable or attempt == HTTP_ATTEMPTS or delay > MAX_RETRY_WAIT_S:
                raise http_error(exc.code, detail, f"Gemini TTS {model}") from exc
            log(f"{model}: HTTP {exc.code}, retrying in {delay:.0f}s ({attempt}/{HTTP_ATTEMPTS})")
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == HTTP_ATTEMPTS:
                raise TTSError(f"Could not reach Gemini TTS: {exc}", E_NETWORK) from exc
            delay = 4.0 * 2 ** (attempt - 1)
        time.sleep(delay)

    audio = None
    for step in data.get("steps", []) or []:
        for part in step.get("content", []) or []:
            if isinstance(part, dict) and part.get("data"):
                audio = part["data"]
    if not audio:
        raise TTSError(f"{model} returned no audio", E_NO_OUTPUT)
    return decode_wav(base64.b64decode(audio))


def decode_wav(raw: bytes) -> np.ndarray:
    """WAV, or headerless 16-bit PCM (older models), -> mono int16 at SAMPLE_RATE."""
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


def loud_frames(samples: np.ndarray) -> np.ndarray:
    """Boolean per 10 ms frame: louder than SILENCE_DB."""
    window = SAMPLE_RATE // 100
    frames = samples[: samples.size // window * window].astype(np.float32).reshape(-1, window)
    rms = np.sqrt((frames ** 2).mean(axis=1)) / 32768.0
    return 20 * np.log10(np.maximum(rms, 1e-9)) > SILENCE_DB


def split_lines(samples: np.ndarray, counts: list[int]) -> tuple[float, float, list[tuple[float, float]], str]:
    """(speech start, speech end, [(line start, line end)], method) in seconds of `samples`."""
    loud = loud_frames(samples)
    if not loud.any():
        raise TTSError("TTS audio is silent", E_NO_OUTPUT)
    first = int(np.argmax(loud))
    last = len(loud) - 1 - int(np.argmax(loud[::-1]))
    start, end = first / 100, (last + 1) / 100
    if len(counts) == 1:
        return start, end, [(start, end)], "single"

    # Silent runs inside the speech.
    runs, i = [], first
    while i <= last:
        if not loud[i]:
            j = i
            while j <= last and not loud[j]:
                j += 1
            if (j - i) / 100 >= MIN_PAUSE_S:
                runs.append((i / 100, j / 100))
            i = j
        else:
            i += 1

    total = sum(counts)
    expected = [start + (end - start) * sum(counts[:k]) / total for k in range(1, len(counts))]
    proportional = list(zip([start] + expected, expected + [end]))

    chosen, after = [], start
    for target in expected:
        candidates = [r for r in runs if r[0] > after]
        if not candidates:
            return start, end, proportional, "proportional"
        best = min(candidates, key=lambda r: abs((r[0] + r[1]) / 2 - target))
        chosen.append(best)
        after = best[1]
    bounds = list(zip([start] + [b for _, b in chosen], [a for a, _ in chosen] + [end]))
    for (a, b), n in zip(bounds, counts):
        length = b - a
        if length < MIN_LINE_S or not LINE_RATE_RANGE[0] <= n / length <= LINE_RATE_RANGE[1]:
            return start, end, proportional, "proportional"
    return start, end, bounds, "silence"


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


def voice_track(lines: list[str], style: str, voice: str, api_key: str, pool: ModelPool) -> tuple[np.ndarray, str]:
    """Audio of all lines, trying the models in order; a daily limit moves to the next model."""
    counts = [syllables(line) for line in lines]
    while True:
        models = pool.available()
        if not models:
            details = "; ".join(f"{m}: {msg}" for m, msg in pool.exhausted.items())
            waits = [parse_retry_after(msg) for msg in pool.exhausted.values()]
            waits = [w for w in waits if w is not None]
            suffix = f"; retry_after={min(waits):.0f}s" if waits else ""
            raise TTSError(f"every TTS model reached its daily limit ({details}){suffix}", E_QUOTA_DAILY)
        model = models[0]
        try:
            for attempt in range(BAD_AUDIO_RETRIES + 1):
                samples = synthesize(model, lines, style, voice, api_key, pool)
                start, end, _, _ = split_lines(samples, counts)
                rate = sum(counts) / max(end - start, 1e-6)
                if TRACK_RATE_RANGE[0] <= rate <= TRACK_RATE_RANGE[1]:
                    return samples, model
                log(f"{model}: {rate:.1f} syllables/s looks wrong (text skipped or repeated?), "
                    f"{'asking again' if attempt < BAD_AUDIO_RETRIES else 'giving up'}")
            raise TTSError(f"{model}: audio does not match the text ({rate:.1f} syllables/s)", E_NO_OUTPUT)
        except TTSError as exc:
            if exc.code != E_QUOTA_DAILY:
                raise
            pool.exhausted[model] = str(exc)
            nxt = pool.available()
            log(f"{model} reached its daily limit" + (f"; falling back to {nxt[0]}" if nxt else ""))


def render_video(video: dict, out_dir: Path, voice: str, accent: str, api_key: str, pool: ModelPool,
                 lead: float, max_speedup: float) -> dict:
    index = video["video_index"]
    segments = video.get("segments") or []
    if not segments:
        raise TTSError(f"Video {index}: no segments in voiceover.json", E_INPUT)
    duration = float(video.get("duration") or 0)
    style = " ".join(p for p in (str(video.get("tts_style") or "").strip(), accent.strip()) if p)
    lines = [str(s["spoken"]).strip() for s in segments]
    counts = [syllables(line) for line in lines]

    def take(style_text: str):
        """Synthesise, split and trim: (track, timings, measured rate, model, split method)."""
        samples, used_model = voice_track(lines, style_text, voice, api_key, pool)
        start, end, bounds, split = split_lines(samples, counts)
        keep = int(KEEP_EDGE_S * SAMPLE_RATE)
        cut_from = max(0, int(start * SAMPLE_RATE) - keep)
        cut_to = min(samples.size, int(end * SAMPLE_RATE) + keep)
        trimmed = np.concatenate([np.zeros(int(lead * SAMPLE_RATE), dtype="<i2"), samples[cut_from:cut_to]])
        shift = lead - cut_from / SAMPLE_RATE
        marks = [{"index": i, "text": seg["text"], "spoken": seg["spoken"], "start": a + shift, "end": b + shift}
                 for i, (seg, (a, b)) in enumerate(zip(segments, bounds), start=1)]
        return trimmed, marks, sum(counts) / max(end - start, 1e-6), used_model, split

    log(f"Video {index}: synthesising {len(lines)} lines in one request ({voice})")
    track, timings, measured_rate, model, method = take(style)
    faster_retry = False
    budget = duration - END_MARGIN_S
    if duration and track.size / SAMPLE_RATE > budget * max_speedup:
        # Too long even at the maximum speed-up (a slow voice): ask once more, same voice,
        # for a faster delivery, and keep the shorter of the two.
        log(f"Video {index}: {track.size / SAMPLE_RATE:.2f}s is too long for {duration:.2f}s "
            f"even at {max_speedup}x; asking {voice} to read faster")
        faster_retry = True
        retry = take(f"{style} {FASTER_STYLE}".strip())
        if retry[0].size < track.size:
            track, timings, measured_rate, model, method = retry

    # Too short: fill the video to (almost) the end, so there is no silent ending.
    # 1) longer pauses between lines (only where real pauses were found), 2) then a slightly
    # slower delivery (pitch kept). Whatever is left after both is accepted.
    added_pause, slowed = 0.0, 1.0
    target_end = duration - FILL_END_MARGIN_S
    natural = track.size / SAMPLE_RATE
    if duration and natural < target_end - FILL_TOLERANCE_S and method == "silence" and len(timings) > 1:
        per_gap = min(MAX_EXTRA_PAUSE_S, (target_end - natural) / (len(timings) - 1))
        gap = np.zeros(int(per_gap * SAMPLE_RATE), dtype="<i2")
        pieces, cursor = [], 0
        for k in range(len(timings) - 1):
            cut = int((timings[k]["end"] + timings[k + 1]["start"]) / 2 * SAMPLE_RATE)
            pieces += [track[cursor:cut], gap]
            cursor = cut
            for t in timings[k + 1:]:
                t["start"] += gap.size / SAMPLE_RATE
                t["end"] += gap.size / SAMPLE_RATE
        track = np.concatenate(pieces + [track[cursor:]])
        added_pause = round(gap.size / SAMPLE_RATE, 3)
        natural = track.size / SAMPLE_RATE
    if duration and natural < target_end - FILL_TOLERANCE_S:
        slowed = max(MIN_SPEED, natural / target_end)
        if slowed < 0.999:
            track = speed_up(track, slowed)  # atempo < 1 = slower
            for t in timings:
                t["start"] /= slowed
                t["end"] /= slowed
            natural = track.size / SAMPLE_RATE
    if added_pause or slowed < 0.999:
        log(f"Video {index}: filled to {natural:.2f}s of {duration:.2f}s "
            f"(+{added_pause:.2f}s per pause, speed {slowed:.3f}x)")

    # Too long: speed up to fit.
    speed, budget = slowed, duration - END_MARGIN_S
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
        "voice": voice,
        "model": model,
        "split": method,
        "audio_file": audio_file.name,
        "video_duration": duration,
        "speech_duration": round(track.size / SAMPLE_RATE, 3),
        "speed": round(speed, 4),
        "added_pause": added_pause,
        "faster_retry": faster_retry,
        "measured_rate": round(measured_rate, 2),
        "segments": timings,
    }
    log(f"Video {index}: {model} / {voice}, {result['speech_duration']:.2f}s of voice for {duration:.2f}s of video "
        f"({measured_rate:.2f} syllables/s, lines split by {method})")
    return result


def assign_voices(indexes: list[int], pool: list[str], kept: dict[int, str]) -> dict[int, str]:
    """A different voice per video: random order, no voice twice in a round.

    Voices of videos kept from an earlier run (kept) count as already used in the
    current round, so a partial re-run does not repeat them. When every voice of the
    pool has been used, a new round starts (reshuffled, not starting with the last one).
    """
    used = [v for v in kept.values() if v in pool][-(len(pool) - 1):] if len(pool) > 1 else []
    bag = [v for v in pool if v not in used]
    random.shuffle(bag)
    assigned, last = {}, (used[-1] if used else None)
    for index in indexes:
        if not bag:
            bag = list(pool)
            random.shuffle(bag)
            if len(bag) > 1 and bag[0] == last:
                bag.append(bag.pop(0))
        last = assigned[index] = bag.pop(0)
    return assigned


def main() -> int:
    parser = argparse.ArgumentParser(description="Make Vietnamese voice tracks from voiceover.json with Gemini TTS.")
    parser.add_argument("voiceover_json", type=Path, help="voiceover.json from gemini_voiceover.py")
    parser.add_argument("output_dir", type=Path, help="Directory for voice_video_<N>.wav")
    parser.add_argument("--voice", help="Use this one voice for every video (default: rotate GEMINI_TTS_VOICES)")
    parser.add_argument("--model", help=f"TTS model (default: GEMINI_TTS_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--fallback-model",
                        help=f"Model used when --model reaches its daily limit "
                             f"(default: GEMINI_TTS_FALLBACK_MODEL or {DEFAULT_FALLBACK_MODEL}; '' = none)")
    parser.add_argument("--accent", help="Accent instruction added to the style "
                                         "(default: GEMINI_TTS_ACCENT or Southern Vietnamese; '' = none)")
    parser.add_argument("--video", type=int, nargs="+", help="Only these video indexes")
    parser.add_argument("--lead", type=float, default=0.15, help="Silence before the voice starts (default 0.15)")
    parser.add_argument("--max-speedup", type=float, default=1.15, help="Max speed-up to fit the video (default 1.15)")
    args = parser.parse_args()

    source = args.voiceover_json.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    tts_file = source.with_name("tts.json")
    model = args.model or os.environ.get("GEMINI_TTS_MODEL", "").strip() or DEFAULT_MODEL
    # Command line > .env (environment) > built-in default. An empty env value is meaningful
    # for the fallback model and the accent (= none), so only a MISSING variable uses the default.
    if args.fallback_model is not None:
        fallback = args.fallback_model.strip()
    else:
        fallback = os.environ.get("GEMINI_TTS_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL).strip()
    if args.voice:
        voice_pool = [args.voice.strip()]
    else:
        voice_pool = [v.strip() for v in os.environ.get("GEMINI_TTS_VOICES", DEFAULT_VOICES).split(",") if v.strip()]
        voice_pool = list(dict.fromkeys(voice_pool)) or [DEFAULT_VOICE]
    if args.accent is None:
        args.accent = os.environ.get("GEMINI_TTS_ACCENT", DEFAULT_ACCENT)
    args.accent = args.accent.strip()

    try:
        try:
            rpm = float(os.environ.get("GEMINI_TTS_RPM", "").strip() or DEFAULT_RPM)
        except ValueError as exc:
            raise TTSError(f"GEMINI_TTS_RPM must be a number: {os.environ.get('GEMINI_TTS_RPM')!r}") from exc
        if rpm <= 0:
            raise TTSError(f"GEMINI_TTS_RPM must be positive: {rpm}")
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

        pool = ModelPool([m for m in dict.fromkeys([model, fallback]) if m], rpm)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        if args.video and tts_file.is_file():
            try:
                results = {v["video_index"]: v for v in json.loads(tts_file.read_text(encoding="utf-8"))["videos"]}
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                results = {}

        # Voices are fixed per video before any request, so a fallback to another model
        # keeps the voice the video was given (the voice names exist in both models).
        to_make = [v["video_index"] for v in videos]
        kept = {k: r.get("voice") for k, r in results.items() if k not in to_make and r.get("voice")}
        voices = assign_voices(to_make, voice_pool, kept)
        log("Voices: " + ", ".join(f"video {k} = {v}" for k, v in voices.items()))

        failed, skipped = [], []
        for position, video in enumerate(videos):
            index = video["video_index"]
            try:
                results[index] = render_video(video, out_dir, voices[index], args.accent, api_key, pool,
                                              args.lead, args.max_speedup)
            except TTSError as exc:
                log_error(exc)  # one "[gemini_tts] E_...: ..." line per failed video
                failed.append(index)
                results.pop(index, None)
                if exc.code == E_QUOTA_DAILY:
                    # Every model is exhausted: further requests would fail too.
                    skipped = [v["video_index"] for v in videos[position + 1:]]
                    break

        tmp = tts_file.with_name(tts_file.name + ".tmp")
        tmp.write_text(json.dumps({
            "model": model, "fallback_model": fallback or None, "voices": voice_pool, "accent": args.accent,
            "sample_rate": SAMPLE_RATE, "audio_dir": str(out_dir),
            "videos": [results[k] for k in sorted(results)],
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
