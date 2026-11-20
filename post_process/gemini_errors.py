"""Shared error codes for the scripts that call Gemini.

Every failure is written to stderr as ONE line:

    [<script>] <CODE>: <message>

<CODE> is one of the constants below, so a workflow can filter by it, e.g. in n8n:
    {{ ($json.stderr.match(/\\] (E_[A-Z_]+):/) || [])[1] }}

    E_QUOTA_DAILY     daily request limit of the model reached: no retry, stops at once.
                      The message ends with "retry_after=<seconds>s" when Google says it.
    E_RATE_LIMIT      per-minute limit still hit after all retries
    E_AUTH            API key missing/invalid or not allowed (HTTP 401/403, or no key)
    E_BAD_REQUEST     request refused (HTTP 400/404/...: wrong model name, bad input)
    E_SERVER          Gemini server error (HTTP 5xx) after all retries
    E_NETWORK         Gemini not reachable (connection error, timeout) after all retries
    E_NO_OUTPUT       Gemini answered without the expected content (no audio, empty answer)
    E_INVALID_ANSWER  Gemini's answer broke the rules after all correction attempts
    E_TOO_LONG        the voice does not fit in the video even at the maximum speed-up
    E_INPUT           missing or invalid input file / argument
    E_UNEXPECTED      anything else (a bug)
"""

import re

E_QUOTA_DAILY = "E_QUOTA_DAILY"
E_RATE_LIMIT = "E_RATE_LIMIT"
E_AUTH = "E_AUTH"
E_BAD_REQUEST = "E_BAD_REQUEST"
E_SERVER = "E_SERVER"
E_NETWORK = "E_NETWORK"
E_NO_OUTPUT = "E_NO_OUTPUT"
E_INVALID_ANSWER = "E_INVALID_ANSWER"
E_TOO_LONG = "E_TOO_LONG"
E_INPUT = "E_INPUT"
E_UNEXPECTED = "E_UNEXPECTED"


class GeminiError(Exception):
    """A failure with an error code and a human-readable reason."""

    def __init__(self, message: str, code: str = E_INPUT) -> None:
        super().__init__(message)
        self.code = code


def parse_retry_after(detail: str) -> float | None:
    """Seconds from Google's 'retry in 27m31s' / 'retry in 41s' / 'retry in 1h2m', or None."""
    match = re.search(r"retry in\s*((?:\d+(?:\.\d+)?\s*[hms]\s*)+)", detail, re.IGNORECASE)
    if not match:
        return None
    seconds = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", match.group(1).lower()):
        seconds += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return seconds


def is_daily_quota(detail: str) -> bool:
    """True when a 429 is about the per-day limit (waiting minutes inside a run is pointless)."""
    text = detail.lower()
    return "per day" in text or "requests per day" in text or "daily" in text


def short_message(detail: str) -> str:
    """The 'message' field of a Gemini error body, else the raw text (shortened)."""
    match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', detail)
    return (match.group(1) if match else detail).strip()[:500]


def http_error(status: int, detail: str, service: str) -> GeminiError:
    """Map a final (non-retried) HTTP error to a GeminiError."""
    message = short_message(detail)
    if status == 429:
        if is_daily_quota(detail):
            wait = parse_retry_after(detail)
            suffix = f"; retry_after={wait:.0f}s" if wait is not None else ""
            return GeminiError(f"{service}: daily limit reached: {message}{suffix}", E_QUOTA_DAILY)
        return GeminiError(f"{service}: rate limit still hit after retries: {message}", E_RATE_LIMIT)
    if status in (401, 403):
        return GeminiError(f"{service}: HTTP {status}: {message}", E_AUTH)
    if status >= 500:
        return GeminiError(f"{service}: HTTP {status} after retries: {message}", E_SERVER)
    return GeminiError(f"{service}: HTTP {status}: {message}", E_BAD_REQUEST)


def format_error(script: str, error: Exception) -> str:
    code = getattr(error, "code", E_UNEXPECTED)
    return f"[{script}] {code}: {error}"
