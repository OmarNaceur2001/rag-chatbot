"""Input/output safety checks and an in-process per-client rate limiter."""

import re
import threading
import time
import unicodedata
from collections import defaultdict, deque

MAX_INPUT_LENGTH = 4000
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60
CANARY_TOKEN = "TN_INTERNAL_CANARY_2026_7f3b91"

_ATTACK_PATTERNS = (
    re.compile(r"\bignore\b.{0,80}\b(instructions|rules|system prompt|developer message)\b"),
    re.compile(r"\b(reveal|print|show|repeat|disclose)\b.{0,80}\b(system prompt|developer message|canary|secret)\b"),
    re.compile(r"\b(oublie|ignore|efface)\b.{0,80}\b(instructions|regles|prompt systeme)\b"),
    re.compile(r"\b(revele|affiche|repete)\b.{0,80}\b(prompt systeme|instructions|canari|secret)\b"),
    re.compile(r"تجاهل.{0,50}(التعليمات|القواعد|النظام)"),
    re.compile(r"(اكشف|اظهر|اطبع|كرر).{0,50}(التعليمات|النظام|الرمز السري)"),
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}\b", re.IGNORECASE),
)


class InputRejected(ValueError):
    pass


class UnsafeOutput(ValueError):
    pass


class RateLimitExceeded(ValueError):
    pass


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        char for char in normalized
        if not unicodedata.combining(char) and unicodedata.category(char) != "Cf"
    )
    return normalized.translate(str.maketrans({
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
    }))


def validate_input(text: str) -> None:
    if len(text) > MAX_INPUT_LENGTH:
        raise InputRejected(f"Message exceeds the {MAX_INPUT_LENGTH}-character limit.")
    normalized = _normalize(text)
    if any(pattern.search(normalized) for pattern in _ATTACK_PATTERNS):
        raise InputRejected("This request was refused by the input safety checks.")


def validate_output(text: str) -> str:
    if CANARY_TOKEN.casefold() in text.casefold():
        raise UnsafeOutput("The generated answer contained the protected canary.")
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        raise UnsafeOutput("The generated answer matched a sensitive-data pattern.")
    return text


_requests_by_client: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()


def check_rate_limit(
    client_id: str,
    *,
    limit: int = RATE_LIMIT_REQUESTS,
    window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
) -> None:
    now = time.monotonic()
    cutoff = now - window_seconds
    with _rate_limit_lock:
        for key in list(_requests_by_client):
            bucket = _requests_by_client[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                del _requests_by_client[key]
        bucket = _requests_by_client[client_id]
        if len(bucket) >= limit:
            raise RateLimitExceeded("Request rate limit exceeded.")
        bucket.append(now)
