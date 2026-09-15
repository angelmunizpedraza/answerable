"""Model providers, with the boring parts that make an LLM safe in CI.

Every provider returns plain text plus a token count. Retries, backoff and
timeouts live here so the rest of the package never has to think about them.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass

RETRY_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)
MAX_ATTEMPTS = 4
BASE_DELAY = 1.0


class ProviderError(RuntimeError):
    """Raised when a provider cannot be reached or refuses to answer."""


class BudgetExceeded(ProviderError):
    """Raised instead of making a call that would spend past --max-cost."""


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


def _sleep(attempt: int, sleeper=time.sleep) -> None:
    # Exponential backoff with jitter. Without the jitter, a batch of questions
    # that hits a rate limit retries in lockstep and hits it again.
    sleeper(BASE_DELAY * (2 ** attempt) * (0.5 + random.random() / 2))


class Provider:
    name = "base"
    # USD per million tokens. Used only to stop a run, never to invoice anyone.
    price_in = 0.0
    price_out = 0.0

    def complete(self, prompt: str, *, model: str, max_tokens: int = 700) -> Completion:
        raise NotImplementedError

    def cost(self, c: Completion) -> float:
        return (c.input_tokens * self.price_in + c.output_tokens * self.price_out) / 1_000_000


class EchoProvider(Provider):
    """Offline provider used by --offline and by the whole test suite.

    It does not pretend to be a model. It looks for the question's content
    words in the page and answers only when it finds a sentence containing
    them, which makes the pipeline runnable, deterministic and free.
    """

    name = "echo"

    def __init__(self, page_text: str = "") -> None:
        self.page_text = page_text

    def complete(self, prompt: str, *, model: str = "echo", max_tokens: int = 700) -> Completion:
        question = ""
        marker = "QUESTION:"
        if marker in prompt:
            question = prompt.split(marker, 1)[1].strip().splitlines()[0].strip()

        body = self.page_text
        if "PAGE:" in prompt:
            body = prompt.split("PAGE:", 1)[1].rsplit("QUESTION:", 1)[0]

        words = [w for w in _content_words(question) if len(w) > 3]
        best, best_hits = "", 0
        for sentence in _sentences(body):
            low = sentence.lower()
            hits = sum(1 for w in words if w in low)
            if not words or hits < max(1, len(words) // 2):
                continue
            # Most matching words wins; ties go to the shorter sentence, which
            # is the more precise quotation.
            if hits > best_hits or (hits == best_hits and len(sentence) < len(best)):
                best, best_hits = sentence, hits

        if best:
            payload = {"answerable": True, "answer": best[:400], "evidence": best[:400]}
        else:
            payload = {"answerable": False, "answer": None, "evidence": None}
        return Completion(text=json.dumps(payload), input_tokens=0, output_tokens=0)


def _content_words(text: str) -> list[str]:
    import re

    stop = {
        "que", "cual", "como", "donde", "cuando", "cuanto", "cuál", "cómo", "dónde",
        "para", "con", "por", "the", "what", "which", "does", "your", "you", "how",
        "and", "are", "can", "una", "uno", "los", "las", "del", "que", "why",
    }
    return [w for w in re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE) if w not in stop]


def _sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


class AnthropicProvider(Provider):
    name = "anthropic"
    price_in = 3.0
    price_out = 15.0
    endpoint = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str | None = None, sleeper=time.sleep) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self._sleep = sleeper

    def complete(self, prompt: str, *, model: str, max_tokens: int = 700) -> Completion:
        import requests

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            # Temperature 0 on purpose: a score that changes between two runs of
            # the same page is not a score, it is a mood.
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        return _post_with_retry(
            requests, self.endpoint, payload, headers, self._sleep, _parse_anthropic
        )


class OpenAIProvider(Provider):
    name = "openai"
    price_in = 2.5
    price_out = 10.0
    endpoint = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str | None = None, sleeper=time.sleep) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._sleep = sleeper

    def complete(self, prompt: str, *, model: str, max_tokens: int = 700) -> Completion:
        import requests

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        return _post_with_retry(
            requests, self.endpoint, payload, headers, self._sleep, _parse_openai
        )


def _post_with_retry(requests_mod, url, payload, headers, sleeper, parse):
    last = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests_mod.post(url, json=payload, headers=headers, timeout=60)
        except Exception as exc:  # network flake
            last = str(exc)
            if attempt == MAX_ATTEMPTS - 1:
                break
            _sleep(attempt, sleeper)
            continue

        if resp.status_code in RETRY_STATUS:
            last = f"HTTP {resp.status_code}"
            if attempt == MAX_ATTEMPTS - 1:
                break
            # Honour the server's own backoff when it sends one.
            wait = resp.headers.get("retry-after")
            if wait and str(wait).isdigit():
                sleeper(int(wait))
            else:
                _sleep(attempt, sleeper)
            continue

        if resp.status_code >= 400:
            raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            body = resp.json()
        except Exception:
            # A 200 carrying an HTML error page from a proxy is not a reply.
            raise ProviderError(
                f"HTTP 200 but the body was not JSON: {resp.text[:200]}"
            ) from None
        if not isinstance(body, dict):
            raise ProviderError(f"unexpected reply shape: {str(body)[:200]}")
        return parse(body)

    raise ProviderError(f"gave up after {MAX_ATTEMPTS} attempts: {last}")


def _int(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _parse_anthropic(data: dict) -> Completion:
    # Every field here is defensive on purpose: a refusal, a stop_reason of
    # max_tokens or a streaming artefact can leave any of them null, and a
    # crash in the parser surfaces as a failed build with a stack trace
    # instead of a readable error.
    blocks = data.get("content") or []
    text = "".join(
        b.get("text") or "" for b in blocks if isinstance(b, dict)
    )
    usage = data.get("usage") or {}
    return Completion(text, _int(usage.get("input_tokens")), _int(usage.get("output_tokens")))


def _parse_openai(data: dict) -> Completion:
    choices = data.get("choices") or [{}]
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    text = message.get("content") or ""
    if not isinstance(text, str):
        text = ""
    usage = data.get("usage") or {}
    return Completion(text, _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens")))


def build(name: str, *, page_text: str = "", api_key: str | None = None) -> Provider:
    name = (name or "").lower()
    if name in ("echo", "offline", ""):
        return EchoProvider(page_text)
    if name == "anthropic":
        return AnthropicProvider(api_key)
    if name == "openai":
        return OpenAIProvider(api_key)
    raise ProviderError(f"unknown provider: {name}")
