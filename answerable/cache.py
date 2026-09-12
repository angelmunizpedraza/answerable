"""A content-addressed cache so a rerun costs nothing and gives the same answer.

Two reasons this exists, and neither is speed:

1. A CI gate that calls a paid API on every push is a CI gate somebody turns
   off in a month. Unchanged pages must be free.
2. A score that moves when nothing changed is not a score. Caching the model's
   reply makes the second run of an unchanged page byte-identical to the first.

The key is a hash of everything that could change the reply: provider, model
and the exact prompt. Change any of them and you get a miss, which is correct.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict

from .providers import BudgetExceeded, Completion

VERSION = 1


def key_for(provider: str, model: str, prompt: str) -> str:
    h = hashlib.sha256()
    for part in (str(VERSION), provider, model, prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")  # so ("ab","c") and ("a","bc") are different keys
    return h.hexdigest()


class Cache:
    """A directory of JSON files, one per key. No index, no locking, no daemon."""

    def __init__(self, path: str | None, enabled: bool = True) -> None:
        self.path = path or os.path.join(".answerable-cache")
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _file(self, key: str) -> str:
        # Two-character shard: 50k entries in one directory is slow to list on
        # every filesystem that matters.
        return os.path.join(self.path, key[:2], key[2:] + ".json")

    def get(self, key: str) -> Completion | None:
        if not self.enabled:
            return None
        try:
            with open(self._file(key), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError, UnicodeDecodeError):
            # Truncated, non-JSON or non-UTF-8: a miss, never a crash. A cache
            # that can break a build is worse than no cache.
            self.misses += 1
            return None
        if not isinstance(data, dict):
            self.misses += 1
            return None
        self.hits += 1
        return Completion(
            text=data.get("text", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cached=True,
        )

    def put(self, key: str, completion: Completion) -> None:
        if not self.enabled:
            return
        target = self._file(key)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        data = asdict(completion)
        data["cached"] = True
        # Write to a temp file and rename: an interrupted run must not leave a
        # half-written JSON file that poisons every future run of that page.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


class CachedProvider:
    """Wraps a provider so identical prompts are answered from disk.

    It deliberately does not wrap `cost()`: a cached reply cost nothing this
    run, and reporting it as spend would make the budget gate lie.
    """

    def __init__(self, inner, cache: Cache, budget: float | None = None) -> None:
        self.inner = inner
        self.cache = cache
        self.budget = budget
        # A provider that charges nothing can never exceed a budget, and
        # --max-cost 0 is the obvious way to write "spend nothing" in CI.
        self.free = not (getattr(inner, "price_in", 0) or getattr(inner, "price_out", 0))
        self.name = getattr(inner, "name", "unknown")
        self.spent = 0.0

    def complete(self, prompt: str, *, model: str, max_tokens: int = 700) -> Completion:
        key = key_for(self.name, model, prompt)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        # Checked here rather than once per question, because a question can
        # cost two calls and the last question would otherwise be unbudgeted.
        # A call already in flight cannot be un-spent, so the budget is a stop
        # line, not a cap: one call may cross it before the run halts.
        if self.budget is not None and not self.free and self.spent >= self.budget:
            raise BudgetExceeded(
                f"spent ${self.spent:.4f} of the ${self.budget:.4f} budget"
            )
        result = self.inner.complete(prompt, model=model, max_tokens=max_tokens)
        self.spent += self.inner.cost(result)
        self.cache.put(key, result)
        return result

    def cost(self, completion: Completion) -> float:
        if completion.cached:
            return 0.0
        return self.inner.cost(completion)
