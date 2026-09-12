"""Load the questions a page is supposed to answer."""

from __future__ import annotations


def parse(text: str) -> list[str]:
    """One question per line. Blank lines and lines starting with # are ignored.

    Duplicates are dropped, keeping the first occurrence: paying a model twice
    for the same question is the easiest money to waste.
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        q = line.strip()
        if not q or q.startswith("#"):
            continue
        key = " ".join(q.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def load(path: str) -> list[str]:
    # errors="replace" on purpose: a questions file saved as Latin-1 by a
    # Spanish-speaking client should cost them one mangled accent, not a
    # stack trace and a red build.
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse(fh.read())
