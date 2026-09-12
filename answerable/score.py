"""Turn a list of verdicts into one number a non-technical person can act on."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from .judge import INVALID, OK, UNANSWERED, UNGROUNDED, UNSUPPORTED, Verdict

BANDS = (
    (90, "excelente", "The page answers almost everything a buyer asks."),
    (75, "bueno", "A few gaps worth filling."),
    (50, "flojo", "Half of what buyers ask is missing from the page."),
    (0, "critico", "The page cannot answer most of what buyers ask."),
)


def _pct(part: int, whole: int) -> int:
    """A percentage rounded half-up.

    round() is half-to-even, so 62.5 becomes 62 while 37.5 becomes 38. A
    --min-score of 63 would then fail a page that answered exactly 62.5% of
    the questions, and pass one that answered 62.5% of a different number.
    """
    if not whole:
        return 0
    return int((Decimal(100 * part) / Decimal(whole)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass
class Score:
    total: int = 0
    ok: int = 0
    unanswered: int = 0
    ungrounded: int = 0
    unsupported: int = 0
    invalid: int = 0
    gaps: list[str] = field(default_factory=list)
    hallucinations: list[Verdict] = field(default_factory=list)

    @property
    def value(self) -> int:
        """0-100: the share of questions the page provably answers.

        Provably is the operative word. An answer whose quotation is not on
        the page counts as a miss, exactly like no answer at all, because a
        buyer reading the page would not have found it either.
        """
        if not self.total:
            return 0
        return _pct(self.ok, self.total)

    @property
    def grounding(self) -> int | None:
        """Of the times the model claimed an answer, how often it could prove it.

        A claim is proved when the quotation is really on the page and the
        answer states no figure the quotation does not contain.

        None when it never claimed one: 0/0 is not 0%, and printing 0% there
        would accuse an honest run of hallucinating.
        """
        claimed = self.ok + self.ungrounded + self.unsupported
        if not claimed:
            return None
        return _pct(self.ok, claimed)

    @property
    def band(self) -> str:
        return next(name for floor, name, _ in BANDS if self.value >= floor)

    @property
    def verdict_line(self) -> str:
        return next(text for floor, _, text in BANDS if self.value >= floor)


def summarise(verdicts: list[Verdict]) -> Score:
    s = Score(total=len(verdicts))
    for v in verdicts:
        if v.status == OK:
            s.ok += 1
        elif v.status == UNANSWERED:
            s.unanswered += 1
            s.gaps.append(v.question)
        elif v.status == UNGROUNDED:
            s.ungrounded += 1
            s.gaps.append(v.question)
            s.hallucinations.append(v)
        elif v.status == UNSUPPORTED:
            s.unsupported += 1
            s.gaps.append(v.question)
            s.hallucinations.append(v)
        elif v.status == INVALID:
            s.invalid += 1
            s.gaps.append(v.question)
    return s
