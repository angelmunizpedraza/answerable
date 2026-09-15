"""The guardrail layer: prompt, strict parsing, and verification.

The interesting part is not asking the model. It is refusing to believe it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

PROMPT = """You are auditing a single web page. Answer using ONLY the text of the page below.
Do not use anything you know from outside the page. Do not guess. Do not infer.

If the page does not contain the answer, say so. Saying "not answerable" is a correct,
useful answer and is never penalised.

Reply with one JSON object and nothing else, in this exact shape:
{{"answerable": true or false, "answer": string or null, "evidence": string or null}}

"evidence" must be copied VERBATIM from ONE paragraph of the page, character for
character. Do not paraphrase it, do not fix its punctuation, do not translate it, do
not stitch together text from different paragraphs. Every figure that appears in your
"answer" must also appear in your "evidence". If you cannot copy a verbatim span that
supports the answer, set "answerable" to false.

PAGE:
{page}

QUESTION: {question}
"""

REPAIR_NOTE = """
IMPORTANT: your previous reply was not a single valid JSON object, so it was discarded.
Answer the question above again, replying with one JSON object and nothing else, in
exactly the shape described above. No explanation, no code fence, no extra text.

Your discarded reply was:
{previous}
"""

# Why a verdict was not counted as an answer.
OK = "ok"
UNANSWERED = "unanswered"
UNGROUNDED = "ungrounded"    # claimed an answer, the quotation is not on the page
UNSUPPORTED = "unsupported"  # the quotation is real, the answer goes beyond it
INVALID = "invalid"          # never returned a usable JSON verdict


@dataclass
class Verdict:
    question: str
    status: str
    answer: str = ""
    evidence: str = ""
    raw: str = ""

    @property
    def answered(self) -> bool:
        return self.status == OK


def build_prompt(page_text: str, question: str, max_chars: int = 60_000) -> str:
    return PROMPT.format(page=page_text[:max_chars], question=question)


def build_repair_prompt(page_text: str, question: str, previous: str,
                        max_chars: int = 60_000) -> str:
    """The repair carries the page and the question again.

    Two reasons, and the second one is the one that bites. A repair prompt made
    only of the model's own garbled reply cannot cite the page except by luck,
    so a cosmetic JSON hiccup would turn into a fake hallucination. And because
    the cache is keyed on the prompt, a repair prompt that did not mention the
    page or the question would let two different questions collide on one key
    and be served each other's answers.
    """
    return build_prompt(page_text, question, max_chars) + REPAIR_NOTE.format(
        previous=previous[:1000]
    )


def _extract_json(text) -> dict | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    # Models sometimes wrap the object in a sentence. Take the outermost braces.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _as_text(value) -> str:
    """Only a real string is text. A dict or a list is a malformed field, not
    something to coerce into one with str()."""
    return value.strip() if isinstance(value, str) else ""


def _as_bool(value):
    """None means the model did not give a usable answer to "is this answerable"."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace(" ", " ")
    return " ".join(text.lower().split())


def _blocks(page_text: str) -> list[str]:
    """The page as the paragraphs it is really made of.

    Matching has to happen inside one paragraph. Flattening the page first
    would let a model quote the end of one bullet and the start of the next as
    if it were one sentence, and that invented sentence would pass.
    """
    return [b for b in (_normalise(block) for block in page_text.split("\n\n")) if b]


def evidence_is_on_the_page(evidence: str, page_text: str,
                            min_chars: int = 12, min_words: int = 3) -> bool:
    """True only if the quoted span really appears in one paragraph of the page.

    This is the point of the tool. A model that invents a quotation is the
    failure mode that matters, and it is cheap to catch: normalise both sides
    and look for the substring, inside one paragraph rather than across the
    whole flattened page.

    A span has to be long enough to mean something. "Sí" appears in every
    Spanish page ever written, and a threshold in characters alone would
    exclude a real table row like "99 EUR al mes", so both a length and a
    word count are required.
    """
    if not evidence:
        return False
    needle = _normalise(evidence)
    if len(needle) < min_chars or len(needle.split()) < min_words:
        return False
    return any(needle in block for block in _blocks(page_text))


UNITS = {
    "cero": 0, "zero": 0,
    "un": 1, "uno": 1, "una": 1, "one": 1,
    "dos": 2, "two": 2, "tres": 3, "three": 3, "cuatro": 4, "four": 4,
    "cinco": 5, "five": 5, "seis": 6, "six": 6, "siete": 7, "seven": 7,
    "ocho": 8, "eight": 8, "nueve": 9, "nine": 9, "diez": 10, "ten": 10,
    "once": 11, "eleven": 11, "doce": 12, "twelve": 12,
    "trece": 13, "thirteen": 13, "catorce": 14, "fourteen": 14,
    "quince": 15, "fifteen": 15, "dieciseis": 16, "sixteen": 16,
    "diecisiete": 17, "seventeen": 17, "dieciocho": 18, "eighteen": 18,
    "diecinueve": 19, "nineteen": 19, "veinte": 20, "twenty": 20,
    "veintiuno": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
    "veintinueve": 29,
    "treinta": 30, "thirty": 30, "cuarenta": 40, "forty": 40,
    "cincuenta": 50, "fifty": 50, "sesenta": 60, "sixty": 60,
    "setenta": 70, "seventy": 70, "ochenta": 80, "eighty": 80,
    "noventa": 90, "ninety": 90,
}
SCALES = {
    "cien": 100, "ciento": 100, "hundred": 100,
    "doscientos": 200, "trescientos": 300, "cuatrocientos": 400,
    "quinientos": 500, "seiscientos": 600, "setecientos": 700,
    "ochocientos": 800, "novecientos": 900,
    "mil": 1000, "thousand": 1000,
    "millon": 1_000_000, "millones": 1_000_000, "million": 1_000_000,
}
MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_JOIN = {"y", "and", "de", "con"}

# Only a dot or a comma groups thousands. A plain space does not: "912 345 678"
# is a phone number, and treating it as 912345678 made every quoted phone
# number look like an invented figure.
_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d\d\d(?!\d))")
_SPACE_GROUPS = re.compile(r"(?<=\d) (?=\d\d\d(?!\d))")
# Not preceded by a letter, so the 2 in "m2" (and in "m²" after NFKC) is a
# unit, not a figure.
_NUMBER = re.compile(r"(?<![^\W\d_])\d+")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _strip_accents(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word)
                   if not unicodedata.combining(c))


def _digit_figures(text: str) -> set[str]:
    # "9:00" is nine o'clock, not the two figures 9 and 0.
    text = _TIME.sub(lambda m: m.group(1) if m.group(2) == "00"
                     else f"{m.group(1)} {m.group(2)}", text)
    flat = _THOUSANDS.sub("", text)
    return {n.lstrip("0") or "0" for n in _NUMBER.findall(flat)}


def _word_figures(text: str) -> set[str]:
    """Numbers written out in words, in Spanish or English.

    Only ever used to widen what the EVIDENCE is taken to contain. A page that
    says "tres años de garantía" does support an answer of "3 años", and
    before this the tool called that an invented figure.
    """
    found: set[str] = set()
    run: list[int] = []

    def flush() -> None:
        if not run:
            return
        total, current = 0, 0
        for value in run:
            if value >= 1000:
                total += (current or 1) * value
                current = 0
            elif value >= 100 and current:
                current *= value
            else:
                current += value
            found.add(str(value))          # each word on its own, too
        found.add(str(total + current))
        run.clear()

    for word in _WORD.findall(text.lower()):
        plain = _strip_accents(word)
        if plain in UNITS:
            run.append(UNITS[plain])
        elif plain in SCALES:
            run.append(SCALES[plain])
        elif plain in MONTHS:
            flush()
            found.add(str(MONTHS[plain]))  # "15 de marzo" supports "15/03"
        elif plain in _JOIN and run:
            continue
        else:
            flush()
    flush()
    return found


def _figures(text: str) -> set[str]:
    """Every number the text states, as digits.

    "2.500", "2,500" and "2500" are the same figure, so the separator is
    dropped; the lookahead insists on exactly three following digits, so a
    decimal comma is not mistaken for one.
    """
    return _digit_figures(_normalise(text))


def _figures_claimed_by(text: str) -> set[str]:
    """What the evidence can be said to contain. Deliberately generous.

    The check should only ever fire on a figure the page really does not
    state, so every reasonable reading of the quotation counts: digits,
    numbers written in words, month names, and thousands grouped with spaces.
    """
    flat = _normalise(text)
    found = _digit_figures(flat)
    found |= _digit_figures(_SPACE_GROUPS.sub("", flat))
    found |= _word_figures(flat)
    return found


def answer_is_supported_by(answer: str, evidence: str) -> bool:
    """Reject an answer that states a figure its own quotation does not contain.

    This does not prove the quotation entails the answer — nothing short of a
    second model would, and a second model has the same problem as the first.
    It catches the failure that actually costs money in this domain: a real
    sentence quoted from the page with an invented price, deadline, percentage
    or number of years attached to it.
    """
    return not (_figures(answer) - _figures_claimed_by(evidence))


def judge(page_text: str, question: str, provider, model: str,
          max_chars: int = 60_000) -> Verdict:
    """Ask once, repair once, then give up. Never coerce a bad reply into a pass.

    The verification runs against exactly the text that was sent, so a
    quotation from a part of the page the model never saw cannot pass.
    """
    page_text = page_text[:max_chars]
    completion = provider.complete(
        build_prompt(page_text, question, max_chars=max_chars), model=model
    )
    raw = completion.text if isinstance(completion.text, str) else ""
    data = _extract_json(raw)
    answerable = _as_bool(data.get("answerable")) if data is not None else None

    if answerable is None:
        repaired = provider.complete(
            build_repair_prompt(page_text, question, raw, max_chars=max_chars),
            model=model,
        )
        raw = repaired.text if isinstance(repaired.text, str) else ""
        data = _extract_json(raw)
        answerable = _as_bool(data.get("answerable")) if data is not None else None
        if answerable is None:
            return Verdict(question, INVALID, raw=raw[:500])

    if not answerable:
        return Verdict(question, UNANSWERED, raw=raw[:500])

    answer = _as_text(data.get("answer"))
    evidence = _as_text(data.get("evidence"))

    if not evidence_is_on_the_page(evidence, page_text):
        # It said yes and could not prove it. That is a worse outcome than no,
        # so it gets its own status instead of being folded into "unanswered".
        return Verdict(question, UNGROUNDED, answer=answer, evidence=evidence,
                       raw=raw[:500])

    if not answer_is_supported_by(answer, evidence):
        return Verdict(question, UNSUPPORTED, answer=answer, evidence=evidence,
                       raw=raw[:500])

    return Verdict(question, OK, answer=answer, evidence=evidence, raw=raw[:500])
