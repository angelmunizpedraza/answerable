"""Two outputs: markdown for a human, JSON for a machine."""

from __future__ import annotations

import json

from .judge import INVALID, OK, UNANSWERED, UNGROUNDED, UNSUPPORTED, Verdict
from .page import Page
from .score import Score

LABEL = {
    OK: "Respondida",
    UNANSWERED: "No responde",
    UNGROUNDED: "Cita inventada",
    UNSUPPORTED: "Dato no respaldado",
    INVALID: "Respuesta ilegible",
}
MARK = {
    OK: "OK",
    UNANSWERED: "FALTA",
    UNGROUNDED: "CITA INVENTADA",
    UNSUPPORTED: "DATO NO RESPALDADO",
    INVALID: "ERROR",
}


def _short(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# Escaped wherever page-supplied text lands in the report. Underscores and
# "#" are left alone: neither can start a construct here, because _short has
# already collapsed the newlines, and escaping them mangles every URL.
_MARKDOWN = ("\\", "`", "*", "[", "]", "<", ">")


def _plain(text: str, limit: int = 200) -> str:
    """Page-supplied text going into a heading, a bullet or a table cell.

    The title of a third-party page is not trusted input: left alone it can
    inject its own markdown heading into the report and into the GitHub job
    summary, which is how a "0/100" report grows a "100/100" headline.
    """
    text = _short(text, limit)
    for char in _MARKDOWN:
        text = text.replace(char, "\\" + char)
    return text


def _cell(text: str) -> str:
    # The backslash is escaped first by _plain, so escaping the pipe here
    # cannot produce "\\|", where the "\\" would consume the escape and the
    # pipe would become a live column separator again.
    return _plain(text, 160).replace("|", "\\|")


def to_markdown(page: Page, verdicts: list[Verdict], score: Score,
                model: str, cost: float = 0.0) -> str:
    out: list[str] = []
    out.append(f"# Answerability: {score.value}/100 ({score.band})")
    out.append("")
    out.append(score.verdict_line)
    out.append("")
    out.append(f"- Página: {_plain(page.title) or '(sin título)'} — {_plain(page.url)}")
    out.append(f"- Palabras analizadas: {page.word_count}")
    out.append(f"- Preguntas: {score.total} · respondidas {score.ok} · "
               f"sin respuesta {score.unanswered} · citas inventadas {score.ungrounded}"
               + (f" · datos no respaldados {score.unsupported}"
                  if score.unsupported else "")
               + (f" · ilegibles {score.invalid}" if score.invalid else ""))
    if score.grounding is not None:
        out.append(f"- De las respuestas afirmadas, {score.grounding}% se pudieron "
                   "verificar literalmente en la página.")
    out.append(f"- Modelo: `{model}` · coste de esta ejecución: ${cost:.4f}")
    out.append("")

    if score.gaps:
        out.append("## Lo que la página no responde")
        out.append("")
        for q in score.gaps:
            out.append(f"- {_plain(q)}")
        out.append("")

    if score.hallucinations:
        out.append("## Respuestas descartadas")
        out.append("")
        out.append("El modelo afirmó una respuesta que no se pudo verificar contra "
                   "la página. No se puntúan.")
        out.append("")
        for v in score.hallucinations:
            if v.status == UNGROUNDED:
                why = f"la cita no aparece en la página: «{_plain(v.evidence)}»"
            else:
                why = (f"la respuesta da una cifra que su propia cita no contiene: "
                       f"«{_plain(v.answer)}» frente a «{_plain(v.evidence)}»")
            out.append(f"- **{_plain(v.question)}** — {why}")
        out.append("")

    out.append("## Detalle")
    out.append("")
    out.append("| Pregunta | Resultado | Respuesta | Evidencia en la página |")
    out.append("| --- | --- | --- | --- |")
    for v in verdicts:
        out.append(
            f"| {_cell(v.question)} | {MARK[v.status]} | "
            f"{_cell(v.answer) or '—'} | {_cell(v.evidence) or '—'} |"
        )
    out.append("")
    return "\n".join(out)


def to_json(page: Page, verdicts: list[Verdict], score: Score,
            model: str, cost: float = 0.0) -> str:
    payload = {
        "url": page.url,
        "title": page.title,
        "word_count": page.word_count,
        "model": model,
        "cost_usd": round(cost, 6),
        "score": score.value,
        "band": score.band,
        "grounding_pct": score.grounding,
        "counts": {
            "total": score.total,
            "answered": score.ok,
            "unanswered": score.unanswered,
            "ungrounded": score.ungrounded,
            "unsupported": score.unsupported,
            "invalid": score.invalid,
        },
        "gaps": score.gaps,
        "questions": [
            {
                "question": v.question,
                "status": v.status,
                "label": LABEL[v.status],
                "answer": v.answer,
                "evidence": v.evidence,
            }
            for v in verdicts
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
