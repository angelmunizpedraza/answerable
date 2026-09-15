import json

from answerable.judge import OK, UNANSWERED, UNGROUNDED, Verdict
from answerable.page import Page
from answerable.report import to_json, to_markdown
from answerable.score import summarise

PAGE = Page(url="https://ejemplo.es/reformas", title="Reformas",
            text="Una reforma de baño tarda entre 10 y 15 días laborables.")
VERDICTS = [
    Verdict("¿Cuánto tarda?", OK, "Entre 10 y 15 días.",
            "Una reforma de baño tarda entre 10 y 15 días laborables."),
    Verdict("¿Cuánto cuesta?", UNANSWERED),
    Verdict("¿Hay garantía?", UNGROUNDED, "Tres años.", "Garantía de tres años"),
]
SCORE = summarise(VERDICTS)


def test_the_markdown_leads_with_the_number():
    assert to_markdown(PAGE, VERDICTS, SCORE, "m").startswith("# Answerability: 33/100")


def test_the_markdown_lists_the_gaps():
    md = to_markdown(PAGE, VERDICTS, SCORE, "m")
    assert "## Lo que la página no responde" in md
    assert "¿Cuánto cuesta?" in md


def test_the_markdown_names_the_invented_quotations():
    md = to_markdown(PAGE, VERDICTS, SCORE, "m")
    assert "Respuestas descartadas" in md
    assert "Garantía de tres años" in md


def test_a_clean_run_does_not_print_an_empty_hallucination_section():
    clean = [VERDICTS[0]]
    md = to_markdown(PAGE, clean, summarise(clean), "m")
    assert "Respuestas descartadas" not in md
    assert "Lo que la página no responde" not in md


def test_a_pipe_in_an_answer_does_not_break_the_table():
    verdicts = [Verdict("¿A | B?", UNANSWERED)]
    md = to_markdown(PAGE, verdicts, summarise(verdicts), "m")
    row = [line for line in md.splitlines()
           if line.startswith("|") and "A \\| B" in line][0]
    unescaped = row.replace("\\|", "")
    assert unescaped.count("|") == 5  # four cells, so five separators


def test_long_text_is_shortened_in_the_table():
    verdicts = [Verdict("¿?", OK, "x" * 500, "y" * 500)]
    md = to_markdown(PAGE, verdicts, summarise(verdicts), "m")
    assert "x" * 500 not in md
    assert "…" in md


def test_the_json_is_machine_readable_and_complete():
    data = json.loads(to_json(PAGE, VERDICTS, SCORE, "modelo-x", cost=0.0123))
    assert data["score"] == 33
    assert data["url"] == "https://ejemplo.es/reformas"
    assert data["model"] == "modelo-x"
    assert data["cost_usd"] == 0.0123
    assert data["counts"] == {"total": 3, "answered": 1, "unanswered": 1,
                              "ungrounded": 1, "unsupported": 0, "invalid": 0}
    assert data["gaps"] == ["¿Cuánto cuesta?", "¿Hay garantía?"]
    assert len(data["questions"]) == 3
    assert data["questions"][0]["status"] == "ok"


def test_the_json_keeps_accents_readable():
    assert "garantía" in to_json(PAGE, VERDICTS, SCORE, "m")


def test_grounding_is_null_in_json_when_unknown():
    verdicts = [Verdict("¿?", UNANSWERED)]
    data = json.loads(to_json(PAGE, verdicts, summarise(verdicts), "m"))
    assert data["grounding_pct"] is None
