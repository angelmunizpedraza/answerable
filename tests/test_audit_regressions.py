"""One test per defect found in the adversarial review of this package.

Every test here failed before the fix. They are kept together so a future
change that reintroduces one is obvious in the diff.
"""

import os

import pytest

from answerable import cli
from answerable.cache import Cache, CachedProvider
from answerable.cli import EXIT_ERROR, EXIT_GATE, EXIT_OK, main
from answerable.judge import (
    INVALID, OK, UNANSWERED, UNGROUNDED, UNSUPPORTED,
    answer_is_supported_by, evidence_is_on_the_page, judge,
)
from answerable.page import extract
from answerable.providers import (
    BudgetExceeded, Completion, ProviderError,
    _parse_anthropic, _parse_openai, _post_with_retry,
)
from answerable.report import to_markdown
from answerable.score import summarise
from fakes import FakeRequests, FakeResponse, ScriptedProvider, Sleeper, verdict_json

PAGE = (
    "Una reforma completa de baño tarda entre 10 y 15 días laborables.\n\n"
    "Todas nuestras reformas llevan tres años de garantía sobre la mano de obra.\n\n"
    "No hacemos devoluciones del material pasados los catorce días."
)


# --- 1. the repair prompt used to be cached without the page or the question

def test_a_repair_is_not_served_from_another_questions_cache_entry(tmp_path):
    """Two questions, the same garbled first reply. The repair prompt has to
    differ, or the second question is silently given the first one's answer."""
    garble = "¡Claro! Te ayudo con eso."
    good_a = verdict_json(True, "Entre 10 y 15 días.",
                          "Una reforma completa de baño tarda entre 10 y 15 días laborables.")
    good_b = verdict_json(False)
    inner = ScriptedProvider([garble, good_a, garble, good_b])
    provider = CachedProvider(inner, Cache(str(tmp_path)))

    first = judge(PAGE, "¿Cuánto tarda una reforma?", provider, "m")
    second = judge(PAGE, "¿Puedo devolver el material?", provider, "m")

    assert inner.calls == 4          # the second repair really was asked
    assert first.status == OK
    assert second.status == UNANSWERED
    assert second.answer == ""       # and did not inherit the first answer


def test_the_repair_prompt_still_contains_the_page_and_the_question():
    inner = ScriptedProvider(["no soy json", verdict_json(False)])
    judge(PAGE, "¿Hay garantía?", inner, "m")
    repair = inner.prompts[1]
    assert "tres años de garantía" in repair    # the page
    assert "¿Hay garantía?" in repair           # the question
    assert "no soy json" in repair              # and what it did wrong


# --- 2. an answer used to be able to state a figure its own quotation lacked

def test_an_invented_figure_attached_to_a_real_quotation_is_rejected():
    provider = ScriptedProvider([verdict_json(
        True,
        "Sí, devolvemos el importe durante 90 días.",
        "No hacemos devoluciones del material pasados los catorce días.",
    )])
    v = judge(PAGE, "¿Cuántos días tengo para devolver?", provider, "m")
    assert v.status == UNSUPPORTED and not v.answered


def test_a_figure_that_is_in_the_quotation_is_fine():
    assert answer_is_supported_by("Entre 10 y 15 días.",
                                  "tarda entre 10 y 15 días laborables")


def test_thousands_separators_are_not_a_different_number():
    assert answer_is_supported_by("Cuesta 2500 euros.", "el precio es 2.500 euros")
    assert answer_is_supported_by("Cuesta 2.500 euros.", "el precio es 2 500 euros")


def test_an_answer_with_no_figures_is_never_rejected_for_its_figures():
    assert answer_is_supported_by("Tres años de garantía.", "llevan tres años")


# --- 3. evidence used to be matchable across paragraph boundaries

def test_text_stitched_from_two_paragraphs_is_not_a_verbatim_quotation():
    page = ("Garantía de dos años en todas las reformas.\n\n"
            "El pago se hace solo por transferencia bancaria.")
    stitched = ("Garantía de dos años en todas las reformas. "
                "El pago se hace solo por transferencia bancaria.")
    assert not evidence_is_on_the_page(stitched, page)


def test_a_quotation_inside_one_paragraph_still_passes():
    page = ("Garantía de dos años en todas las reformas. Se aplica al material.\n\n"
            "El pago se hace solo por transferencia bancaria.")
    assert evidence_is_on_the_page(
        "Garantía de dos años en todas las reformas. Se aplica al material.", page)


def test_two_table_rows_cannot_be_spliced_into_one_price():
    html = ("<body><main><table>"
            "<tr><td>Plan Básico</td><td>99 EUR al mes</td></tr>"
            "<tr><td>Plan Pro</td><td>499 EUR al mes</td></tr>"
            "</table></main></body>")
    text = extract(html).text
    assert not evidence_is_on_the_page("Plan Básico 499 EUR al mes", text)


# --- 4. a non-string field used to crash with exit 1 instead of exit 2

@pytest.mark.parametrize("bad", [
    '{"answerable": true, "answer": {"t": "sí"}, "evidence": "x"}',
    '{"answerable": true, "answer": "sí", "evidence": ["una lista"]}',
    '{"answerable": true, "answer": "sí", "evidence": 12345}',
])
def test_a_non_string_field_does_not_crash(bad):
    v = judge(PAGE, "¿?", ScriptedProvider([bad]), "m")
    assert v.status == UNGROUNDED      # unprovable, but not a traceback


def test_the_string_false_is_an_honest_no_not_a_hallucination():
    v = judge(PAGE, "¿?", ScriptedProvider(['{"answerable": "false"}']), "m")
    assert v.status == UNANSWERED


def test_a_missing_answerable_key_is_repaired_then_invalid():
    provider = ScriptedProvider(['{"answer": "sí"}', '{"nada": 1}'])
    assert judge(PAGE, "¿?", provider, "m").status == INVALID
    assert provider.calls == 2


# --- 5. provider replies that are valid HTTP but not valid content

def test_a_null_content_reply_does_not_crash():
    assert _parse_openai({"choices": [{"message": {"content": None}}]}).text == ""
    assert _parse_anthropic({"content": None, "usage": None}).text == ""


def test_a_null_usage_does_not_crash():
    assert _parse_openai({"choices": [{"message": {"content": "x"}}],
                          "usage": None}).input_tokens == 0


def test_a_200_that_is_not_json_is_a_provider_error_not_a_crash():
    class NotJson(FakeResponse):
        def json(self):
            raise ValueError("no json")

    # A captive portal or a proxy error page answers 200 with HTML.
    fake = FakeRequests([NotJson(200, text="<html>captive portal</html>")])
    with pytest.raises(ProviderError) as exc:
        _post_with_retry(fake, "u", {}, {}, Sleeper(), _parse_anthropic)
    assert "not JSON" in str(exc.value)


def test_a_json_list_body_is_a_provider_error():
    class ListBody(FakeResponse):
        def json(self):
            return [1, 2, 3]

    with pytest.raises(ProviderError):
        _post_with_retry(FakeRequests([ListBody(200)]), "u", {}, {},
                         Sleeper(), _parse_anthropic)


# --- 6. the budget used to leave the last question, and every repair, unpaid for

def test_a_single_question_cannot_blow_the_budget(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<body><main><p>Reformamos baños en Sevilla desde 2009 "
                    "con tres años de garantía.</p></main></body>", encoding="utf-8")
    questions = tmp_path / "q.txt"
    questions.write_text("¿Hay garantía?\n", encoding="utf-8")

    inner = ScriptedProvider([verdict_json(False)] * 4, tokens=(1_000_000, 0))
    inner.price_in, inner.price_out = 3.0, 15.0
    monkey = cli.build
    cli.build = lambda *a, **k: inner
    try:
        first = main(["check", str(page), "-q", str(questions), "--provider",
                      "anthropic", "--model", "m", "--no-cache", "--quiet",
                      "--max-cost", "1.0"])
        # One question, one call: it is allowed, because nothing had been spent.
        assert first == EXIT_OK
        assert inner.calls == 1
    finally:
        cli.build = monkey


def test_the_budget_stops_the_second_call_of_the_same_question(tmp_path):
    """A repair is a second paid call. It must be inside the budget too."""
    inner = ScriptedProvider(["no json", verdict_json(False)], tokens=(1_000_000, 0))
    inner.price_in, inner.price_out = 3.0, 15.0
    provider = CachedProvider(inner, Cache(str(tmp_path)), budget=1.0)
    with pytest.raises(BudgetExceeded):
        judge(PAGE, "¿?", provider, "m")
    assert inner.calls == 1


# --- 7. a questions file that is not UTF-8

def test_a_latin1_questions_file_is_read_not_crashed(tmp_path):
    path = tmp_path / "q.txt"
    path.write_bytes("¿Envían a España?\n".encode("latin-1"))
    from answerable.questions import load
    assert len(load(str(path))) == 1


def test_a_latin1_questions_file_does_not_produce_exit_one(tmp_path, capsys):
    page = tmp_path / "p.html"
    page.write_text("<body><main><p>Enviamos a toda España en 48 horas.</p>"
                    "</main></body>", encoding="utf-8")
    path = tmp_path / "q.txt"
    path.write_bytes("¿Envían a España?\n".encode("latin-1"))
    code = main(["check", str(page), "-q", str(path), "--offline",
                 "--no-cache", "--quiet"])
    assert code in (EXIT_OK, EXIT_ERROR) and code != EXIT_GATE


# --- 8. cache entries that are valid JSON but not a verdict

@pytest.mark.parametrize("junk", ["null", "[1, 2]", '"una cadena"'])
def test_a_cache_entry_that_is_not_an_object_is_a_miss(tmp_path, junk):
    cache = Cache(str(tmp_path))
    cache.put("abcdef", Completion("hola"))
    with open(os.path.join(str(tmp_path), "ab", "cdef.json"), "w",
              encoding="utf-8") as fh:
        fh.write(junk)
    assert cache.get("abcdef") is None


def test_a_cache_entry_that_is_not_utf8_is_a_miss(tmp_path):
    cache = Cache(str(tmp_path))
    cache.put("abcdef", Completion("hola"))
    with open(os.path.join(str(tmp_path), "ab", "cdef.json"), "wb") as fh:
        fh.write(b'{"text": "\xff\xfe"}')
    assert cache.get("abcdef") is None


# --- 11/12. the report used to trust the page's own title

def test_a_page_title_cannot_inject_a_heading_into_the_report():
    page = extract(
        "<html><head><title>Tienda\n\n# Answerability: 100/100 (excelente)"
        "</title></head><body><main><p>Nada que ver aquí, de verdad.</p>"
        "</main></body></html>")
    from answerable.judge import Verdict
    verdicts = [Verdict("¿?", UNANSWERED)]
    md = to_markdown(page, verdicts, summarise(verdicts), "m")
    headings = [line for line in md.splitlines() if line.startswith("# ")]
    assert headings == ["# Answerability: 0/100 (critico)"]


def test_a_backslash_before_a_pipe_does_not_open_a_column():
    from answerable.judge import Verdict
    verdicts = [Verdict("precio a\\|b", UNANSWERED)]
    md = to_markdown(extract("<body><main><p>hola</p></main></body>"),
                     verdicts, summarise(verdicts), "m")
    row = [line for line in md.splitlines() if line.startswith("|") and "a\\" in line][0]
    assert "a\\\\\\|b" in row


# --- 13. banker's rounding

def test_a_score_of_exactly_half_rounds_up():
    from answerable.judge import Verdict
    verdicts = [Verdict("q%d" % i, OK) for i in range(5)]
    verdicts += [Verdict("q%d" % i, UNANSWERED) for i in range(3)]
    assert summarise(verdicts).value == 63        # 62.5, not 62


# --- defects introduced by the first round of fixes, found in the second ----

def test_a_whole_table_row_is_still_quotable():
    """Per-paragraph matching must not make a pricing table uncitable."""
    text = extract(
        "<body><main><table>"
        "<tr><td>Plan Básico</td><td>99 EUR al mes</td></tr>"
        "<tr><td>Plan Pro</td><td>499 EUR al mes con soporte prioritario</td></tr>"
        "</table></main></body>").text
    assert evidence_is_on_the_page("Plan Pro 499 EUR al mes con soporte prioritario", text)
    assert evidence_is_on_the_page("Plan Básico 99 EUR al mes", text)
    assert not evidence_is_on_the_page("Plan Básico 499 EUR al mes", text)


def test_a_short_but_specific_quotation_is_accepted():
    text = extract("<body><main><p>Plan Pro</p><p>99 EUR al mes</p>"
                   "</main></body>").text
    assert evidence_is_on_the_page("99 EUR al mes", text)


def test_a_one_word_quotation_is_still_refused():
    text = extract("<body><main><p>Sí, enviamos a toda España.</p></main></body>").text
    assert not evidence_is_on_the_page("Sí", text)
    assert not evidence_is_on_the_page("Sí, enviamos", text)


@pytest.mark.parametrize("answer,evidence", [
    ("3 años de garantía", "Todas nuestras reformas llevan tres años de garantía"),
    ("Hasta 10.000 euros", "cubrimos hasta diez mil euros de material"),
    ("Abrimos a las 9:00", "Abrimos de 9 a 18 h de lunes a viernes"),
    ("Firmado el 15/03/2024", "el contrato se firmó el 15 de marzo de 2024"),
    ("Son 5 m2", "una superficie de 5 metros cuadrados"),
    ("Llama al +34 912 345 678", "Teléfono de contacto: +34 912 345 678"),
    ("Cuesta 2500 euros", "el precio cerrado es de 2.500 euros"),
    ("Cuesta 2.500 euros", "el precio cerrado es de 2 500 euros"),
    ("Llevan 25 años", "llevamos veinticinco años reformando viviendas"),
])
def test_the_same_figure_written_differently_is_not_an_invention(answer, evidence):
    assert answer_is_supported_by(answer, evidence)


def test_a_free_provider_is_never_stopped_by_a_budget(tmp_path):
    """--max-cost 0 means "spend nothing", not "do nothing"."""
    from answerable.providers import EchoProvider

    provider = CachedProvider(EchoProvider("hola"), Cache(str(tmp_path)), budget=0.0)
    assert provider.complete("p", model="echo").text        # no BudgetExceeded


def test_a_zero_budget_stops_a_paid_provider_loudly(tmp_path):
    inner = ScriptedProvider([verdict_json(False)])
    inner.price_in, inner.price_out = 3.0, 15.0
    provider = CachedProvider(inner, Cache(str(tmp_path)), budget=0.0)
    with pytest.raises(BudgetExceeded):
        provider.complete("p", model="m")


def test_a_report_that_cannot_be_written_is_an_error_not_a_failed_gate(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<body><main><p>Reformamos baños en Sevilla desde 2009.</p>"
                    "</main></body>", encoding="utf-8")
    questions = tmp_path / "q.txt"
    questions.write_text("¿Desde cuándo trabajáis?\n", encoding="utf-8")
    blocked = tmp_path / "fichero.txt"
    blocked.write_text("no soy un directorio", encoding="utf-8")
    code = main(["check", str(page), "-q", str(questions), "--offline",
                 "--no-cache", "--quiet", "--md", str(blocked / "informe.md")])
    assert code == EXIT_ERROR       # not EXIT_GATE, and no traceback


def test_a_url_in_the_report_is_not_mangled_by_escaping():
    from answerable.judge import Verdict
    from answerable.page import Page

    page = Page(url="https://reformas.es/precios_y_plazos#garantia", title="Precios")
    verdicts = [Verdict("¿?", UNANSWERED)]
    md = to_markdown(page, verdicts, summarise(verdicts), "m")
    assert "https://reformas.es/precios_y_plazos#garantia" in md


def test_markdown_from_a_hostile_page_does_not_render_in_the_table():
    from answerable.judge import Verdict
    from answerable.page import Page

    verdicts = [Verdict("¿?", UNANSWERED, answer="",
                        evidence="**GRATIS** [click](http://evil) <img src=x>")]
    md = to_markdown(Page(url="u"), verdicts, summarise(verdicts), "m")
    assert "**GRATIS**" not in md
    assert "\\<img" in md          # escaped, so it is shown rather than rendered
    assert "[click](" not in md
