from answerable.judge import (
    INVALID, OK, UNANSWERED, UNGROUNDED,
    _extract_json, build_prompt, evidence_is_on_the_page, judge,
)
from fakes import ScriptedProvider, verdict_json

PAGE = (
    "Una reforma completa de baño tarda entre 10 y 15 días laborables.\n\n"
    "Todas nuestras reformas llevan tres años de garantía sobre la mano de obra."
)


# --- the guardrail itself -------------------------------------------------

def test_a_verbatim_quotation_is_accepted():
    assert evidence_is_on_the_page(
        "Una reforma completa de baño tarda entre 10 y 15 días laborables.", PAGE
    )


def test_an_invented_quotation_is_rejected():
    assert not evidence_is_on_the_page(
        "Una reforma completa de baño tarda entre 3 y 4 días laborables.", PAGE
    )


def test_typographic_differences_do_not_count_as_invented():
    # A model that swaps a straight quote for a curly one has not lied.
    page = 'La garantía es de "tres años" - sin excepciones para el cliente.'
    quoted = "La garantía es de “tres años” – sin excepciones para el cliente."
    assert evidence_is_on_the_page(quoted, page)


def test_case_and_spacing_differences_do_not_count_as_invented():
    assert evidence_is_on_the_page(
        "TODAS NUESTRAS   REFORMAS LLEVAN TRES AÑOS DE GARANTÍA", PAGE
    )


def test_a_span_too_short_to_mean_anything_is_rejected():
    # "de" appears in every Spanish page ever written.
    assert not evidence_is_on_the_page("de", PAGE)
    assert not evidence_is_on_the_page("tres años", PAGE)


def test_empty_evidence_is_rejected():
    assert not evidence_is_on_the_page("", PAGE)
    assert not evidence_is_on_the_page("   ", PAGE)


# --- parsing whatever the model actually sent ------------------------------

def test_plain_json_is_parsed():
    assert _extract_json('{"answerable": true}') == {"answerable": True}


def test_a_fenced_block_is_parsed():
    assert _extract_json('```json\n{"answerable": false}\n```') == {"answerable": False}


def test_json_wrapped_in_chatter_is_parsed():
    text = 'Claro, aquí tienes:\n{"answerable": true, "answer": "sí"}\nEspero que sirva.'
    assert _extract_json(text)["answer"] == "sí"


def test_prose_is_not_parsed():
    assert _extract_json("No puedo responder a eso.") is None


def test_a_json_list_is_not_accepted_as_a_verdict():
    assert _extract_json('[{"answerable": true}]') is None


# --- the decision ----------------------------------------------------------

def test_a_grounded_answer_passes():
    provider = ScriptedProvider([
        verdict_json(True, "Entre 10 y 15 días laborables.",
                     "Una reforma completa de baño tarda entre 10 y 15 días laborables.")
    ])
    v = judge(PAGE, "¿Cuánto tarda?", provider, "fake")
    assert v.status == OK and v.answered
    assert provider.calls == 1


def test_an_honest_no_is_not_a_failure():
    provider = ScriptedProvider([verdict_json(False)])
    v = judge(PAGE, "¿Cuánto cuesta?", provider, "fake")
    assert v.status == UNANSWERED and not v.answered


def test_an_answer_whose_quotation_is_not_on_the_page_is_rejected():
    provider = ScriptedProvider([
        verdict_json(True, "Cuesta 3.000 euros.",
                     "Una reforma de baño cuesta 3.000 euros con todo incluido.")
    ])
    v = judge(PAGE, "¿Cuánto cuesta?", provider, "fake")
    assert v.status == UNGROUNDED and not v.answered
    # The invented quotation is kept so a human can see what it made up.
    assert "3.000 euros" in v.evidence


def test_an_answer_with_no_quotation_at_all_is_rejected():
    provider = ScriptedProvider([verdict_json(True, "Sí, claro.", None)])
    assert judge(PAGE, "¿Hay garantía?", provider, "fake").status == UNGROUNDED


def test_a_malformed_reply_is_repaired_once():
    good = verdict_json(True, "Tres años.",
                        "Todas nuestras reformas llevan tres años de garantía sobre la mano de obra.")
    provider = ScriptedProvider(["Pues mira, depende del caso.", good])
    v = judge(PAGE, "¿Hay garantía?", provider, "fake")
    assert v.status == OK
    assert provider.calls == 2


def test_a_reply_that_stays_malformed_is_invalid_not_a_pass():
    provider = ScriptedProvider(["no", "sigo sin hacerte caso"])
    v = judge(PAGE, "¿Hay garantía?", provider, "fake")
    assert v.status == INVALID and not v.answered
    assert provider.calls == 2


def test_the_page_is_truncated_before_it_is_sent():
    prompt = build_prompt("ñ" * 100_000, "¿?", max_chars=500)
    assert prompt.count("ñ") == 500


def test_the_prompt_carries_the_question_and_the_page():
    prompt = build_prompt(PAGE, "¿Cuánto tarda?")
    assert "¿Cuánto tarda?" in prompt
    assert "tres años de garantía" in prompt
