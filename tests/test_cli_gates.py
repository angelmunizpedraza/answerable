"""The gates that need a provider the offline stub cannot imitate:
one that invents quotations, and one that costs money.
"""

import pytest

from answerable import cli
from answerable.cli import EXIT_ERROR, EXIT_GATE, EXIT_OK, main
from fakes import ScriptedProvider, verdict_json

HTML = """<body><main>
<p>Una reforma completa de baño tarda entre 10 y 15 días laborables desde que empieza la obra.</p>
<p>Todas nuestras reformas llevan tres años de garantía sobre la mano de obra.</p>
</main></body>"""

GROUNDED = verdict_json(
    True, "Entre 10 y 15 días.",
    "Una reforma completa de baño tarda entre 10 y 15 días laborables desde que empieza la obra.")
INVENTED = verdict_json(
    True, "Cuesta 2.500 euros.",
    "Una reforma completa de baño cuesta 2.500 euros con todo incluido.")


@pytest.fixture()
def site(tmp_path):
    page = tmp_path / "pagina.html"
    page.write_text(HTML, encoding="utf-8")
    questions = tmp_path / "preguntas.txt"
    questions.write_text("¿Cuánto tarda?\n¿Cuánto cuesta?\n", encoding="utf-8")
    return str(page), str(questions)


def use(monkeypatch, replies, tokens=(0, 0)):
    provider = ScriptedProvider(replies, tokens=tokens)
    monkeypatch.setattr(cli, "build", lambda *a, **k: provider)
    return provider


def run(page, questions, *extra):
    return main(["check", page, "-q", questions, "--provider", "anthropic",
                 "--model", "un-modelo", "--no-cache", "--quiet"] + list(extra))


def test_an_invented_quotation_can_fail_the_build(site, monkeypatch, capsys):
    page, questions = site
    use(monkeypatch, [GROUNDED, INVENTED])
    assert run(page, questions, "--fail-on-hallucination") == EXIT_GATE
    assert "could not be verified" in capsys.readouterr().err


def test_without_that_flag_an_invented_quotation_only_lowers_the_score(
        site, monkeypatch, capsys):
    page, questions = site
    use(monkeypatch, [GROUNDED, INVENTED])
    code = main(["check", page, "-q", questions, "--provider", "anthropic",
                 "--model", "un-modelo", "--no-cache"])
    report = capsys.readouterr().out
    assert code == EXIT_OK
    assert "50/100" in report              # the invented answer did not count
    assert "citas inventadas 1" in report
    assert "2.500 euros" in report         # and the human is shown what it made up


def test_a_clean_run_passes_the_hallucination_gate(site, monkeypatch):
    page, questions = site
    use(monkeypatch, [GROUNDED, verdict_json(False)])
    assert run(page, questions, "--fail-on-hallucination") == EXIT_OK


def test_the_budget_stops_the_run_instead_of_scoring_half_a_page(site, monkeypatch, capsys):
    page, questions = site
    # One million input tokens per call at $3/M: the first question blows a $1 budget.
    use(monkeypatch, [GROUNDED, GROUNDED], tokens=(1_000_000, 0))
    code = run(page, questions, "--max-cost", "1.0")
    assert code == EXIT_ERROR      # not a score, not a pass: an unfinished run
    assert "budget" in capsys.readouterr().err


def test_a_budget_that_is_never_reached_does_not_interfere(site, monkeypatch):
    page, questions = site
    use(monkeypatch, [GROUNDED, verdict_json(False)], tokens=(10, 5))
    assert run(page, questions, "--max-cost", "5.0") == EXIT_OK


def test_a_provider_failure_is_an_error_not_a_zero_score(site, monkeypatch, capsys):
    from answerable.providers import ProviderError

    page, questions = site

    class Broken:
        name = "broken"

        def complete(self, *a, **k):
            raise ProviderError("HTTP 500")

        def cost(self, c):
            return 0.0

    monkeypatch.setattr(cli, "build", lambda *a, **k: Broken())
    assert run(page, questions) == EXIT_ERROR
    assert "provider failed" in capsys.readouterr().err
