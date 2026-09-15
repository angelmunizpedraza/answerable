import json

import pytest

from answerable.cli import EXIT_ERROR, EXIT_GATE, EXIT_OK, main

HTML = """<body><main>
<h1>Reformas Guadaíra</h1>
<p>Una reforma completa de baño tarda entre 10 y 15 días laborables desde que empieza la obra.</p>
<p>Todas nuestras reformas llevan tres años de garantía sobre la mano de obra.</p>
</main></body>"""

QUESTIONS = """¿Cuánto tarda una reforma completa de baño?
¿Cuántos años de garantía llevan las reformas?
¿Aceptáis pago fraccionado con financiación bancaria?
"""


@pytest.fixture()
def site(tmp_path):
    page = tmp_path / "pagina.html"
    page.write_text(HTML, encoding="utf-8")
    questions = tmp_path / "preguntas.txt"
    questions.write_text(QUESTIONS, encoding="utf-8")
    return tmp_path, str(page), str(questions)


def run(page, questions, *extra, cache=None):
    args = ["check", page, "-q", questions, "--offline", "--quiet"]
    args += ["--cache-dir", cache] if cache else ["--no-cache"]
    return main(args + list(extra))


def test_a_page_that_answers_enough_passes(site, capsys):
    _, page, questions = site
    assert run(page, questions, "--min-score", "60") == EXIT_OK


def test_a_page_that_answers_too_little_fails_the_build(site, capsys):
    _, page, questions = site
    assert run(page, questions, "--min-score", "100") == EXIT_GATE
    assert "below the required 100" in capsys.readouterr().err


def test_without_a_gate_it_only_reports(site):
    _, page, questions = site
    assert run(page, questions) == EXIT_OK


def test_a_missing_page_is_an_error_not_a_pass(site):
    _, _, questions = site
    assert run("no-existe.html", questions) == EXIT_ERROR


def test_a_missing_questions_file_is_an_error(site):
    _, page, _ = site
    assert run(page, "no-existe.txt") == EXIT_ERROR


def test_an_empty_questions_file_is_an_error(tmp_path, site):
    _, page, _ = site
    empty = tmp_path / "vacio.txt"
    empty.write_text("# solo comentarios\n", encoding="utf-8")
    assert run(page, str(empty)) == EXIT_ERROR


def test_a_page_with_no_readable_text_is_an_error(tmp_path, site):
    _, _, questions = site
    empty = tmp_path / "vacia.html"
    empty.write_text("<body><nav>menu</nav></body>", encoding="utf-8")
    assert run(str(empty), questions) == EXIT_ERROR


def test_a_paid_provider_without_a_model_stops_before_spending(site, capsys):
    _, page, questions = site
    code = main(["check", page, "-q", questions, "--provider", "anthropic",
                 "--no-cache", "--quiet"])
    assert code == EXIT_ERROR
    assert "--model is required" in capsys.readouterr().err


def test_the_reports_are_written_where_asked(tmp_path, site):
    _, page, questions = site
    md = tmp_path / "out" / "informe.md"
    js = tmp_path / "out" / "informe.json"
    assert run(page, questions, "--md", str(md), "--json", str(js)) == EXIT_OK
    assert "Answerability" in md.read_text(encoding="utf-8")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["counts"]["total"] == 3
    assert data["gaps"] == ["¿Aceptáis pago fraccionado con financiación bancaria?"]


def test_the_job_summary_is_appended_for_github(tmp_path, site, monkeypatch):
    _, page, questions = site
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert run(page, questions, "--job-summary") == EXIT_OK
    assert "Answerability" in summary.read_text(encoding="utf-8")


def test_the_second_run_of_an_unchanged_page_is_served_from_cache(tmp_path, site, capsys):
    _, page, questions = site
    cache = str(tmp_path / "cache")
    assert main(["check", page, "-q", questions, "--offline",
                 "--cache-dir", cache]) == EXIT_OK
    capsys.readouterr()
    assert main(["check", page, "-q", questions, "--offline",
                 "--cache-dir", cache]) == EXIT_OK
    assert "came from the cache" in capsys.readouterr().out


def test_a_report_is_printed_unless_quiet(site, capsys):
    _, page, questions = site
    main(["check", page, "-q", questions, "--offline", "--no-cache"])
    assert "# Answerability" in capsys.readouterr().out


def test_quiet_prints_nothing_on_success(site, capsys):
    _, page, questions = site
    run(page, questions)
    assert capsys.readouterr().out == ""


def test_a_long_page_warns_that_it_was_truncated(tmp_path, site, capsys):
    _, _, questions = site
    long_page = tmp_path / "larga.html"
    body = "<p>Una reforma completa de baño tarda entre 10 y 15 días.</p>" * 60
    long_page.write_text(f"<body><main>{body}</main></body>", encoding="utf-8")
    run(str(long_page), questions, "--max-chars", "1000")
    assert "only the first 1000" in capsys.readouterr().err


def test_a_max_chars_too_small_to_be_real_is_rejected(site, capsys):
    _, page, questions = site
    assert run(page, questions, "--max-chars", "0") == EXIT_ERROR
    assert "at least 1000" in capsys.readouterr().err


def test_stdin_can_be_the_page(site, monkeypatch, capsys):
    import io

    _, _, questions = site
    monkeypatch.setattr("sys.stdin", io.StringIO(HTML))
    assert run("-", questions, "--min-score", "60") == EXIT_OK


def test_the_version_flag_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_no_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
