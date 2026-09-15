from answerable.questions import parse


def test_one_question_per_line():
    assert parse("¿Cuánto cuesta?\n¿Cuánto tarda?") == ["¿Cuánto cuesta?", "¿Cuánto tarda?"]


def test_comments_and_blank_lines_are_ignored():
    text = "# las preguntas del cliente\n\n¿Cuánto cuesta?\n\n   \n# fin\n"
    assert parse(text) == ["¿Cuánto cuesta?"]


def test_duplicates_are_dropped_keeping_the_first():
    text = "¿Cuánto cuesta?\n  ¿CUÁNTO   CUESTA?  \n¿Y el plazo?"
    assert parse(text) == ["¿Cuánto cuesta?", "¿Y el plazo?"]


def test_empty_file_gives_no_questions():
    assert parse("") == []


def test_load_reads_a_file(tmp_path):
    from answerable.questions import load

    path = tmp_path / "q.txt"
    path.write_text("# c\n¿Hay garantía?\n", encoding="utf-8")
    assert load(str(path)) == ["¿Hay garantía?"]
