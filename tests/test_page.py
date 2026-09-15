from answerable.page import extract


HTML = """
<!doctype html><html><head><title>  Taller Ruiz  </title>
<style>.a{color:red}</style></head>
<body>
<nav>Inicio Contacto Blog</nav>
<header>Llama ya al 900 000 000</header>
<main>
  <h1>Cambio de aceite en Sevilla</h1>
  <p>El cambio de aceite tarda
  cuarenta minutos y cuesta
  sesenta euros.</p>
  <h2>Horario</h2>
  <p>Abrimos de lunes a viernes.</p>
  <script>var x = "abrimos los domingos";</script>
</main>
<footer>Aviso legal</footer>
</body></html>
"""


def test_title_is_stripped():
    assert extract(HTML).title == "Taller Ruiz"


def test_scripts_and_boilerplate_are_removed():
    text = extract(HTML).text
    assert "abrimos los domingos" not in text   # script contents
    assert "Aviso legal" not in text            # footer
    assert "900 000 000" not in text            # header
    assert "Inicio Contacto Blog" not in text   # nav


def test_a_paragraph_arrives_as_one_line():
    # The whole tool depends on verbatim quoting, so the source file's line
    # wrapping must not survive into the text the model reads.
    text = extract(HTML).text
    assert "El cambio de aceite tarda cuarenta minutos y cuesta sesenta euros." in text


def test_blocks_stay_separated():
    text = extract(HTML).text
    assert "sesenta euros. Horario" not in text


def test_headings_are_collected():
    assert extract(HTML).headings == ["Cambio de aceite en Sevilla", "Horario"]


def test_word_count_ignores_numbers_and_punctuation():
    page = extract("<body><main><p>Cuesta 60 euros, sí.</p></main></body>")
    assert page.word_count == 3  # Cuesta, euros, sí — the 60 is not a word
    assert "60" in page.text


def test_page_without_main_falls_back_to_body():
    page = extract("<body><p>Solo un parrafo.</p></body>")
    assert "Solo un parrafo." in page.text


def test_br_becomes_a_line_break():
    page = extract("<body><main><p>Lunes<br>Martes</p></main></body>")
    assert "Lunes" in page.text and "Martes" in page.text
