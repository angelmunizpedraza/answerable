"""Turn a URL or an HTML file into the text a model is allowed to read."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

# Stripped entirely: they are never the answer to a buyer's question.
NON_CONTENT = ("script", "style", "noscript", "template", "svg", "iframe")
# Stripped too: a cookie banner or a mega-menu is not page content, and leaving
# them in lets a model "answer" from boilerplate that appears on every URL.
BOILERPLATE = ("nav", "header", "footer", "aside", "form")
# Text is joined with spaces inside these and broken between them, so a
# paragraph arrives as one line. It matters twice over: the model is asked to
# quote verbatim, and a quotation chopped by the HTML source indentation is a
# quotation nobody can check; and the verifier refuses to match across these
# breaks, so a quote cannot be stitched out of two unrelated paragraphs.
BLOCKS = (
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr",
    "blockquote", "pre", "dt", "dd", "figcaption", "div", "section", "article",
)
# Note what is NOT in that tuple: td and th. A table row is one readable line,
# so its cells are joined with a space. Breaking between them would make
# "Plan Pro 499 EUR al mes" impossible to quote, while breaking between rows
# still stops "Plan Basico" being welded to the next row's price.

_WS = re.compile(r"[ \t ]+")
_BLANK = re.compile(r"\n{3,}")


@dataclass
class Page:
    url: str
    title: str = ""
    text: str = ""
    headings: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(re.findall(r"[^\W\d_]+", self.text, flags=re.UNICODE))


def _clean(raw: str) -> str:
    """Collapse a block's internal line breaks, keep the breaks between blocks.

    HTML sources wrap their paragraphs wherever the editor's margin fell. Those
    line breaks are not in the sentence, so they must not end up in the text a
    model is asked to quote verbatim from.
    """
    raw = _WS.sub(" ", raw)
    raw = "\n".join(line.strip() for line in raw.splitlines())
    raw = _BLANK.sub("\n\n", raw)
    parts = [_WS.sub(" ", block.replace("\n", " ")).strip()
             for block in raw.split("\n\n")]
    return "\n\n".join(part for part in parts if part).strip()


def extract(html: str, url: str = "") -> Page:
    """Extract the readable body of a page.

    Boilerplate is removed on purpose: if the model is allowed to quote the
    footer, every page in the world "answers" a question about opening hours.
    """
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup.find_all(NON_CONTENT + BOILERPLATE):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    headings = [h.get_text(" ", strip=True) for h in main.find_all(["h1", "h2", "h3"])]
    headings = [h for h in headings if h]

    for br in main.find_all("br"):
        br.replace_with("\n")
    for tag in main.find_all(BLOCKS):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")

    return Page(url=url, title=title, text=_clean(main.get_text(" ")), headings=headings)


def read_file(path: str) -> Page:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return extract(fh.read(), url=path)


def fetch(url: str, timeout: int = 20) -> Page:
    import requests  # imported here so the offline path needs no network stack

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; answerable/0.1; "
            "+https://github.com/angelmunizpedraza/answerable)"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return extract(resp.text, url=url)
