# answerable

**Your page ranks. Can it actually answer the question the buyer typed?**
`answerable` sends one page and a list of real buyer questions to a model, forces a structured verdict, and then **checks in code that the quotation the model gave is really on the page, and that the answer states no figure that quotation does not contain.** An answer it cannot verify is thrown away, not scored.

[![CI](https://github.com/angelmunizpedraza/answerable/actions/workflows/ci.yml/badge.svg)](https://github.com/angelmunizpedraza/answerable/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why this exists

An AI answer engine does not rank your page. It reads it and tries to write a sentence. If the page does not contain the sentence, you are not in the answer — however well you rank.

So the useful question is no longer "does this page target the keyword". It is: **of the twenty things a buyer asks before paying, how many can this page actually answer?**

That is a question a model can answer, and there is exactly one reason people do not trust a model to answer it: it will happily say yes to anything. Ask a model "does this page say what the warranty is?" and it will find you a warranty, because finding one is what it was rewarded for.

This tool is built around that problem instead of around the model.

## The idea

Every answer must arrive with a span **copied verbatim from one paragraph of the page**. Nothing is taken on trust afterwards. Two checks run in plain Python, with no model involved:

**1. Is the quotation real?**

```python
def evidence_is_on_the_page(evidence: str, page_text: str,
                            min_chars: int = 12, min_words: int = 3) -> bool:
    if not evidence:
        return False
    needle = _normalise(evidence)
    if len(needle) < min_chars or len(needle.split()) < min_words:
        return False
    return any(needle in block for block in _blocks(page_text))
```

Both sides are normalised first — NFKC, curly quotes, dashes, non-breaking spaces, case, whitespace — because a model that turns `"tres años"` into `“tres años”` has not lied. The match has to land inside **one paragraph**: flattening the page first would let a model quote the end of one table row and the start of the next as though it were a sentence, and that invented sentence would pass. A table *row* is one paragraph, though, so a real price line stays quotable — being too strict here is not the safe side, it just accuses honest models.

**2. Does the answer stay inside its own quotation?**

A real sentence lifted from the page, with an invented price welded onto it, survives the first check. So every figure in the answer must also appear in the evidence.

The comparison is strict on the answer and generous on the quotation, because the only acceptable false result here is the one that lets a page off lightly. `2.500`, `2,500`, `2 500` and `2500` are the same figure; a decimal comma is not a thousands separator; `9:00` is nine o'clock and not the figures 9 and 0; the `2` in `m²` is a unit; a quotation saying *tres años*, *diez mil* or *15 de marzo* supports an answer of `3 años`, `10.000` or `15/03`.

This is deliberately not a claim of full entailment: proving that a quotation *implies* an answer would need a second model, and a second model has the first one's problem. It catches the failure that costs money in this domain — an invented price, deadline, percentage or number of years.

A verdict lands in one of five states, and only the first one scores:

| State | Meaning |
|---|---|
| `ok` | Answered, quotation verified, figures check out. |
| `unanswered` | The model said the page does not cover it. Honest, useful, never penalised. |
| `ungrounded` | It claimed an answer and quoted something **not on the page**. |
| `unsupported` | The quotation is real, but the answer states a figure the quotation does not contain. |
| `invalid` | It never returned a usable JSON verdict, even after one repair. Never coerced into a pass. |

Everything except `ok` counts as a miss. A buyer reading that page would not have found the answer either.

## Install

```bash
pip install "answerable @ git+https://github.com/angelmunizpedraza/answerable"
```

## Use

```bash
answerable check https://tu-web.es/reformas \
  --questions preguntas.txt \
  --provider anthropic --model <model-id> \
  --min-score 70 --max-cost 0.50 \
  --md informe.md --json informe.json
```

`preguntas.txt` is one question per line; `#` starts a comment:

```
# Lo que pregunta un cliente antes de pedir presupuesto.
¿Cuánto tarda una reforma de baño completa?
¿El presupuesto es gratuito?
¿Cuánta garantía tienen las reformas?
¿Hay que pagar algo por adelantado?
¿Trabajáis en Dos Hermanas?
¿Cuánto cuesta reformar un baño de 5 metros cuadrados?
¿Puedo pagar la reforma a plazos sin intereses?
¿Retiráis los escombros vosotros?
```

Output, abridged — a `## Detalle` table with one row per question follows:

```
# Answerability: 63/100 (flojo)

Half of what buyers ask is missing from the page.

- Página: Reformas Guadaíra — Reformas integrales de baño y cocina en Sevilla — examples/pagina.html
- Palabras analizadas: 207
- Preguntas: 8 · respondidas 5 · sin respuesta 3 · citas inventadas 0
- De las respuestas afirmadas, 100% se pudieron verificar literalmente en la página.
- Modelo: `echo` · coste de esta ejecución: $0.0000

## Lo que la página no responde

- ¿Cuánto cuesta reformar un baño de 5 metros cuadrados?
- ¿Puedo pagar la reforma a plazos sin intereses?
- ¿Retiráis los escombros vosotros?
```

That list is the deliverable. It is not an audit finding to argue about; it is the brief for the next three paragraphs of the page.

Run it with no API key and no cost:

```bash
answerable check examples/pagina.html -q examples/preguntas.txt --offline
```

## Putting an LLM in CI without it costing you money or your sanity

Four things have to be true before a model belongs in a pipeline, and each one is a module here.

**1. It must be cheap to be wrong.** Replies are cached in a content-addressed directory keyed on `provider + model + prompt`. Rerun an unchanged page and every answer comes off disk, free. Change the model and you correctly get a miss. The repair prompt carries the page and the question too — a repair keyed only on the model's garbled reply would let two different questions collide on one cache entry and be served each other's answers.

**2. It must not be able to spend your money.** `--max-cost` is checked inside the provider, before every paid call — including the repair call, which is why the check does not live in the question loop. It is a stop line rather than a cap: a call already in flight cannot be un-spent, so the run can cross the line by one call and then **halts with exit code 2**. An unfinished run is an error, never a score.

**3. It must not flake the build.** Retries with exponential backoff and jitter on 408/409/425/429/500/502/503/504, honouring the server's own integer `retry-after`. A 401 is *not* retried — retrying a bad key four times wastes a minute and still fails. A 200 carrying a proxy's HTML error page, a `null` content block and a missing `usage` object are all handled as errors, not as crashes: a stack trace in CI reads as a failed gate, which is a lie.

**4. It must give the same answer twice.** `temperature: 0`, deterministic prompts, and the cache. A score that moves when nothing changed is not a score, it is a mood.

```yaml
- uses: actions/cache@v4
  with:
    path: .answerable-cache
    key: answerable-${{ hashFiles('preguntas.txt') }}

- uses: angelmunizpedraza/answerable@v1.0.0
  with:
    target: https://tu-web.es/servicios/reformas-bano
    questions-file: preguntas.txt
    provider: anthropic
    model: <model-id>
    api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    min-score: "70"
    fail-on-hallucination: "true"
    max-cost: "0.50"
```

Pin the exact tag, not a moving one. A moving `@v1` can change under you between two runs of the same pipeline, and a gate that changes without you changing anything is not a gate.

Exit codes are the contract: `0` passed, `1` a gate failed, `2` the run could not be completed.

## What it does not do

- **It does not crawl.** One page per run, on purpose. Answerability is a property of a page, not of a domain.
- **It does not rewrite your page.** It tells you which questions are missing. Writing the answer is the job the model is worst at and you are best at.
- **It does not prove entailment.** See check 2 above: it proves the quotation is real and that the answer invents no figures. That is a floor, not a guarantee.
- **`--offline` is not a model.** The `echo` provider is a keyword-matching stub, so the pipeline can be wired up, tested and demoed for free. It is what the CLI tests and the whole CI run on. It is not an evaluator, and this README will not pretend it is.
- **It only reads the page.** Boilerplate (`nav`, `header`, `footer`, `aside`, `form`) is stripped before the model sees anything, because otherwise every page in the world "answers" a question from its own mega-menu.

## Tests

149 tests, all offline: no API key, no network, nothing to stub at your end.

```bash
pip install -e ".[dev]" && pytest -q
```

46 of them are in `tests/test_audit_regressions.py`, one per defect found when this package was reviewed adversarially against its own claims, in two rounds — because the first round of fixes introduced defects of its own, and that is the part people leave out.

Round one: text stitched across two paragraphs passing as a verbatim quotation; a real quotation carrying an invented price; the repair prompt colliding in the cache and serving one question's answer to another; `"false"` as a string being truthy, turning an honest "no" into an accusation of hallucination; the last question of a run escaping the budget; a provider reply with `content: null` crashing with exit 1, which CI reads as a failed gate.

Round two, all caused by round one: a pricing table becoming impossible to quote, so an honest model scored 0/100; a page that writes *tres años* rejecting an answer of `3 años` as invented; a quoted phone number read as an eleven-digit price; `--max-cost 0` blocking a run that costs nothing.

## Part of a set

- [seo-audit](https://github.com/angelmunizpedraza/seo-audit) — the technical baseline
- [render-gap](https://github.com/angelmunizpedraza/render-gap) — what AI crawlers see before JavaScript
- [geo-check](https://github.com/angelmunizpedraza/geo-check) — whether they are allowed to read it
- **answerable** — whether what they read can answer anything

---

MIT · [Ángel Muñiz Pedraza](https://www.linkedin.com/in/angel-muniz-seo)
