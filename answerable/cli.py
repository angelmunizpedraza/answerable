"""Command line entry point.

    answerable check <url|file> --questions questions.txt --provider anthropic --model <id>

Exit codes are the contract with CI:
    0  every gate passed
    1  a gate failed (score below --min-score, or a hallucination with --fail-on-hallucination)
    2  the run could not be completed (bad input, provider down, budget exhausted)
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .cache import Cache, CachedProvider
from .judge import judge
from .page import Page, extract, fetch, read_file
from .providers import BudgetExceeded, ProviderError, build
from .questions import load
from .report import to_json, to_markdown
from .score import summarise

EXIT_OK, EXIT_GATE, EXIT_ERROR = 0, 1, 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="answerable",
        description="Ask a model whether a page can answer your buyers' questions, "
                    "and verify every answer against the page before believing it.",
    )
    p.add_argument("--version", action="version", version=f"answerable {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="check one page against a list of questions")
    c.add_argument("target", help="URL, or path to a local HTML file")
    c.add_argument("-q", "--questions", required=True,
                   help="file with one question per line (# starts a comment)")
    c.add_argument("--provider", default="echo",
                   choices=["echo", "anthropic", "openai"],
                   help="echo runs offline with no API key and no cost (default)")
    c.add_argument("--model", default="",
                   help="model id, required for anthropic and openai")
    c.add_argument("--offline", action="store_true",
                   help="force the offline provider, ignoring --provider")
    c.add_argument("--max-chars", type=int, default=60_000,
                   help="how much page text the model may read (default 60000)")

    c.add_argument("--min-score", type=int, default=None,
                   help="exit 1 if the answerability score is below this")
    c.add_argument("--fail-on-hallucination", action="store_true",
                   help="exit 1 if any answer could not be verified against the page")
    c.add_argument("--max-cost", type=float, default=None,
                   help="stop the run, with exit 2, as soon as this many USD "
                        "have been spent")

    c.add_argument("--cache-dir", default=".answerable-cache")
    c.add_argument("--no-cache", action="store_true")
    c.add_argument("--md", help="write the markdown report to this path")
    c.add_argument("--json", dest="json_path", help="write the JSON report to this path")
    c.add_argument("--job-summary", action="store_true",
                   help="append the markdown report to $GITHUB_STEP_SUMMARY")
    c.add_argument("--quiet", action="store_true")
    c.add_argument("--debug", action="store_true",
                   help="re-raise unexpected errors instead of reporting them")
    return p


def _load_page(target: str, timeout: int = 20) -> Page:
    if target.startswith(("http://", "https://")):
        return fetch(target, timeout=timeout)
    if target == "-":
        return extract(sys.stdin.read(), url="(stdin)")
    if not os.path.exists(target):
        raise FileNotFoundError(target)
    return read_file(target)


def _run_check(args, out=None, err=None) -> int:
    # Resolved here, not in the signature: binding sys.stdout at import time
    # makes the streams impossible to capture or redirect later.
    out = out or sys.stdout
    err = err or sys.stderr
    try:
        page = _load_page(args.target)
    except FileNotFoundError:
        print(f"answerable: no such file: {args.target}", file=err)
        return EXIT_ERROR
    except Exception as exc:
        print(f"answerable: could not read {args.target}: {exc}", file=err)
        return EXIT_ERROR

    if not page.text.strip():
        print(f"answerable: {args.target} has no readable text after stripping "
              "navigation and scripts", file=err)
        return EXIT_ERROR

    try:
        questions = load(args.questions)
    except (OSError, ValueError) as exc:
        print(f"answerable: could not read questions: {exc}", file=err)
        return EXIT_ERROR
    if not questions:
        print(f"answerable: {args.questions} contains no questions", file=err)
        return EXIT_ERROR

    if args.max_chars < 1000:
        # Silently truncating to nothing would score every page 0 and fail
        # every gate, which looks exactly like a real finding.
        print(f"answerable: --max-chars must be at least 1000, got "
              f"{args.max_chars}", file=err)
        return EXIT_ERROR

    if len(page.text) > args.max_chars:
        # Saying nothing here would be dishonest: questions answered in the
        # discarded tail come back as gaps that are not gaps.
        print(f"answerable: warning: the page has {len(page.text)} characters and "
              f"only the first {args.max_chars} were sent to the model. Raise "
              "--max-chars if the page is long.", file=err)

    provider_name = "echo" if args.offline else args.provider
    if provider_name in ("anthropic", "openai") and not args.model:
        print(f"answerable: --model is required with --provider {provider_name}",
              file=err)
        return EXIT_ERROR
    model = args.model or "echo"

    try:
        base = build(provider_name, page_text=page.text)
    except ProviderError as exc:
        print(f"answerable: {exc}", file=err)
        return EXIT_ERROR

    cache = Cache(args.cache_dir, enabled=not args.no_cache)
    provider = CachedProvider(base, cache, budget=args.max_cost)

    verdicts = []
    for question in questions:
        try:
            verdicts.append(
                judge(page.text, question, provider, model,
                      max_chars=args.max_chars)
            )
        except BudgetExceeded as exc:
            # An unfinished run is an error, never a score: reporting the
            # questions that did get asked would understate the page.
            print(f"answerable: stopped after {len(verdicts)} of {len(questions)} "
                  f"questions: {exc}. Raise --max-cost or ask fewer questions.",
                  file=err)
            return EXIT_ERROR
        except ProviderError as exc:
            print(f"answerable: provider failed on {question!r}: {exc}", file=err)
            return EXIT_ERROR
        except Exception as exc:  # a bug here must not read as a failed gate
            if args.debug:
                raise
            print(f"answerable: unexpected failure on {question!r}: "
                  f"{type(exc).__name__}: {exc} (rerun with --debug for the "
                  "stack trace)", file=err)
            return EXIT_ERROR

    score = summarise(verdicts)
    cost = provider.spent
    markdown = to_markdown(page, verdicts, score, model, cost)

    try:
        if args.md:
            _write(args.md, markdown)
        if args.json_path:
            _write(args.json_path, to_json(page, verdicts, score, model, cost))
        if args.job_summary:
            summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary:
                with open(summary, "a", encoding="utf-8") as fh:
                    fh.write(markdown + "\n")
    except OSError as exc:
        # Exit 1 means "a gate failed". A disk that would not take the report
        # is not a gate, and must not be reported as one.
        if args.debug:
            raise
        print(f"answerable: could not write the report: {exc}", file=err)
        return EXIT_ERROR

    if not args.quiet:
        print(markdown, file=out)

    looked_up = cache.hits + cache.misses
    if cache.hits and not args.quiet:
        # Calls, not questions: a question that needed a repair made two.
        print(f"({cache.hits} of {looked_up} model calls came from the cache)",
              file=out)

    failed = False
    if args.min_score is not None and score.value < args.min_score:
        print(f"answerable: score {score.value} is below the required "
              f"{args.min_score}", file=err)
        failed = True
    unproven = score.ungrounded + score.unsupported
    if args.fail_on_hallucination and unproven:
        print(f"answerable: {unproven} answer(s) could not be verified against "
              "the page", file=err)
        failed = True
    return EXIT_GATE if failed else EXIT_OK


def _write(path: str, text: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "check":
        return _run_check(args)
    return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
