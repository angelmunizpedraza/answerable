import pytest

from answerable import providers
from answerable.providers import (
    MAX_ATTEMPTS, Completion, EchoProvider, ProviderError,
    _parse_anthropic, _parse_openai, _post_with_retry, build,
)
from fakes import FakeRequests, FakeResponse, Sleeper


def _post(responses, sleeper=None, parse=_parse_anthropic):
    fake = FakeRequests(responses)
    return fake, _post_with_retry(
        fake, "https://example.invalid", {}, {}, sleeper or Sleeper(), parse
    )


def test_a_good_reply_costs_one_call():
    payload = {"content": [{"text": "hola"}],
               "usage": {"input_tokens": 10, "output_tokens": 4}}
    fake, result = _post([FakeResponse(200, payload)])
    assert result.text == "hola" and result.input_tokens == 10
    assert len(fake.calls) == 1


def test_a_rate_limit_is_retried():
    payload = {"content": [{"text": "ok"}], "usage": {}}
    fake, result = _post([FakeResponse(429), FakeResponse(200, payload)])
    assert result.text == "ok"
    assert len(fake.calls) == 2


def test_the_servers_own_retry_after_is_honoured():
    sleeper = Sleeper()
    payload = {"content": [{"text": "ok"}], "usage": {}}
    _post([FakeResponse(429, headers={"retry-after": "7"}),
           FakeResponse(200, payload)], sleeper)
    assert sleeper.waits == [7]


def test_backoff_grows_between_attempts():
    sleeper = Sleeper()
    with pytest.raises(ProviderError):
        _post([FakeResponse(503)] * MAX_ATTEMPTS, sleeper)
    assert len(sleeper.waits) == MAX_ATTEMPTS - 1
    assert sleeper.waits == sorted(sleeper.waits)
    assert sleeper.waits[-1] > sleeper.waits[0]


def test_it_gives_up_instead_of_retrying_forever():
    fake = FakeRequests([FakeResponse(500)] * MAX_ATTEMPTS)
    with pytest.raises(ProviderError) as exc:
        _post_with_retry(fake, "u", {}, {}, Sleeper(), _parse_anthropic)
    assert len(fake.calls) == MAX_ATTEMPTS
    assert "500" in str(exc.value)


def test_a_bad_api_key_is_not_retried():
    # Retrying a 401 four times just wastes a minute and still fails.
    fake = FakeRequests([FakeResponse(401, text="invalid x-api-key")])
    with pytest.raises(ProviderError) as exc:
        _post_with_retry(fake, "u", {}, {}, Sleeper(), _parse_anthropic)
    assert len(fake.calls) == 1
    assert "401" in str(exc.value)


def test_a_network_error_is_retried_then_reported():
    fake = FakeRequests([ConnectionError("reset")] * MAX_ATTEMPTS)
    with pytest.raises(ProviderError) as exc:
        _post_with_retry(fake, "u", {}, {}, Sleeper(), _parse_anthropic)
    assert "reset" in str(exc.value)


def test_anthropic_and_openai_payloads_are_parsed():
    a = _parse_anthropic({"content": [{"text": "uno"}, {"text": " dos"}],
                          "usage": {"input_tokens": 3, "output_tokens": 2}})
    assert (a.text, a.input_tokens, a.output_tokens) == ("uno dos", 3, 2)

    o = _parse_openai({"choices": [{"message": {"content": "hola"}}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    assert (o.text, o.input_tokens, o.output_tokens) == ("hola", 5, 1)


def test_an_empty_reply_does_not_crash_the_parser():
    assert _parse_anthropic({}).text == ""
    assert _parse_openai({}).text == ""


def test_cost_is_per_million_tokens():
    class P(providers.Provider):
        price_in, price_out = 3.0, 15.0

    cost = P().cost(Completion("", input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == pytest.approx(18.0)


def test_the_offline_provider_answers_from_the_page():
    page = ("Una reforma completa de baño tarda entre 10 y 15 días laborables "
            "desde que empieza la obra.")
    reply = EchoProvider(page).complete(
        f"PAGE:\n{page}\nQUESTION: ¿Cuánto tarda una reforma de baño?"
    )
    assert '"answerable": true' in reply.text
    assert "10 y 15" in reply.text


def test_the_offline_provider_says_no_when_the_page_does_not_say():
    page = "Reformamos baños en Sevilla y su área metropolitana desde 2009."
    reply = EchoProvider(page).complete(
        f"PAGE:\n{page}\nQUESTION: ¿Aceptáis pago fraccionado con financiación?"
    )
    assert '"answerable": false' in reply.text


def test_the_offline_provider_is_free_and_deterministic():
    page = "Reformamos baños en Sevilla desde 2009 con garantía de tres años."
    prompt = f"PAGE:\n{page}\nQUESTION: ¿Cuánta garantía hay?"
    p = EchoProvider(page)
    first, second = p.complete(prompt), p.complete(prompt)
    assert first.text == second.text
    assert p.cost(first) == 0.0


def test_build_picks_a_provider_and_rejects_unknown_ones():
    assert build("echo").name == "echo"
    assert build("").name == "echo"
    with pytest.raises(ProviderError):
        build("gemini")


def test_a_paid_provider_without_a_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        build("anthropic")
