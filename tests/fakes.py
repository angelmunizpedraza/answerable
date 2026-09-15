"""Fakes used across the suite. Nothing here touches the network."""

import json

from answerable.providers import Completion, Provider


class ScriptedProvider(Provider):
    """Returns the replies it was given, in order, and records the prompts."""

    name = "scripted"
    price_in = 1000.0   # 0.001 USD per 1k input tokens: easy to assert on
    price_out = 2000.0

    def __init__(self, replies, tokens=(100, 50)):
        self.replies = list(replies)
        self.prompts = []
        self.tokens = tokens

    def complete(self, prompt, *, model="fake", max_tokens=700):
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else ""
        return Completion(text, self.tokens[0], self.tokens[1])

    @property
    def calls(self):
        return len(self.prompts)


def verdict_json(answerable, answer=None, evidence=None):
    return json.dumps(
        {"answerable": answerable, "answer": answer, "evidence": evidence}
    )


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeRequests:
    """Stands in for the `requests` module inside _post_with_retry."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Sleeper:
    """Collects the sleeps instead of taking them."""

    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)
