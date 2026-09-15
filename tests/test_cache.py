import json
import os

from answerable.cache import Cache, CachedProvider, key_for
from answerable.providers import Completion
from fakes import ScriptedProvider


def test_the_key_changes_with_everything_that_changes_the_reply():
    base = key_for("anthropic", "m1", "prompt")
    assert key_for("openai", "m1", "prompt") != base
    assert key_for("anthropic", "m2", "prompt") != base
    assert key_for("anthropic", "m1", "prompt!") != base
    assert key_for("anthropic", "m1", "prompt") == base


def test_the_key_cannot_be_confused_by_moving_a_boundary():
    # Without a separator, ("ab","c") and ("a","bc") would hash the same.
    assert key_for("ab", "c", "x") != key_for("a", "bc", "x")


def test_a_stored_reply_comes_back(tmp_path):
    cache = Cache(str(tmp_path))
    cache.put("abc123", Completion("hola", 10, 2))
    got = cache.get("abc123")
    assert got.text == "hola" and got.input_tokens == 10
    assert got.cached is True


def test_a_missing_key_is_a_miss(tmp_path):
    assert Cache(str(tmp_path)).get("nope") is None


def test_a_corrupted_entry_is_a_miss_not_a_crash(tmp_path):
    cache = Cache(str(tmp_path))
    cache.put("deadbeef", Completion("hola"))
    path = os.path.join(str(tmp_path), "de", "adbeef.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{ this is not json")
    assert cache.get("deadbeef") is None


def test_writing_leaves_no_temporary_files_behind(tmp_path):
    cache = Cache(str(tmp_path))
    cache.put("abcdef", Completion("hola"))
    leftovers = [f for _, _, files in os.walk(str(tmp_path))
                 for f in files if f.endswith(".tmp")]
    assert leftovers == []


def test_a_disabled_cache_stores_nothing(tmp_path):
    target = tmp_path / "never-created"
    cache = Cache(str(target), enabled=False)
    cache.put("abcdef", Completion("hola"))
    assert cache.get("abcdef") is None
    assert not os.path.exists(str(target))


def test_the_second_identical_question_does_not_reach_the_model(tmp_path):
    inner = ScriptedProvider(['{"answerable": false}', '{"answerable": true}'])
    provider = CachedProvider(inner, Cache(str(tmp_path)))
    first = provider.complete("misma pregunta", model="m")
    second = provider.complete("misma pregunta", model="m")
    assert inner.calls == 1
    assert first.text == second.text
    assert second.cached is True


def test_a_different_model_is_asked_again(tmp_path):
    inner = ScriptedProvider(['{"a": 1}', '{"a": 2}'])
    provider = CachedProvider(inner, Cache(str(tmp_path)))
    provider.complete("p", model="m1")
    provider.complete("p", model="m2")
    assert inner.calls == 2


def test_a_cached_reply_is_not_charged_to_the_budget(tmp_path):
    inner = ScriptedProvider(['{"a": 1}'], tokens=(1_000_000, 0))
    provider = CachedProvider(inner, Cache(str(tmp_path)))
    provider.complete("p", model="m")
    spent_after_first = provider.spent
    provider.complete("p", model="m")
    assert spent_after_first > 0
    assert provider.spent == spent_after_first


def test_two_runs_of_the_same_page_give_the_same_bytes(tmp_path):
    # The reproducibility promise, tested rather than asserted in a README.
    inner = ScriptedProvider(['{"answerable": true, "answer": "x"}'])
    cache = Cache(str(tmp_path))
    first = CachedProvider(inner, cache).complete("p", model="m").text
    second = CachedProvider(ScriptedProvider([]), cache).complete("p", model="m").text
    assert first == second


def test_entries_are_sharded_so_one_directory_never_holds_them_all(tmp_path):
    cache = Cache(str(tmp_path))
    cache.put("ab" + "0" * 62, Completion("x"))
    assert os.path.isdir(os.path.join(str(tmp_path), "ab"))


def test_the_stored_file_is_readable_json(tmp_path):
    cache = Cache(str(tmp_path))
    cache.put("ffeedd", Completion("hola", 1, 2))
    with open(os.path.join(str(tmp_path), "ff", "eedd.json"), encoding="utf-8") as fh:
        assert json.load(fh)["text"] == "hola"
