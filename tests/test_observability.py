import pytest

from swiftrag import RAG
from swiftrag._retry import retry_call
from swiftrag.llms import LLMProvider


def test_retry_call_succeeds_after_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    result = retry_call(flaky, retries=3, sleep=lambda _: None, rng=lambda: 0.0)
    assert result == "ok"
    assert calls["n"] == 3


def test_retry_call_raises_after_exhausting():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        retry_call(always_fails, retries=2, sleep=lambda _: None, rng=lambda: 0.0)
    assert calls["n"] == 3  # 1 initial + 2 retries


def test_retry_call_respects_zero_retries():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise OSError("boom")

    with pytest.raises(OSError):
        retry_call(fn, retries=0, sleep=lambda _: None)
    assert calls["n"] == 1


def test_on_retrieve_callback_invoked():
    seen = []
    rag = RAG(
        documents=["Apples are fruit.", "Cars have wheels."],
        on_retrieve=lambda q, sources: seen.append((q, len(sources))),
    )
    rag.query("apples")
    assert seen and seen[0][0] == "apples"
    assert seen[0][1] >= 1


class _UsageLLM(LLMProvider):
    def __init__(self):
        self.last_usage = {}

    def generate(self, messages, **kwargs):
        self.last_usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        return "an answer"


def test_usage_is_attached_to_response_and_on_generate():
    captured = {}
    rag = RAG(
        documents="content",
        llm_model=_UsageLLM(),
        on_generate=lambda q, a, usage: captured.update(usage),
    )
    resp = rag.query("anything")
    assert resp.usage["total_tokens"] == 18
    assert captured["completion_tokens"] == 7


def test_callback_errors_do_not_break_query():
    def boom(*args):
        raise RuntimeError("callback failure")

    rag = RAG(documents="content", on_retrieve=boom, on_generate=boom)
    # Should not raise despite both hooks failing.
    resp = rag.query("content")
    assert resp.answer


def test_stream_triggers_on_generate():
    captured = {}
    rag = RAG(
        documents="Tigers are big cats.",
        on_generate=lambda q, a, usage: captured.update({"q": q, "a": a}),
    )
    tokens = list(rag.stream("tigers"))
    assert "".join(tokens)
    assert captured["q"] == "tigers"
    assert captured["a"]
