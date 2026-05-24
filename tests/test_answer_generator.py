"""Tests for sme.eval.answer_generator.

The reader wraps an OpenAI-shaped client; tests mock that client so they
run without ``openai`` installed and without network access. Mirrors the
``_FakeClient`` shape used by ``test_longmemeval_judge.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

from sme.eval.answer_generator import (
    DEFAULT_READER_MODEL,
    generate_answer,
)


def _fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                              total_tokens=15),
    )


class _FakeClient:
    """Minimal stand-in for openai.OpenAI()."""

    def __init__(self, response_or_exc):
        self._resp = response_or_exc
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, *, model, messages, temperature=0.0):
                outer.calls.append({"model": model, "messages": messages,
                                    "temperature": temperature})
                if isinstance(outer._resp, Exception):
                    raise outer._resp
                return outer._resp

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_default_reader_model_is_gpt_4_1_mini():
    assert DEFAULT_READER_MODEL == "gpt-4.1-mini"


def test_generate_answer_returns_stripped_content():
    client = _FakeClient(_fake_response("  You bought a kayak.  \n"))
    out = generate_answer(
        "What did I buy?",
        "User bought a kayak yesterday.",
        client=client,
    )
    assert out == "You bought a kayak."
    assert client.calls[0]["model"] == DEFAULT_READER_MODEL
    assert client.calls[0]["temperature"] == 0.0
    sent = client.calls[0]["messages"][0]["content"]
    # Prompt should contain both the question and the context.
    assert "What did I buy?" in sent
    assert "User bought a kayak yesterday." in sent


def test_generate_answer_honors_reader_model_override():
    client = _FakeClient(_fake_response("answer"))
    generate_answer(
        "q", "ctx",
        reader_model="gpt-4o-mini",
        client=client,
    )
    assert client.calls[0]["model"] == "gpt-4o-mini"


def test_generate_answer_returns_empty_string_when_no_client_and_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = generate_answer("q", "ctx", client=None)
    assert out == ""


def test_generate_answer_returns_empty_string_on_api_exception():
    client = _FakeClient(RuntimeError("network down"))
    out = generate_answer("q", "ctx", client=client)
    assert out == ""


def test_generate_answer_handles_blank_completion():
    """OpenAI sometimes returns content=None on safety blocks; tolerate it."""
    client = _FakeClient(SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0,
                              total_tokens=1),
    ))
    out = generate_answer("q", "ctx", client=client)
    assert out == ""


def test_generate_answer_truncates_context_when_max_chars_set():
    client = _FakeClient(_fake_response("ok"))
    big_context = "X" * 10000
    generate_answer(
        "q",
        big_context,
        client=client,
        max_context_chars=200,
    )
    sent = client.calls[0]["messages"][0]["content"]
    # The original 10k of X is no longer present; only 200 should be.
    assert "X" * 10000 not in sent
    # Truncated context is still in the prompt.
    assert "X" * 200 in sent


def test_generate_answer_no_truncation_by_default():
    client = _FakeClient(_fake_response("ok"))
    big_context = "X" * 10000
    generate_answer("q", big_context, client=client)
    sent = client.calls[0]["messages"][0]["content"]
    assert "X" * 10000 in sent
