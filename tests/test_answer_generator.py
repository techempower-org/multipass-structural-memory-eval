"""Tests for sme.eval.answer_generator.

The reader wraps an OpenAI-shaped client; tests mock that client so they
run without ``openai`` installed and without network access. Mirrors the
``_FakeClient`` shape used by ``test_longmemeval_judge.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import sys
from types import ModuleType

from sme.eval.answer_generator import (
    DEFAULT_READER_MODEL,
    _default_client,
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
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_API_BASE", raising=False)
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


# --- Azure OpenAI client path -----------------------------------------------

def _install_fake_openai_module(monkeypatch):
    """Inject a stub ``openai`` module exposing OpenAI / AzureOpenAI.

    Each returned class records its kwargs on a class attribute so the
    test can assert what _default_client passed. We install at the
    module level so ``from openai import AzureOpenAI`` inside
    ``_default_client`` resolves to our stub.
    """
    fake = ModuleType("openai")

    class _FakeOpenAI:
        last_kwargs: dict | None = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs

    class _FakeAzureOpenAI:
        last_kwargs: dict | None = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs

    fake.OpenAI = _FakeOpenAI
    fake.AzureOpenAI = _FakeAzureOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    return _FakeOpenAI, _FakeAzureOpenAI


def test_default_client_prefers_azure_when_both_env_vars_set(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.azure.com/")
    monkeypatch.setenv("AZURE_API_VERSION", "2099-12-01-preview")
    # Even if OPENAI_API_KEY is set, Azure should win.
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    _, FakeAzure = _install_fake_openai_module(monkeypatch)

    client = _default_client()

    assert isinstance(client, FakeAzure)
    assert FakeAzure.last_kwargs == {
        "azure_endpoint": "https://example.azure.com/",
        "api_key": "azure-secret",
        "api_version": "2099-12-01-preview",
    }


def test_default_client_azure_uses_default_api_version(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.azure.com/")
    monkeypatch.delenv("AZURE_API_VERSION", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _, FakeAzure = _install_fake_openai_module(monkeypatch)

    _default_client()

    assert FakeAzure.last_kwargs["api_version"] == "2024-12-01-preview"


def test_default_client_falls_back_to_openai_when_azure_partial(monkeypatch):
    # Only one Azure var set → fall through to OPENAI_API_KEY path.
    monkeypatch.setenv("AZURE_API_KEY", "azure-secret")
    monkeypatch.delenv("AZURE_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    FakeOpenAI, _ = _install_fake_openai_module(monkeypatch)

    client = _default_client()

    assert isinstance(client, FakeOpenAI)


def test_default_client_returns_none_when_neither_configured(monkeypatch):
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _default_client() is None
