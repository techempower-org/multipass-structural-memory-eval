"""Tests for sme.eval.longmemeval_judge — canonical type-specific prompts.

The judge wraps an OpenAI-shaped client; we mock that client so the tests
work without ``openai`` installed and without a network call. The fakes match
the SDK's response object shape minimally: an object with
``.choices[0].message.content`` and ``.usage.{prompt,completion,total}_tokens``.

The canonical LongMemEval judge (xiaowu0162/LongMemEval evaluate_qa.py) is a
*binary* yes/no grader: the verdict is ``'yes' in reply.lower()``. There is no
PARTIAL class. These tests assert (1) the question_type -> template mapping
selects the canonical preference / abstention / temporal / knowledge-update
templates, and (2) the binary verdict maps onto SME's CORRECT / INCORRECT /
ABSTAIN label contract.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Optional

from sme.eval.longmemeval_judge import (
    DEFAULT_JUDGE_MODEL,
    _build_prompt,
    _default_client,
    _parse_judge_reply,
    _template_for_question_type,
    grade_answer,
)


def _fake_response(content: str,
                   prompt_tokens: int = 10,
                   completion_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class _FakeClient:
    """Minimal stand-in for openai.OpenAI(). Accepts the canonical max_tokens
    kwarg so the judge's tight completion cap is exercised in tests too."""

    def __init__(self, response_or_factory):
        self._resp = response_or_factory
        self.calls: list[dict] = []

        outer = self

        class _Completions:
            def create(self, *, model, messages, temperature=0.0,
                       max_tokens=None):
                outer.calls.append({"model": model, "messages": messages,
                                    "temperature": temperature,
                                    "max_tokens": max_tokens})
                resp = outer._resp
                if callable(resp):
                    return resp(model=model, messages=messages)
                if isinstance(resp, Exception):
                    raise resp
                return resp

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


# --- template selection (Selene deliverable #3) ----------------------------

def test_preference_selects_rubric_template():
    """single-session-preference must use the rubric-based template — the
    template the old generic judge was missing (collapsed preference to 0.0)."""
    tmpl = _template_for_question_type(
        "single-session-preference", is_abstention=False
    )
    assert "rubric for desired personalized response" in tmpl
    assert "does not need to reflect all the points in the rubric" in tmpl
    assert "Rubric: {}" in tmpl


def test_abstention_selects_unanswerable_template():
    """Abstention (any _abs question) uses the unanswerable-detection template,
    overriding the type-specific template."""
    tmpl = _template_for_question_type("abstention", is_abstention=False)
    assert "unanswerable question" in tmpl
    assert "correctly identify the question as unanswerable" in tmpl
    # The abstention flag overrides any type-specific template.
    forced = _template_for_question_type(
        "single-session-preference", is_abstention=True
    )
    assert forced == tmpl


def test_temporal_selects_offbyone_template():
    tmpl = _template_for_question_type("temporal-reasoning", is_abstention=False)
    assert "off-by-one errors" in tmpl


def test_knowledge_update_selects_update_template():
    tmpl = _template_for_question_type("knowledge-update", is_abstention=False)
    assert "previous information along with an updated answer" in tmpl


def test_base_template_shared_across_ie_and_multisession():
    """user / assistant / multi-session share one base correctness template."""
    base_u = _template_for_question_type("single-session-user", is_abstention=False)
    base_a = _template_for_question_type(
        "single-session-assistant", is_abstention=False
    )
    base_m = _template_for_question_type("multi-session", is_abstention=False)
    assert base_u == base_a == base_m
    # Base template is generic correctness, NOT rubric/unanswerable.
    assert "Correct Answer: {}" in base_u
    assert "Rubric" not in base_u
    assert "unanswerable" not in base_u


def test_unknown_type_falls_back_to_base():
    base = _template_for_question_type("single-session-user", is_abstention=False)
    assert _template_for_question_type("made-up-type", is_abstention=False) == base


def test_build_prompt_fills_slots_in_upstream_order():
    """Slot order is (question, correct-answer/rubric, response)."""
    prompt = _build_prompt(
        "single-session-user", "What did I buy?", "A kayak",
        "You bought a kayak.", is_abstention=False,
    )
    # Upstream fills .format(question, answer, response) in that order.
    assert "Question: What did I buy?" in prompt
    assert "Correct Answer: A kayak" in prompt
    assert "Model Response: You bought a kayak." in prompt


def test_preference_prompt_labels_gold_as_rubric():
    prompt = _build_prompt(
        "single-session-preference", "How should I be greeted?",
        "Greet warmly, mention the user is vegan.", "Hi! Here are vegan picks.",
        is_abstention=False,
    )
    assert "Rubric: Greet warmly, mention the user is vegan." in prompt


def test_abstention_prompt_labels_gold_as_explanation():
    prompt = _build_prompt(
        "abstention", "What is my dog's name?",
        "The user never mentioned a dog.", "I don't have that information.",
        is_abstention=True,
    )
    assert "Explanation: The user never mentioned a dog." in prompt


# --- binary verdict parsing ------------------------------------------------

def test_parse_judge_reply_yes_is_true():
    is_yes, rationale = _parse_judge_reply("Yes")
    assert is_yes is True
    assert rationale == "Yes"


def test_parse_judge_reply_no_is_false():
    is_yes, _ = _parse_judge_reply("no")
    assert is_yes is False


def test_parse_judge_reply_empty_is_false():
    is_yes, rationale = _parse_judge_reply("")
    assert is_yes is False
    assert "empty" in rationale.lower()


def test_parse_judge_reply_substring_yes():
    # Upstream uses substring containment, not exact match.
    is_yes, _ = _parse_judge_reply("Yes, the response is correct.")
    assert is_yes is True


# --- grade_answer end-to-end -----------------------------------------------

def test_grade_answer_yes_maps_to_correct():
    client = _FakeClient(_fake_response("yes"))
    out = grade_answer(
        question_type="single-session-user",
        question="What did I buy?",
        gold_answer="A kayak",
        hypothesis="You bought a kayak.",
        client=client,
    )
    assert out["autoeval_label"] == "CORRECT"
    assert out["judge_model"] == DEFAULT_JUDGE_MODEL
    assert out["usage"]["prompt_tokens"] == 10
    assert out["usage"]["total_tokens"] == 15
    # Canonical call shape: tight max_tokens cap on a non-reasoning judge.
    assert client.calls[0]["max_tokens"] == 10
    assert client.calls[0]["temperature"] == 0.0
    # The base correctness template was used.
    sent = client.calls[0]["messages"][0]["content"]
    assert "Is the model response correct? Answer yes or no only." in sent
    assert "A kayak" in sent


def test_grade_answer_no_maps_to_incorrect():
    client = _FakeClient(_fake_response("no"))
    out = grade_answer(
        question_type="single-session-user",
        question="What did I buy?",
        gold_answer="A kayak",
        hypothesis="A submarine",
        client=client,
    )
    assert out["autoeval_label"] == "INCORRECT"


def test_grade_answer_abstention_yes_maps_to_abstain():
    """A correct 'yes' on an abstention question (model identified it as
    unanswerable) maps to ABSTAIN — which the aggregator scores as right."""
    client = _FakeClient(_fake_response("yes"))
    out = grade_answer(
        question_type="abstention",
        question="What did I say about my submarine?",
        gold_answer="The user never mentioned a submarine.",
        hypothesis="I don't have that information.",
        client=client,
    )
    assert out["autoeval_label"] == "ABSTAIN"
    sent = client.calls[0]["messages"][0]["content"]
    assert "unanswerable" in sent


def test_grade_answer_abstention_no_maps_to_incorrect():
    """A 'no' on an abstention question (model fabricated an answer) is
    INCORRECT, not ABSTAIN."""
    client = _FakeClient(_fake_response("no"))
    out = grade_answer(
        question_type="abstention",
        question="What did I say about my submarine?",
        gold_answer="The user never mentioned a submarine.",
        hypothesis="You said it was yellow.",
        client=client,
    )
    assert out["autoeval_label"] == "INCORRECT"


def test_grade_answer_is_abstention_flag_forces_abstention_template():
    """Passing is_abstention=True forces the abstention template even when the
    question_type is a normal fact type (the reader-sweep passes the flag for
    _abs records that retain their original question_type)."""
    client = _FakeClient(_fake_response("yes"))
    out = grade_answer(
        question_type="single-session-user",
        question="What is my cat's name?",
        gold_answer="The user has no cat.",
        hypothesis="You don't have a cat.",
        client=client,
        is_abstention=True,
    )
    assert out["autoeval_label"] == "ABSTAIN"
    sent = client.calls[0]["messages"][0]["content"]
    assert "unanswerable" in sent


def test_grade_answer_preference_uses_rubric_template():
    client = _FakeClient(_fake_response("yes"))
    grade_answer(
        question_type="single-session-preference",
        question="How should responses address me?",
        gold_answer="Use a warm tone; the user is a vegan chef.",
        hypothesis="Here are some warm vegan suggestions, chef!",
        client=client,
    )
    sent = client.calls[0]["messages"][0]["content"]
    assert "rubric for desired personalized response" in sent
    assert "Rubric: Use a warm tone; the user is a vegan chef." in sent


def test_grade_answer_temporal_uses_temporal_template():
    client = _FakeClient(_fake_response("no"))
    out = grade_answer(
        question_type="temporal-reasoning",
        question="How many days between the events?",
        gold_answer="18 days",
        hypothesis="20 days",
        client=client,
    )
    assert out["autoeval_label"] == "INCORRECT"
    sent = client.calls[0]["messages"][0]["content"]
    assert "off-by-one errors" in sent


def test_grade_answer_knowledge_update_uses_ku_template():
    client = _FakeClient(_fake_response("yes"))
    grade_answer(
        question_type="knowledge-update",
        question="Where do I work?",
        gold_answer="Acme",
        hypothesis="You used to work at Beta, but now you work at Acme.",
        client=client,
    )
    sent = client.calls[0]["messages"][0]["content"]
    assert "previous information along with an updated answer" in sent


# --- API key handling -------------------------------------------------------

def test_grade_answer_no_api_key_returns_error_label(monkeypatch):
    """When client is None and no key is present, return ERROR with a clear
    rationale rather than raising."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_API_BASE", raising=False)
    out = grade_answer(
        question_type="single-session-user",
        question="q",
        gold_answer="g",
        hypothesis="h",
        client=None,
    )
    assert out["autoeval_label"] == "ERROR"
    assert "OPENAI_API_KEY" in out["rationale"]


# --- retries / transient errors --------------------------------------------

class _FlakyClient:
    """Fails N times, then succeeds with the given content."""

    def __init__(self, fail_n: int, then_content: str,
                 final_exc: Optional[Exception] = None):
        self.fail_n = fail_n
        self.then_content = then_content
        self.final_exc = final_exc
        self.attempts = 0
        outer = self

        class _Completions:
            def create(self, *, model, messages, temperature=0.0,
                       max_tokens=None):
                outer.attempts += 1
                if outer.attempts <= outer.fail_n:
                    raise RuntimeError(f"transient {outer.attempts}")
                if outer.final_exc is not None:
                    raise outer.final_exc
                return _fake_response(outer.then_content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_grade_answer_retries_then_succeeds(monkeypatch):
    # Make sleep instant so the test isn't slow.
    monkeypatch.setattr("sme.eval.longmemeval_judge.time.sleep", lambda *_: None)
    client = _FlakyClient(fail_n=2, then_content="yes")
    out = grade_answer(
        question_type="single-session-user",
        question="q", gold_answer="g", hypothesis="h",
        client=client,
    )
    assert out["autoeval_label"] == "CORRECT"
    assert client.attempts == 3


def test_grade_answer_returns_error_after_max_retries(monkeypatch):
    monkeypatch.setattr("sme.eval.longmemeval_judge.time.sleep", lambda *_: None)
    client = _FlakyClient(fail_n=99, then_content="never")
    out = grade_answer(
        question_type="single-session-user",
        question="q", gold_answer="g", hypothesis="h",
        client=client,
    )
    assert out["autoeval_label"] == "ERROR"
    assert "judge call failed" in out["rationale"]
    # Three retry attempts before bailing.
    assert client.attempts == 3


# --- Azure OpenAI client path -----------------------------------------------

def _install_fake_openai_module(monkeypatch):
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


def test_grade_answer_unknown_question_type_still_calls_judge():
    client = _FakeClient(_fake_response("yes"))
    out = grade_answer(
        question_type="made-up-type",
        question="q", gold_answer="g", hypothesis="h",
        client=client,
    )
    assert out["autoeval_label"] == "CORRECT"
    sent = client.calls[0]["messages"][0]["content"]
    # Falls back to the base correctness template.
    assert "Is the model response correct? Answer yes or no only." in sent
