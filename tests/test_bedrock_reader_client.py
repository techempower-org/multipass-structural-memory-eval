"""Tests for the Claude/Bedrock reader client shim — #116 Opus-4.8 reader arm.

The shim lets Claude (Bedrock) readers reuse the same injected-client seam as
Azure/OpenAI: ``client.chat.completions.create(...)`` → ``.choices[0].message.content``.
"""
from types import SimpleNamespace

from sme.eval.answer_generator import (
    _BedrockOpenAIShim, _is_claude_model, _CLAUDE_BEDROCK_IDS, generate_answer,
)


class _FakeBedrock:
    """Mimics anthropic.AnthropicBedrock: .messages.create(...) -> content blocks."""
    def __init__(self):
        self.last = None

        class _Messages:
            def create(inner, **kw):  # noqa: N805
                self.last = kw
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="PONG")]
                )
        self.messages = _Messages()


def test_is_claude_model():
    assert _is_claude_model("claude-opus-4-8")
    assert _is_claude_model("us.anthropic.claude-opus-4-8")
    assert _is_claude_model("anthropic.claude-sonnet-4-6")
    assert not _is_claude_model("o4-mini")
    assert not _is_claude_model("gpt-5.3-chat")


def test_shim_translates_openai_shape_to_bedrock():
    fake = _FakeBedrock()
    shim = _BedrockOpenAIShim(fake)
    resp = shim.chat.completions.create(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
    )
    # OpenAI-shaped response
    assert resp.choices[0].message.content == "PONG"
    # mapped the friendly id to the us. inference profile
    assert fake.last["model"] == _CLAUDE_BEDROCK_IDS["claude-opus-4-8"]
    assert fake.last["model"] == "us.anthropic.claude-opus-4-8"
    # max_tokens defaulted (anthropic requires it), temperature passed through
    assert fake.last["max_tokens"] == 1024
    assert fake.last["temperature"] == 0.0
    # single user turn, no system
    assert fake.last["messages"] == [{"role": "user", "content": "hi"}]
    assert "system" not in fake.last


def test_shim_splits_system_message():
    fake = _FakeBedrock()
    shim = _BedrockOpenAIShim(fake)
    shim.chat.completions.create(
        model="claude-opus-4-8",
        messages=[{"role": "system", "content": "be terse"},
                  {"role": "user", "content": "q"}],
    )
    assert fake.last["system"] == "be terse"
    assert fake.last["messages"] == [{"role": "user", "content": "q"}]


def test_passthrough_bedrock_id_unmapped():
    fake = _FakeBedrock()
    shim = _BedrockOpenAIShim(fake)
    shim.chat.completions.create(
        model="us.anthropic.claude-opus-4-8",
        messages=[{"role": "user", "content": "q"}],
    )
    assert fake.last["model"] == "us.anthropic.claude-opus-4-8"


def test_generate_answer_through_shim():
    """End-to-end: generate_answer with an injected Bedrock shim returns text."""
    shim = _BedrockOpenAIShim(_FakeBedrock())
    ans = generate_answer("What is the capital?", "Paris is the capital of France.",
                          reader_model="claude-opus-4-8", client=shim)
    assert ans == "PONG"


class _TemperatureRejectingBedrock:
    """Mimics a model (e.g. opus-4-8) that 400s when `temperature` is sent.

    Records every create() kwarg set so the test can assert which calls carried
    temperature and how many requests were made.
    """
    def __init__(self):
        self.calls: list[dict] = []

        class _Messages:
            def create(inner, **kw):  # noqa: N805
                self.calls.append(kw)
                if "temperature" in kw:
                    raise ValueError(
                        "400 Bad Request: temperature is not supported by this model"
                    )
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="OK")]
                )
        self.messages = _Messages()


def test_temperature_deprecation_cached_per_model():
    """First temperature call to a deprecating model pays the retry (2 requests);
    every later call skips temperature entirely (1 request each). The cache is
    keyed on the resolved Bedrock id, so a different model is unaffected."""
    import sme.eval.answer_generator as ag
    # Isolate the module-level cache for this test.
    ag._TEMPERATURE_DEPRECATED.clear()
    fake = _TemperatureRejectingBedrock()
    shim = _BedrockOpenAIShim(fake)
    bid = _CLAUDE_BEDROCK_IDS["claude-opus-4-8"]

    # First call: sends temperature, 400s, retries without it → 2 requests.
    r1 = shim.chat.completions.create(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "q"}],
        temperature=0.0,
    )
    assert r1.choices[0].message.content == "OK"
    assert len(fake.calls) == 2
    assert "temperature" in fake.calls[0]
    assert "temperature" not in fake.calls[1]
    assert bid in ag._TEMPERATURE_DEPRECATED

    # Second call: model is cached, so temperature is never sent → 1 request.
    fake.calls.clear()
    shim.chat.completions.create(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "q2"}],
        temperature=0.0,
    )
    assert len(fake.calls) == 1
    assert "temperature" not in fake.calls[0]

    ag._TEMPERATURE_DEPRECATED.clear()


def test_temperature_passed_through_when_model_accepts_it():
    """A model that accepts temperature is never added to the cache, and every
    call keeps sending temperature (single request, no retry)."""
    import sme.eval.answer_generator as ag
    ag._TEMPERATURE_DEPRECATED.clear()
    fake = _FakeBedrock()  # accepts temperature, returns PONG
    shim = _BedrockOpenAIShim(fake)
    shim.chat.completions.create(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "q"}],
        temperature=0.0,
    )
    assert fake.last["temperature"] == 0.0
    assert _CLAUDE_BEDROCK_IDS["claude-opus-4-8"] not in ag._TEMPERATURE_DEPRECATED


def test_judge_routes_claude_to_bedrock(monkeypatch):
    """#116 Opus-judge: _client_for_judge sends claude-* judges to the Bedrock
    shim and everything else to the Azure/OpenAI default. Monkeypatched so the
    test never needs live AWS/Azure creds (CI-safe)."""
    import sme.eval.answer_generator as ag
    from sme.eval import longmemeval_judge as lj
    sentinel_bedrock = object()
    sentinel_default = object()
    monkeypatch.setattr(ag, "_bedrock_client", lambda: sentinel_bedrock)
    monkeypatch.setattr(lj, "_default_client", lambda: sentinel_default)
    assert lj._client_for_judge("claude-opus-4-8") is sentinel_bedrock
    assert lj._client_for_judge("us.anthropic.claude-opus-4-8") is sentinel_bedrock
    assert lj._client_for_judge("gpt-5.3-chat") is sentinel_default
    assert lj._client_for_judge("o4-mini") is sentinel_default
