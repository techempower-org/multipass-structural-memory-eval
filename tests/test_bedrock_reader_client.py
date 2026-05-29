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
