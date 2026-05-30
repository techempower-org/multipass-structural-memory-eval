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


# --- local / ollama lane --------------------------------------------------


def test_is_local_model_detection():
    from sme.eval.answer_generator import _is_local_model
    # bare ollama tags + the explicit prefix are local
    assert _is_local_model("phi4")
    assert _is_local_model("qwen2.5:14b-instruct-q4_K_M")
    assert _is_local_model("qwen3.5:4b")
    assert _is_local_model("gemma4:e4b")
    assert _is_local_model("ollama/phi4")
    assert _is_local_model("llama3.1:8b")  # any ':' tag is the ollama convention
    # cloud ids (no ':') are NOT local
    assert not _is_local_model("gpt-4o")
    assert not _is_local_model("gpt-4o-2024-08-06")
    assert not _is_local_model("gpt-5.3-chat")
    assert not _is_local_model("o4-mini")
    assert not _is_local_model("claude-opus-4-8")
    assert not _is_local_model("us.anthropic.claude-opus-4-8")


def test_ollama_model_id_strips_prefix():
    from sme.eval.answer_generator import _ollama_model_id
    assert _ollama_model_id("ollama/phi4") == "phi4"
    assert _ollama_model_id("ollama/qwen2.5:14b") == "qwen2.5:14b"
    # no prefix -> unchanged
    assert _ollama_model_id("phi4") == "phi4"
    assert _ollama_model_id("qwen2.5:14b") == "qwen2.5:14b"


def test_reader_routes_local_to_ollama(monkeypatch):
    """_client_for_model sends ollama ids to the ollama client, claude to
    bedrock, and cloud ids to the default — monkeypatched, no live servers."""
    import sme.eval.answer_generator as ag
    ag._provider_cache.clear()
    sentinel_ollama = object()
    sentinel_bedrock = object()
    sentinel_default = object()
    monkeypatch.setattr(ag, "_ollama_client", lambda: sentinel_ollama)
    monkeypatch.setattr(ag, "_bedrock_client", lambda: sentinel_bedrock)
    monkeypatch.setattr(ag, "_default_client", lambda: sentinel_default)
    assert ag._client_for_model("phi4") is sentinel_ollama
    assert ag._client_for_model("qwen2.5:14b-instruct-q4_K_M") is sentinel_ollama
    assert ag._client_for_model("ollama/gemma4:e4b") is sentinel_ollama
    assert ag._client_for_model("claude-opus-4-8") is sentinel_bedrock
    assert ag._client_for_model("gpt-4o") is sentinel_default
    ag._provider_cache.clear()


def test_judge_routes_local_to_ollama(monkeypatch):
    import sme.eval.answer_generator as ag
    from sme.eval import longmemeval_judge as lj
    sentinel_ollama = object()
    sentinel_default = object()
    monkeypatch.setattr(ag, "_ollama_client", lambda: sentinel_ollama)
    monkeypatch.setattr(lj, "_default_client", lambda: sentinel_default)
    assert lj._client_for_judge("phi4") is sentinel_ollama
    assert lj._client_for_judge("qwen3.5:4b") is sentinel_ollama
    assert lj._client_for_judge("ollama/phi4") is sentinel_ollama
    assert lj._client_for_judge("gpt-4o-2024-08-06") is sentinel_default


def test_ollama_client_uses_env_base_url(monkeypatch):
    """_ollama_client points an OpenAI client at OLLAMA_BASE_URL (default
    localhost:11434/v1) with a placeholder key. We fake the OpenAI ctor so the
    test asserts the wiring without importing the real SDK behaviour."""
    import sme.eval.answer_generator as ag
    captured = {}

    class _FakeOpenAIModule:
        @staticmethod
        def OpenAI(*, base_url, api_key):  # noqa: N802
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            return ("ollama-client", base_url)

    monkeypatch.setitem(__import__("sys").modules, "openai", _FakeOpenAIModule)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://katana.local:11434/v1")
    client = ag._ollama_client()
    assert client == ("ollama-client", "http://katana.local:11434/v1")
    assert captured["base_url"] == "http://katana.local:11434/v1"
    assert captured["api_key"] == "ollama"


def test_generate_answer_routes_local_model_to_ollama_with_bare_id():
    """End-to-end (no monkeypatch of the client — we INJECT a fake client to
    prove generate_answer strips the ``ollama/`` prefix before the call so the
    ollama server receives the bare tag, and returns the answer text."""
    from sme.eval.answer_generator import generate_answer

    class _FakeOpenAILike:
        def __init__(self):
            self.last_model = None
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create))

        def _create(self, *, model, messages, **kw):
            self.last_model = model
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Paris"))]
            )

    fake = _FakeOpenAILike()
    ans = generate_answer(
        "What is the capital of France?",
        "Paris is the capital of France.",
        reader_model="ollama/phi4", client=fake,
    )
    assert ans == "Paris"
    # the ``ollama/`` prefix was stripped before hitting the client
    assert fake.last_model == "phi4"
