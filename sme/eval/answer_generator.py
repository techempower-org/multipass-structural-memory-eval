"""Reader-pass answer generation for the LongMemEval cross-validation harness.

Wraps a single LLM call that turns retrieved context + a question into an
answer string the LongMemEval judge can score. Extracted from the
cross_validation harness so it's importable from the CLI subcommand and
tests can mock the client without spawning the harness module.

Design notes:

- The default reader is ``gpt-4.1-mini`` per issue #17 — cheaper than the
  judge model, but recent enough to handle the multi-session synthesis
  task. The harness still defaults to ``gpt-4o-mini`` for back-compat with
  the cross-validation script's prior contract; the new CLI defaults to
  ``gpt-4.1-mini``.
- All failure modes return the empty string. The downstream judge will
  then label that turn INCORRECT, which is the right signal — we couldn't
  produce an answer.
- The client interface is the OpenAI-SDK ``chat.completions.create``
  shape so tests can inject a ``SimpleNamespace`` stub identical to
  ``test_longmemeval_judge.py``'s ``_FakeClient``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)

DEFAULT_READER_MODEL = "gpt-4.1-mini"

READER_PROMPT_TEMPLATE = (
    "Answer the user's question using only the conversation history "
    "below. If the answer is not present, say 'I don't know.'\n\n"
    "Conversation history:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)


def _default_client() -> Optional[Any]:
    """Lazily construct an OpenAI or AzureOpenAI client, or None if unavailable.

    Prefers Azure when ``AZURE_API_KEY`` and ``AZURE_API_BASE`` are both
    set (JP's homelab uses Azure-deployed gpt-4o-mini / gpt-4o); falls
    back to ``OPENAI_API_KEY`` → vanilla ``OpenAI()`` for upstream users.
    Returns ``None`` if neither path is configured or the SDK isn't
    installed — callers degrade gracefully from there.
    """
    azure_key = os.environ.get("AZURE_API_KEY")
    azure_base = os.environ.get("AZURE_API_BASE")
    if azure_key and azure_base:
        try:
            from openai import AzureOpenAI  # type: ignore[import-not-found]
        except ImportError:
            log.info("answer_generator: openai SDK not installed")
            return None
        api_version = os.environ.get(
            "AZURE_API_VERSION", "2024-12-01-preview"
        )
        return AzureOpenAI(
            azure_endpoint=azure_base,
            api_key=azure_key,
            api_version=api_version,
        )
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        log.info("answer_generator: openai SDK not installed")
        return None
    return OpenAI()


# --- Claude / Bedrock reader support (#116 Opus-4.8 reader arm) -------------
# Lazy + optional, exactly like the openai import above: boto3 + the Bedrock
# client are imported only when a Claude reader is actually requested, so the
# core stays lightweight (constitutional principle — no hard boto3 dep).
#
# Friendly reader-model id -> Bedrock inference-profile id. On-demand throughput
# for opus-4.x is NOT served by the bare ``anthropic.claude-opus-4-8`` model id;
# it requires the cross-region ``us.`` inference profile.
_CLAUDE_BEDROCK_IDS = {
    "claude-opus-4-8": "us.anthropic.claude-opus-4-8",
    "claude-opus-4-7": "us.anthropic.claude-opus-4-7",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5",
}


def _is_claude_model(model: str) -> bool:
    return model.startswith(
        ("claude", "anthropic.", "us.anthropic.", "global.anthropic.")
    )


# Models (keyed by resolved Bedrock id) known to reject `temperature` with a
# 400. Newer models (e.g. opus-4-8) deprecate the param; once a model 400s on
# it we record it here and stop sending temperature on subsequent calls, so
# only the FIRST call to such a model pays the retry — not every call.
_TEMPERATURE_DEPRECATED: set[str] = set()


class _BedrockOpenAIShim:
    """Expose an ``AnthropicBedrock`` client with the OpenAI
    ``client.chat.completions.create(...)`` shape that ``generate_answer`` and
    the LongMemEval judge already speak — so Claude readers drop into the same
    injected-client seam as Azure/OpenAI with zero call-site changes."""

    def __init__(self, bedrock: Any):
        self._b = bedrock
        self.chat = self  # so client.chat.completions.create works

    @property
    def completions(self):
        return self

    def create(self, *, model: str, messages: list, temperature: Optional[float] = None,
               max_tokens: int = 1024, **_ignored: Any):
        from types import SimpleNamespace
        bid = _CLAUDE_BEDROCK_IDS.get(model, model)
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        conv = [{"role": m["role"], "content": m["content"]}
                for m in messages if m.get("role") != "system"]
        kw: dict[str, Any] = dict(model=bid, max_tokens=max_tokens, messages=conv)
        if system:
            kw["system"] = system
        # Only send temperature if this model hasn't already 400'd on it. Once a
        # model is in the deprecation cache every later call sends one request.
        if temperature is not None and bid not in _TEMPERATURE_DEPRECATED:
            kw["temperature"] = temperature
        try:
            resp = self._b.messages.create(**kw)
        except Exception as e:  # noqa: BLE001
            # Newer models (e.g. opus-4-8) deprecate `temperature`; retry once
            # without it and cache the deprecation so the retry is paid once
            # per model, not on every call.
            if "temperature" in str(e) and "temperature" in kw:
                _TEMPERATURE_DEPRECATED.add(bid)
                kw.pop("temperature")
                resp = self._b.messages.create(**kw)
            else:
                raise
        text = "".join(getattr(b, "text", "") for b in resp.content
                       if getattr(b, "type", None) == "text")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )


_provider_cache: dict[str, Any] = {}


def _bedrock_client() -> Optional[Any]:
    try:
        from anthropic import AnthropicBedrock  # type: ignore[import-not-found]
    except ImportError:
        log.info("answer_generator: anthropic SDK not installed — Bedrock reader unavailable")
        return None
    region = (os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION")
              or os.environ.get("AWS_DEFAULT_REGION") or "us-west-1")
    try:
        return _BedrockOpenAIShim(AnthropicBedrock(aws_region=region))
    except Exception as e:  # noqa: BLE001
        log.warning("answer_generator: AnthropicBedrock init failed: %s", e)
        return None


def _client_for_model(model: str) -> Optional[Any]:
    """Pick a reader client by model id: Claude/Bedrock models route to the
    AnthropicBedrock shim, everything else to the Azure/OpenAI default. Cached
    per provider so a mixed sweep builds each client at most once."""
    key = "bedrock" if _is_claude_model(model) else "default"
    if key not in _provider_cache:
        _provider_cache[key] = _bedrock_client() if key == "bedrock" else _default_client()
    return _provider_cache[key]


_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


def generate_answer(
    question: str,
    context_string: str,
    *,
    reader_model: str = DEFAULT_READER_MODEL,
    client: Optional[Any] = None,
    max_context_chars: Optional[int] = None,
    prompt_template: Optional[str] = None,
) -> str:
    """Generate an answer to ``question`` using ``context_string`` as evidence.

    Args:
        question: Natural-language question.
        context_string: Retrieved evidence text (typically an adapter's
            ``QueryResult.context_string``).
        reader_model: Model id to call. Defaults to ``gpt-4.1-mini``.
        client: An OpenAI-SDK-shaped client. When ``None``, one is
            constructed lazily from ``OPENAI_API_KEY``. Tests pass a fake.
        max_context_chars: Optional cap on the size of ``context_string``
            inserted into the prompt. Useful when an adapter returns a
            very large context that would otherwise exceed the reader's
            input window. ``None`` (default) keeps the full context.
        prompt_template: Optional reader-prompt template with ``{context}``
            and ``{question}`` fields. Defaults to ``READER_PROMPT_TEMPLATE``.
            The #116 reader sweep varies this to compare extraction prompts.

    Returns:
        Stripped answer string. Empty string on any failure (missing
        client, API error, blank completion).
    """
    if client is None:
        client = _client_for_model(reader_model)
    if client is None:
        return ""

    ctx = context_string or ""
    if max_context_chars is not None and len(ctx) > max_context_chars:
        ctx = ctx[:max_context_chars]

    template = prompt_template or READER_PROMPT_TEMPLATE
    prompt = template.format(context=ctx, question=question)
    try:
        kwargs: dict[str, Any] = dict(
            model=reader_model,
            messages=[{"role": "user", "content": prompt}],
        )
        if not _is_reasoning_model(reader_model):
            kwargs["temperature"] = 0.0
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 — degrade gracefully
        log.warning("answer_generator: reader call failed: %s", e)
        return ""
