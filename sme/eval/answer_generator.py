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
        client = _default_client()
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
