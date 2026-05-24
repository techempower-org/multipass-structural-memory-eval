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
    """Lazily construct an OpenAI client, or None if unavailable.

    Treats both "package not installed" and "OPENAI_API_KEY not set" as
    None. Callers decide how to handle the absence — typically by
    returning the empty string so the judge can mark the answer wrong.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        log.info("answer_generator: openai SDK not installed")
        return None
    return OpenAI()


def generate_answer(
    question: str,
    context_string: str,
    *,
    reader_model: str = DEFAULT_READER_MODEL,
    client: Optional[Any] = None,
    max_context_chars: Optional[int] = None,
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

    prompt = READER_PROMPT_TEMPLATE.format(context=ctx, question=question)
    try:
        resp = client.chat.completions.create(
            model=reader_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 — degrade gracefully
        log.warning("answer_generator: reader call failed: %s", e)
        return ""
