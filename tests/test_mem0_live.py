"""Live end-to-end smoke for the Mem0 adapter against the REAL mem0ai lib.

Drives ``mem0ai`` for real — ``add`` + ``search`` — on a throwaway on-disk
Qdrant under ``tmp_path``, with extraction + embeddings wired to **local
ollama** (phi4 + nomic-embed-text). NO cloud: ``OPENAI_API_KEY`` is cleared
for the duration so a misconfigured stack fails loudly rather than silently
billing OpenAI.

Auto-skips when:
  - ``mem0ai`` isn't installed (minimal CI env), or
  - no ollama daemon answers at ``OLLAMA_BASE_URL`` (default localhost:11434),
    or the required models aren't pulled.

Marked ``live`` so it deselects with ``-m "not live"``.

This is the antidote to mock-only coverage and the analogue of
``tests/test_omega_live.py`` / ``tests/test_hindsight_live.py``: it proves the
adapter actually stores + recalls through mem0's real extraction path, and —
critically — that a **paraphrased** query (disjoint vocabulary) returns the
right memory, which only passes when the local extractor + embedder genuinely
work (the qwen-markdown silent-degradation trap from the Hindsight finding
would fail this).
"""

from __future__ import annotations

import os
import tempfile

import pytest

mem0 = pytest.importorskip(
    "mem0", reason="mem0ai not installed; live Mem0 smoke skipped"
)

pytestmark = pytest.mark.live

_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_LLM_MODEL = os.environ.get("MEM0_LIVE_LLM", "phi4:latest")
_EMBED_MODEL = os.environ.get("MEM0_LIVE_EMBED", "nomic-embed-text:v1.5")


def _ollama_ready() -> bool:
    """True iff an ollama daemon is reachable and has both required models."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=3) as r:
            import json

            tags = json.loads(r.read().decode("utf-8"))
        names = {m.get("name", "") for m in tags.get("models", [])}
        return _LLM_MODEL in names and _EMBED_MODEL in names
    except Exception:  # noqa: BLE001 — any failure = not ready, skip
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _ollama_ready(),
        reason=(
            f"ollama not ready at {_OLLAMA_URL} with {_LLM_MODEL} + "
            f"{_EMBED_MODEL}; live Mem0 smoke skipped"
        ),
    ),
]


def _local_config(store_path: str) -> dict:
    """All-local mem0 config: ollama LLM + ollama embedder + on-disk Qdrant."""
    return {
        "llm": {
            "provider": "ollama",
            "config": {"model": _LLM_MODEL, "ollama_base_url": _OLLAMA_URL},
        },
        "embedder": {
            "provider": "ollama",
            "config": {"model": _EMBED_MODEL, "ollama_base_url": _OLLAMA_URL},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": store_path,
                "collection_name": "sme_live_smoke",
                "embedding_model_dims": 768,  # nomic-embed-text dim
            },
        },
    }


def test_live_ingest_and_paraphrase_retrieve(monkeypatch):
    """Ingest a tiny corpus through mem0's real extraction path and prove a
    paraphrased query (zero token overlap) surfaces the right memory — the
    extractor-health gate. Single mem0 instance (on-disk Qdrant locks
    ~/.mem0/migrations_qdrant to one process, so no concurrent instances)."""
    # Hard-guarantee no cloud spend: a missing key makes the OpenAI default
    # fail loudly; we override with ollama anyway, but clear it for safety.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from sme.adapters.mem0 import Mem0Adapter

    with tempfile.TemporaryDirectory(prefix="mem0_live_") as store:
        a = Mem0Adapter(config=_local_config(store), user_id="live", n_results=5)
        try:
            report = a.ingest_corpus(
                [
                    {"content": "Maria adopted a golden retriever named "
                                "Biscuit in March."},
                    {"content": "The team standup moved to 9:30am on "
                                "Tuesdays."},
                ]
            )
            # mem0's add() reasons about store/update/skip, so entities_created
            # counts add() calls that didn't raise, not stored-memory count.
            assert report["errors"] == [], report
            assert report["entities_created"] >= 1

            # Paraphrase — disjoint vocabulary from "golden retriever Biscuit".
            result = a.query("What pet does Maria have?", n_results=5)
            assert result.error is None, result.error
            assert result.context_string, "empty context from live mem0 query"
            assert "Biscuit" in result.context_string, result.context_string
            assert result.retrieved_entities
            e0 = result.retrieved_entities[0]
            assert e0.id.startswith("mem0:")
            assert e0.entity_type.startswith("memory:")
        finally:
            a.close()
