"""Live end-to-end smoke for the Hindsight adapter against a REAL server.

Drives a running Hindsight backend (Docker: ghcr.io/vectorize-io/hindsight)
through the adapter — ``ingest_corpus`` (retain + inline LLM fact extraction)
+ ``query`` (recall) — and asserts retrieval actually works, not just that the
normaliser handles a fixture shape. This is the antidote to mock-only coverage
and the guard against the "silently degraded" failure mode (#184): if the
backend's extractor is misconfigured, retain succeeds but recall returns
nothing — these assertions catch that.

Skipped automatically when:
  - hindsight-client isn't installed, or
  - no Hindsight server answers /health at HINDSIGHT_BASE_URL
    (default http://localhost:8888).

So CI on a minimal env stays green. Marked ``live`` for ``-m "not live"``.

Isolation: each test uses a unique ``bank_id`` so it never collides with
other banks in the same server, and never touches any production store.
"""
from __future__ import annotations

import os
import urllib.request
import uuid

import pytest

pytest.importorskip(
    "hindsight_client", reason="hindsight-client not installed; live smoke skipped"
)

_BASE_URL = os.environ.get("HINDSIGHT_BASE_URL", "http://localhost:8888")


def _server_up(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _server_up(_BASE_URL),
        reason=f"no Hindsight server at {_BASE_URL}; live smoke skipped",
    ),
]


# A tiny corpus with disjoint-vocab queries: the questions deliberately
# avoid the stored wording so a pass proves SEMANTIC retrieval over the
# extracted facts, not keyword overlap.
_SMOKE_CORPUS = [
    {"content": "Maria adopted a golden retriever named Biscuit in March 2023.",
     "document_id": "D1"},
    {"content": "The weekly team standup moved to 9:30am on Tuesdays.",
     "document_id": "D2"},
    {"content": "Biscuit's vet appointment is scheduled for next Friday.",
     "document_id": "D3"},
]


@pytest.fixture
def live_adapter():
    from sme.adapters.hindsight import HindsightAdapter

    a = HindsightAdapter(
        base_url=_BASE_URL,
        bank_id=f"smetest_{uuid.uuid4().hex[:8]}",
        n_results=5,
    )
    yield a
    a.close()


def test_live_ingest_and_paraphrase_recall(live_adapter):
    """Ingest a small corpus, then recall with a paraphrased query whose
    tokens don't overlap the stored text. A non-empty, on-topic result
    proves the real extract→embed→recall path works (not degraded)."""
    report = live_adapter.ingest_corpus(_SMOKE_CORPUS)
    assert report["errors"] == [], report
    assert report["entities_created"] == 3, report

    # "pet" never appears in the stored text ("golden retriever") — a hit
    # here is semantic, not keyword.
    res = live_adapter.query("What kind of pet does Maria have?", n_results=5)
    assert res.error is None, res.error
    assert res.context_string, "empty context — extractor likely degraded"
    assert "Biscuit" in res.context_string, res.context_string
    assert res.retrieved_entities


def test_live_document_id_maps_back_to_ingest_unit(live_adapter):
    """Recall hits must carry the document_id we supplied at retain time —
    that's the join key for session-level R@K."""
    live_adapter.ingest_corpus(_SMOKE_CORPUS)
    # disjoint-vocab temporal query → the vet-appointment fact (D3)
    res = live_adapter.query("When is the dog seeing the doctor?", n_results=5)
    assert res.error is None, res.error
    doc_ids = {
        e.properties.get("document_id")
        for e in (res.retrieved_entities or [])
        if e.properties.get("document_id")
    }
    # At least one retrieved fact must trace back to a known ingest unit.
    assert doc_ids, "no document_id on any recall hit — R@K mapping broken"
    assert doc_ids <= {"D1", "D2", "D3"}, doc_ids
