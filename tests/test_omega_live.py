"""Live end-to-end smoke for the OMEGA adapter against the REAL package.

These tests drive ``omega-memory`` for real — ``store`` + ``query`` +
``get_graph_snapshot`` — on a throwaway ``OMEGA_HOME`` under ``tmp_path``.
They are the antidote to mock-only coverage: they prove the adapter
actually retrieves what it ingested through OMEGA's real SQLite + FTS5 /
sqlite-vec path, not just that our normaliser handles a fixture shape.

Skipped automatically when omega-memory isn't installed, so CI on a
minimal env stays green. Marked ``live`` so they can be deselected with
``-m "not live"`` when offline-only runs are wanted.

Isolation: the ``live_omega_home`` fixture sets OMEGA_HOME to a tmp dir
*before* the adapter imports omega, and the adapter restores the prior
value on close(). No test ever touches the user's real ~/.omega.
"""

from __future__ import annotations

import os

import pytest

omega = pytest.importorskip(
    "omega", reason="omega-memory not installed; live OMEGA smoke skipped"
)

pytestmark = pytest.mark.live


@pytest.fixture
def live_omega_home(monkeypatch, tmp_path):
    """Point OMEGA at an isolated tmp store for the duration of the test."""
    home = tmp_path / "omega_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OMEGA_HOME", str(home))
    return home


# A tiny LongMemEval-shaped corpus: a few facts, one of which a query
# should clearly surface.
_SMOKE_CORPUS = [
    {"content": "Maria adopted a golden retriever named Biscuit in March.",
     "type": "summary"},
    {"content": "The team standup moved to 9:30am on Tuesdays.",
     "type": "decision"},
    {"content": "Biscuit's vet appointment is scheduled for next Friday.",
     "type": "summary"},
    {"content": "JP prefers conventional commit messages.",
     "type": "user_preference"},
]


def test_live_ingest_query_and_snapshot(live_omega_home):
    """End-to-end against the real package in ONE adapter lifecycle:
    ingest a small corpus, retrieve through query_structured, and read
    the SQLite snapshot.

    These three assertions live in a single test (one store binding) on
    purpose. OMEGA caches a process-global SQLiteStore singleton and runs
    auto-relate on a daemon thread; re-binding it to a fresh OMEGA_HOME
    repeatedly within one process (which separate test functions would
    do) is inherently racy — a documented limitation, see
    docs/omega_adapter.md. The benchmark runner rebuilds an adapter per
    question but only calls query() (which reads through OMEGA's own live
    store), so it is unaffected; this test exercises the full surface
    once, deterministically.
    """
    from sme.adapters.omega import OmegaAdapter

    a = OmegaAdapter(omega_home=str(live_omega_home), n_results=5)
    try:
        report = a.ingest_corpus(_SMOKE_CORPUS)
        assert report["entities_created"] == 4, report
        assert report["errors"] == []

        # --- retrieval ---
        # NOTE: in this CI env OMEGA's ONNX embedding model isn't loaded,
        # so retrieval degrades to FTS5 keyword matching (OMEGA logs a
        # "hash-fallback" warning). A natural-language question whose
        # tokens don't overlap the stored text ("What pet does Maria
        # have?") returns nothing under keyword matching — that's an
        # OMEGA property, not an adapter bug. We query with terms that
        # appear in the corpus so the smoke proves the adapter's ingest →
        # query_structured → context_string path works end-to-end. (When
        # the ONNX model IS available, semantic retrieval handles the
        # paraphrased form too; the head-to-head benches run with it.)
        result = a.query("Maria golden retriever", n_results=5)
        assert result.error is None, result.error
        assert result.context_string, "empty context_string from live query"
        assert "Biscuit" in result.context_string, result.context_string
        assert result.retrieved_entities
        e0 = result.retrieved_entities[0]
        assert e0.id.startswith("omega:")
        assert e0.entity_type.startswith("memory:")

        # --- topology snapshot ---
        assert os.path.exists(a.db_path), a.db_path
        entities, edges = a.get_graph_snapshot()
        assert len(entities) == 4, [e.name for e in entities]
        # event_type must be carried through, not collapsed to "memory".
        types_seen = {e.entity_type for e in entities}
        assert "memory:user_preference" in types_seen, types_seen
        assert "memory:decision" in types_seen, types_seen
        # edges is a list (may be empty — OMEGA auto-relate is async); the
        # call must not raise on the real schema.
        assert isinstance(edges, list)
    finally:
        a.close()


def test_live_close_restores_prior_omega_home(monkeypatch, tmp_path):
    """close() must restore the OMEGA_HOME we found, so the live adapter
    never leaves the process pointed at the benchmark store."""
    from sme.adapters.omega import OmegaAdapter

    sentinel = str(tmp_path / "prior_home")
    monkeypatch.setenv("OMEGA_HOME", sentinel)
    a = OmegaAdapter(omega_home=str(tmp_path / "bench_home"))
    assert os.environ["OMEGA_HOME"] != sentinel
    a.close()
    assert os.environ["OMEGA_HOME"] == sentinel
