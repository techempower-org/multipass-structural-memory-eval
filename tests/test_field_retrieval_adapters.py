"""Tests for the field retrieval-cluster adapters (ai-memory, agentmemory).

HTTP-mocked — no live daemon. Each test stubs the adapter's ``_http_post`` so
the request shapes and response parsing are exercised without a server.
"""

from __future__ import annotations

from sme.adapters.ai_memory import AiMemoryAdapter
from sme.adapters.agentmemory import AgentMemoryAdapter, _chunk


# --- ai-memory ------------------------------------------------------


def test_ai_memory_construction_defaults():
    a = AiMemoryAdapter()
    assert a.api_url == "http://127.0.0.1:9077"
    assert a.namespace == "sme_bench"
    assert a.tier == "mid"


def test_ai_memory_url_trailing_slash_stripped():
    a = AiMemoryAdapter(api_url="http://host:9077/")
    assert a.api_url == "http://host:9077"


def test_ai_memory_ingest_forgets_then_bulk_loads(monkeypatch):
    a = AiMemoryAdapter()
    calls: list[tuple[str, object]] = []

    def fake_post(url, payload):
        calls.append((url, payload))
        if url.endswith("/forget"):
            return {"deleted": 0}
        if url.endswith("/memories/bulk"):
            return {"created": len(payload), "errors": []}
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(a, "_http_post", fake_post)
    corpus = [
        {"id": "S0", "document": "alpha session", "session_id": "S0"},
        {"id": "S1", "document": "beta session", "session_id": "S1"},
    ]
    report = a.ingest_corpus(corpus)
    assert report["entities_created"] == 2
    # First call forgets the namespace, second bulk-loads.
    assert calls[0][0].endswith("/forget")
    assert calls[0][1] == {"namespace": "sme_bench"}
    assert calls[1][0].endswith("/memories/bulk")
    # session_id round-trips through metadata AND title.
    item0 = calls[1][1][0]
    assert item0["content"] == "alpha session"
    assert item0["metadata"]["session_id"] == "S0"
    assert item0["title"] == "S0"
    assert item0["namespace"] == "sme_bench"


def test_ai_memory_query_parses_recall_and_surfaces_session_id(monkeypatch):
    a = AiMemoryAdapter()

    def fake_post(url, payload):
        assert url.endswith("/recall")
        # query field is `context`, NOT `query`.
        assert payload["context"] == "where did I go?"
        assert payload["namespace"] == "sme_bench"
        return {
            "memories": [
                {"id": "m1", "content": "Paris trip", "score": 0.9,
                 "metadata": {"session_id": "S0"}, "tier": "mid",
                 "namespace": "sme_bench"},
                {"id": "m2", "content": "Lyon trip", "score": 0.7,
                 "title": "S1", "tier": "mid"},
            ],
            "count": 2,
        }

    monkeypatch.setattr(a, "_http_post", fake_post)
    res = a.query("where did I go?", n_results=5)
    assert res.error is None
    assert len(res.retrieved_entities) == 2
    # session_id from metadata, then title fallback.
    assert res.retrieved_entities[0].properties["session_id"] == "S0"
    assert res.retrieved_entities[1].properties["session_id"] == "S1"
    assert "Paris trip" in res.context_string


def test_ai_memory_query_no_results(monkeypatch):
    a = AiMemoryAdapter()
    monkeypatch.setattr(a, "_http_post", lambda url, payload: {"memories": []})
    res = a.query("anything")
    assert res.error == "NO_RESULTS"


def test_ai_memory_graph_snapshot_empty():
    a = AiMemoryAdapter()
    assert a.get_graph_snapshot() == ([], [])


# --- agentmemory ----------------------------------------------------


def test_agentmemory_construction_defaults():
    b = AgentMemoryAdapter()
    assert b.api_url == "http://127.0.0.1:3111"
    assert b.project.startswith("sme_bench_")


def test_chunk_splits_long_text_on_whitespace():
    chunks = _chunk("word " * 200, 380)  # ~1000 chars
    assert len(chunks) > 1
    assert all(len(c) <= 380 for c in chunks)
    # No mid-word split — chunks shouldn't end on a partial "wor".
    assert all(not c.endswith("wor") for c in chunks)


def test_chunk_short_text_single_chunk():
    assert _chunk("short", 380) == ["short"]
    assert _chunk("", 380) == []


def test_agentmemory_ingest_rotates_project_and_observes_chunks(monkeypatch):
    b = AgentMemoryAdapter()
    project_before = b.project
    calls: list[object] = []

    def fake_post(url, payload):
        assert url.endswith("/observe")
        calls.append(payload)
        return {"observationId": f"obs{len(calls)}"}

    monkeypatch.setattr(b, "_http_post", fake_post)
    # A long document chunks into multiple observations, all same sessionId.
    corpus = [{"id": "S0", "document": "lorem ipsum " * 80, "session_id": "S0"}]
    report = b.ingest_corpus(corpus)
    assert b.project != project_before  # rotated for isolation
    assert report["entities_created"] == len(calls) >= 2
    for p in calls:
        assert p["hookType"] == "prompt_submit"
        assert p["sessionId"] == "S0"
        assert p["project"] == b.project
        assert "prompt" in p["data"]


def test_agentmemory_query_uses_search_with_project_filter(monkeypatch):
    b = AgentMemoryAdapter()

    def fake_post(url, payload):
        # /search (not /smart-search) so the project filter isolates results.
        assert url.endswith("/agentmemory/search")
        assert payload["query"] == "trip?"
        assert payload["project"] == b.project
        assert payload["format"] == "compact"
        return {
            "format": "compact",
            "results": [
                {"obsId": "o1", "sessionId": "S0", "title": "observation",
                 "type": "synthetic", "score": 0.88},
                {"obsId": "o2", "sessionId": "S2", "title": "observation",
                 "type": "synthetic", "score": 0.5},
            ],
        }

    monkeypatch.setattr(b, "_http_post", fake_post)
    res = b.query("trip?", n_results=5)
    assert res.error is None
    assert [e.properties["session_id"] for e in res.retrieved_entities] == ["S0", "S2"]


def test_agentmemory_query_no_results(monkeypatch):
    b = AgentMemoryAdapter()
    monkeypatch.setattr(
        b, "_http_post", lambda url, payload: {"format": "compact", "results": []}
    )
    assert b.query("x").error == "NO_RESULTS"


def test_agentmemory_graph_snapshot_empty():
    b = AgentMemoryAdapter()
    assert b.get_graph_snapshot() == ([], [])
