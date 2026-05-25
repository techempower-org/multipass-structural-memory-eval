"""Tests for sme.adapters.hindsight — HTTP-mocked, no live server, no SDK."""

from __future__ import annotations

import json
import urllib.error

from sme.adapters.hindsight import HindsightAdapter


# Ensure the SDK probe finds nothing in the test environment. (We never
# install hindsight-client in tests; the import inside _try_load_sdk
# returns None silently when the module isn't present.)


# --- construction --------------------------------------------------


def test_construction_default_base_url(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_BASE_URL", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    a = HindsightAdapter()
    assert a.base_url == "http://localhost:8888"
    assert a.bank_id == "sme"
    assert a.api_key is None


def test_construction_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_BASE_URL", "http://h.example:9000")
    monkeypatch.setenv("HINDSIGHT_API_KEY", "key-from-env")
    a = HindsightAdapter()
    assert a.base_url == "http://h.example:9000"
    assert a.api_key == "key-from-env"


def test_construction_explicit_kwargs_win(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_BASE_URL", "http://env")
    a = HindsightAdapter(base_url="http://explicit", api_key="xyz")
    assert a.base_url == "http://explicit"
    assert a.api_key == "xyz"


def test_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_BASE_URL", raising=False)
    a = HindsightAdapter(base_url="http://host/")
    assert a.base_url == "http://host"


# --- query() via recall ---------------------------------------------


_RECALL_OK = {
    "results": [
        {
            "id": "h1",
            "content": "first hindsight hit",
            "type": "World",
            "score": 0.92,
        },
        {
            "id": "h2",
            "content": "second hindsight hit",
            "type": "Experiences",
            "score": 0.81,
            "relationships": [
                {"source": "h2", "target": "h1", "type": "follows"},
            ],
        },
    ]
}


def _adapter(monkeypatch, **kwargs):
    monkeypatch.delenv("HINDSIGHT_BASE_URL", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    defaults = dict(base_url="http://h", bank_id="b1")
    defaults.update(kwargs)
    return HindsightAdapter(**defaults)


def test_query_success_builds_context(monkeypatch, fake_urlopen_factory):
    fake_urlopen_factory({"POST http://h/recall": _RECALL_OK})
    a = _adapter(monkeypatch)
    result = a.query("what happened")
    assert result.error is None
    assert "[1] [World] first hindsight hit" in result.context_string
    assert "[2] [Experiences] second hindsight hit" in result.context_string


def test_query_retrieved_entities_carry_score_and_bank(monkeypatch, fake_urlopen_factory):
    fake_urlopen_factory({"POST http://h/recall": _RECALL_OK})
    a = _adapter(monkeypatch)
    result = a.query("q")
    assert len(result.retrieved_entities) == 2
    e0 = result.retrieved_entities[0]
    assert e0.id == "hindsight:h1"
    assert e0.properties["score"] == 0.92
    assert e0.properties["bank_id"] == "b1"


def test_query_extracts_relationships_into_edges(monkeypatch, fake_urlopen_factory):
    fake_urlopen_factory({"POST http://h/recall": _RECALL_OK})
    a = _adapter(monkeypatch)
    result = a.query("q")
    assert len(result.retrieved_edges) == 1
    e = result.retrieved_edges[0]
    assert e.source_id == "hindsight:h2"
    assert e.target_id == "hindsight:h1"
    assert e.edge_type == "follows"


def test_query_uses_reflect_endpoint_when_configured(monkeypatch, fake_urlopen_factory):
    fake_urlopen_factory({
        "POST http://h/reflect": {"results": [{"id": "r1", "content": "reflected"}]},
    })
    a = _adapter(monkeypatch, use_reflect=True)
    result = a.query("deep question")
    assert "reflected" in result.context_string
    assert any("reflect" in step for step in result.retrieval_path)


def test_query_reflect_with_answer_only_envelope(monkeypatch, fake_urlopen_factory):
    fake_urlopen_factory({
        "POST http://h/reflect": {"answer": "Synthesized response."},
    })
    a = _adapter(monkeypatch, use_reflect=True)
    result = a.query("synth")
    assert result.error is None
    assert "Synthesized response." in result.context_string


def test_query_request_body_includes_bank_and_top_k(monkeypatch, fake_urlopen_factory):
    captured = {}

    def capture(req):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _RECALL_OK

    fake_urlopen_factory({"POST http://h/recall": capture})
    a = _adapter(monkeypatch, n_results=7)
    a.query("hello")
    assert captured["body"]["bank_id"] == "b1"
    assert captured["body"]["query"] == "hello"
    assert captured["body"]["top_k"] == 7


def test_query_api_key_sent_as_bearer(monkeypatch, fake_urlopen_factory):
    captured = {}

    def capture(req):
        captured["headers"] = dict(req.header_items())
        return _RECALL_OK

    fake_urlopen_factory({"POST http://h/recall": capture})
    a = _adapter(monkeypatch, api_key="hunter2")
    a.query("x")
    auth = captured["headers"].get("Authorization")
    assert auth == "Bearer hunter2"


def test_query_no_results(monkeypatch, fake_urlopen_factory):
    fake_urlopen_factory({"POST http://h/recall": {"results": []}})
    a = _adapter(monkeypatch)
    result = a.query("x")
    assert result.error == "NO_RESULTS"


def test_query_auth_failure(monkeypatch, fake_urlopen_factory):
    err = urllib.error.HTTPError("http://h/recall", 401, "Unauthorized", {}, None)
    fake_urlopen_factory({"POST http://h/recall": err})
    a = _adapter(monkeypatch)
    result = a.query("x")
    assert result.error.startswith("AUTH:")


def test_query_5xx(monkeypatch, fake_urlopen_factory):
    err = urllib.error.HTTPError("http://h/recall", 500, "boom", {}, None)
    fake_urlopen_factory({"POST http://h/recall": err})
    a = _adapter(monkeypatch)
    result = a.query("x")
    assert result.error.startswith("HTTP 500")


def test_query_connection_refused(monkeypatch, fake_urlopen_factory):
    err = urllib.error.URLError("refused")
    fake_urlopen_factory({"POST http://h/recall": err})
    a = _adapter(monkeypatch)
    result = a.query("x")
    assert result.error.startswith("CONNECTION:")


# --- ingest_corpus -------------------------------------------------


def test_ingest_posts_to_retain(monkeypatch, fake_urlopen_factory):
    captured = []

    def capture(req):
        captured.append(json.loads(req.data.decode("utf-8")))
        return {"ok": True}

    fake_urlopen_factory({"POST http://h/retain": capture})
    a = _adapter(monkeypatch)
    report = a.ingest_corpus([
        {"content": "Alice works at Google"},
        {"content": "Bob hikes", "context": "hobbies", "timestamp": "2026-01-01T00:00:00Z"},
    ])
    assert report["entities_created"] == 2
    assert captured[0]["bank_id"] == "b1"
    assert captured[0]["content"] == "Alice works at Google"
    assert captured[1]["context"] == "hobbies"
    assert captured[1]["timestamp"] == "2026-01-01T00:00:00Z"


def test_ingest_skips_empty_content(monkeypatch, fake_urlopen_factory):
    fake_urlopen_factory({"POST http://h/retain": {"ok": True}})
    a = _adapter(monkeypatch)
    report = a.ingest_corpus([{"content": ""}, {"text": "ok"}])
    assert report["entities_created"] == 1


def test_ingest_captures_failures(monkeypatch, fake_urlopen_factory):
    err = urllib.error.HTTPError("http://h/retain", 500, "x", {}, None)
    fake_urlopen_factory({"POST http://h/retain": err})
    a = _adapter(monkeypatch)
    report = a.ingest_corpus([{"content": "x"}])
    assert report["entities_created"] == 0
    assert len(report["errors"]) == 1


# --- get_graph_snapshot --------------------------------------------


def test_snapshot_returns_empty_on_404(monkeypatch, fake_urlopen_factory):
    err = urllib.error.HTTPError(
        "http://h/banks/b1/stats", 404, "Not Found", {}, None
    )
    fake_urlopen_factory({"GET http://h/banks/b1/stats": err})
    a = _adapter(monkeypatch)
    entities, edges = a.get_graph_snapshot()
    assert entities == []
    assert edges == []


def test_snapshot_projects_entities_and_relationships(monkeypatch, fake_urlopen_factory):
    stats = {
        "entities": [
            {"id": "alice", "name": "Alice", "type": "person"},
            {"id": "google", "name": "Google", "type": "company"},
        ],
        "relationships": [
            {"source": "alice", "target": "google", "type": "works_at"},
        ],
    }
    fake_urlopen_factory({"GET http://h/banks/b1/stats": stats})
    a = _adapter(monkeypatch)
    entities, edges = a.get_graph_snapshot()
    assert {e.id for e in entities} == {"hindsight:alice", "hindsight:google"}
    assert len(edges) == 1
    assert edges[0].source_id == "hindsight:alice"
    assert edges[0].edge_type == "works_at"


def test_snapshot_connection_refused_returns_empty(monkeypatch, fake_urlopen_factory):
    err = urllib.error.URLError("nope")
    fake_urlopen_factory({"GET http://h/banks/b1/stats": err})
    a = _adapter(monkeypatch)
    entities, edges = a.get_graph_snapshot()
    assert entities == []
    assert edges == []


# --- ontology ------------------------------------------------------


def test_ontology_lists_three_operations(monkeypatch):
    a = _adapter(monkeypatch)
    ont = a.get_ontology_source()
    ops = next(s for s in ont["schema"] if s["kind"] == "operations")
    assert set(ops["values"]) == {"retain", "recall", "reflect"}
    strategies = next(s for s in ont["schema"] if s["kind"] == "retrieval_strategies")
    assert "graph" in strategies["values"]
