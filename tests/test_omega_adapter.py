"""Tests for sme.adapters.omega.

Two layers:
  * Mocked tests inject a fake ``omega`` module so the adapter's
    normalisation logic is covered without omega-memory installed.
  * Live tests (``test_omega_live.py``) drive the real package on an
    isolated ``OMEGA_HOME`` and are skipped when omega-memory is absent.

The fake module mirrors the *verified* omega-memory 1.4.x surface:
``store(content, event_type, ...)`` returns a status string, and
``query_structured(text, limit=...)`` returns a list[dict] with
``id`` / ``content`` / ``event_type`` / ``relevance`` fields.
``query()`` returns a human-readable string (the adapter only falls
back to it when ``query_structured`` is absent).
"""

from __future__ import annotations

import os
import sqlite3
import sys
import types

import pytest


def _install_fake_omega(
    monkeypatch,
    store_fn=None,
    query_structured_fn=None,
    query_fn=None,
    with_query_structured=True,
):
    """Inject a fake ``omega`` module into sys.modules so the adapter's
    import succeeds without omega-memory installed."""
    fake = types.ModuleType("omega")

    def default_store(content, event_type="memory", **kwargs):
        default_store.calls.append((content, event_type))
        return f"Stored mem-fake ({event_type})"

    default_store.calls = []

    def default_query_structured(text, limit=10, **kwargs):
        default_query_structured.calls.append((text, limit))
        return [
            {"id": "m1", "content": "first omega memory",
             "event_type": "decision", "relevance": 0.9},
            {"id": "m2", "content": "second omega memory",
             "event_type": "lesson", "relevance": 0.7},
        ]

    default_query_structured.calls = []

    def default_query(text, **kwargs):
        default_query.calls.append(text)
        return (
            "Results: 1\n"
            "## 1. [decision] `mem-strfallback` (str: 0.88)\n"
            "string fallback content\n"
            "*2026-05-30T06:33*\n"
        )

    default_query.calls = []

    fake.store = store_fn or default_store
    if with_query_structured:
        fake.query_structured = query_structured_fn or default_query_structured
    fake.query = query_fn or default_query
    fake.remember = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "omega", fake)
    return fake


@pytest.fixture
def isolate_omega_home(monkeypatch, tmp_path):
    """Keep every OMEGA_HOME mutation inside the test's tmp dir.

    The adapter sets os.environ["OMEGA_HOME"] in __init__; monkeypatch's
    setenv baseline ensures the process env is restored after the test so
    nothing leaks into a real ~/.omega.
    """
    monkeypatch.setenv("OMEGA_HOME", str(tmp_path / "_baseline_home"))
    return tmp_path


@pytest.fixture
def fake_omega(monkeypatch, isolate_omega_home):
    return _install_fake_omega(monkeypatch)


# --- construction ---------------------------------------------------


def test_construction_requires_omega_package(monkeypatch, isolate_omega_home):
    monkeypatch.delitem(sys.modules, "omega", raising=False)

    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def faux_import(name, *a, **kw):
        if name == "omega":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", faux_import)

    from sme.adapters.omega import OmegaAdapter
    with pytest.raises(ImportError, match="omega-memory"):
        OmegaAdapter(omega_home=str(isolate_omega_home / "h"))


def test_construction_succeeds_with_omega_present(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    assert a.default_memory_type == "summary"
    assert a.n_results == 10
    a.close()


def test_construction_sets_and_restores_omega_home(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    home = str(isolate_omega_home / "isolated")
    prior = os.environ.get("OMEGA_HOME")
    a = OmegaAdapter(omega_home=home)
    # While the adapter is live, OMEGA_HOME points at our isolated dir.
    assert os.path.realpath(os.environ["OMEGA_HOME"]) == os.path.realpath(home)
    a.close()
    # After close, the prior value is restored.
    assert os.environ.get("OMEGA_HOME") == prior


def test_db_path_parent_becomes_home(fake_omega, isolate_omega_home):
    """A db_path (CLI parity) resolves its PARENT dir as OMEGA_HOME and
    always names the file omega.db."""
    from sme.adapters.omega import OmegaAdapter
    db = isolate_omega_home / "custom_dir" / "whatever.db"
    a = OmegaAdapter(db_path=str(db))
    assert a.omega_home == str((isolate_omega_home / "custom_dir").resolve())
    assert a.db_path.endswith("omega.db")
    a.close()


# --- ingest_corpus -------------------------------------------------


def test_ingest_calls_omega_store_per_row(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    report = a.ingest_corpus([
        {"content": "fact one", "type": "lesson"},
        {"content": "fact two"},  # uses default_memory_type
    ])
    assert report["entities_created"] == 2
    assert fake_omega.store.calls == [
        ("fact one", "lesson"),
        ("fact two", "summary"),
    ]
    a.close()


def test_ingest_accepts_event_type_field(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    a.ingest_corpus([{"content": "x", "event_type": "error"}])
    assert fake_omega.store.calls == [("x", "error")]
    a.close()


def test_ingest_skips_empty_content(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    report = a.ingest_corpus([{"content": ""}, {"content": "ok"}])
    assert report["entities_created"] == 1
    assert any("empty content" in w for w in report["warnings"])
    a.close()


def test_ingest_captures_store_failures(monkeypatch, isolate_omega_home):
    def boom(content, event_type="memory", **kwargs):
        raise RuntimeError("write failed")
    _install_fake_omega(monkeypatch, store_fn=boom)
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    report = a.ingest_corpus([{"content": "x"}])
    assert report["entities_created"] == 0
    assert len(report["errors"]) == 1
    assert "write failed" in report["errors"][0]
    a.close()


# --- query ----------------------------------------------------------


def test_query_uses_query_structured(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    result = a.query("anything")
    assert result.error is None
    assert "[1] [decision] first omega memory" in result.context_string
    assert "[2] [lesson] second omega memory" in result.context_string
    # query_structured got called, not the string query()
    assert fake_omega.query_structured.calls
    assert "omega_query_structured" in result.retrieval_path[0]
    a.close()


def test_query_retrieved_entities_carry_type_and_score(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    result = a.query("anything")
    assert len(result.retrieved_entities) == 2
    e0 = result.retrieved_entities[0]
    assert e0.id == "omega:m1"
    assert e0.entity_type == "memory:decision"
    assert e0.properties["score"] == 0.9
    a.close()


def test_query_falls_back_to_string_when_no_query_structured(monkeypatch, isolate_omega_home):
    """Older OMEGA without query_structured: parse the string form."""
    _install_fake_omega(monkeypatch, with_query_structured=False)
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    result = a.query("anything")
    assert result.error is None
    assert "string fallback content" in result.context_string
    assert "[decision]" in result.context_string
    a.close()


def test_query_no_results(monkeypatch, isolate_omega_home):
    _install_fake_omega(monkeypatch, query_structured_fn=lambda t, limit=10, **kw: [])
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    result = a.query("nothing")
    assert result.error == "NO_RESULTS"
    a.close()


def test_query_normalises_dict_envelope(monkeypatch, isolate_omega_home):
    _install_fake_omega(
        monkeypatch,
        query_structured_fn=lambda t, limit=10, **kw: {
            "results": [{"id": "r1", "content": "hi"}]
        },
    )
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    result = a.query("x")
    assert "hi" in result.context_string
    a.close()


def test_query_normalises_list_of_strings(monkeypatch, isolate_omega_home):
    _install_fake_omega(
        monkeypatch,
        query_structured_fn=lambda t, limit=10, **kw: ["alpha", "beta"],
    )
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    result = a.query("x")
    assert "alpha" in result.context_string
    assert "beta" in result.context_string
    a.close()


def test_query_n_results_truncates(monkeypatch, isolate_omega_home):
    _install_fake_omega(
        monkeypatch,
        query_structured_fn=lambda t, limit=10, **kw: [
            {"id": f"m{i}", "content": f"c{i}"} for i in range(20)
        ],
    )
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"), n_results=3)
    result = a.query("x")
    assert len(result.retrieved_entities) == 3
    a.close()


def test_query_internal_error_captured(monkeypatch, isolate_omega_home):
    def boom(t, limit=10, **kw):
        raise RuntimeError("query crash")
    _install_fake_omega(monkeypatch, query_structured_fn=boom)
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    result = a.query("x")
    assert result.error.startswith("INTERNAL:")
    assert "query crash" in result.error
    a.close()


# --- get_graph_snapshot --------------------------------------------


def _seed_omega_db(path):
    """Build a fake OMEGA SQLite database matching the real 1.4.x schema:
    memories.event_type and edges.edge_type."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, event_type TEXT)"
    )
    conn.execute(
        "CREATE TABLE edges (id INTEGER, source_id TEXT, target_id TEXT, edge_type TEXT)"
    )
    conn.executemany(
        "INSERT INTO memories VALUES (?,?,?)",
        [
            ("m1", "first", "decision"),
            ("m2", "second", "lesson"),
            ("m3", "third", "user_preference"),
        ],
    )
    conn.executemany(
        "INSERT INTO edges VALUES (?,?,?,?)",
        [
            (1, "m1", "m2", "related"),
            (2, "m2", "m3", "supersedes"),
        ],
    )
    conn.commit()
    conn.close()


def test_snapshot_reads_memories_and_edges(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    home = isolate_omega_home / "snap"
    home.mkdir(parents=True, exist_ok=True)
    _seed_omega_db(home / "omega.db")
    a = OmegaAdapter(omega_home=str(home))
    entities, edges = a.get_graph_snapshot()
    assert {e.id for e in entities} == {"omega:m1", "omega:m2", "omega:m3"}
    assert {e.entity_type for e in entities} == {
        "memory:decision",
        "memory:lesson",
        "memory:user_preference",
    }
    edge_pairs = {(e.source_id, e.target_id, e.edge_type) for e in edges}
    assert ("omega:m1", "omega:m2", "related") in edge_pairs
    assert ("omega:m2", "omega:m3", "supersedes") in edge_pairs
    a.close()


def test_snapshot_missing_db_returns_empty(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "missing_home"))
    entities, edges = a.get_graph_snapshot()
    assert entities == []
    assert edges == []
    a.close()


def test_snapshot_schema_drift_no_edges_table(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    home = isolate_omega_home / "drift"
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(home / "omega.db"))
    conn.execute(
        "CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, event_type TEXT)"
    )
    conn.execute("INSERT INTO memories VALUES ('only', 'lone', 'summary')")
    conn.commit()
    conn.close()
    a = OmegaAdapter(omega_home=str(home))
    entities, edges = a.get_graph_snapshot()
    assert len(entities) == 1
    assert edges == []  # no edges table is graceful, not fatal
    a.close()


def test_snapshot_alternate_table_names(fake_omega, isolate_omega_home):
    """Older/newer OMEGA may use 'memory' table singular with a 'kind' col."""
    from sme.adapters.omega import OmegaAdapter
    home = isolate_omega_home / "alt"
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(home / "omega.db"))
    conn.execute("CREATE TABLE memory (id TEXT, text TEXT, kind TEXT)")
    conn.execute("INSERT INTO memory VALUES ('z', 'zed', 'lesson')")
    conn.commit()
    conn.close()
    a = OmegaAdapter(omega_home=str(home))
    entities, _ = a.get_graph_snapshot()
    assert len(entities) == 1
    assert entities[0].entity_type == "memory:lesson"
    a.close()


# --- reset ----------------------------------------------------------


def test_reset_refuses_default_home(fake_omega, monkeypatch):
    from sme.adapters.omega import OmegaAdapter, DEFAULT_OMEGA_HOME
    # Point at the default home explicitly; reset must refuse without the
    # override env var.
    a = OmegaAdapter(omega_home=DEFAULT_OMEGA_HOME)
    monkeypatch.delenv("OMEGA_ALLOW_DEFAULT_RESET", raising=False)
    with pytest.raises(RuntimeError, match="refusing to reset"):
        a.reset()
    a.close()


def test_reset_deletes_file_for_custom_home(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    home = isolate_omega_home / "resettable"
    home.mkdir(parents=True, exist_ok=True)
    db = home / "omega.db"
    db.write_text("seed")
    a = OmegaAdapter(omega_home=str(home))
    a.reset()
    assert not db.exists()
    a.close()


# --- ontology -------------------------------------------------------


def test_ontology_lists_omega_type_vocabulary(fake_omega, isolate_omega_home):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(omega_home=str(isolate_omega_home / "h"))
    ont = a.get_ontology_source()
    assert ont["type"] == "readme"
    type_entry = next(s for s in ont["schema"] if s["kind"] == "memory_types")
    assert "decision" in type_entry["values"]
    assert "user_preference" in type_entry["values"]
    edge_entry = next(s for s in ont["schema"] if s["kind"] == "edge_types")
    assert "related" in edge_entry["values"]
    assert "supersedes" in edge_entry["values"]
    a.close()


# --- string-parse fallback unit ------------------------------------


def test_parse_query_string_extracts_hits():
    from sme.adapters.omega import _parse_query_string
    text = (
        "Results: 2\n"
        "## 1. [user_preference] `mem-aaa` (str: 1.00)\n"
        "JP prefers donkeys.\n"
        "*2026-05-30T06:33*\n"
        "\n"
        "## 2. [lesson] `mem-bbb` (str: 0.50)\n"
        "graph endpoint is slow.\n"
        "*2026-05-30T06:34*\n"
    )
    hits = _parse_query_string(text)
    assert len(hits) == 2
    assert hits[0]["id"] == "mem-aaa"
    assert hits[0]["event_type"] == "user_preference"
    assert hits[0]["content"] == "JP prefers donkeys."
    assert hits[0]["relevance"] == 1.0
    assert hits[1]["id"] == "mem-bbb"
