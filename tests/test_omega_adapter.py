"""Tests for sme.adapters.omega — fully mocked, no live OMEGA install."""

from __future__ import annotations

import sqlite3
import sys
import types

import pytest


def _install_fake_omega(monkeypatch, store_fn=None, query_fn=None):
    """Inject a fake ``omega`` module into sys.modules so the adapter's
    import succeeds without omega-memory installed."""
    fake = types.ModuleType("omega")

    def default_store(content, mem_type):
        default_store.calls.append((content, mem_type))

    default_store.calls = []

    def default_query(text):
        default_query.calls.append(text)
        return [
            {"id": "m1", "content": "first omega memory", "type": "decision", "score": 0.9},
            {"id": "m2", "content": "second omega memory", "type": "lesson", "score": 0.7},
        ]

    default_query.calls = []

    fake.store = store_fn or default_store
    fake.query = query_fn or default_query
    fake.remember = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "omega", fake)
    return fake


@pytest.fixture
def fake_omega(monkeypatch):
    return _install_fake_omega(monkeypatch)


# --- construction ---------------------------------------------------


def test_construction_requires_omega_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "omega", None)
    # Force ImportError by injecting a sentinel that fails import
    import importlib

    # Remove and prevent import
    monkeypatch.delitem(sys.modules, "omega", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def faux_import(name, *a, **kw):
        if name == "omega":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", faux_import)

    # Have to re-import the adapter module since _require_omega is module-level
    from sme.adapters.omega import OmegaAdapter
    with pytest.raises(ImportError, match="omega-memory"):
        OmegaAdapter(db_path="/tmp/nope.db")


def test_construction_succeeds_with_omega_present(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    assert a.default_memory_type == "summary"
    assert a.n_results == 10


def test_db_path_env_var_used(fake_omega, monkeypatch, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    monkeypatch.setenv("OMEGA_DB_PATH", str(tmp_path / "from_env.db"))
    a = OmegaAdapter()
    assert "from_env.db" in a.db_path


# --- ingest_corpus -------------------------------------------------


def test_ingest_calls_omega_store_per_row(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    report = a.ingest_corpus([
        {"content": "fact one", "type": "lesson"},
        {"content": "fact two"},  # uses default_memory_type
    ])
    assert report["entities_created"] == 2
    assert fake_omega.store.calls == [
        ("fact one", "lesson"),
        ("fact two", "summary"),
    ]


def test_ingest_skips_empty_content(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    report = a.ingest_corpus([{"content": ""}, {"content": "ok"}])
    assert report["entities_created"] == 1
    assert any("empty content" in w for w in report["warnings"])


def test_ingest_captures_store_failures(monkeypatch, tmp_path):
    def boom(content, mem_type):
        raise RuntimeError("write failed")
    _install_fake_omega(monkeypatch, store_fn=boom)
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    report = a.ingest_corpus([{"content": "x"}])
    assert report["entities_created"] == 0
    assert len(report["errors"]) == 1
    assert "write failed" in report["errors"][0]


# --- query ----------------------------------------------------------


def test_query_builds_context_string(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    result = a.query("anything")
    assert result.error is None
    assert "[1] [decision] first omega memory" in result.context_string
    assert "[2] [lesson] second omega memory" in result.context_string


def test_query_retrieved_entities_carry_type_and_score(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    result = a.query("anything")
    assert len(result.retrieved_entities) == 2
    e0 = result.retrieved_entities[0]
    assert e0.id == "omega:m1"
    assert e0.entity_type == "memory:decision"
    assert e0.properties["score"] == 0.9


def test_query_no_results(monkeypatch, tmp_path):
    _install_fake_omega(monkeypatch, query_fn=lambda t: [])
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    result = a.query("nothing")
    assert result.error == "NO_RESULTS"


def test_query_normalises_dict_envelope(monkeypatch, tmp_path):
    _install_fake_omega(
        monkeypatch,
        query_fn=lambda t: {"results": [{"id": "r1", "content": "hi"}]},
    )
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    result = a.query("x")
    assert "hi" in result.context_string


def test_query_normalises_list_of_strings(monkeypatch, tmp_path):
    _install_fake_omega(monkeypatch, query_fn=lambda t: ["alpha", "beta"])
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    result = a.query("x")
    assert "alpha" in result.context_string
    assert "beta" in result.context_string


def test_query_n_results_truncates(monkeypatch, tmp_path):
    _install_fake_omega(
        monkeypatch,
        query_fn=lambda t: [{"id": f"m{i}", "content": f"c{i}"} for i in range(20)],
    )
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"), n_results=3)
    result = a.query("x")
    assert len(result.retrieved_entities) == 3


def test_query_internal_error_captured(monkeypatch, tmp_path):
    def boom(t):
        raise RuntimeError("query crash")
    _install_fake_omega(monkeypatch, query_fn=boom)
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    result = a.query("x")
    assert result.error.startswith("INTERNAL:")
    assert "query crash" in result.error


# --- get_graph_snapshot --------------------------------------------


def _seed_omega_db(path):
    """Build a fake OMEGA SQLite database with memories + edges."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, type TEXT)"
    )
    conn.execute(
        "CREATE TABLE edges (source_id TEXT, target_id TEXT, type TEXT)"
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
        "INSERT INTO edges VALUES (?,?,?)",
        [
            ("m1", "m2", "related"),
            ("m2", "m3", "supersedes"),
        ],
    )
    conn.commit()
    conn.close()


def test_snapshot_reads_memories_and_edges(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    db = tmp_path / "om.db"
    _seed_omega_db(db)
    a = OmegaAdapter(db_path=str(db))
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


def test_snapshot_missing_db_returns_empty(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "missing.db"))
    entities, edges = a.get_graph_snapshot()
    assert entities == []
    assert edges == []


def test_snapshot_schema_drift_no_edges_table(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    db = tmp_path / "om.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, type TEXT)"
    )
    conn.execute("INSERT INTO memories VALUES ('only', 'lone', 'summary')")
    conn.commit()
    conn.close()
    a = OmegaAdapter(db_path=str(db))
    entities, edges = a.get_graph_snapshot()
    assert len(entities) == 1
    assert edges == []  # no edges table is graceful, not fatal


def test_snapshot_alternate_table_names(fake_omega, tmp_path):
    """Older/newer OMEGA may use 'memory' table singular."""
    from sme.adapters.omega import OmegaAdapter
    db = tmp_path / "om.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE memory (id TEXT, text TEXT, kind TEXT)"
    )
    conn.execute("INSERT INTO memory VALUES ('z', 'zed', 'lesson')")
    conn.commit()
    conn.close()
    a = OmegaAdapter(db_path=str(db))
    entities, _ = a.get_graph_snapshot()
    assert len(entities) == 1
    assert entities[0].entity_type == "memory:lesson"


# --- reset ----------------------------------------------------------


def test_reset_refuses_default_path(fake_omega):
    from sme.adapters.omega import OmegaAdapter, DEFAULT_DB_PATH
    a = OmegaAdapter()  # uses default path
    a.db_path = str(__import__("pathlib").Path(__import__("os").path.expanduser(DEFAULT_DB_PATH)).resolve())
    with pytest.raises(RuntimeError, match="refusing to reset"):
        a.reset()


def test_reset_deletes_file_for_custom_path(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    db = tmp_path / "custom.db"
    db.write_text("seed")
    a = OmegaAdapter(db_path=str(db))
    a.reset()
    assert not db.exists()


# --- ontology -------------------------------------------------------


def test_ontology_lists_omega_type_vocabulary(fake_omega, tmp_path):
    from sme.adapters.omega import OmegaAdapter
    a = OmegaAdapter(db_path=str(tmp_path / "om.db"))
    ont = a.get_ontology_source()
    assert ont["type"] == "readme"
    type_entry = next(s for s in ont["schema"] if s["kind"] == "memory_types")
    assert "decision" in type_entry["values"]
    assert "user_preference" in type_entry["values"]
    edge_entry = next(s for s in ont["schema"] if s["kind"] == "edge_types")
    assert "related" in edge_entry["values"]
    assert "supersedes" in edge_entry["values"]
