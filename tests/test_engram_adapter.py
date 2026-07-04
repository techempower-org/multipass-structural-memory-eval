"""Unit tests for EngramAdapter (engram TypeScript MCP server).

engram (@199-bio/engram) is a local-first MCP memory server backed by
SQLite. This adapter spawns it over stdio (its default transport) and
speaks MCP JSON-RPC, while reading its knowledge-graph tables directly
from the SQLite file (which the adapter owns via ENGRAM_DB_PATH) for the
structural categories.

The MCP transport is injected (``transport=``) so these tests need no
Node runtime: a ``FakeTransport`` returns canned tool responses. The
graph / contradiction paths are exercised against a real temp SQLite DB
built with the stdlib ``sqlite3`` — again, no Node. The one live test is
opt-in (``ENGRAM_LIVE=1``) and skips cleanly when the runtime is absent.
"""

from __future__ import annotations

import os
import shutil
import sqlite3

import pytest

from sme.adapters.base import ContradictionPair, HarnessDescriptor, QueryResult
from sme.adapters.engram_adapter import EngramAdapter, EngramTransportError

_LIVE = os.environ.get("ENGRAM_LIVE") == "1"


# --- Test doubles -----------------------------------------------------


class FakeTransport:
    """Injectable stand-in for the stdio MCP transport.

    ``responses`` maps a tool name to a dict (the parsed
    ``result.content[0].text`` JSON) or a callable ``(arguments) -> dict``.
    Set ``raises`` to an exception to simulate a dead/failed subprocess on
    every call.
    """

    def __init__(self, responses=None, raises=None):
        self.responses = responses or {}
        self.raises = raises
        self.calls = []
        self.closed = False

    def call(self, tool, arguments):
        self.calls.append((tool, arguments))
        if self.raises is not None:
            raise self.raises
        r = self.responses.get(tool)
        if callable(r):
            return r(arguments)
        if r is None:
            raise EngramTransportError(f"FakeTransport: no canned response for {tool!r}")
        return r

    def close(self):
        self.closed = True


def _make_engram_db(db_dir, *, entities=(), relations=(), contradictions=(), memories=()):
    """Create a minimal engram.db (subset of the real schema) for the
    direct-read paths. Returns the db file path."""
    os.makedirs(db_dir, exist_ok=True)
    db_file = os.path.join(db_dir, "engram.db")
    con = sqlite3.connect(db_file)
    con.executescript(
        """
        CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, disabled INTEGER DEFAULT 0);
        CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT, type TEXT, metadata TEXT);
        CREATE TABLE relations (id TEXT PRIMARY KEY, from_entity TEXT, to_entity TEXT,
                                type TEXT, properties TEXT);
        CREATE TABLE contradictions (id TEXT PRIMARY KEY, entity_id TEXT,
                                     memory_id_a TEXT, memory_id_b TEXT, description TEXT,
                                     resolved INTEGER DEFAULT 0);
        """
    )
    con.executemany("INSERT INTO memories(id, content) VALUES (?, ?)", list(memories))
    con.executemany(
        "INSERT INTO entities(id, name, type, metadata) VALUES (?, ?, ?, ?)", list(entities)
    )
    con.executemany(
        "INSERT INTO relations(id, from_entity, to_entity, type, properties) "
        "VALUES (?, ?, ?, ?, ?)",
        list(relations),
    )
    con.executemany(
        "INSERT INTO contradictions(id, entity_id, memory_id_a, memory_id_b, description) "
        "VALUES (?, ?, ?, ?, ?)",
        list(contradictions),
    )
    con.commit()
    con.close()
    return db_file


def _adapter(tmp_path, transport=None, **over):
    """Build an adapter with an injected transport and a real temp db dir;
    reset-on-ingest off so ingest tests don't wipe first."""
    kwargs = dict(
        engram_path=None,
        db_path=str(tmp_path / "engramdb"),
        transport=transport if transport is not None else FakeTransport(),
        reset_before_ingest=False,
    )
    kwargs.update(over)
    return EngramAdapter(**kwargs)


# --- Construction -----------------------------------------------------


def test_creates_owned_temp_db_when_no_db_path(monkeypatch):
    monkeypatch.delenv("ENGRAM_DB_PATH", raising=False)
    a = EngramAdapter(engram_path=None, transport=FakeTransport())
    assert a._owns_db is True
    assert a.db_dir and os.path.isdir(a.db_dir)
    a.close()
    assert not os.path.exists(a.db_dir)  # owned temp dir cleaned on close


def test_explicit_db_path_is_not_owned(tmp_path):
    a = _adapter(tmp_path)
    assert a._owns_db is False
    assert a.db_dir == str(tmp_path / "engramdb")


def test_env_engram_path(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGRAM_PATH", "/opt/engram")
    a = EngramAdapter(db_path=str(tmp_path), transport=FakeTransport())
    assert a.engram_path == "/opt/engram"


def test_n_results_default(tmp_path):
    assert _adapter(tmp_path).n_results == 5


# --- ingest_corpus ----------------------------------------------------


def test_ingest_maps_content_and_records_id_map(tmp_path):
    t = FakeTransport({"remember": lambda args: {
        "success": True, "memory_id": "m-42",
        "entities_stored": [], "relationships_stored": [],
    }})
    a = _adapter(tmp_path, transport=t)
    result = a.ingest_corpus([{"id": "doc-1", "text": "Paris is the capital of France."}])
    assert result["entities_created"] == 1
    assert result["edges_created"] == 0
    assert result["errors"] == []
    # remember got the content
    tool, args = t.calls[0]
    assert tool == "remember"
    assert args["content"] == "Paris is the capital of France."
    # engram memory id → corpus id recorded for retrieval attribution
    assert a._id_map["m-42"] == "doc-1"


def test_ingest_content_field_precedence(tmp_path):
    sent = []
    t = FakeTransport({"remember": lambda args: (sent.append(args["content"]),
                                                 {"success": True, "memory_id": "x"})[1]})
    a = _adapter(tmp_path, transport=t)
    a.ingest_corpus([
        {"id": "1", "document": "from-document", "content": "c", "text": "t"},
        {"id": "2", "content": "from-content", "text": "t"},
        {"id": "3", "text": "from-text"},
    ])
    assert sent == ["from-document", "from-content", "from-text"]


def test_ingest_forwards_entities_and_relationships(tmp_path):
    t = FakeTransport({"remember": lambda args: {
        "success": True, "memory_id": "m1",
        "entities_stored": ["Sarah"], "relationships_stored": ["Sarah -[works_at]-> Acme"],
    }})
    a = _adapter(tmp_path, transport=t)
    result = a.ingest_corpus([{
        "id": "d1", "text": "Sarah works at Acme.",
        "entities": [{"name": "Sarah", "type": "person"}, {"name": "Acme", "type": "organization"}],
        "relationships": [{"from": "Sarah", "to": "Acme", "type": "works_at"}],
    }])
    _, args = t.calls[0]
    assert args["entities"] == [{"name": "Sarah", "type": "person"},
                                {"name": "Acme", "type": "organization"}]
    assert args["relationships"] == [{"from": "Sarah", "to": "Acme", "type": "works_at"}]
    assert result["edges_created"] == 1  # one relationship stored


def test_ingest_duplicate_counts_as_existed_not_error(tmp_path):
    t = FakeTransport({"remember": lambda args: {
        "success": False, "duplicate": True, "existing_id": "m-old",
    }})
    a = _adapter(tmp_path, transport=t)
    result = a.ingest_corpus([{"id": "1", "text": "dup"}])
    assert result["entities_created"] == 0
    assert result["errors"] == []
    assert any("duplicate" in w.lower() for w in result["warnings"])


def test_ingest_skips_blank_content(tmp_path):
    t = FakeTransport({"remember": {"success": True, "memory_id": "x"}})
    a = _adapter(tmp_path, transport=t)
    result = a.ingest_corpus([{"id": "1", "text": "  "}, {"id": "2"}])
    assert result["entities_created"] == 0
    assert t.calls == []  # nothing sent


def test_ingest_transport_error_captured_not_raised(tmp_path):
    t = FakeTransport(raises=EngramTransportError("subprocess died"))
    a = _adapter(tmp_path, transport=t)
    result = a.ingest_corpus([{"id": "1", "text": "boom"}])
    assert result["entities_created"] == 0
    assert result["errors"] and "subprocess died" in result["errors"][0]


# --- query ------------------------------------------------------------


def _recall(context, ids):
    return {"recall": {"context": context, "_ids": ids}}


def test_query_happy_path(tmp_path):
    t = FakeTransport(_recall(
        ["Jan 5: The Eiffel Tower is in Paris.", "Jan 6: France is in Europe."],
        ["m-1", "m-2"],
    ))
    a = _adapter(tmp_path, transport=t)
    r = a.query("Where is the Eiffel Tower?", n_results=3)
    assert r.error is None
    assert "Eiffel Tower" in r.context_string
    assert [e.id for e in r.retrieved_entities] == ["m-1", "m-2"]
    assert r.retrieved_entities[0].properties["rank"] == 1
    assert all(isinstance(p, (str, int, float)) for p in r.retrieval_path)
    # recall got include_graph default True
    _, args = t.calls[0]
    assert args["include_graph"] is True
    assert args["limit"] == 3


def test_query_without_n_results_kwarg(tmp_path):
    t = FakeTransport(_recall([], []))
    r = _adapter(tmp_path, transport=t).query("anything")
    assert isinstance(r, QueryResult)


def test_query_uses_constructor_n_results_default(tmp_path):
    t = FakeTransport(_recall([], []))
    _adapter(tmp_path, transport=t, n_results=9).query("q")
    _, args = t.calls[0]
    assert args["limit"] == 9


def test_query_no_results_sets_error(tmp_path):
    t = FakeTransport(_recall([], []))
    r = _adapter(tmp_path, transport=t).query("nothing")
    assert r.context_string == ""
    assert r.error == "NO_RESULTS"


def test_query_maps_engram_id_to_corpus_id(tmp_path):
    t = FakeTransport({
        "remember": {"success": True, "memory_id": "m-7"},
        "recall": {"context": ["Jan 1: hi"], "_ids": ["m-7"]},
    })
    a = _adapter(tmp_path, transport=t)
    a.ingest_corpus([{"id": "corpus-99", "text": "hi"}])
    r = a.query("hi")
    # retrieved entity carries the original corpus id for R@K scoring
    assert r.retrieved_entities[0].properties["source_id"] == "corpus-99"


def test_query_transport_error_no_raise(tmp_path):
    t = FakeTransport(raises=EngramTransportError("connect failed"))
    r = _adapter(tmp_path, transport=t).query("q")
    assert isinstance(r, QueryResult)
    assert r.error and "connect failed" in r.error


def test_get_flat_retrieval_disables_graph(tmp_path):
    t = FakeTransport(_recall(["Jan 1: x"], ["m-1"]))
    a = _adapter(tmp_path, transport=t)
    a.get_flat_retrieval("x")
    _, args = t.calls[0]
    assert args["include_graph"] is False


# --- get_graph_snapshot (direct SQLite) -------------------------------


def test_graph_snapshot_from_sqlite(tmp_path):
    db_dir = str(tmp_path / "engramdb")
    _make_engram_db(
        db_dir,
        entities=[("e1", "Sarah", "person", None), ("e2", "Acme", "organization", None)],
        relations=[("r1", "e1", "e2", "works_at", None)],
    )
    a = _adapter(tmp_path, transport=FakeTransport())
    entities, edges = a.get_graph_snapshot()
    ids = {e.id for e in entities}
    assert "entity:e1" in ids and "entity:e2" in ids
    assert len(edges) == 1
    edge = edges[0]
    assert edge.source_id == "entity:e1" and edge.target_id == "entity:e2"
    assert edge.edge_type == "works_at"
    # internal consistency
    for e in edges:
        assert e.source_id in ids and e.target_id in ids


def test_graph_snapshot_synthesizes_missing_endpoint(tmp_path):
    db_dir = str(tmp_path / "engramdb")
    _make_engram_db(
        db_dir,
        entities=[("e1", "Sarah", "person", None)],
        relations=[("r1", "e1", "ghost", "knows", None)],  # 'ghost' not in entities
    )
    a = _adapter(tmp_path, transport=FakeTransport())
    entities, edges = a.get_graph_snapshot()
    ids = {e.id for e in entities}
    for e in edges:
        assert e.source_id in ids and e.target_id in ids  # ghost synthesized


def test_graph_snapshot_missing_db_returns_empty(tmp_path):
    a = _adapter(tmp_path, transport=FakeTransport())  # no db file created
    assert a.get_graph_snapshot() == ([], [])


# --- get_contradiction_pairs (direct SQLite) --------------------------


def test_contradiction_pairs_from_sqlite(tmp_path):
    db_dir = str(tmp_path / "engramdb")
    _make_engram_db(
        db_dir,
        memories=[("m1", "Alice earns 50k"), ("m2", "Alice earns 80k")],
        contradictions=[("c1", None, "m1", "m2", "salary conflict")],
    )
    a = _adapter(tmp_path, transport=FakeTransport())
    pairs = a.get_contradiction_pairs()
    assert len(pairs) == 1
    p = pairs[0]
    assert isinstance(p, ContradictionPair)
    assert {p.source_a, p.source_b} == {"m1", "m2"}


def test_contradiction_pairs_empty_when_no_db(tmp_path):
    assert _adapter(tmp_path, transport=FakeTransport()).get_contradiction_pairs() == []


# --- reset ------------------------------------------------------------


def test_reset_user_db_deletes_rows_and_counts(tmp_path):
    db_dir = str(tmp_path / "engramdb")
    _make_engram_db(
        db_dir,
        memories=[("m1", "a"), ("m2", "b")],
        entities=[("e1", "X", "person", None)],
    )
    t = FakeTransport()
    a = _adapter(tmp_path, transport=t)
    n = a.reset()
    assert n == 2  # counted the two memories
    assert t.closed is True  # subprocess released before wipe
    con = sqlite3.connect(os.path.join(db_dir, "engram.db"))
    assert con.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
    con.close()


def test_reset_owned_db_deletes_files(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGRAM_DB_PATH", raising=False)
    a = EngramAdapter(engram_path=None, transport=FakeTransport())  # owns temp db
    _make_engram_db(a.db_dir, memories=[("m1", "a")])
    n = a.reset()
    assert n == 1
    assert not os.path.exists(os.path.join(a.db_dir, "engram.db"))
    shutil.rmtree(a.db_dir, ignore_errors=True)


def test_reset_no_db_returns_zero(tmp_path):
    assert _adapter(tmp_path, transport=FakeTransport()).reset() == 0


# --- optional methods -------------------------------------------------


def test_get_ontology_source_typed(tmp_path):
    src = _adapter(tmp_path).get_ontology_source()
    assert src["type"] in {"declared", "readme", "inferred"}
    assert isinstance(src["schema"], list)
    assert isinstance(src["documentation"], str)


def test_get_harness_manifest_shape_and_versions(tmp_path):
    a = _adapter(tmp_path, engram_path="/opt/engram")
    manifest = a.get_harness_manifest()
    assert isinstance(manifest, list) and manifest
    assert all(isinstance(d, HarnessDescriptor) for d in manifest)
    mcp = next(d for d in manifest if d.kind == "mcp_resource")
    # version attribution (Sandman pin #1) present in properties
    assert "engram_version" in mcp.properties
    assert "node_version" in mcp.properties


def test_get_introspection_report_none(tmp_path):
    assert _adapter(tmp_path).get_introspection_report() is None


def test_close_idempotent(tmp_path):
    a = _adapter(tmp_path)
    a.close()
    a.close()


# --- Real stdio transport (hermetic — a Python fake MCP server) -------
#
# FakeTransport bypasses the actual subprocess/JSON-RPC plumbing, so these
# tests drive the REAL EngramStdioTransport against a tiny stdlib-only MCP
# server spoken over stdin/stdout. This exercises spawn + initialize
# handshake + newline framing + select-based reads + non-JSON-line
# skipping + content parsing — with only python3, no Node/engram/models.

_FAKE_MCP_SERVER = r'''
import sys, json
# A non-JSON banner on stdout — the transport must skip it (real servers
# occasionally print startup noise before protocol).
sys.stdout.write("fake-engram starting (not json)\n"); sys.stdout.flush()

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue
    method = msg.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "serverInfo": {"name": "fake-engram", "version": "0"}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/call":
        name = msg["params"]["name"]; args = msg["params"].get("arguments", {})
        if name == "remember":
            payload = {"success": True, "memory_id": "mem-" + str(args.get("content", ""))[:6],
                       "relationships_stored": []}
        elif name == "recall":
            payload = {"context": ["fake: " + args.get("query", "")], "_ids": ["mem-1"]}
        else:
            payload = {"ok": True}
        send({"jsonrpc": "2.0", "id": msg["id"],
              "result": {"content": [{"type": "text", "text": json.dumps(payload)}]}})
'''


def _write_fake_server(path):
    path.write_text(_FAKE_MCP_SERVER)
    return str(path)


def test_real_stdio_transport_handshake_and_call(tmp_path):
    import sys as _sys

    from sme.adapters.engram_adapter import EngramStdioTransport

    server = _write_fake_server(tmp_path / "fake_mcp.py")
    t = EngramStdioTransport(
        entry=server, db_dir=str(tmp_path), node_bin=_sys.executable,
        startup_timeout=10, call_timeout=10,
    )
    try:
        r = t.call("recall", {"query": "hello", "limit": 5})
        assert r == {"context": ["fake: hello"], "_ids": ["mem-1"]}
        r2 = t.call("remember", {"content": "abcdefghij"})
        assert r2["success"] is True and r2["memory_id"].startswith("mem-")
    finally:
        t.close()


def test_real_stdio_transport_reports_dead_process(tmp_path):
    import sys as _sys

    from sme.adapters.engram_adapter import EngramStdioTransport

    # A server that exits immediately → handshake must fail, not hang.
    dead = tmp_path / "dead.py"
    dead.write_text("import sys; sys.exit(0)\n")
    t = EngramStdioTransport(
        entry=str(dead), db_dir=str(tmp_path), node_bin=_sys.executable,
        startup_timeout=5, call_timeout=5,
    )
    with pytest.raises(EngramTransportError):
        t.call("recall", {"query": "x"})
    t.close()


def test_adapter_end_to_end_via_real_transport(tmp_path):
    """Full adapter path (ingest → query) over the REAL transport against
    the fake server — engram_path/dist/index.js is the fake server, node is
    python3. Validates the spawn + JSON-RPC round-trip end to end."""
    import sys as _sys

    dist = tmp_path / "engram_home" / "dist"
    dist.mkdir(parents=True)
    _write_fake_server(dist / "index.js")
    a = EngramAdapter(
        engram_path=str(tmp_path / "engram_home"),
        node_bin=_sys.executable,
        db_path=str(tmp_path / "db"),
        reset_before_ingest=False,
    )
    try:
        res = a.ingest_corpus([{"id": "d1", "text": "hello world"}])
        assert res["entities_created"] == 1, res
        q = a.query("hello", n_results=3)
        assert q.error is None
        assert "fake: hello" in q.context_string
        assert q.retrieved_entities[0].id == "mem-1"
    finally:
        a.close()


# --- Live integration (opt-in) ----------------------------------------


@pytest.mark.skipif(not _LIVE, reason="set ENGRAM_LIVE=1 (needs node + built dist/) to run")
def test_live_ingest_recall_snapshot_reset(tmp_path):
    """Real engram over stdio. Skips cleanly if node/engram_path/dist are
    absent. See morpheus-engram.md for the one-time `npm install && build`."""
    engram_path = os.environ.get("ENGRAM_PATH")
    if not engram_path or not os.path.exists(os.path.join(engram_path, "dist", "index.js")):
        pytest.skip("engram dist/ not built — see morpheus-engram.md setup")
    if not shutil.which("node"):
        pytest.skip("node not on PATH")
    a = EngramAdapter(engram_path=engram_path, db_path=str(tmp_path / "live"),
                      reset_before_ingest=True)
    try:
        res = a.ingest_corpus([
            {"id": "L1", "text": "The Eiffel Tower is in Paris.",
             "entities": [{"name": "Eiffel Tower", "type": "place"},
                          {"name": "Paris", "type": "place"}],
             "relationships": [{"from": "Eiffel Tower", "to": "Paris", "type": "located_in"}]},
            {"id": "L2", "text": "Marie Curie won two Nobel Prizes."},
        ])
        assert res["entities_created"] >= 1, res
        q = a.query("Where is the Eiffel Tower?", n_results=5)
        assert q.error in (None, "NO_RESULTS"), q.error
        entities, edges = a.get_graph_snapshot()
        assert isinstance(entities, list) and isinstance(edges, list)
    finally:
        a.close()
