"""Unit tests for MemPalaceServerAdapter (Go MemPalace server).

Talks to sefodo26's Go reimplementation of the MemPalace server
(fork: techempower-org/mempalace-server) over its REST API
(``/mp/api/v1``, bearer auth). All HTTP is mocked via the shared
``fake_urlopen_factory`` fixture — no live server required.

The adapter is also exercised through the parametric contract suite in
``tests/test_fork_adapters_contract.py`` (unreachable-URL factory).
"""

from __future__ import annotations

import json
import os
import urllib.error

import pytest

from sme.adapters.base import HarnessDescriptor, ProbeResult, QueryResult
from sme.adapters.mempalace_server_adapter import MemPalaceServerAdapter

BASE = "http://mp-test:8000"

# Set MEMPALACE_SERVER_LIVE=1 (and, if not using docker-compose defaults,
# MEMPALACE_SERVER_URL / MEMPALACE_SERVER_API_KEY) to run the live
# round-trip against a real Go server. Off by default so CI stays hermetic.
_LIVE = os.environ.get("MEMPALACE_SERVER_LIVE") == "1"


def _adapter(**over):
    """Build an adapter against the fake host; reset-on-ingest off by
    default so ingest tests don't have to mock the list/delete round-trip.
    """
    kwargs = dict(
        api_url=BASE,
        api_key="test-key",
        reset_before_ingest=False,
    )
    kwargs.update(over)
    return MemPalaceServerAdapter(**kwargs)


# --- Construction / config resolution ---------------------------------


def test_default_construction_uses_docker_compose_defaults(monkeypatch):
    for var in ("MEMPALACE_SERVER_URL", "MEMPALACE_SERVER_API_KEY", "MEMPALACE_SERVER_TENANT"):
        monkeypatch.delenv(var, raising=False)
    a = MemPalaceServerAdapter()
    assert a.api_url == "http://localhost:8000"
    assert a.api_key == "local-dev-key-change-me"
    assert a.tenant == "default"
    assert a.n_results == 5


def test_env_construction(monkeypatch):
    monkeypatch.setenv("MEMPALACE_SERVER_URL", "http://env-host:9000/")
    monkeypatch.setenv("MEMPALACE_SERVER_API_KEY", "env-key")
    monkeypatch.setenv("MEMPALACE_SERVER_TENANT", "env-tenant")
    a = MemPalaceServerAdapter()
    assert a.api_url == "http://env-host:9000"  # trailing slash stripped
    assert a.api_key == "env-key"
    assert a.tenant == "env-tenant"


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("MEMPALACE_SERVER_URL", "http://env-host:9000")
    monkeypatch.setenv("MEMPALACE_SERVER_API_KEY", "env-key")
    a = MemPalaceServerAdapter(api_url="http://explicit:1", api_key="explicit")
    assert a.api_url == "http://explicit:1"
    assert a.api_key == "explicit"


def test_base_url_trailing_slash_stripped():
    a = _adapter(api_url="http://mp-test:8000/")
    assert a.api_url == "http://mp-test:8000"


# --- ingest_corpus ----------------------------------------------------


def test_ingest_maps_fields_and_counts(fake_urlopen_factory):
    captured = {}

    def add_route(req):
        captured["body"] = json.loads(req.data.decode())
        captured["auth"] = req.get_header("Authorization")
        return {"success": True, "drawer_id": "abc", "wing": "w", "room": "r"}

    fake_urlopen_factory({f"POST {BASE}/mp/api/v1/drawers": add_route})
    a = _adapter(wing="fallback-wing")
    result = a.ingest_corpus(
        [
            {
                "id": "doc-1",
                "text": "hello world",
                "wing": "projects",
                "room": "decisions",
                "source_file": "d1.md",
            }
        ]
    )
    assert result["entities_created"] == 1
    assert result["edges_created"] == 0
    assert result["errors"] == []
    assert captured["body"] == {
        "wing": "projects",
        "room": "decisions",
        "content": "hello world",
        "source_file": "d1.md",
        "added_by": "sme",
    }
    assert captured["auth"] == "Bearer test-key"


def test_ingest_content_field_precedence(fake_urlopen_factory):
    bodies = []

    def add_route(req):
        bodies.append(json.loads(req.data.decode()))
        return {"success": True, "drawer_id": "x"}

    fake_urlopen_factory({f"POST {BASE}/mp/api/v1/drawers": add_route})
    a = _adapter()
    a.ingest_corpus(
        [
            {"id": "1", "document": "from-document", "content": "c", "text": "t"},
            {"id": "2", "content": "from-content", "text": "t"},
            {"id": "3", "text": "from-text"},
        ]
    )
    assert [b["content"] for b in bodies] == [
        "from-document",
        "from-content",
        "from-text",
    ]


def test_ingest_room_falls_back_to_session_then_id(fake_urlopen_factory):
    bodies = []

    def add_route(req):
        bodies.append(json.loads(req.data.decode()))
        return {"success": True, "drawer_id": "x"}

    fake_urlopen_factory({f"POST {BASE}/mp/api/v1/drawers": add_route})
    a = _adapter(wing="w", room=None)
    a.ingest_corpus(
        [
            {"id": "id-1", "text": "a", "session_id": "sess-9"},  # session wins
            {"id": "id-2", "text": "b"},  # id used
        ]
    )
    assert [b["room"] for b in bodies] == ["sess-9", "id-2"]


def test_ingest_counts_bullets(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/api/v1/drawers": {
                "success": True,
                "bullets_stored": 3,
                "bullets_total": 3,
                "wing": "w",
                "room": "r",
            },
        }
    )
    a = _adapter()
    result = a.ingest_corpus([{"id": "1", "text": "- a\n- b\n- c"}])
    assert result["entities_created"] == 3


def test_ingest_already_exists_counts_zero(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/api/v1/drawers": {
                "success": True,
                "reason": "already_exists",
                "drawer_id": "abc",
            },
        }
    )
    a = _adapter()
    result = a.ingest_corpus([{"id": "1", "text": "dup"}])
    assert result["entities_created"] == 0
    assert any("already_exists" in w or "existed" in w for w in result["warnings"])


def test_ingest_skips_blank_content(fake_urlopen_factory):
    fake_urlopen_factory({})  # no route needed — nothing should POST
    a = _adapter()
    result = a.ingest_corpus([{"id": "1", "text": "   "}, {"id": "2"}])
    assert result["entities_created"] == 0


def test_ingest_http_error_captured_not_raised(fake_urlopen_factory):
    err = urllib.error.HTTPError(f"{BASE}/mp/api/v1/drawers", 500, "Server Error", {}, None)
    fake_urlopen_factory({f"POST {BASE}/mp/api/v1/drawers": err})
    a = _adapter()
    result = a.ingest_corpus([{"id": "1", "text": "boom"}])
    assert result["entities_created"] == 0
    assert result["errors"]
    assert "500" in result["errors"][0]


def test_ingest_resets_wing_first_when_enabled(fake_urlopen_factory):
    seq = []
    pages = [{"drawers": [{"drawer_id": "old1"}], "count": 1}, {"drawers": []}]

    def list_route(req):
        seq.append(("list", req.full_url))
        return pages.pop(0)

    def del_route(req):
        seq.append(("delete", req.full_url))
        return {"success": True}

    def add_route(req):
        seq.append(("add", req.full_url))
        return {"success": True, "drawer_id": "new"}

    fake_urlopen_factory(
        {
            f"GET {BASE}/mp/api/v1/drawers": list_route,
            f"DELETE {BASE}/mp/api/v1/drawers/old1": del_route,
            f"POST {BASE}/mp/api/v1/drawers": add_route,
        }
    )
    a = _adapter(wing="scoped", reset_before_ingest=True)
    result = a.ingest_corpus([{"id": "1", "text": "fresh"}])
    kinds = [k for k, _ in seq]
    # reset (list+delete) must happen before the add
    assert kinds.index("delete") < kinds.index("add")
    # reset list must be scoped to the configured wing
    assert any("wing=scoped" in url for k, url in seq if k == "list")
    assert result["entities_created"] == 1


# --- query ------------------------------------------------------------


def _search_body(results):
    return {"results": results, "count": len(results)}


def test_query_happy_path(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/api/v1/search": _search_body(
                [
                    {
                        "drawer_id": "d-1",
                        "wing": "projects",
                        "room": "decisions",
                        "similarity": 0.87,
                        "distance": 0.13,
                        "content": "We use JWT.",
                        "filed_at": "2026-07-03T00:00:00Z",
                        "source_file": "auth.md",
                        "metadata": {"wing": "projects", "room": "decisions"},
                    },
                ]
            ),
        }
    )
    a = _adapter()
    r = a.query("auth?", n_results=3)
    assert r.error is None
    assert "We use JWT." in r.context_string
    assert len(r.retrieved_entities) == 1
    e = r.retrieved_entities[0]
    # Retrieval scorers match Entity.id against corpus/session ids, which
    # ingest stores in source_file — so the id is the source stem, with the
    # server's content-hash drawer id kept in properties for provenance.
    assert e.id == "auth"
    assert e.properties["drawer_id"] == "d-1"
    assert e.properties["wing"] == "projects"
    assert e.properties["room"] == "decisions"
    assert e.properties["similarity"] == 0.87
    assert e.properties["distance"] == 0.13
    assert e.properties["rank"] == 1
    # retrieval_path items must be str/int/float (contract)
    assert all(isinstance(p, (str, int, float)) for p in r.retrieval_path)


def test_query_without_n_results_kwarg(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/api/v1/search": _search_body([]),
        }
    )
    a = _adapter()
    r = a.query("anything")
    assert isinstance(r, QueryResult)


def test_query_no_results_sets_error(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/api/v1/search": _search_body([]),
        }
    )
    a = _adapter()
    r = a.query("nothing here")
    assert r.context_string == ""
    assert r.error == "NO_RESULTS"


def test_query_sends_limit_and_filters(fake_urlopen_factory):
    captured = {}

    def search_route(req):
        captured["body"] = json.loads(req.data.decode())
        return _search_body([])

    fake_urlopen_factory({f"POST {BASE}/mp/api/v1/search": search_route})
    a = _adapter()
    a.query("q", n_results=7, wing="projects", room="decisions")
    assert captured["body"]["query"] == "q"
    assert captured["body"]["limit"] == 7
    assert captured["body"]["wing"] == "projects"
    assert captured["body"]["room"] == "decisions"
    # max_distance omitted by default (matches reference benchmark → server 1.5)
    assert "max_distance" not in captured["body"]


def test_query_default_limit_uses_constructor_n_results(fake_urlopen_factory):
    captured = {}

    def search_route(req):
        captured["body"] = json.loads(req.data.decode())
        return _search_body([])

    fake_urlopen_factory({f"POST {BASE}/mp/api/v1/search": search_route})
    a = _adapter(n_results=9)
    a.query("q")  # no explicit n_results → falls back to constructor value
    assert captured["body"]["limit"] == 9


def test_query_max_distance_included_when_set(fake_urlopen_factory):
    captured = {}

    def search_route(req):
        captured["body"] = json.loads(req.data.decode())
        return _search_body([])

    fake_urlopen_factory({f"POST {BASE}/mp/api/v1/search": search_route})
    a = _adapter(max_distance=0.4)
    a.query("q")
    assert captured["body"]["max_distance"] == 0.4


def test_query_auth_error_no_raise(fake_urlopen_factory):
    err = urllib.error.HTTPError(f"{BASE}/mp/api/v1/search", 401, "Unauthorized", {}, None)
    fake_urlopen_factory({f"POST {BASE}/mp/api/v1/search": err})
    a = _adapter()
    r = a.query("q")
    assert isinstance(r, QueryResult)
    assert r.error and "401" in r.error


def test_query_connection_error_no_raise(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/api/v1/search": urllib.error.URLError("Connection refused"),
        }
    )
    a = _adapter()
    r = a.query("q")
    assert r.error and r.error.startswith("CONNECTION:")


# --- get_graph_snapshot ----------------------------------------------


def _mcp_route(entities=None, relations_by_entity=None, error=None):
    """Build a fake_urlopen callable for POST /mp/mcp that dispatches on the
    JSON-RPC tool name (params.name) and returns the standard MCP envelope
    ({result:{content:[{type:text,text:<json>}]}}), or a JSON-RPC error."""
    entities = entities or []
    relations_by_entity = relations_by_entity or {}

    def route(req):
        env = json.loads(req.data.decode())
        rid = env.get("id")
        if error is not None:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": error}}
        name = env["params"]["name"]
        args = env["params"].get("arguments", {})
        if name == "mempalace_kg_search_entities":
            payload = {"entities": entities, "count": len(entities)}
        elif name == "mempalace_kg_get_entity":
            ename = args.get("name")
            payload = {"entity": {"name": ename}, "relations": relations_by_entity.get(ename, [])}
        else:
            payload = {"ok": True}
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
        }

    return route


def test_graph_snapshot_from_taxonomy(fake_urlopen_factory):
    # KG empty → deterministic fallback to the wing/room taxonomy snapshot.
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/mcp": _mcp_route(entities=[]),
            f"GET {BASE}/mp/api/v1/taxonomy": {
                "total": 6,
                "taxonomy": {
                    "projects": {"decisions": 3, "notes": 2},
                    "people": {"jp": 1},
                },
            },
        }
    )
    a = _adapter()
    entities, edges = a.get_graph_snapshot()
    ids = {e.id for e in entities}
    assert "wing:projects" in ids
    assert "wing:people" in ids
    assert "room:decisions" in ids
    # every edge endpoint resolves to an entity (internal consistency)
    for edge in edges:
        assert edge.source_id in ids
        assert edge.target_id in ids
    # rooms link to their wing via member_of
    assert any(e.edge_type == "member_of" for e in edges)
    assert a._graph_basis == "taxonomy"


def test_graph_snapshot_error_returns_empty(fake_urlopen_factory):
    err = urllib.error.HTTPError(f"{BASE}/mp/api/v1/taxonomy", 500, "Server Error", {}, None)
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/mcp": _mcp_route(entities=[]),  # KG empty
            f"GET {BASE}/mp/api/v1/taxonomy": err,  # taxonomy errors
        }
    )
    a = _adapter()
    entities, edges = a.get_graph_snapshot()
    assert entities == []
    assert edges == []


# --- get_graph_snapshot: real KG via MCP (kg-first, taxonomy fallback) ---


def test_graph_snapshot_from_kg(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/mcp": _mcp_route(
                entities=[
                    {"name": "Sarah", "entity_type": "person"},
                    {"name": "Acme", "entity_type": "organization"},
                ],
                relations_by_entity={
                    "Sarah": [{"type": "works_at", "from": "Sarah", "to": "Acme"}],
                    "Acme": [{"type": "works_at", "from": "Sarah", "to": "Acme"}],
                },
            ),
            # taxonomy present too — but KG is non-empty so it must NOT be used.
            f"GET {BASE}/mp/api/v1/taxonomy": {"total": 9, "taxonomy": {"w": {"r": 9}}},
        }
    )
    a = _adapter()
    entities, edges = a.get_graph_snapshot()
    ids = {e.id for e in entities}
    assert a._graph_basis == "kg"
    assert "kg:Sarah" in ids and "kg:Acme" in ids
    assert "wing:w" not in ids  # taxonomy NOT used
    assert len(edges) == 1  # relation deduped across both endpoints
    assert edges[0].edge_type == "works_at"
    for e in edges:
        assert e.source_id in ids and e.target_id in ids


def test_graph_snapshot_kg_unavailable_falls_back_to_taxonomy(fake_urlopen_factory):
    # AGE not installed → kg_search_entities returns a JSON-RPC error.
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/mcp": _mcp_route(error="knowledge graph not available"),
            f"GET {BASE}/mp/api/v1/taxonomy": {"total": 1, "taxonomy": {"w": {"r": 1}}},
        }
    )
    a = _adapter()
    entities, edges = a.get_graph_snapshot()
    assert a._graph_basis == "taxonomy"
    assert any(e.id == "wing:w" for e in entities)


def test_graph_snapshot_kg_synthesizes_missing_endpoint(fake_urlopen_factory):
    # A relation points to an entity beyond the enumerated set → synthesize it.
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/mcp": _mcp_route(
                entities=[{"name": "Sarah", "entity_type": "person"}],
                relations_by_entity={
                    "Sarah": [{"type": "knows", "from": "Sarah", "to": "Ghost"}],
                },
            ),
        }
    )
    a = _adapter()
    entities, edges = a.get_graph_snapshot()
    ids = {e.id for e in entities}
    assert "kg:Ghost" in ids  # synthesized
    for e in edges:
        assert e.source_id in ids and e.target_id in ids


def test_kg_contradiction_pairs_via_base_default(fake_urlopen_factory):
    # A CONTRADICTS-typed KG relation → base-class get_contradiction_pairs
    # derives one pair from the KG snapshot (no override needed).
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/mcp": _mcp_route(
                entities=[
                    {"name": "salary_50k", "entity_type": "claim"},
                    {"name": "salary_80k", "entity_type": "claim"},
                ],
                relations_by_entity={
                    "salary_50k": [
                        {"type": "CONTRADICTS", "from": "salary_50k", "to": "salary_80k"}
                    ],
                    "salary_80k": [
                        {"type": "CONTRADICTS", "from": "salary_50k", "to": "salary_80k"}
                    ],
                },
            ),
        }
    )
    a = _adapter()
    pairs = a.get_contradiction_pairs()
    assert len(pairs) == 1
    assert {pairs[0].source_a, pairs[0].source_b} == {"kg:salary_50k", "kg:salary_80k"}


def test_graph_basis_recorded_in_harness_manifest(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/mcp": _mcp_route(
                entities=[{"name": "A", "entity_type": "concept"}],
                relations_by_entity={"A": []},
            ),
        }
    )
    a = _adapter()
    a.get_graph_snapshot()  # sets basis = kg
    mcp = next(d for d in a.get_harness_manifest() if d.kind == "mcp_resource")
    assert mcp.properties["graph_basis"] == "kg"


# --- reset ------------------------------------------------------------


def test_reset_lists_and_deletes(fake_urlopen_factory):
    pages = [
        {"drawers": [{"drawer_id": "d1"}, {"drawer_id": "d2"}], "count": 2},
        {"drawers": [], "count": 0},
    ]
    deleted = []

    def list_route(req):
        return pages.pop(0)

    def del_route(req):
        deleted.append(req.full_url.rsplit("/", 1)[-1])
        return {"success": True, "drawer_id": deleted[-1]}

    fake_urlopen_factory(
        {
            f"GET {BASE}/mp/api/v1/drawers": list_route,
            f"DELETE {BASE}/mp/api/v1/drawers/d1": del_route,
            f"DELETE {BASE}/mp/api/v1/drawers/d2": del_route,
        }
    )
    a = _adapter()
    n = a.reset()
    assert n == 2
    assert set(deleted) == {"d1", "d2"}


def test_reset_unreachable_returns_zero_no_raise(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"GET {BASE}/mp/api/v1/drawers": urllib.error.URLError("Connection refused"),
        }
    )
    a = _adapter()
    assert a.reset() == 0


# --- optional methods -------------------------------------------------


def test_get_ontology_source_typed():
    a = _adapter()
    src = a.get_ontology_source()
    assert src["type"] in {"declared", "readme", "inferred"}
    assert isinstance(src["schema"], list)
    assert isinstance(src["documentation"], str)


def test_get_flat_retrieval_delegates_to_query(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"POST {BASE}/mp/api/v1/search": _search_body(
                [
                    {"drawer_id": "d1", "content": "hi", "wing": "w", "room": "r"},
                ]
            ),
        }
    )
    a = _adapter()
    r = a.get_flat_retrieval("hi")
    assert isinstance(r, QueryResult)
    assert r.retrieved_entities and r.retrieved_entities[0].id == "d1"


def test_get_harness_manifest_shape():
    a = _adapter()
    manifest = a.get_harness_manifest()
    assert isinstance(manifest, list)
    assert manifest and all(isinstance(d, HarnessDescriptor) for d in manifest)
    kinds = {d.kind for d in manifest}
    assert "mcp_resource" in kinds


def test_harness_mcp_probe_success(fake_urlopen_factory):
    fake_urlopen_factory({f"GET {BASE}/mp/mcp/health": {"status": "ok"}})
    a = _adapter()
    mcp = next(d for d in a.get_harness_manifest() if d.kind == "mcp_resource")
    res = mcp.probe_fn()
    assert isinstance(res, ProbeResult)
    assert res.success is True


def test_harness_probe_failure_no_raise(fake_urlopen_factory):
    fake_urlopen_factory(
        {
            f"GET {BASE}/mp/mcp/health": urllib.error.URLError("down"),
        }
    )
    a = _adapter()
    mcp = next(d for d in a.get_harness_manifest() if d.kind == "mcp_resource")
    res = mcp.probe_fn()
    assert res.success is False
    assert res.error


def test_get_introspection_report_is_none():
    assert _adapter().get_introspection_report() is None


def test_close_is_idempotent():
    a = _adapter()
    a.close()
    a.close()


# --- Live integration (opt-in) ----------------------------------------


@pytest.mark.skipif(not _LIVE, reason="set MEMPALACE_SERVER_LIVE=1 to run against a live Go server")
def test_live_ingest_search_reset_roundtrip():
    """End-to-end against a real server: ingest → search → snapshot → reset.

    Uses an isolated ``sme_selftest`` wing so it never touches other data,
    and always resets it in a ``finally``. Skips cleanly (not fails) if the
    server is unreachable even when the opt-in flag is set."""
    a = MemPalaceServerAdapter(wing="sme_selftest", reset_before_ingest=True)
    mcp = next(d for d in a.get_harness_manifest() if d.kind == "mcp_resource")
    if not mcp.probe_fn().success:
        pytest.skip("Go MemPalace server not reachable")
    try:
        res = a.ingest_corpus(
            [
                {"id": "live-1", "text": "The capital of France is Paris.", "room": "geo-1"},
                {"id": "live-2", "text": "The Eiffel Tower stands in Paris.", "room": "geo-2"},
            ]
        )
        assert res["entities_created"] >= 1, res
        q = a.query("Where is the Eiffel Tower?", n_results=5)
        # Either we retrieved something, or the server honestly reports none;
        # both are valid non-error outcomes for a smoke test.
        assert q.error in (None, "NO_RESULTS"), q.error
        entities, edges = a.get_graph_snapshot()
        assert isinstance(entities, list) and isinstance(edges, list)
    finally:
        a.reset(wing="sme_selftest")


@pytest.mark.skipif(not _LIVE, reason="set MEMPALACE_SERVER_LIVE=1 to run against a live Go server")
def test_live_kg_snapshot_reads_real_entity_graph():
    """Seed the AGE entity graph explicitly (kg_add_entity/relation), then
    confirm get_graph_snapshot reads it back with basis='kg'. Proves the
    real-KG path end-to-end (the server does no auto-extraction, so we must
    populate the KG ourselves to exercise it). Cleans up its own entities."""
    a = MemPalaceServerAdapter()
    mcp = next(d for d in a.get_harness_manifest() if d.kind == "mcp_resource")
    if not mcp.probe_fn().success:
        pytest.skip("Go MemPalace server not reachable")
    seeded = a._mcp_call(
        "mempalace_kg_add_entity", {"name": "SME_KGProbe_Sarah", "entity_type": "person"}
    )
    if seeded is None:
        pytest.skip("AGE entity graph not available on this server")
    try:
        a._mcp_call(
            "mempalace_kg_add_entity", {"name": "SME_KGProbe_Acme", "entity_type": "organization"}
        )
        a._mcp_call(
            "mempalace_kg_add_relation",
            {
                "from_entity": "SME_KGProbe_Sarah",
                "relation_type": "works_at",
                "to_entity": "SME_KGProbe_Acme",
            },
        )
        entities, edges = a.get_graph_snapshot()
        assert a._graph_basis == "kg", a._graph_basis
        names = {e.name for e in entities}
        assert "SME_KGProbe_Sarah" in names and "SME_KGProbe_Acme" in names
        assert any(
            e.source_id == "kg:SME_KGProbe_Sarah" and e.target_id == "kg:SME_KGProbe_Acme"
            for e in edges
        )
    finally:
        a._mcp_call("mempalace_kg_delete_entity", {"name": "SME_KGProbe_Sarah"})
        a._mcp_call("mempalace_kg_delete_entity", {"name": "SME_KGProbe_Acme"})
