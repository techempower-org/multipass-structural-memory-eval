"""Tests for sme.adapters.mempalace_daemon — HTTP-mocked, no live daemon."""

from __future__ import annotations

import json
import urllib.error

import pytest

from sme.adapters.mempalace_daemon import MemPalaceDaemonAdapter


# --- Auth resolution -------------------------------------------------


def test_auth_explicit_kwargs_win(monkeypatch, tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("PALACE_API_KEY=from-file\nPALACE_DAEMON_URL=http://from-file\n")
    monkeypatch.setenv("PALACE_API_KEY", "from-env")
    monkeypatch.setenv("PALACE_DAEMON_URL", "http://from-env")

    a = MemPalaceDaemonAdapter(
        api_url="http://explicit",
        api_key="explicit-key",
        env_file=env_file,
    )
    assert a.api_url == "http://explicit"
    assert a.api_key == "explicit-key"


def test_auth_env_file_used_when_no_kwargs(monkeypatch, tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text(
        'PALACE_API_KEY="from-file"\nPALACE_DAEMON_URL=http://from-file:8085\n'
    )
    monkeypatch.delenv("PALACE_API_KEY", raising=False)
    monkeypatch.delenv("PALACE_DAEMON_URL", raising=False)

    a = MemPalaceDaemonAdapter(env_file=env_file)
    assert a.api_url == "http://from-file:8085"
    assert a.api_key == "from-file"


def test_auth_process_env_used_when_env_file_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PALACE_API_KEY", "from-env")
    monkeypatch.setenv("PALACE_DAEMON_URL", "http://from-env:8085")

    a = MemPalaceDaemonAdapter(env_file=tmp_path / "does-not-exist")
    assert a.api_url == "http://from-env:8085"
    assert a.api_key == "from-env"


def test_auth_raises_when_nothing_resolves(monkeypatch, tmp_path):
    monkeypatch.delenv("PALACE_API_KEY", raising=False)
    monkeypatch.delenv("PALACE_DAEMON_URL", raising=False)

    with pytest.raises(ValueError, match="api_url"):
        MemPalaceDaemonAdapter(env_file=tmp_path / "nope")


def test_auth_url_trailing_slash_is_stripped(monkeypatch, tmp_path):
    monkeypatch.delenv("PALACE_API_KEY", raising=False)
    monkeypatch.delenv("PALACE_DAEMON_URL", raising=False)
    a = MemPalaceDaemonAdapter(
        api_url="http://example/",
        api_key="k",
        env_file=tmp_path / "nope",
    )
    assert a.api_url == "http://example"


# --- query() ---------------------------------------------------------


_OK_ENVELOPE = {
    "query": "memory",
    "filters": {"wing": None, "room": None},
    "total_before_filter": 3,
    "available_in_scope": 150811,
    "warnings": [],
    "results": [
        {
            "text": "first chunk text",
            "metadata": {
                "wing": "memorypalace",
                "room": "architecture",
                "source_file": "/path/to/notes.md",
            },
            "score": 0.91,
        },
        {
            "text": "second chunk",
            "metadata": {
                "wing": "memorypalace",
                "room": "diary",
                "source_file": "/path/to/diary.md",
            },
            "score": 0.84,
        },
    ],
}


def _adapter(monkeypatch, tmp_path, **kwargs):
    monkeypatch.delenv("PALACE_API_KEY", raising=False)
    monkeypatch.delenv("PALACE_DAEMON_URL", raising=False)
    defaults = dict(
        api_url="http://daemon",
        api_key="key",
        env_file=tmp_path / "no-env",
    )
    defaults.update(kwargs)
    return MemPalaceDaemonAdapter(**defaults)


def test_query_success_builds_context_string(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory")
    assert result.error is None
    assert "[1] [memorypalace/architecture]" in result.context_string
    assert "first chunk text" in result.context_string
    assert "[2] [memorypalace/diary]" in result.context_string
    assert "second chunk" in result.context_string
    # Source filename basenames, not full paths
    assert "notes.md" in result.context_string
    assert "/path/to/notes.md" not in result.context_string


def test_query_retrieved_entities_have_wing_room_score(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory")
    assert len(result.retrieved_entities) == 2
    e0 = result.retrieved_entities[0]
    assert e0.entity_type == "drawer:architecture"
    assert e0.properties["wing"] == "memorypalace"
    assert e0.properties["room"] == "architecture"
    assert e0.properties["score"] == 0.91


def test_query_retrieval_path_includes_kind_and_counts(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory")
    path_str = "; ".join(result.retrieval_path)
    assert "kind=content" in path_str
    assert "available_in_scope=150811" in path_str
    assert "total_before_filter=3" in path_str


def test_query_kind_kwarg_overrides_default(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=all": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path)  # kind defaults to "content"
    result = a.query("memory", kind="all")
    assert "kind=all" in "; ".join(result.retrieval_path)


def test_query_n_results_threads_through_to_limit(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=12&kind=content": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory", n_results=12)
    assert result.error is None  # would AssertionError in fake_urlopen otherwise


def test_query_candidate_strategy_constructor_default(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """#57 — candidate_strategy on constructor flows to /search query param."""
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content&candidate_strategy=hybrid": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path, candidate_strategy="hybrid")
    result = a.query("memory")
    assert result.error is None  # fake_urlopen would AssertionError on URL mismatch


def test_query_candidate_strategy_per_call_overrides_default(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """#57 — per-call kwarg overrides the constructor default."""
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content&candidate_strategy=union": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path, candidate_strategy="hybrid")
    result = a.query("memory", candidate_strategy="union")
    assert result.error is None


def test_query_no_candidate_strategy_omitted_from_url(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """#57 — when neither ctor nor call supplies a strategy, the param is
    omitted so the daemon picks its default."""
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path)  # no candidate_strategy
    result = a.query("memory")
    assert result.error is None


def test_invalid_candidate_strategy_raises_in_ctor(monkeypatch, tmp_path):
    """#57 + Gemini PR #68 review: client-side validation guards against
    typos like 'hybird' silently falling back to daemon default."""
    with pytest.raises(ValueError, match="Invalid candidate_strategy"):
        _adapter(monkeypatch, tmp_path, candidate_strategy="hybird")


def test_invalid_candidate_strategy_raises_in_query(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """Same validation at the call site."""
    fake_urlopen_factory({})
    a = _adapter(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Invalid candidate_strategy"):
        a.query("memory", candidate_strategy="hybird")


def test_query_question_is_url_quoted(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        # spaces and ampersands must be quoted in the URL
        "GET http://daemon/search?q=hello+world+%26+more&limit=5&kind=content": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("hello world & more")
    assert result.error is None


# --- query() error paths --------------------------------------------


def test_query_warnings_emit_soft_error_with_results(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    envelope = {
        **_OK_ENVELOPE,
        "warnings": ["vector search unavailable: Error finding id"],
    }
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": envelope,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory")
    # Soft signal: error set, but context_string still populated
    assert result.error is not None
    assert result.error.startswith("WARN:")
    assert "vector search unavailable" in result.error
    assert "first chunk text" in result.context_string


def test_query_warnings_with_empty_results(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    envelope = {
        "query": "memory",
        "filters": {"wing": None, "room": None},
        "total_before_filter": 0,
        "available_in_scope": 150811,
        "warnings": ["vector search unavailable: Error finding id"],
        "results": [],
    }
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": envelope,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory")
    assert result.error.startswith("WARN:")
    assert result.context_string == ""


def test_query_no_results_returns_no_results(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    envelope = {**_OK_ENVELOPE, "results": [], "warnings": []}
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": envelope,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory")
    assert result.error == "NO_RESULTS"


def test_query_auth_error_returns_AUTH(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    err = urllib.error.HTTPError(
        "http://daemon/search", 401, "Unauthorized", {}, None
    )
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": err,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory")
    assert result.error.startswith("AUTH:")
    assert "401" in result.error


def test_query_5xx_returns_HTTP_error(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    err = urllib.error.HTTPError(
        "http://daemon/search", 500, "Server Error", {}, None
    )
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": err,
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory")
    assert result.error.startswith("HTTP 500")


def test_query_connection_refused_returns_CONNECTION(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": (
            urllib.error.URLError("Connection refused")
        ),
    })
    a = _adapter(monkeypatch, tmp_path)
    result = a.query("memory")
    assert result.error.startswith("CONNECTION:")


def test_query_sends_x_api_key_header(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    captured = {}

    def capture(req):
        captured["headers"] = dict(req.header_items())
        captured["url"] = req.full_url
        return _OK_ENVELOPE  # factory will wrap into a _FakeResponse

    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": capture,
    })
    a = _adapter(monkeypatch, tmp_path, api_key="my-secret")
    a.query("memory")
    # urllib normalises header names; check both casings
    api_key_value = (
        captured["headers"].get("X-api-key")
        or captured["headers"].get("X-Api-Key")
    )
    assert api_key_value == "my-secret"


# --- /search/age-fused POST endpoint (#45) --------------------------


_AGE_FUSED_ENVELOPE = {
    "query": "memory",
    "total_before_filter": 2,
    "available_in_scope": 4242,
    "warnings": [],
    # /search/age-fused returns flat per-hit fields (drawer_id, wing,
    # room, source_file at top level) instead of nesting them under
    # `metadata`. The adapter parsing must tolerate this shape.
    "results": [
        {
            "drawer_id": "drawer-1",
            "wing": "memorypalace",
            "room": "architecture",
            "source_file": "/path/to/notes.md",
            "text": "first chunk text",
            "score": 0.91,
        },
        {
            "drawer_id": "drawer-2",
            "wing": "memorypalace",
            "room": "diary",
            "source_file": "/path/to/diary.md",
            "text": "second chunk",
            "score": 0.84,
        },
    ],
}


def test_query_search_endpoint_age_fused_uses_post(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """#45 — alternate search endpoint (e.g. /search/age-fused) is a
    POST + JSON body with field name `query` instead of `q`. The fake
    urlopen factory keys on METHOD + URL, so registering only the POST
    route is enough to assert the adapter uses POST."""
    captured = {}

    def capture(req):
        import json as _json
        captured["method"] = req.get_method()
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = _json.loads(req.data.decode("utf-8"))
        return _AGE_FUSED_ENVELOPE

    fake_urlopen_factory({
        "POST http://daemon/search/age-fused": capture,
    })
    a = _adapter(
        monkeypatch, tmp_path, search_endpoint="/search/age-fused",
    )
    result = a.query("memory", wing="lme_q1")
    assert result.error is None
    # POST, not GET
    assert captured["method"] == "POST"
    # JSON body uses `query` (not `q`), `limit`, `wing`
    assert captured["payload"]["query"] == "memory"
    assert captured["payload"]["limit"] == 5
    assert captured["payload"]["wing"] == "lme_q1"
    # Content-Type set by _http_post
    ct = (
        captured["headers"].get("Content-type")
        or captured["headers"].get("Content-Type")
    )
    assert ct == "application/json"


def test_query_search_endpoint_age_fused_parses_flat_hits(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """#45 — /search/age-fused puts wing/room/source_file at the hit's
    top level (not under hit['metadata']). The hit-parsing path must
    tolerate both shapes."""
    fake_urlopen_factory({
        "POST http://daemon/search/age-fused": _AGE_FUSED_ENVELOPE,
    })
    a = _adapter(
        monkeypatch, tmp_path, search_endpoint="/search/age-fused",
    )
    result = a.query("memory")
    assert result.error is None
    # Context string reflects the flat per-hit metadata
    assert "[1] [memorypalace/architecture]" in result.context_string
    assert "first chunk text" in result.context_string
    assert "[2] [memorypalace/diary]" in result.context_string
    # Entity ids come from the daemon's flat `drawer_id` field
    assert len(result.retrieved_entities) == 2
    assert result.retrieved_entities[0].id == "drawer-1"
    assert result.retrieved_entities[0].properties["wing"] == "memorypalace"
    assert result.retrieved_entities[0].properties["room"] == "architecture"
    assert result.retrieved_entities[0].properties["source_file"] == "/path/to/notes.md"


def test_query_default_search_endpoint_still_uses_get(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """Default search_endpoint='/search' must keep the existing GET path
    so /search consumers are unaffected by the #45 plumbing."""
    fake_urlopen_factory({
        "GET http://daemon/search?q=memory&limit=5&kind=content": _OK_ENVELOPE,
    })
    a = _adapter(monkeypatch, tmp_path)  # default endpoint
    result = a.query("memory")
    assert result.error is None


def test_query_age_fused_http_error_translates(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """The new _http_post helper shares the same error-translation
    contract as _http_get — 5xx becomes 'HTTP NNN' error string."""
    err = urllib.error.HTTPError(
        "http://daemon/search/age-fused", 500, "Server Error", {}, None
    )
    fake_urlopen_factory({
        "POST http://daemon/search/age-fused": err,
    })
    a = _adapter(
        monkeypatch, tmp_path, search_endpoint="/search/age-fused",
    )
    result = a.query("memory")
    assert result.error.startswith("HTTP 500")


def test_query_age_fused_auth_error_returns_AUTH(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    err = urllib.error.HTTPError(
        "http://daemon/search/age-fused", 401, "Unauthorized", {}, None
    )
    fake_urlopen_factory({
        "POST http://daemon/search/age-fused": err,
    })
    a = _adapter(
        monkeypatch, tmp_path, search_endpoint="/search/age-fused",
    )
    result = a.query("memory")
    assert result.error.startswith("AUTH:")
    assert "401" in result.error


# --- get_graph_snapshot — /graph fast path --------------------------


_GRAPH_RESPONSE = {
    "wings": {
        "memorypalace": 427,
        "projects": 106183,
        "umbra": 82,
    },
    "rooms": [
        {"wing": "memorypalace", "rooms": {"architecture": 17, "diary": 235}},
        {"wing": "projects", "rooms": {"architecture": 9, "general": 100}},
        {"wing": "umbra", "rooms": {"diary": 12}},
    ],
    "tunnels": [
        {"room": "architecture", "wings": ["memorypalace", "projects"]},
        {"room": "diary", "wings": ["memorypalace", "umbra"]},
    ],
    "kg_entities": [
        {"id": "e1", "name": "Multipass", "type": "concept", "properties": {}}
    ],
    "kg_triples": [
        {
            "subject": "e1",
            "predicate": "described_by",
            "object": "e1",
            "valid_from": "2026-04-25",
            "valid_to": None,
            "confidence": 1.0,
            "source_file": "README.md",
        }
    ],
    "kg_stats": {"entities": 1, "triples": 1},
}


def test_snapshot_graph_endpoint_creates_wing_entities(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/graph": _GRAPH_RESPONSE,
    })
    a = _adapter(monkeypatch, tmp_path)
    entities, edges = a.get_graph_snapshot()

    wing_entities = [e for e in entities if e.entity_type == "wing"]
    assert {e.name for e in wing_entities} == {"memorypalace", "projects", "umbra"}


def test_snapshot_graph_endpoint_creates_room_entities_with_wings(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/graph": _GRAPH_RESPONSE,
    })
    a = _adapter(monkeypatch, tmp_path)
    entities, _ = a.get_graph_snapshot()

    rooms_by_name = {e.name: e for e in entities if e.id.startswith("room:")}
    assert "architecture" in rooms_by_name
    assert sorted(rooms_by_name["architecture"].properties["wings"]) == [
        "memorypalace",
        "projects",
    ]
    # 'general' is a catch-all and should be skipped, mirroring the
    # existing direct adapter's filter.
    assert "general" not in rooms_by_name


def test_snapshot_graph_endpoint_member_of_edges(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/graph": _GRAPH_RESPONSE,
    })
    a = _adapter(monkeypatch, tmp_path)
    _, edges = a.get_graph_snapshot()

    member_of = [e for e in edges if e.edge_type == "member_of"]
    pairs = {(e.source_id, e.target_id) for e in member_of}
    assert ("room:architecture", "wing:memorypalace") in pairs
    assert ("room:architecture", "wing:projects") in pairs
    assert ("room:diary", "wing:memorypalace") in pairs


def test_snapshot_graph_endpoint_tunnel_edges(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/graph": _GRAPH_RESPONSE,
    })
    a = _adapter(monkeypatch, tmp_path)
    _, edges = a.get_graph_snapshot()
    tunnels = [e for e in edges if e.edge_type == "tunnel"]
    pairs = {
        tuple(sorted([e.source_id, e.target_id]))
        for e in tunnels
    }
    # architecture connects memorypalace<->projects
    assert ("wing:memorypalace", "wing:projects") in pairs
    # diary connects memorypalace<->umbra
    assert ("wing:memorypalace", "wing:umbra") in pairs


def test_snapshot_graph_endpoint_kg_entities_and_triples(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/graph": _GRAPH_RESPONSE,
    })
    a = _adapter(monkeypatch, tmp_path)
    entities, edges = a.get_graph_snapshot()
    kg_ents = [e for e in entities if e.id.startswith("kg:")]
    assert len(kg_ents) == 1
    assert kg_ents[0].name == "Multipass"

    kg_edges = [e for e in edges if e.source_id.startswith("kg:")]
    assert len(kg_edges) == 1
    assert kg_edges[0].edge_type == "described_by"


# --- get_graph_snapshot — MCP fallback ------------------------------


def _mcp_envelope(payload) -> dict:
    """Build an MCP tools/call response envelope wrapping a JSON payload."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
    }


def _mcp_request_router(routes_by_tool: dict):
    """Returns a callable that fake_urlopen_factory can hand back as the
    response for ``POST http://daemon/mcp``.

    Inspects the request body to dispatch on (tool_name, arguments) and
    returns the matching MCP envelope. Unknown tools raise AssertionError.
    """
    def _route(req, *, _routes=routes_by_tool):
        body = req.data
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        rpc = json.loads(body)
        params = rpc.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        # Per-wing list_rooms: key on tool:wing
        if name == "mempalace_list_rooms":
            key = f"mempalace_list_rooms:{args.get('wing')}"
        else:
            key = name
        if key not in _routes:
            raise AssertionError(f"unrouted MCP call: {key}")
        result = _routes[key]
        if isinstance(result, Exception):
            raise result
        return _mcp_envelope(result)
    return _route


def test_snapshot_falls_back_to_mcp_on_404(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/graph": (
            urllib.error.HTTPError(
                "http://daemon/graph", 404, "Not Found", {}, None
            )
        ),
        "POST http://daemon/mcp": _mcp_request_router({
            "mempalace_list_wings": {
                "wings": {"memorypalace": 427, "umbra": 82}
            },
            "mempalace_list_tunnels": [
                {"room": "diary", "wings": ["memorypalace", "umbra"]}
            ],
            "mempalace_list_rooms:memorypalace": {
                "wing": "memorypalace",
                "rooms": {"diary": 235, "architecture": 17},
            },
            "mempalace_list_rooms:umbra": {
                "wing": "umbra",
                "rooms": {"diary": 12},
            },
        }),
    })
    a = _adapter(monkeypatch, tmp_path)
    entities, edges = a.get_graph_snapshot()
    wing_names = {e.name for e in entities if e.entity_type == "wing"}
    assert wing_names == {"memorypalace", "umbra"}
    tunnels = [e for e in edges if e.edge_type == "tunnel"]
    assert len(tunnels) == 1
    pair = tuple(sorted([tunnels[0].source_id, tunnels[0].target_id]))
    assert pair == ("wing:memorypalace", "wing:umbra")


def test_snapshot_force_mcp_with_prefer_graph_false(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "POST http://daemon/mcp": _mcp_request_router({
            "mempalace_list_wings": {"wings": {"only": 1}},
            "mempalace_list_tunnels": [],
            "mempalace_list_rooms:only": {"wing": "only", "rooms": {}},
        }),
    })
    a = _adapter(
        monkeypatch, tmp_path, prefer_graph_endpoint=False
    )
    entities, _ = a.get_graph_snapshot()
    # Should NOT have hit /graph at all (no route registered for it)
    wing_names = {e.name for e in entities if e.entity_type == "wing"}
    assert wing_names == {"only"}


def test_snapshot_partial_on_list_rooms_failure(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """If list_rooms fails for one wing, the snapshot still returns
    every other wing's data."""
    fake_urlopen_factory({
        "GET http://daemon/graph": urllib.error.HTTPError(
            "http://daemon/graph", 404, "Not Found", {}, None
        ),
        "POST http://daemon/mcp": _mcp_request_router({
            "mempalace_list_wings": {"wings": {"good": 1, "bad": 1}},
            "mempalace_list_tunnels": [],
            "mempalace_list_rooms:good": {"wing": "good", "rooms": {"r1": 5}},
            # 'bad' wing's list_rooms raises
            "mempalace_list_rooms:bad": urllib.error.HTTPError(
                "http://daemon/mcp", 500, "tool error", {}, None
            ),
        }),
    })
    a = _adapter(monkeypatch, tmp_path)
    entities, _ = a.get_graph_snapshot()
    room_names = {e.name for e in entities if e.id.startswith("room:")}
    assert "r1" in room_names  # the good wing's room is present


# --- ontology + ingest -----------------------------------------------


def test_get_ontology_source_matches_existing_adapter(monkeypatch, tmp_path):
    a = _adapter(monkeypatch, tmp_path)
    ont = a.get_ontology_source()
    assert ont["type"] == "readme"
    declared = {entry["kind"] for entry in ont["schema"]}
    assert "structural" in declared
    assert "hall_vocabulary" in declared


def test_ingest_corpus_raises_with_helpful_message(monkeypatch, tmp_path):
    a = _adapter(monkeypatch, tmp_path)
    with pytest.raises(NotImplementedError, match="diagnostic-only"):
        a.ingest_corpus([])


# --- introspection (GET /ontology) -----------------------------------


_ONTOLOGY_BODY = {
    "declared": {
        "entity_types": ["wing", "hall", "room", "drawer", "closet", "tunnel"],
        "edge_types": ["hall", "tunnel", "member_of"],
    },
    "effective": {
        "edge_types": ["MENTIONS"],
        "entity_kinds": {"PROPER_NOUN": 12000, "TECH_IDENT": 7000},
        "entities": 267519,
        "triples": 0,
        "mentions": 5580000,
    },
    "drift": {
        "declared_edge_types_present": [],
        "declared_edge_types_absent": ["hall", "tunnel", "member_of"],
        "entity_kinds_undeclared": ["PROPER_NOUN", "TECH_IDENT"],
        "structure_claim": "hierarchical",
        "structure_observed": "not_computed",
        "drift_score": 1.0,
    },
}


def test_get_introspection_report_returns_ontology_payload(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """The adapter fetches GET /ontology and returns the daemon's
    declared-vs-effective drift report verbatim."""
    fake_urlopen_factory({
        "GET http://daemon:8085/ontology": _ONTOLOGY_BODY,
    })
    a = _adapter(monkeypatch, tmp_path, api_url="http://daemon:8085")
    report = a.get_introspection_report()
    assert report is not None
    assert report["drift"]["drift_score"] == 1.0
    assert report["effective"]["entity_kinds"]["PROPER_NOUN"] == 12000


def test_get_introspection_report_none_on_404(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """An older daemon without /ontology returns 404 → None, preserving the
    introspection-0.0 baseline (system scores like any other without the
    capability)."""
    fake_urlopen_factory({
        "GET http://daemon:8085/ontology": urllib.error.HTTPError(
            "http://daemon:8085/ontology", 404, "Not Found", {}, None
        ),
    })
    a = _adapter(monkeypatch, tmp_path, api_url="http://daemon:8085")
    assert a.get_introspection_report() is None


def test_get_introspection_report_none_on_503(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """Under the chroma backend / unreachable AGE the daemon answers 503 →
    None, so the chroma access path scores introspection 0.0 honestly."""
    fake_urlopen_factory({
        "GET http://daemon:8085/ontology": urllib.error.HTTPError(
            "http://daemon:8085/ontology", 503, "Service Unavailable", {}, None
        ),
    })
    a = _adapter(monkeypatch, tmp_path, api_url="http://daemon:8085")
    assert a.get_introspection_report() is None


def test_get_introspection_report_none_on_malformed_body(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """A 200 response missing the 'drift' key isn't a usable introspection
    report → None rather than a half-credited score."""
    fake_urlopen_factory({
        "GET http://daemon:8085/ontology": {"declared": {}, "effective": {}},
    })
    a = _adapter(monkeypatch, tmp_path, api_url="http://daemon:8085")
    assert a.get_introspection_report() is None


# --- #147 real-KG structural measurement -----------------------------


_GRAPH_BODY = {
    "wings": {"a": 2, "b": 1},
    "rooms": [
        {"wing": "a", "rooms": {"shared": 2}},
        {"wing": "b", "rooms": {"shared": 1}},
    ],
    "tunnels": [{"room": "shared", "wings": ["a", "b"]}],
    "kg_entities": [
        {"id": "Alice", "name": "Alice", "type": "entity"},
        {"id": "Acme", "name": "Acme", "type": "entity"},
    ],
    "kg_triples": [
        {"subject": "Alice", "object": "Acme", "predicate": "works_at"},
    ],
}


def test_graph_kg_only_excludes_structural_edges(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """With graph_kg_only=True the snapshot is the real KG only — no tunnel /
    member_of structural edges that otherwise swamp Cat 4/5/8 topology."""
    fake_urlopen_factory({"GET http://daemon/graph": _GRAPH_BODY})
    a = _adapter(monkeypatch, tmp_path, graph_kg_only=True)
    entities, edges = a.get_graph_snapshot()
    assert all(e.id.startswith("kg:") for e in entities)
    assert {ed.edge_type for ed in edges} == {"works_at"}
    assert all(ed.edge_type not in ("tunnel", "member_of") for ed in edges)


def test_graph_kg_only_false_keeps_full_projection(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """Default (graph_kg_only=False) is the full snapshot — structural edges
    plus the KG layer. Guards the default path against the #147 flag."""
    fake_urlopen_factory({"GET http://daemon/graph": _GRAPH_BODY})
    a = _adapter(monkeypatch, tmp_path)  # default False
    _, edges = a.get_graph_snapshot()
    edtypes = {ed.edge_type for ed in edges}
    assert "tunnel" in edtypes
    assert "member_of" in edtypes
    assert "works_at" in edtypes


def test_graph_limit_threads_into_graph_url(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """graph_limit raises the /graph KG sample cap by adding ?limit=N — so the
    daemon's CTE-bounded RELATION read returns a representative sample (#147)."""
    fake_urlopen_factory({"GET http://daemon/graph?limit=5000": _GRAPH_BODY})
    a = _adapter(monkeypatch, tmp_path, graph_kg_only=True, graph_limit=5000)
    _, edges = a.get_graph_snapshot()
    # The mock only registers the ?limit=5000 URL; a bare /graph would raise
    # AssertionError, so reaching here proves the limit was applied.
    assert {ed.edge_type for ed in edges} == {"works_at"}


# --- #147 follow-up: exact RELATION distribution for Cat 4 -----------


def test_get_edge_type_distribution_aggregates_cypher_rows(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """POST /cypher returns {rows:[{rt,n}]} from the full RELATION GROUP BY;
    the adapter projects it to an exact {relation_type: count} map. agtype
    quoting is stripped."""
    fake_urlopen_factory({
        "POST http://daemon/cypher": {
            "graph": "mempalace_kg",
            "rows": [
                {"rt": "other", "n": 1057935},
                {"rt": "contains", "n": 348375},
                {"rt": '"is_a"', "n": 41975},  # quoted agtype string
            ],
            "count": 3,
        },
    })
    a = _adapter(monkeypatch, tmp_path)
    dist = a.get_edge_type_distribution()
    assert dist == {"other": 1057935, "contains": 348375, "is_a": 41975}


def test_get_edge_type_distribution_none_when_cypher_unavailable(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """A 503 (chroma backend) / older daemon → None, so Cat 4 falls back to
    counting the sampled edges."""
    fake_urlopen_factory({
        "POST http://daemon/cypher": urllib.error.HTTPError(
            "http://daemon/cypher", 503, "Service Unavailable", {}, None
        ),
    })
    a = _adapter(monkeypatch, tmp_path)
    assert a.get_edge_type_distribution() is None


def test_get_edge_type_distribution_none_on_malformed(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """A 200 response without a usable rows list → None rather than a bogus
    empty distribution."""
    fake_urlopen_factory({
        "POST http://daemon/cypher": {"graph": "mempalace_kg", "count": 0},
    })
    a = _adapter(monkeypatch, tmp_path)
    assert a.get_edge_type_distribution() is None


# --- #147 Cat 5 MENTIONS context ------------------------------------


def test_get_mentions_context_reads_kg_stats(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """Cat 5's MENTIONS context comes straight off /graph kg_stats — entity /
    RELATION / MENTIONS totals, no extra OOM-prone walk."""
    fake_urlopen_factory({
        "GET http://daemon/graph": {
            "wings": {}, "rooms": [], "tunnels": [],
            "kg_entities": [], "kg_triples": [], "kg_mentions": [],
            "kg_stats": {"entities": 1156232, "triples": 1921600, "mentions": 6691737},
        },
    })
    a = _adapter(monkeypatch, tmp_path)
    ctx = a.get_mentions_context()
    assert ctx == {
        "entities": 1156232,
        "relation_edges": 1921600,
        "mentions_edges": 6691737,
    }


def test_get_mentions_context_threads_graph_limit(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """When graph_limit is set the context read uses ?limit=N too (same /graph
    call the snapshot uses)."""
    fake_urlopen_factory({
        "GET http://daemon/graph?limit=5000": {
            "kg_stats": {"entities": 10, "triples": 20, "mentions": 30},
        },
    })
    a = _adapter(monkeypatch, tmp_path, graph_kg_only=True, graph_limit=5000)
    ctx = a.get_mentions_context()
    assert ctx["mentions_edges"] == 30


def test_get_mentions_context_none_when_graph_unavailable(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/graph": urllib.error.HTTPError(
            "http://daemon/graph", 503, "Service Unavailable", {}, None
        ),
    })
    a = _adapter(monkeypatch, tmp_path)
    assert a.get_mentions_context() is None


def test_get_mentions_context_none_when_no_kg_stats(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """A /graph payload without kg_stats.mentions → None (don't fabricate)."""
    fake_urlopen_factory({
        "GET http://daemon/graph": {"wings": {}, "kg_stats": {}},
    })
    a = _adapter(monkeypatch, tmp_path)
    assert a.get_mentions_context() is None


# --- #152 exact full-graph Cat 5/8 stats (GET /graph/structural-stats) ----


_STRUCTURAL_STATS = {
    "entities": 1156241,
    "edges": 1921600,
    "component_count": 50000,
    "largest_component_size": 800000,
    "largest_component_fraction": 0.69,
    "isolate_count": 12000,
    "component_size_histogram": [800000, 1200],
    "modularity": 0.72,
    "modularity_communities": 41,
    "modularity_note": None,
}


def test_get_structural_stats_reads_cached_daemon_stats(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """The adapter GETs the daemon's cached exact full-graph stats verbatim."""
    fake_urlopen_factory({
        "GET http://daemon/graph/structural-stats": _STRUCTURAL_STATS,
    })
    a = _adapter(monkeypatch, tmp_path)
    s = a.get_structural_stats()
    assert s["modularity"] == 0.72
    assert s["largest_component_size"] == 800000


def test_get_structural_stats_none_on_404_not_yet_computed(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """404 (daemon hasn't run the gated POST) → None; Cat 5/8 fall back to the
    sampled snapshot. The adapter NEVER POSTs to trigger the heavy compute."""
    fake_urlopen_factory({
        "GET http://daemon/graph/structural-stats": urllib.error.HTTPError(
            "http://daemon/graph/structural-stats", 404, "Not Found", {}, None
        ),
    })
    a = _adapter(monkeypatch, tmp_path)
    assert a.get_structural_stats() is None


def test_get_structural_stats_none_on_503(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    fake_urlopen_factory({
        "GET http://daemon/graph/structural-stats": urllib.error.HTTPError(
            "http://daemon/graph/structural-stats", 503, "Service Unavailable", {}, None
        ),
    })
    a = _adapter(monkeypatch, tmp_path)
    assert a.get_structural_stats() is None


def test_get_structural_stats_none_on_malformed(
    monkeypatch, tmp_path, fake_urlopen_factory
):
    """A 200 without component_count isn't a usable stats payload → None."""
    fake_urlopen_factory({
        "GET http://daemon/graph/structural-stats": {"modularity": 0.5},
    })
    a = _adapter(monkeypatch, tmp_path)
    assert a.get_structural_stats() is None
