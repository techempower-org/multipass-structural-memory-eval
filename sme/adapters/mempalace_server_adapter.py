"""SME adapter for the Go MemPalace server (sefodo26's reimplementation).

Talks to the Go rewrite of the MemPalace server — upstream
``github.com/sefodo26/mempalace-server``, our fork
``techempower-org/mempalace-server`` — over its optional REST/JSON API
(``/mp/api/v1``, bearer auth). The REST layer is a thin wrapper over the
exact same storage/validation logic the MCP endpoint uses, so measuring
through it measures the server itself; it also matches the shape the Go
project's own benchmark harness uses, keeping SME numbers comparable to
that project's published R@K figures.

Enable the REST API on the server with ``ENABLE_REST_API=true`` (on by
default in the project's ``docker-compose.yml``). The always-on MCP
endpoint (``POST /mp/mcp``) is exposed here only as a Category 9 harness
surface (``get_harness_manifest``), not as the ingest/query transport.

Config (constructor arg → env var → docker-compose default):
    api_url : MEMPALACE_SERVER_URL      → http://localhost:8000
    api_key : MEMPALACE_SERVER_API_KEY  → local-dev-key-change-me
    tenant  : MEMPALACE_SERVER_TENANT   → default

Tenancy note: the Go server's ``MEMPALACE_TENANT_ID`` is **server-level**
configuration, not a per-request header — the server code exposes no
tenant header. This adapter therefore cannot switch tenant per request;
``tenant`` is accepted for parity/labelling only (surfaced in the
retrieval path). Isolation between corpora is achieved by ``reset()``
(delete-all, optionally wing-scoped), mirroring the Go project's own
benchmark ``reset_store()``.

Two server behaviours the ingest path accounts for:
  * Deterministic drawer id ``sha256(wing/room/content[:500])[:16]`` with
    idempotent add — re-adding the same (wing, room, content-prefix)
    stores nothing and returns ``reason=already_exists``. Ingest keys the
    room off the row's ``session_id``/``id`` so distinct documents don't
    collide.
  * Pure bullet lists are split server-side into one drawer per bullet;
    ``add_drawer`` then returns ``bullets_stored``/``bullets_total``
    instead of a single ``drawer_id``. The ingest counter handles both.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from sme.adapters._graph_mapping import project_graph
from sme.adapters.base import (
    Edge,
    Entity,
    HarnessDescriptor,
    ProbeResult,
    QueryResult,
    SMEAdapter,
    annotate_superseded_edges,
)

log = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_API_KEY = "local-dev-key-change-me"  # docker-compose default; override in prod
DEFAULT_TENANT = "default"
DEFAULT_WING = "sme"
DEFAULT_TIMEOUT = 60.0
REST_BASE = "/mp/api/v1"
MCP_ENDPOINT = "/mp/mcp"  # stateless JSON-RPC (no initialize/session needed)
DEFAULT_KG_ENTITY_LIMIT = 100  # server caps kg_search_entities at 100
_LIST_PAGE = 100  # server clamps list limit to <=100
_RESET_MAX_PAGES = 100_000  # safety cap so a stuck delete can't spin forever


class MemPalaceServerAdapter(SMEAdapter):
    """SMEAdapter against a running Go MemPalace server's REST API.

    Construction never connects and never raises for missing config —
    unset values fall back to env vars then the docker-compose defaults,
    so a zero-config adapter points at a fresh ``docker compose up``. The
    first network call happens in ``ingest_corpus``/``query``/
    ``get_graph_snapshot``/``reset``.

    Args:
        api_url: Base URL (trailing slash stripped). Default resolves via
            ``MEMPALACE_SERVER_URL`` then ``http://localhost:8000``.
        api_key: Bearer token. Default resolves via
            ``MEMPALACE_SERVER_API_KEY`` then the docker-compose dev key.
        tenant: Informational only (server-level, not per-request).
        wing: Default wing for ingested drawers when a row omits one.
        room: Default room for ingested drawers. ``None`` (default) keys
            the room off each row's ``session_id``/``id`` for uniqueness,
            falling back to ``"general"``.
        n_results: Default top-K for ``query()``.
        max_distance: Optional cosine-distance ceiling for ``/search``.
            ``None`` (default) omits the field so the server applies its
            own default (1.5), matching the reference benchmark. ``0.0``
            disables the filter server-side.
        api_timeout: Per-request HTTP timeout (seconds).
        reset_before_ingest: When True (default), ``ingest_corpus`` wipes
            the configured wing before loading — the per-question-haystack
            isolation pattern the retrieval benchmarks rely on.
        read_only: Accepted for CLI parity; ignored (ingest/reset mutate).
    """

    def __init__(
        self,
        *,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant: Optional[str] = None,
        wing: str = DEFAULT_WING,
        room: Optional[str] = None,
        n_results: int = 5,
        max_distance: Optional[float] = None,
        api_timeout: float = DEFAULT_TIMEOUT,
        reset_before_ingest: bool = True,
        kg_entity_limit: int = DEFAULT_KG_ENTITY_LIMIT,
        read_only: bool = False,
    ) -> None:
        resolved_url = (
            api_url
            or os.environ.get("MEMPALACE_SERVER_URL")
            or DEFAULT_API_URL
        )
        resolved_key = (
            api_key
            or os.environ.get("MEMPALACE_SERVER_API_KEY")
            or DEFAULT_API_KEY
        )
        resolved_tenant = (
            tenant
            or os.environ.get("MEMPALACE_SERVER_TENANT")
            or DEFAULT_TENANT
        )
        self.api_url = resolved_url.rstrip("/")
        self.api_key = resolved_key
        self.tenant = resolved_tenant
        self.wing = wing
        self.room = room
        self.n_results = n_results
        self.max_distance = max_distance
        self.api_timeout = api_timeout
        self.reset_before_ingest = reset_before_ingest
        self.kg_entity_limit = kg_entity_limit
        # Which substrate the last get_graph_snapshot returned: "kg" (real
        # AGE entity graph), "taxonomy" (wing/room scaffold fallback),
        # "empty", or "unknown" (not yet called). Surfaced in the harness
        # manifest so a run's structural basis is attributable.
        self._graph_basis = "unknown"

    # --- SMEAdapter required methods ----------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """Load ``corpus`` into the server as drawers, one POST per row.

        Each row maps content ← ``document``|``content``|``text``, wing ←
        row/``metadata`` ``wing`` or the configured default, room ←
        row/``metadata`` ``room`` or ``session_id``/``id`` or the
        configured default or ``"general"``, and ``source_file`` ←
        ``source_file``|``source``|``id``. Blank content is skipped.

        Never raises on transport failure — HTTP/connection errors are
        recorded in ``errors`` and the 4-key result dict is always
        returned, so the contract suite can exercise it against an
        unreachable server.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if self.reset_before_ingest:
            removed = self.reset(wing=self.wing, _errors=errors)
            warnings.append(f"reset wing={self.wing!r} before ingest: {removed} removed")

        created = 0
        existed = 0
        for row in corpus:
            content = (
                row.get("document")
                or row.get("content")
                or row.get("text")
                or ""
            )
            if not str(content).strip():
                continue
            meta = row.get("metadata") or {}
            wing = row.get("wing") or meta.get("wing") or self.wing
            room = (
                row.get("room")
                or meta.get("room")
                or row.get("session_id")
                or meta.get("session_id")
                or row.get("id")
                or self.room
                or "general"
            )
            source_file = (
                row.get("source_file")
                or row.get("source")
                or (str(row.get("id")) if row.get("id") is not None else "")
            )
            payload = {
                "wing": str(wing),
                "room": str(room),
                "content": str(content),
                "source_file": str(source_file),
                "added_by": "sme",
            }
            body = self._http("POST", f"{REST_BASE}/drawers", payload)
            if isinstance(body, QueryResult):
                errors.append(f"add failed ({room}): {body.error}")
                continue
            if not isinstance(body, dict):
                errors.append(f"add returned non-dict for room={room}")
                continue
            if "bullets_stored" in body:
                created += int(body.get("bullets_stored") or 0)
            elif body.get("reason") == "already_exists":
                existed += 1
            elif body.get("success"):
                created += 1
            else:
                errors.append(f"add unsuccessful ({room}): {body}")

        if existed:
            warnings.append(f"{existed} drawer(s) already_exists (skipped)")

        return {
            "entities_created": created,
            "edges_created": 0,
            "errors": errors,
            "warnings": warnings,
        }

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        wing: Optional[str] = None,
        room: Optional[str] = None,
        max_distance: Optional[float] = None,
    ) -> QueryResult:
        """Semantic search via ``POST /mp/api/v1/search``.

        Populates ``context_string`` with the ranked hits (the exact text
        SME tokenises for Cat 7). ``n_results`` defaults to the adapter's
        configured ``n_results``. Returns a ``QueryResult`` with ``error``
        set — never raises — on transport failure or no results.
        """
        k = n_results if n_results is not None else self.n_results
        payload: dict[str, Any] = {"query": question, "limit": k}
        if wing:
            payload["wing"] = wing
        if room:
            payload["room"] = room
        chosen_dist = max_distance if max_distance is not None else self.max_distance
        if chosen_dist is not None:
            payload["max_distance"] = chosen_dist

        body = self._http("POST", f"{REST_BASE}/search", payload)
        if isinstance(body, QueryResult):
            return body

        results = body.get("results") if isinstance(body, dict) else None
        retrieval_path = [
            f"search:tenant={self.tenant}",
            f"wing={wing or '*'}",
            f"room={room or '*'}",
            f"k={k}",
        ]
        if not results:
            return QueryResult(
                answer="",
                context_string="",
                error="NO_RESULTS",
                retrieval_path=retrieval_path,
            )

        context_parts: list[str] = []
        retrieved: list[Entity] = []
        for i, hit in enumerate(results):
            hit = hit or {}
            content = hit.get("content") or ""
            meta = hit.get("metadata") or {}
            wing_name = hit.get("wing") or meta.get("wing") or "?"
            room_name = hit.get("room") or meta.get("room") or "?"
            source_file = hit.get("source_file") or meta.get("source_file") or ""
            source_label = Path(source_file).name if source_file else ""
            raw_id = hit.get("drawer_id") or hit.get("id")
            drawer_id = str(raw_id) if raw_id is not None else f"drawer_hit:{i}"
            label = source_label or drawer_id
            context_parts.append(
                f"[{i + 1}] [{wing_name}/{room_name}] {label}\n{content}"
            )
            retrieved.append(
                Entity(
                    id=drawer_id,
                    name=label,
                    entity_type=f"drawer:{room_name}",
                    properties={
                        "_table": "mempalace_server_hit",
                        "wing": wing_name,
                        "room": room_name,
                        "similarity": hit.get("similarity"),
                        "distance": hit.get("distance"),
                        "source_file": source_file,
                        "rank": i + 1,
                    },
                )
            )

        context_string = "\n\n".join(context_parts)
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=retrieved,
            retrieval_path=retrieval_path,
        )

    def get_graph_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        """Return the server's real knowledge graph, falling back to the
        wing/room taxonomy scaffold.

        Primary basis is the **AGE entity graph** (entities + typed
        relations), enumerated over MCP — the honest analog of an emergent
        graph and comparable to other adapters' KG snapshots. When that
        graph is empty or unavailable (older server / AGE not installed),
        the wing/room ``/taxonomy`` projection is used instead so the
        structural categories still have a substrate. ``self._graph_basis``
        records which was returned (``kg`` / ``taxonomy`` / ``empty``) and
        is surfaced in the harness manifest for run attribution.

        Note: the Go server performs NO automatic entity extraction from
        drawer content — its KG is populated only by explicit
        ``kg_add_entity`` / ``kg_add_relation`` calls. So a drawer-only
        ingest yields an empty KG here (→ taxonomy basis); a populated KG
        is read faithfully.

        Never raises — returns ``([], [])`` if both bases are unavailable.
        """
        kg = self._kg_snapshot_via_mcp()
        if kg is not None and kg[0]:
            self._graph_basis = "kg"
            return kg
        tax = self._taxonomy_snapshot()
        self._graph_basis = "taxonomy" if tax[0] else "empty"
        return tax

    def _kg_snapshot_via_mcp(self) -> Optional[tuple[list[Entity], list[Edge]]]:
        """Enumerate the AGE entity graph over MCP → ``(entities, edges)``.

        ``kg_search_entities`` with an empty query lists all entities (the
        server's Cypher does ``name CONTAINS ''`` → matches everything, cap
        ``kg_entity_limit`` ≤ 100); ``kg_get_entity`` per entity yields its
        in/out relations, deduplicated across endpoints. Returns ``None``
        when the KG layer is unavailable (AGE not installed / transport
        error) so the caller falls back, or ``([], [])`` when the graph is
        reachable but empty.
        """
        search = self._mcp_call(
            "mempalace_kg_search_entities",
            {"query": "", "limit": self.kg_entity_limit},
        )
        if search is None:
            return None  # AGE unavailable / transport error → caller falls back
        ent_list = search.get("entities") if isinstance(search, dict) else None
        if not ent_list:
            return [], []  # reachable but empty

        entities: list[Entity] = []
        present: set[str] = set()
        names: list[str] = []
        for e in ent_list:
            if not isinstance(e, dict):
                continue
            name = e.get("name")
            if not name:
                continue
            nid = f"kg:{name}"
            if nid in present:
                continue
            present.add(nid)
            names.append(name)
            entities.append(
                Entity(
                    id=nid,
                    name=str(name),
                    entity_type=f"kg:{e.get('entity_type') or 'entity'}",
                    properties={
                        "_table": "kg_entity",
                        "description": e.get("description"),
                    },
                )
            )

        edges: list[Edge] = []
        seen: set[tuple] = set()
        for name in names:
            got = self._mcp_call("mempalace_kg_get_entity", {"name": name})
            if not isinstance(got, dict):
                continue
            for r in got.get("relations") or []:
                if not isinstance(r, dict):
                    continue
                frm, to, rtype = r.get("from"), r.get("to"), r.get("type")
                if not frm or not to:
                    continue
                key = (frm, rtype, to)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    Edge(
                        source_id=f"kg:{frm}",
                        target_id=f"kg:{to}",
                        edge_type=rtype or "related",
                        properties={"_table": "kg_relation"},
                    )
                )

        # Internal consistency: a relation may reference an entity beyond the
        # enumerated cap — synthesize a minimal node so every edge endpoint
        # resolves (mirrors _graph_mapping's kg_only endpoint synthesis).
        for e in edges:
            for endpoint in (e.source_id, e.target_id):
                if endpoint not in present:
                    present.add(endpoint)
                    entities.append(
                        Entity(
                            id=endpoint,
                            name=endpoint.split("kg:", 1)[-1],
                            entity_type="kg:entity",
                            properties={"_table": "kg_entity", "_endpoint_only": True},
                        )
                    )
        # Cat 6 plumbing parity with project_graph: stamp _superseded_by on
        # edges from a superseded entity (derived from supersedes-typed edges).
        annotate_superseded_edges(edges)
        return entities, edges

    def _taxonomy_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        """Wing/room structural scaffold from ``GET /mp/api/v1/taxonomy``,
        projected via the shared ``project_graph`` (identical to the daemon
        and familiar adapters). ``([], [])`` on transport error."""
        body = self._http("GET", f"{REST_BASE}/taxonomy")
        if isinstance(body, QueryResult) or not isinstance(body, dict):
            return [], []
        tree = body.get("taxonomy") or {}
        if not isinstance(tree, dict):
            return [], []
        wings: dict[str, int] = {}
        rooms: list[dict] = []
        for wing_name, room_map in tree.items():
            room_map = room_map or {}
            wings[wing_name] = sum(int(v or 0) for v in room_map.values())
            rooms.append({"wing": wing_name, "rooms": room_map})
        snapshot = {
            "wings": wings,
            "rooms": rooms,
            "tunnels": [],
            "kg_entities": [],
            "kg_triples": [],
        }
        return project_graph(snapshot)

    # --- Isolation ----------------------------------------------------

    def reset(self, wing: Optional[str] = None, *, _errors: Optional[list] = None) -> int:
        """Delete drawers to isolate the next corpus. Returns the count
        removed. Optionally scoped to ``wing``.

        Lists a page (offset 0), deletes it, and re-lists — the delete
        shrinks the set, so paging from 0 each round walks the whole store
        (mirrors the Go project's own ``reset_store``). Bounded by
        ``_RESET_MAX_PAGES`` so a delete that never takes can't spin
        forever. Transport errors stop the loop and are recorded (into
        ``_errors`` when ingest supplies it); ``reset`` never raises.
        """
        removed = 0
        params = {"limit": _LIST_PAGE, "offset": 0}
        if wing:
            params["wing"] = wing
        list_url = f"{REST_BASE}/drawers?{urllib.parse.urlencode(params)}"
        for _ in range(_RESET_MAX_PAGES):
            body = self._http("GET", list_url)
            if isinstance(body, QueryResult):
                if _errors is not None:
                    _errors.append(f"reset list failed: {body.error}")
                break
            drawers = body.get("drawers") if isinstance(body, dict) else None
            if not drawers:
                break
            progressed = False
            for d in drawers:
                did = d.get("drawer_id") or d.get("id")
                if not did:
                    continue
                del_body = self._http(
                    "DELETE", f"{REST_BASE}/drawers/{urllib.parse.quote(str(did))}"
                )
                if isinstance(del_body, QueryResult):
                    if _errors is not None:
                        _errors.append(f"reset delete {did} failed: {del_body.error}")
                    continue
                removed += 1
                progressed = True
            if not progressed:
                # nothing in this page could be deleted — avoid an infinite loop
                break
        return removed

    # --- Optional SMEAdapter methods ----------------------------------

    def get_flat_retrieval(self, question: str, k: int = 5) -> QueryResult:
        """Vector/FTS retrieval with no graph traversal — same path as
        ``query`` (the server has one hybrid retrieval pipeline)."""
        return self.query(question, n_results=k)

    def get_ontology_source(self) -> dict:
        """Declared MemPalace ontology, as documented for the Go server.

        Same documented vocabulary as the palace-daemon / familiar
        adapters so Cat 8 is comparable across access paths; the backend
        implementation differs, the documented ontology does not."""
        return {
            "type": "declared",
            "schema": [
                {
                    "kind": "structural",
                    "entities": ["wing", "room", "drawer", "tunnel"],
                },
                {
                    "kind": "knowledge_graph",
                    "entities": ["entity", "relation", "fact"],
                },
                {
                    "kind": "hall_vocabulary",
                    "values": [
                        "facts", "events", "discoveries",
                        "preferences", "advice",
                    ],
                },
            ],
            "documentation": (
                "MemPalace server (Go) organises memories into Wings, Rooms, "
                "and Drawers, with a pgvector semantic index, an Apache AGE "
                "entity/relation graph, a bi-temporal knowledge graph of "
                "subject-predicate-object facts, and cross-wing Tunnels. "
                "This adapter accesses it over the REST API (/mp/api/v1); the "
                "documented ontology is unchanged from the Python MemPalace."
            ),
        }

    def get_harness_manifest(self) -> list[HarnessDescriptor]:
        """Category 9 surfaces the server actually exposes.

        The Go server *is* an MCP server, so its primary harness surface
        is the MCP endpoint; the optional REST search is a second
        tool-call surface. Each ``probe_fn`` does a real, non-mutating
        call-through and reports success/latency without raising — feeding
        Cat 9b (call-through success)."""
        return [
            HarnessDescriptor(
                name="mempalace_mcp",
                kind="mcp_resource",
                probe_fn=self._probe_mcp_health,
                description="MCP-over-HTTP endpoint (JSON-RPC tools/call).",
                properties={
                    "endpoint": f"{self.api_url}{MCP_ENDPOINT}",
                    "transport": "streamable-http",
                    "health": f"{self.api_url}/mp/mcp/health",
                    # Structural basis the last get_graph_snapshot used
                    # (kg / taxonomy / empty / unknown) — run attribution.
                    "graph_basis": self._graph_basis,
                },
            ),
            HarnessDescriptor(
                name="mempalace_rest_search",
                kind="tool_call",
                probe_fn=self._probe_rest_search,
                description="REST semantic-search tool call.",
                properties={"endpoint": f"{self.api_url}{REST_BASE}/search"},
            ),
        ]

    def _probe_mcp_health(self) -> ProbeResult:
        """GET the unauthenticated MCP liveness endpoint."""
        t0 = time.perf_counter()
        body = self._http("GET", "/mp/mcp/health", auth=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(body, QueryResult):
            return ProbeResult(success=False, latency_ms=latency_ms, error=body.error)
        return ProbeResult(
            success=True, latency_ms=latency_ms, output=json.dumps(body)[:200]
        )

    def _probe_rest_search(self) -> ProbeResult:
        """A minimal authenticated search — read-only, mutates nothing."""
        t0 = time.perf_counter()
        body = self._http("POST", f"{REST_BASE}/search", {"query": "ping", "limit": 1})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(body, QueryResult):
            return ProbeResult(success=False, latency_ms=latency_ms, error=body.error)
        return ProbeResult(success=True, latency_ms=latency_ms)

    # --- HTTP plumbing ------------------------------------------------

    def _http(self, method: str, path: str, payload: Optional[dict] = None,
              *, auth: bool = True) -> Any:
        """Issue an HTTP request, returning parsed JSON on 2xx or a
        ``QueryResult`` carrying an ``error`` string on any failure.

        The QueryResult-as-error sentinel keeps every public method's
        no-raise contract: callers check ``isinstance(body, QueryResult)``.
        """
        url = f"{self.api_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.api_timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = str(e)
            if e.code in (401, 403):
                err = f"AUTH: {e.code} {detail}"
            else:
                err = f"HTTP {e.code}: {detail}"
            return QueryResult(answer="", context_string="", error=err)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return QueryResult(
                answer="", context_string="", error=f"CONNECTION: {e}"
            )
        except Exception as e:  # pragma: no cover - defensive
            return QueryResult(
                answer="", context_string="", error=f"INTERNAL: {e}"
            )

    def _mcp_call(self, tool: str, arguments: dict) -> Optional[dict]:
        """Call an MCP tool via ``POST /mp/mcp`` (stateless JSON-RPC — no
        initialize/session needed) and return the parsed JSON object from
        ``result.content[0].text``.

        Returns ``None`` on any failure (transport error, JSON-RPC error —
        e.g. AGE not installed, missing/blank content, bad JSON) so callers
        can treat "KG unavailable" uniformly and fall back.
        """
        envelope = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        body = self._http("POST", MCP_ENDPOINT, envelope)
        if isinstance(body, QueryResult) or not isinstance(body, dict):
            return None
        if body.get("error"):
            return None
        result = body.get("result")
        if not isinstance(result, dict):
            return None
        content = result.get("content") or []
        if not content or not isinstance(content[0], dict):
            return None
        text = content[0].get("text", "")
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def close(self) -> None:
        """Stateless HTTP client — nothing to release."""
        pass
