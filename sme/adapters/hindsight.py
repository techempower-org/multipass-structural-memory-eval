"""Hindsight adapter for SME.

Hindsight (https://github.com/vectorize-io/hindsight) is a Docker-based
memory system from Vectorize. It exposes three operations on top of a
biomimetic store (World facts / Experiences / Mental Models):

  - ``retain(bank_id, content, ...)`` — ingest a string; an LLM extracts
    entities, relationships, and temporal data inline.
  - ``recall(bank_id, query)`` — fast multi-strategy retrieval (semantic
    + BM25 + graph + temporal, fused via reciprocal-rank + cross-encoder
    reranking).
  - ``reflect(bank_id, query)`` — deep multi-hop analysis.

Default ports: 8888 (API), 9999 (UI). No auth is documented for the
self-hosted deployment — only the upstream LLM provider key is needed.

This adapter prefers the official ``hindsight-client`` Python SDK and
falls back to a thin urllib HTTP client when the SDK isn't installed.
The fallback keeps SME's optional-dependency footprint minimal and
makes the adapter testable without pulling in the full client.

There is no documented standalone graph-query API. Graph relationships
are surfaced *inside* recall results — Hindsight's README confirms the
graph strategy is one of the four parallel retrieval strategies, not a
separately addressable endpoint. ``get_graph_snapshot()`` therefore
returns the entities + edges we can reconstruct from the per-result
metadata that recall exposes. When that metadata is unavailable
(older Hindsight versions), the snapshot is empty and the adapter
logs a warning rather than failing.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from sme.adapters.base import Edge, Entity, QueryResult, SMEAdapter

log = logging.getLogger(__name__)


DEFAULT_BASE_URL = "http://localhost:8888"
DEFAULT_BANK_ID = "sme"
DEFAULT_TIMEOUT = 60.0


class HindsightAdapter(SMEAdapter):
    """SMEAdapter against a running Hindsight server.

    Args:
        base_url: Hindsight API base URL. Defaults to
            ``http://localhost:8888`` per the README's docker example.
            Honors ``HINDSIGHT_BASE_URL`` env var when unset.
        bank_id: Memory bank/namespace name. Hindsight isolates per-user
            memories via ``bank_id``. Defaults to ``"sme"`` so multiple
            SME runs don't collide with whatever the user's app keeps in
            the default bank.
        api_key: Optional API key. Not required for self-hosted; sent as
            ``Authorization: Bearer <key>`` when provided. Honors
            ``HINDSIGHT_API_KEY`` env var when unset.
        n_results: Default top-K for ``query()``.
        use_reflect: If True, route ``query()`` through the slower but
            deeper ``/reflect`` endpoint. Default False (uses ``recall``)
            because reflect is documented as on-demand deep analysis,
            not the typical fast-path.
        api_timeout: Per-request HTTP timeout in seconds.
        read_only: Accepted for CLI parity. Ignored.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        bank_id: str = DEFAULT_BANK_ID,
        api_key: Optional[str] = None,
        n_results: int = 10,
        use_reflect: bool = False,
        recall_max_tokens: int = 4096,
        recall_budget: str = "mid",
        api_timeout: float = DEFAULT_TIMEOUT,
        read_only: bool = True,
    ) -> None:
        resolved_url = base_url or os.environ.get("HINDSIGHT_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = resolved_url.rstrip("/")
        self.bank_id = bank_id
        self.api_key = api_key or os.environ.get("HINDSIGHT_API_KEY")
        self.n_results = n_results
        self.use_reflect = use_reflect
        # recall has no top_k — it budgets retrieval by token count + a
        # qualitative budget tier ('low'|'mid'|'high'). We over-fetch and
        # slice to n_results in query().
        self.recall_max_tokens = recall_max_tokens
        self.recall_budget = recall_budget
        self.api_timeout = api_timeout
        # Lazy SDK probe — adapter still works without the SDK.
        self._client = _try_load_sdk(self.base_url, self.api_key)

    # --- SMEAdapter required ------------------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """Seed Hindsight via ``retain`` for each corpus row.

        Each row needs at least ``content``. Optional fields:
          - ``context``: short label shown to the LLM during extraction
          - ``timestamp``: ISO-8601 string (Hindsight uses this for the
            temporal retrieval strategy)
          - ``metadata``: dict of custom fields (Hindsight allows
            per-user / per-app filtering via these)
        """
        errors: list[str] = []
        warnings: list[str] = []
        stored = 0
        for i, row in enumerate(corpus):
            content = row.get("content") or row.get("text") or ""
            if not content:
                warnings.append(f"row {i}: empty content, skipped")
                continue
            try:
                self._retain(
                    content=content,
                    context=row.get("context"),
                    timestamp=row.get("timestamp"),
                    metadata=row.get("metadata"),
                    document_id=row.get("document_id") or row.get("id"),
                )
                stored += 1
            except Exception as e:
                errors.append(f"row {i}: {e}")
        return {
            "entities_created": stored,  # Hindsight extracts entities server-side
            "edges_created": 0,  # not surfaced by retain
            "errors": errors,
            "warnings": warnings,
        }

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        route: bool = False,  # accepted for CLI parity; Hindsight handles routing
    ) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        endpoint = "reflect" if self.use_reflect else "recall"
        try:
            raw = self._recall_or_reflect(endpoint, question, k)
        except _HindsightError as e:
            return QueryResult(answer="", context_string="", error=str(e))

        hits = _extract_hits(raw)
        if not hits:
            return QueryResult(answer="", context_string="", error="NO_RESULTS")

        context_parts: list[str] = []
        retrieved: list[Entity] = []
        edges: list[Edge] = []
        for i, hit in enumerate(hits[:k]):
            text = hit.get("text") or hit.get("content") or hit.get("memory", "")
            mem_id = str(hit.get("id") or f"hindsight_hit:{i}")
            mem_kind = hit.get("type") or hit.get("kind") or "memory"
            score = hit.get("score") or hit.get("relevance")
            # document_id is the ingest-unit id we supplied at retain time
            # (e.g. the LongMemEval/LoCoMo session id). It's the join key
            # for drawer-style R@K — Entity.id is set to it when present so
            # the harness's hit_at_K logic compares like-for-like against
            # expected session ids.
            doc_id = hit.get("document_id")
            entity_id = doc_id if doc_id else f"hindsight:{mem_id}"
            context_parts.append(f"[{i + 1}] [{mem_kind}] {text}")
            retrieved.append(
                Entity(
                    id=str(entity_id),
                    name=mem_id,
                    entity_type=f"hindsight:{mem_kind}",
                    properties={
                        "_table": "hindsight_memory",
                        "kind": mem_kind,
                        "score": score,
                        "document_id": doc_id,
                        "bank_id": self.bank_id,
                    },
                )
            )
            # Hindsight may embed entity/relationship metadata in each
            # hit (per the README's biomimetic data structures). When
            # present, project it into SME's Edge shape so Cat 5/6
            # structural reads have something to chew on.
            for rel in hit.get("relationships", []) or []:
                if not isinstance(rel, dict):
                    continue
                edges.append(
                    Edge(
                        source_id=f"hindsight:{rel.get('source', mem_id)}",
                        target_id=f"hindsight:{rel.get('target', mem_id)}",
                        edge_type=str(rel.get("type", "related")),
                        properties={"_source": "hindsight_recall_hit"},
                    )
                )

        context_string = "\n\n".join(context_parts)
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=retrieved,
            retrieved_edges=edges,
            retrieval_path=[f"hindsight_{endpoint}:k={k}", f"bank_id={self.bank_id}"],
        )

    def get_graph_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        """Return what we can reconstruct of the Hindsight graph.

        Hindsight has no documented standalone graph endpoint. The
        ``stats`` and ``list`` endpoints (when present) are the closest
        equivalents. This method tries them, falls back to empty when
        nothing is reachable, and never raises — Cat 5/6 will simply
        score zero against an empty snapshot, which is the correct
        answer for a system that doesn't expose its graph.
        """
        # Best-effort: try a stats endpoint that some self-hosted memory
        # systems expose for diagnostics. If it doesn't exist (most
        # likely on current Hindsight), return empty.
        try:
            body = self._http_get(f"{self.base_url}/banks/{self.bank_id}/stats")
        except _HindsightError as e:
            log.info("hindsight stats endpoint unavailable: %s", e)
            return [], []

        # The stats response shape is undocumented; defensive parsing.
        if not isinstance(body, dict):
            return [], []
        entities: list[Entity] = []
        edges: list[Edge] = []
        for ent in body.get("entities", []) or []:
            if not isinstance(ent, dict):
                continue
            eid = str(ent.get("id") or ent.get("name") or "?")
            entities.append(
                Entity(
                    id=f"hindsight:{eid}",
                    name=str(ent.get("name", eid)),
                    entity_type=str(ent.get("type", "entity")),
                    properties={"_source": "hindsight_stats"},
                )
            )
        for rel in body.get("relationships", []) or []:
            if not isinstance(rel, dict):
                continue
            edges.append(
                Edge(
                    source_id=f"hindsight:{rel.get('source', '?')}",
                    target_id=f"hindsight:{rel.get('target', '?')}",
                    edge_type=str(rel.get("type", "related")),
                    properties={"_source": "hindsight_stats"},
                )
            )
        return entities, edges

    def get_ontology_source(self) -> dict:
        return {
            "type": "readme",
            "schema": [
                {
                    "kind": "memory_types",
                    "values": ["World", "Experiences", "MentalModels"],
                },
                {
                    "kind": "retrieval_strategies",
                    "values": ["semantic", "keyword_bm25", "graph", "temporal"],
                },
                {
                    "kind": "operations",
                    "values": ["retain", "recall", "reflect"],
                },
            ],
            "documentation": (
                "Hindsight uses biomimetic memory structures: World "
                "(facts), Experiences (agent's own history), and "
                "Mental Models (learned understanding formed via "
                "reflect). Retrieval is multi-strategy (semantic + BM25 "
                "+ graph + temporal) with reciprocal-rank fusion and "
                "cross-encoder reranking. Per-user isolation is via "
                "custom metadata on retain. No standalone graph-query "
                "endpoint is documented."
            ),
        }

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()  # SDK may or may not have a close
            except (AttributeError, Exception):
                pass
            self._client = None

    # --- HTTP / SDK plumbing ------------------------------------------

    def _retain(
        self,
        *,
        content: str,
        context: Optional[str] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[dict] = None,
        document_id: Optional[str] = None,
    ) -> None:
        # Verified against hindsight-client v0.7.1:
        # retain(bank_id, content, timestamp=None, context=None,
        #        document_id=None, metadata=None, ...). document_id is the
        # caller-supplied ingest-unit id that recall hits echo back, so it's
        # what makes R@K against the originating session possible.
        kwargs: dict[str, Any] = {"bank_id": self.bank_id, "content": content}
        if context is not None:
            kwargs["context"] = context
        if timestamp is not None:
            kwargs["timestamp"] = timestamp
        if metadata is not None:
            kwargs["metadata"] = metadata
        if document_id is not None:
            kwargs["document_id"] = document_id
        if self._client is not None:
            self._client.retain(**kwargs)
            return
        self._http_post(f"{self.base_url}/retain", kwargs)

    def _recall_or_reflect(self, endpoint: str, query: str, k: int) -> Any:
        # NOTE: the real hindsight-client recall/reflect (verified against
        # v0.7.1) does NOT take a top_k. recall budgets by max_tokens; ranking
        # is internal (RRF + cross-encoder rerank). We over-fetch with a
        # generous max_tokens and slice to k in query(). include_source_facts
        # surfaces the originating raw fact (and its document_id) so R@K can
        # map a recall hit back to the ingested unit.
        if self._client is not None:
            method = getattr(self._client, endpoint, None)
            if method is None:
                raise _HindsightError(
                    f"INTERNAL: hindsight client has no {endpoint!r} method"
                )
            if endpoint == "recall":
                return method(
                    bank_id=self.bank_id, query=query,
                    max_tokens=self.recall_max_tokens,
                    budget=self.recall_budget,
                    include_source_facts=True,
                )
            return method(bank_id=self.bank_id, query=query, budget=self.recall_budget)
        # HTTP fallback — the real REST surface mirrors the SDK kwargs.
        body: dict[str, Any] = {"bank_id": self.bank_id, "query": query}
        if endpoint == "recall":
            body.update(max_tokens=self.recall_max_tokens,
                        budget=self.recall_budget, include_source_facts=True)
        else:
            body["budget"] = self.recall_budget
        return self._http_post(f"{self.base_url}/{endpoint}", body)

    def _http_get(self, url: str) -> Any:
        req = urllib.request.Request(url, method="GET", headers=self._headers())
        return self._http_send(req)

    def _http_post(self, url: str, body: dict) -> Any:
        data = json.dumps(body).encode("utf-8")
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        return self._http_send(req)

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _http_send(self, req: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=self.api_timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = str(e)
            if e.code in (401, 403):
                raise _HindsightError(f"AUTH: invalid credentials ({e.code})") from e
            raise _HindsightError(f"HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise _HindsightError(f"CONNECTION: {e}") from e
        except json.JSONDecodeError as e:
            raise _HindsightError(f"INTERNAL: invalid JSON from hindsight: {e}") from e


# --- helpers -------------------------------------------------------


class _HindsightError(Exception):
    """Carries the SME-formatted error string for QueryResult.error."""


def _try_load_sdk(base_url: str, api_key: Optional[str]):
    """Try to import and construct the hindsight-client. Return None
    if unavailable. Failure to import is silent — we have a working
    urllib fallback."""
    try:
        from hindsight_client import Hindsight  # type: ignore
    except ImportError:
        return None
    try:
        kwargs: dict[str, Any] = {"base_url": base_url}
        if api_key:
            kwargs["api_key"] = api_key
        return Hindsight(**kwargs)
    except Exception as e:  # pragma: no cover
        log.warning("hindsight SDK present but construction failed: %s", e)
        return None


def _hit_to_dict(hit: Any) -> dict:
    """Normalise one recall hit to a plain dict. Handles both the SDK's
    Pydantic ``RecallResult`` (verified v0.7.1: id/text/type/document_id/
    metadata/source_fact_ids) and the HTTP-fallback dict."""
    if isinstance(hit, dict):
        return hit
    # Pydantic model → dict
    dump = getattr(hit, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001 — fall through to attr scrape
            pass
    return {
        k: getattr(hit, k, None)
        for k in ("id", "text", "type", "document_id", "context",
                  "metadata", "source_fact_ids", "entities", "tags")
    }


def _extract_hits(raw: Any) -> list[dict]:
    """Normalise a recall/reflect response into a list of hit dicts.

    The real hindsight-client returns a Pydantic ``RecallResponse`` with a
    ``.results`` list of ``RecallResult`` (recall) or a ``ReflectResponse``
    with ``.answer``/``.facts`` (reflect). The HTTP fallback returns the
    equivalent JSON dict. Tolerate both, plus a bare list.
    """
    if raw is None:
        return []
    # SDK Pydantic objects expose attributes, not dict keys.
    results_attr = getattr(raw, "results", None)
    if isinstance(results_attr, list):
        return [_hit_to_dict(h) for h in results_attr]
    answer_attr = getattr(raw, "answer", None)
    if isinstance(answer_attr, str) and answer_attr:
        return [{"text": answer_attr, "type": "reflect_answer"}]
    if isinstance(raw, list):
        return [_hit_to_dict(h) for h in raw]
    if isinstance(raw, dict):
        for key in ("results", "memories", "hits"):
            inner = raw.get(key)
            if isinstance(inner, list):
                return [_hit_to_dict(h) for h in inner]
        if isinstance(raw.get("answer"), str):
            return [{"text": raw["answer"], "type": "reflect_answer"}]
    return []
