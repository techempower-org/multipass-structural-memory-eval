"""ai-memory (alphaonedev/ai-memory-mcp) HTTP adapter for SME.

ai-memory is a Rust persistent-memory system: pure SQLite FTS5 + a local
MiniLM embedding model (Candle), exposed as a local HTTP API on
``127.0.0.1:9077`` under ``/api/v1/``. There is **no write-time LLM** — ingest
is embedding + FTS only (the optional LLM query-expansion is a ``smart+`` tier
feature requiring Ollama and stays OFF at the default ``semantic`` tier). That
makes it a $0, fully-local, zero-cloud retrieval-only competitor that publishes
**R@5** (97.8% on LongMemEval-S per the project's README) — directly comparable
to mempalace's R@5 headline with no metric mixing.

This adapter talks to a running ``ai-memory serve`` over HTTP. Like the
``postgres_ingest`` adapter it is built for the LongMemEval shape where each
question has its own small haystack and the store must reset between questions:
``ingest_corpus`` first ``POST /api/v1/forget``s the working namespace, then
``POST /api/v1/memories/bulk``-loads the corpus. ``query`` hits
``POST /api/v1/recall``.

Endpoints used (verified against ai-memory v0.6.4):
  * ``POST /api/v1/memories/bulk`` — array of CreateMemory objects (cap 1000).
    Each: ``{"content", "title", "namespace", "tier", "metadata"}``. The
    session id is carried in BOTH ``metadata.session_id`` and ``title`` so a
    LongMemEval runner can compute session-level R@K regardless of which field
    ``recall`` echoes back.
  * ``POST /api/v1/recall`` — body ``{"context", "limit", "namespace"}`` →
    ``{"memories": [Memory + score], "count"}``. The query field is ``context``
    (NOT ``query``). recall mutates the DB (auto-promote/touch) but that is
    harmless here because the namespace is forgotten before the next ingest.
  * ``POST /api/v1/forget`` — body ``{"namespace"}`` → ``{"deleted": N}``.

Retrieval-only: ``get_graph_snapshot`` returns ``([], [])`` like the other
verbatim-first adapters (flat_baseline, postgres_ingest, omega's session row).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from sme.adapters.base import Entity, QueryResult, SMEAdapter

log = logging.getLogger(__name__)

DEFAULT_API_URL = "http://127.0.0.1:9077"
DEFAULT_NAMESPACE = "sme_bench"
DEFAULT_TIER = "mid"
DEFAULT_TIMEOUT = 60.0
_BULK_CAP = 1000  # ai-memory caps /memories/bulk at 1000 items


class AiMemoryAdapter(SMEAdapter):
    """SMEAdapter against a running ``ai-memory serve`` HTTP daemon.

    Args:
        api_url: Base URL of the ai-memory server (default
            ``http://127.0.0.1:9077``). Trailing slash stripped.
        namespace: Working namespace. ``ingest_corpus`` forgets and reloads
            this namespace per call so per-question haystacks don't bleed.
        tier: Storage tier applied to ingested rows (``short`` | ``mid`` |
            ``long``). ``mid`` is a neutral default; the tier does not change
            which embedding/FTS path retrieval takes at the ``semantic`` tier.
        n_results: Default top-K for ``query()``.
        api_timeout: Per-request HTTP timeout in seconds.
        read_only: Accepted for CLI parity. ``query`` still issues recall
            (recall mutates auto-promote counters server-side, not the corpus).
    """

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_API_URL,
        namespace: str = DEFAULT_NAMESPACE,
        tier: str = DEFAULT_TIER,
        n_results: int = 5,
        api_timeout: float = DEFAULT_TIMEOUT,
        read_only: bool = False,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.namespace = namespace
        self.tier = tier
        self.n_results = n_results
        self.api_timeout = api_timeout

    # --- SMEAdapter required ------------------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """Forget the working namespace, then bulk-load the corpus.

        Each corpus row is ``{id, document, [session_id], [metadata]}``. The
        ``session_id`` (taken from the row or its metadata) is stored in
        ``metadata.session_id`` and echoed into the title, so the runner can
        compute session-level R@K. Returns ``{entities_created, edges_created,
        errors, warnings}`` per the SMEAdapter contract.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Reset per-question: wipe the namespace so prior haystacks can't leak.
        forget = self._http_post(
            f"{self.api_url}/api/v1/forget", {"namespace": self.namespace}
        )
        if isinstance(forget, QueryResult) and forget.error:
            # A namespace that was never created yields nothing to forget;
            # only surface a genuine transport failure.
            if not forget.error.startswith("HTTP 4"):
                warnings.append(f"forget failed: {forget.error}")

        items: list[dict[str, Any]] = []
        for row in corpus:
            content = row.get("document") or row.get("content") or row.get("text") or ""
            if not content.strip():
                continue
            meta_in = row.get("metadata") or {}
            session_id = (
                row.get("session_id")
                or meta_in.get("session_id")
                or row.get("id")
            )
            metadata = dict(meta_in)
            if session_id is not None:
                metadata["session_id"] = session_id
            item: dict[str, Any] = {
                "content": content,
                "namespace": self.namespace,
                "tier": self.tier,
                "metadata": metadata,
            }
            if session_id is not None:
                item["title"] = str(session_id)
            items.append(item)

        if not items:
            return {"entities_created": 0, "edges_created": 0,
                    "errors": errors, "warnings": warnings}

        created = 0
        for start in range(0, len(items), _BULK_CAP):
            batch = items[start:start + _BULK_CAP]
            body = self._http_post(
                f"{self.api_url}/api/v1/memories/bulk", batch
            )
            if isinstance(body, QueryResult):
                errors.append(f"bulk insert failed: {body.error}")
                continue
            # Response: {"created": N, "errors": [...]}
            created += int(body.get("created", 0)) if isinstance(body, dict) else 0
            if isinstance(body, dict):
                errors.extend(body.get("errors", []) or [])

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
        route: bool = False,  # accepted for CLI parity; ai-memory ranks itself
    ) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        payload = {
            "context": question,  # ai-memory's recall query field is `context`
            "limit": k,
            "namespace": self.namespace,
        }
        body = self._http_post(f"{self.api_url}/api/v1/recall", payload)
        if isinstance(body, QueryResult):
            return body  # transport/HTTP error already wrapped

        memories = body.get("memories") if isinstance(body, dict) else None
        if not memories:
            return QueryResult(answer="", context_string="", error="NO_RESULTS")

        context_parts: list[str] = []
        retrieved: list[Entity] = []
        for i, hit in enumerate(memories[:k]):
            hit = hit or {}
            text = hit.get("content") or ""
            meta = hit.get("metadata") or {}
            # session_id round-trips via metadata first, title as fallback.
            session_id = meta.get("session_id") or hit.get("title")
            mem_id = str(hit.get("id") or f"ai_memory_hit:{i}")
            score = hit.get("score")
            context_parts.append(f"[{i + 1}] {text}")
            retrieved.append(
                Entity(
                    id=f"ai_memory:{mem_id}",
                    name=str(session_id) if session_id is not None else mem_id,
                    entity_type=f"memory:{hit.get('tier', '?')}",
                    properties={
                        "_table": "ai_memory_hit",
                        "namespace": hit.get("namespace"),
                        "score": score,
                        "session_id": session_id,
                        "rank": i + 1,
                    },
                )
            )

        context_string = "\n\n".join(context_parts)
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=retrieved,
            retrieval_path=[f"recall:namespace={self.namespace};k={k}"],
        )

    def get_graph_snapshot(self):
        # Retrieval-only system — no graph, like the other verbatim-first rows.
        return [], []

    def get_flat_retrieval(self, question: str) -> QueryResult:
        return self.query(question)

    def get_ontology_source(self) -> dict:
        return {
            "type": "inferred",
            "schema": [],
            "documentation": (
                "ai-memory: Rust SQLite FTS5 + local MiniLM (Candle) "
                "retrieval-only store. No write-time LLM; semantic tier uses "
                "local embeddings only. Accessed via the local HTTP API."
            ),
        }

    # --- HTTP plumbing ------------------------------------------------

    def _http_post(self, url: str, payload: Any) -> Any:
        """POST JSON, return parsed JSON on 2xx or a QueryResult on error."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.api_timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = str(e)
            return QueryResult(
                answer="", context_string="", error=f"HTTP {e.code}: {detail}"
            )
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return QueryResult(
                answer="", context_string="", error=f"CONNECTION: {e}"
            )
        except Exception as e:  # pragma: no cover
            return QueryResult(
                answer="", context_string="", error=f"INTERNAL: {e}"
            )

    def close(self) -> None:
        # Stateless HTTP client; nothing to release. The daemon owns the file.
        pass
