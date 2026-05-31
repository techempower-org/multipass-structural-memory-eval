"""agentmemory (rohitg00/agentmemory) REST adapter for SME.

agentmemory is a Node/TypeScript local memory server (REST on :3111) built for
AI coding agents. Retrieval is **hybrid BM25 + vector + graph RRF** over
*observations* captured during a coding session; embeddings are local MiniLM
(``all-MiniLM-L6-v2``) via ``@xenova/transformers``, no cloud key. The
auto-compress LLM hook is OFF by default — kept OFF here to satisfy SME's
$0 / no-write-time-LLM constraint.

Data model (the load-bearing facts, verified against the api.ts / observe.ts
source at the published commit):

  * The searchable store is **observations**, ingested via
    ``POST /agentmemory/observe`` with a hook-style payload
    ``{hookType, sessionId, project, cwd, timestamp, data}``. For a corpus
    document we send ``hookType="prompt_submit"`` with ``data={"prompt": <text>}``;
    ``mem::observe`` lifts that into ``raw.userPrompt``.
  * With auto-compress OFF (our case), ``observe`` indexes a **synthetic**
    compression: the BM25 + vector index entry is ``title + " " + narrative``
    where ``narrative = truncate(prompt, 400)`` — i.e. **only the first ~400
    characters of each observation are searchable**. This is a documented
    architectural property of the no-LLM path; agentmemory's published 95.2%
    R@5 was measured with its own compression pipeline. To keep the full
    session text searchable WITHOUT enabling an LLM, this adapter **chunks each
    corpus document into ≤``CHUNK_CHARS`` observations**, all tagged with the
    same ``sessionId``. Session-level R@K is unaffected because
    ``smart-search`` returns ``sessionId`` per hit.
  * Retrieval: ``POST /agentmemory/search`` body ``{query, limit, project,
    format:"compact"}`` → ``{format:"compact", results:[{obsId, sessionId,
    title, type, score, timestamp}], ...}``. Each hit carries ``sessionId``
    directly — the LongMemEval R@K signal. We use ``/search`` rather than
    ``/smart-search`` because ``mem::search`` is the only retrieval path that
    accepts a ``project`` filter (smart-search ignores project), and the
    project filter is what makes per-question isolation work (see below).

Per-question reset: each ``ingest_corpus`` call uses a **fresh project name**
(monotonic counter), so a new question's haystack lands in a clean project
scope. The search index is global, but ``mem::search`` post-filters hits by
``project`` (resolving each hit's session → project), so a project-scoped
``/search`` can't surface a prior question's observations. Project-rotation +
the ``project`` search filter is the isolation mechanism (the agentmemory
analogue of the daemon's per-question wing). ``smart-search`` would NOT isolate
— it has no project filter — which is why this adapter uses ``/search``.

Retrieval-only: ``get_graph_snapshot`` returns ``([], [])``.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from sme.adapters.base import Entity, QueryResult, SMEAdapter

log = logging.getLogger(__name__)

DEFAULT_API_URL = "http://127.0.0.1:3111"
DEFAULT_TIMEOUT = 60.0
# Match the synthetic-compression narrative cap so each observation is fully
# indexed (no silent head-truncation). 380 < 400 leaves headroom for the
# " | " joiner the synthetic compressor inserts between narrative parts.
CHUNK_CHARS = 380
_PROJECT_COUNTER = itertools.count(1)


def _chunk(text: str, size: int) -> list[str]:
    """Split text into ``size``-char chunks on whitespace where possible."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            # Back off to the last whitespace so we don't split mid-word.
            ws = text.rfind(" ", i, end)
            if ws > i:
                end = ws
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = end if end > i else end + 1
    return chunks


class AgentMemoryAdapter(SMEAdapter):
    """SMEAdapter against a running ``agentmemory`` REST server (:3111).

    Args:
        api_url: Base URL (default ``http://127.0.0.1:3111``).
        project: Base project name. Each ``ingest_corpus`` call appends a
            monotonic suffix so per-question haystacks are isolated; if None a
            counter-derived name is used.
        n_results: Default top-K for ``query()``.
        api_timeout: Per-request HTTP timeout in seconds.
        include_lessons: Accepted for CLI/registry parity. Unused on the
            ``/search`` path (that flag was a ``smart-search`` knob); kept so
            an existing CLI invocation doesn't break.
        read_only: Accepted for CLI parity.
    """

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_API_URL,
        project: Optional[str] = None,
        n_results: int = 5,
        api_timeout: float = DEFAULT_TIMEOUT,
        include_lessons: bool = True,
        read_only: bool = False,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.base_project = project or "sme_bench"
        self.n_results = n_results
        self.api_timeout = api_timeout
        self.include_lessons = include_lessons
        # Resolved per ingest_corpus call so each question is isolated.
        self.project = f"{self.base_project}_{next(_PROJECT_COUNTER)}"

    # --- SMEAdapter required ------------------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """Rotate to a fresh project, then observe each document's chunks.

        Each corpus row ``{id, document, [session_id]}`` becomes one or more
        ``prompt_submit`` observations (chunked to stay under the synthetic
        narrative cap), all tagged with the row's ``session_id`` so
        ``smart-search`` can return it for session-level R@K.
        """
        self.project = f"{self.base_project}_{next(_PROJECT_COUNTER)}"
        errors: list[str] = []
        warnings: list[str] = []
        observations = 0
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()

        for row in corpus:
            content = row.get("document") or row.get("content") or row.get("text") or ""
            if not content.strip():
                continue
            session_id = (
                row.get("session_id")
                or (row.get("metadata") or {}).get("session_id")
                or row.get("id")
            )
            if session_id is None:
                warnings.append("row without session_id/id; skipped")
                continue
            for chunk in _chunk(content, CHUNK_CHARS):
                payload = {
                    "hookType": "prompt_submit",
                    "sessionId": str(session_id),
                    "project": self.project,
                    "cwd": "/sme",
                    "timestamp": ts,
                    "data": {"prompt": chunk},
                }
                body = self._http_post(
                    f"{self.api_url}/agentmemory/observe", payload
                )
                if isinstance(body, QueryResult):
                    errors.append(f"observe failed (session {session_id}): {body.error}")
                    continue
                observations += 1

        return {
            "entities_created": observations,
            "edges_created": 0,
            "errors": errors,
            "warnings": warnings,
        }

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        route: bool = False,  # accepted for CLI parity; agentmemory ranks itself
    ) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        # /search (mem::search) is the only retrieval path that honours a
        # `project` filter — required for per-question isolation. compact
        # format returns one row per hit with sessionId + score.
        payload = {
            "query": question,
            "limit": k,
            "project": self.project,
            "format": "compact",
        }
        body = self._http_post(
            f"{self.api_url}/agentmemory/search", payload
        )
        if isinstance(body, QueryResult):
            return body

        results = body.get("results") if isinstance(body, dict) else None
        if not results:
            return QueryResult(answer="", context_string="", error="NO_RESULTS")

        context_parts: list[str] = []
        retrieved: list[Entity] = []
        for i, hit in enumerate(results[:k]):
            hit = hit or {}
            session_id = hit.get("sessionId")
            obs_id = str(hit.get("obsId") or f"agentmemory_hit:{i}")
            title = hit.get("title") or ""
            score = hit.get("score")
            context_parts.append(f"[{i + 1}] [{session_id}] {title}")
            retrieved.append(
                Entity(
                    id=f"agentmemory:{obs_id}",
                    name=str(session_id) if session_id is not None else obs_id,
                    entity_type=f"observation:{hit.get('type', '?')}",
                    properties={
                        "_table": "agentmemory_hit",
                        "session_id": session_id,
                        "score": score,
                        "rank": i + 1,
                    },
                )
            )

        context_string = "\n\n".join(context_parts)
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=retrieved,
            retrieval_path=[f"search:project={self.project};k={k}"],
        )

    def get_graph_snapshot(self):
        return [], []

    def get_flat_retrieval(self, question: str) -> QueryResult:
        return self.query(question)

    def get_ontology_source(self) -> dict:
        return {
            "type": "inferred",
            "schema": [],
            "documentation": (
                "agentmemory: Node REST server, hybrid BM25 + vector + graph "
                "RRF over observations. Local MiniLM embeddings, no cloud key. "
                "No-LLM (synthetic) compression truncates each observation to "
                "~400 chars; this adapter chunks documents to keep full text "
                "searchable."
            ),
        }

    # --- HTTP plumbing ------------------------------------------------

    def _http_post(self, url: str, payload: Any) -> Any:
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
        pass
