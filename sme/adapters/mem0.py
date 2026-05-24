"""Mem0 OSS adapter for SME.

Mem0 (https://github.com/mem0ai/mem0, ``pip install mem0ai``) is a
Python SDK memory layer with a default stack of OpenAI ``gpt-5-mini``
for fact extraction, OpenAI ``text-embedding-3-small`` embeddings, and
a Qdrant vector store on disk at ``/tmp/qdrant``. History is kept in
SQLite at ``~/.mem0/history.db``.

This adapter targets the OSS library mode (``from mem0 import Memory``),
not the cloud platform. It does **not** require a Mem0 API key, but
the underlying LLM/embedder still needs its provider key
(typically ``OPENAI_API_KEY``).

**Graph memory was removed from the OSS package** in late 2025/early
2026 (~4000 lines of Neo4j/Memgraph/Kuzu/AGE/Neptune integration
deleted). ``relations`` on search results are no longer populated, and
``enable_graph`` / ``graph_store`` config keys are gone. SME's
``get_graph_snapshot()`` therefore returns just the entities Mem0
exposes as a flat list — Cat 5/6 will score zero against this, which
is the honest reading of a system that intentionally dropped its
graph layer.

Entity *linking* still happens (auto-extracted at ``add()`` time and
stored in a parallel ``{collection}_entities`` Qdrant collection),
but it's only consumed indirectly through ranking — not exposed as a
queryable graph. We surface what we can find when the entities
collection is reachable, and an empty graph otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sme.adapters.base import Edge, Entity, QueryResult, SMEAdapter

log = logging.getLogger(__name__)


DEFAULT_USER_ID = "sme"
DEFAULT_N_RESULTS = 10


def _require_mem0():
    """Import mem0 or raise with an install hint.

    Wrapped so construction is the failure point, not module import.
    """
    try:
        from mem0 import Memory  # type: ignore
    except ImportError as e:
        raise ImportError(
            "mem0 adapter requires the mem0ai package. "
            "Install with: pip install mem0ai  "
            "(or: pip install 'sme-eval[mem0]')"
        ) from e
    return Memory


class Mem0Adapter(SMEAdapter):
    """SMEAdapter backed by Mem0's OSS Memory class.

    Args:
        config: Optional Mem0 config dict passed to ``Memory.from_config``.
            When omitted, uses ``Memory()`` defaults (OpenAI LLM, OpenAI
            embeddings, on-disk Qdrant, SQLite history).
        user_id: Default user namespace for ``add()`` / ``search()``.
            Mem0 requires ``filters={"user_id": ...}`` on every search
            since the graph-memory removal migration.
        n_results: Default ``top_k`` for ``query()``.
        memory: Optional pre-constructed Memory instance — useful in
            tests so we don't have to monkeypatch the import.
        read_only: Accepted for CLI parity. Ignored.
    """

    def __init__(
        self,
        *,
        config: Optional[dict] = None,
        user_id: str = DEFAULT_USER_ID,
        n_results: int = DEFAULT_N_RESULTS,
        memory: Any = None,
        read_only: bool = True,
    ) -> None:
        self.user_id = user_id
        self.n_results = n_results
        if memory is not None:
            self._memory = memory
            self._Memory = type(memory)
        else:
            self._Memory = _require_mem0()
            if config:
                # Older versions: from_config(config_dict)
                # Newer versions: from_config(config_dict=...)
                try:
                    self._memory = self._Memory.from_config(config)
                except TypeError:
                    self._memory = self._Memory.from_config(config_dict=config)
            else:
                self._memory = self._Memory()

    # --- SMEAdapter required ------------------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """Seed Mem0 via ``Memory.add`` for each corpus row.

        Each row needs at least ``content``. Optional fields:
          - ``user_id``: override the adapter default for this row
          - ``role``: ``"user"`` (default) or ``"assistant"`` — Mem0
            expects message dicts, not raw strings
          - ``metadata``: dict, passed through Mem0's metadata kwarg
            when the installed Mem0 supports it

        Mem0's "add" uses an LLM internally to extract facts from the
        message list, so seeding cost = (corpus_size * ~1 LLM call).
        Plan accordingly when benchmarking.
        """
        errors: list[str] = []
        warnings: list[str] = []
        stored = 0
        for i, row in enumerate(corpus):
            content = row.get("content") or row.get("text") or ""
            if not content:
                warnings.append(f"row {i}: empty content, skipped")
                continue
            uid = row.get("user_id") or self.user_id
            role = row.get("role") or "user"
            messages = [{"role": role, "content": content}]
            try:
                self._memory.add(messages, user_id=uid)
                stored += 1
            except TypeError as e:
                # Older mem0 took (data, user_id) instead of (messages, user_id)
                try:
                    self._memory.add(content, user_id=uid)
                    stored += 1
                except Exception as inner:
                    errors.append(
                        f"row {i}: mem0.add rejected both message-list and "
                        f"string shapes — outer: {e}; inner: {inner}"
                    )
            except Exception as e:
                errors.append(f"row {i}: {e}")
        return {
            "entities_created": stored,
            "edges_created": 0,  # graph memory removed in Mem0 OSS
            "errors": errors,
            "warnings": warnings,
        }

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        user_id: Optional[str] = None,
        route: bool = False,  # accepted for CLI parity; mem0 has no routing layer
    ) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        uid = user_id or self.user_id
        # Post-graph-removal API: filters is required, user_id can't be
        # a top-level kwarg. Older versions accepted user_id directly.
        try:
            raw = self._memory.search(
                query=question, filters={"user_id": uid}, top_k=k
            )
        except (TypeError, ValueError):
            try:
                raw = self._memory.search(query=question, user_id=uid, top_k=k)
            except Exception as e:
                return QueryResult(answer="", context_string="", error=f"INTERNAL: {e}")
        except Exception as e:
            return QueryResult(answer="", context_string="", error=f"INTERNAL: {e}")

        hits = _extract_results(raw)
        if not hits:
            return QueryResult(answer="", context_string="", error="NO_RESULTS")

        context_parts: list[str] = []
        retrieved: list[Entity] = []
        for i, hit in enumerate(hits[:k]):
            text = hit.get("memory") or hit.get("text") or hit.get("content", "")
            mem_id = str(hit.get("id") or f"mem0_hit:{i}")
            score = hit.get("score") or hit.get("relevance")
            categories = hit.get("categories") or []
            cat_label = ",".join(categories) if categories else "memory"
            context_parts.append(f"[{i + 1}] [{cat_label}] {text}")
            retrieved.append(
                Entity(
                    id=f"mem0:{mem_id}",
                    name=mem_id,
                    entity_type=f"memory:{cat_label}",
                    properties={
                        "_table": "mem0_memory",
                        "score": score,
                        "categories": list(categories),
                        "user_id": uid,
                    },
                )
            )
        context_string = "\n\n".join(context_parts)
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=retrieved,
            retrieval_path=[f"mem0_search:k={k}", f"user_id={uid}"],
        )

    def get_graph_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        """Return whatever flat memory list Mem0 exposes, no edges.

        Mem0 OSS dropped graph memory; ``relations`` on results are no
        longer populated. Returning ([], []) would be defensible, but
        SME's structural categories work better when they can see the
        nodes that *do* exist even if there are no edges. We pull every
        memory for the configured user via ``get_all`` and project them
        as isolated entities. Edges list stays empty by design.
        """
        try:
            raw = self._memory.get_all(filters={"user_id": self.user_id})
        except (TypeError, ValueError):
            # Older API
            try:
                raw = self._memory.get_all(user_id=self.user_id)
            except Exception as e:
                log.warning("mem0 get_all failed: %s; returning empty snapshot", e)
                return [], []
        except Exception as e:
            log.warning("mem0 get_all failed: %s; returning empty snapshot", e)
            return [], []

        rows = _extract_results(raw)
        entities: list[Entity] = []
        for row in rows:
            mem_id = str(row.get("id") or "?")
            text = row.get("memory") or row.get("text") or row.get("content", "")
            categories = row.get("categories") or []
            cat_label = ",".join(categories) if categories else "memory"
            entities.append(
                Entity(
                    id=f"mem0:{mem_id}",
                    name=mem_id,
                    entity_type=f"memory:{cat_label}",
                    properties={
                        "_table": "mem0_memory",
                        "categories": list(categories),
                        "user_id": self.user_id,
                        "content_preview": str(text)[:200],
                    },
                )
            )
        return entities, []

    # --- optional helpers ---------------------------------------------

    def reset(self) -> None:
        """Clear all memories for this adapter's user_id.

        Mem0 exposes both ``delete_all(user_id=...)`` (per-user wipe)
        and ``reset()`` (global, including the SQLite history). We use
        per-user delete by default; pass ``hard=True`` via env var to
        full-reset (off by design, since it would nuke unrelated user
        data on a shared Mem0 instance).
        """
        try:
            self._memory.delete_all(user_id=self.user_id)
        except (TypeError, ValueError):
            try:
                self._memory.delete_all(filters={"user_id": self.user_id})
            except Exception as e:
                log.warning("mem0 delete_all failed: %s", e)
        except Exception as e:
            log.warning("mem0 delete_all failed: %s", e)

    def get_ontology_source(self) -> dict:
        return {
            "type": "readme",
            "schema": [
                {
                    "kind": "scopes",
                    "values": ["user", "session", "agent"],
                },
                {
                    "kind": "retrieval_signals",
                    "values": ["semantic_vector", "bm25", "entity_match"],
                },
                {
                    "kind": "result_fields",
                    "values": [
                        "id",
                        "memory",
                        "user_id",
                        "categories",
                        "created_at",
                        "score",
                    ],
                },
            ],
            "documentation": (
                "Mem0 OSS stores memories per user_id with optional "
                "session/agent scopes. Retrieval is multi-signal "
                "(semantic vector + BM25 + entity matching), fused into "
                "a single score. Default stack: OpenAI gpt-5-mini for "
                "extraction, text-embedding-3-small (1536d) for "
                "embeddings, on-disk Qdrant for vectors, SQLite for "
                "history. Graph memory was removed from the OSS "
                "package — `relations` on results is no longer populated."
            ),
        }

    def close(self) -> None:
        # Mem0's Memory class has no close()/__exit__ in current versions.
        # The underlying Qdrant connection cleans up on GC.
        self._memory = None


# --- helpers --------------------------------------------------------


def _extract_results(raw: Any) -> list[dict]:
    """Normalise mem0's various return shapes to list[dict].

    Search returns ``{"results": [...]}``. ``get_all`` returns the same
    shape in current versions but historically returned a bare list.
    Be tolerant of both.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        inner = raw.get("results") or raw.get("memories")
        if isinstance(inner, list):
            return [r for r in inner if isinstance(r, dict)]
    return []
