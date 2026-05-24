"""OMEGA adapter for SME.

OMEGA (https://github.com/omega-memory/omega-memory, ``pip install
omega-memory``) is a local-only persistent memory system for AI coding
agents. It stores typed memories in a SQLite database under
``~/.omega/omega.db`` and serves semantic search via 384-dim
``bge-small-en-v1.5`` ONNX embeddings + ``sqlite-vec``. Memories carry
a type (``decision``, ``lesson``, ``error``, ``summary``,
``user_preference``) and are auto-linked into a typed relationship
graph (``related``, ``supersedes``, ``contradicts``).

This adapter goes through OMEGA's Python library API
(``from omega import store, query``) rather than its MCP server, so it
inherits OMEGA's actual storage + retrieval path with no extra hops.

Mode A (seed via ``ingest_corpus``) is supported because OMEGA exposes
``store()`` directly. ``get_graph_snapshot()`` walks OMEGA's SQLite
tables (``memories`` + ``edges``) to project memories + typed edges into
SME's ``Entity``/``Edge`` shape — OMEGA's edges are exactly the
structural relationships SME is asking about.

Optional dependency. Construction raises ImportError with install hint
if ``omega-memory`` isn't installed.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from sme.adapters.base import Edge, Entity, QueryResult, SMEAdapter

log = logging.getLogger(__name__)


DEFAULT_DB_PATH = "~/.omega/omega.db"


def _require_omega():
    """Import omega-memory or raise with an install hint.

    Wrapped in a function so adapter construction is the failure point
    (loud, early) rather than module import.
    """
    try:
        import omega  # type: ignore
    except ImportError as e:
        raise ImportError(
            "omega adapter requires the omega-memory package. "
            "Install with: pip install omega-memory  "
            "(or: pip install 'sme-eval[omega]')"
        ) from e
    return omega


class OmegaAdapter(SMEAdapter):
    """SMEAdapter backed by OMEGA's local SQLite + ONNX memory store.

    Args:
        db_path: Path to the OMEGA SQLite file. Defaults to
            ``~/.omega/omega.db``. Accepted for diagnostic reads against
            an alternate palace; OMEGA's own library API uses the env
            var ``OMEGA_DB_PATH`` if set, otherwise the default.
        default_memory_type: Type string applied to corpus rows that
            don't specify one. OMEGA's documented vocabulary is
            ``decision`` | ``lesson`` | ``error`` | ``summary`` |
            ``user_preference``. ``summary`` is the safest catch-all
            for arbitrary seeded content.
        n_results: Default top-K for ``query()``.
        read_only: Accepted for CLI parity. OMEGA opens the SQLite file
            in WAL mode; we open our diagnostic snapshot connection
            read-only.
    """

    def __init__(
        self,
        *,
        db_path: Optional[str] = None,
        default_memory_type: str = "summary",
        n_results: int = 10,
        read_only: bool = True,
    ) -> None:
        self._omega = _require_omega()
        resolved = db_path or os.environ.get("OMEGA_DB_PATH") or DEFAULT_DB_PATH
        self.db_path = str(Path(os.path.expanduser(resolved)).resolve())
        self.default_memory_type = default_memory_type
        self.n_results = n_results
        # Lazy — only opened when get_graph_snapshot needs it.
        self._kg_conn: Optional[sqlite3.Connection] = None

    # --- SMEAdapter required ------------------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """Seed OMEGA from a corpus list.

        Each corpus row is expected to be a dict with at least ``content``
        (the text to store). Optional fields:
          - ``type``: OMEGA memory type override (defaults to
            ``default_memory_type``)
          - ``tags``: list[str], joined into the OMEGA tag field if
            the installed OMEGA version supports keyword args (older
            versions accept only ``(content, type)``)
        """
        errors: list[str] = []
        warnings: list[str] = []
        stored = 0
        for i, row in enumerate(corpus):
            content = row.get("content") or row.get("text") or ""
            if not content:
                warnings.append(f"row {i}: empty content, skipped")
                continue
            mem_type = row.get("type") or self.default_memory_type
            try:
                self._omega.store(content, mem_type)
                stored += 1
            except TypeError:
                # Older signature may differ; surface and stop.
                errors.append(
                    f"row {i}: omega.store(content, type) signature rejected — "
                    "check installed omega-memory version"
                )
                break
            except Exception as e:
                errors.append(f"row {i}: {e}")
        return {
            "entities_created": stored,
            "edges_created": 0,  # OMEGA creates `related` edges
            # asynchronously after auto-relate runs; we can't count them
            # from the ingest call alone.
            "errors": errors,
            "warnings": warnings,
        }

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        route: bool = False,  # accepted for CLI parity; OMEGA does its own ranking
    ) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        try:
            raw = self._omega.query(question)
        except Exception as e:
            return QueryResult(
                answer="",
                context_string="",
                error=f"INTERNAL: {e}",
            )

        # OMEGA's query() return shape is not documented in the README.
        # Defensive normalisation: accept list-of-dicts, list-of-strings,
        # or a dict with a "results" key.
        hits = _normalise_query_hits(raw)
        if not hits:
            return QueryResult(answer="", context_string="", error="NO_RESULTS")

        hits = hits[:k]
        context_parts: list[str] = []
        retrieved: list[Entity] = []
        for i, hit in enumerate(hits):
            text = hit.get("content") or hit.get("memory") or hit.get("text", "")
            mem_type = hit.get("type") or hit.get("memory_type") or "?"
            score = hit.get("score") or hit.get("similarity") or hit.get("rank")
            mem_id = str(hit.get("id") or f"omega_hit:{i}")
            context_parts.append(f"[{i + 1}] [{mem_type}] {text}")
            retrieved.append(
                Entity(
                    id=f"omega:{mem_id}",
                    name=mem_id,
                    entity_type=f"memory:{mem_type}",
                    properties={
                        "_table": "omega_memory",
                        "type": mem_type,
                        "score": score,
                    },
                )
            )
        context_string = "\n\n".join(context_parts)
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=retrieved,
            retrieval_path=[f"omega_query:k={k}"],
        )

    def get_graph_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        """Project OMEGA's SQLite store into SME (entities, edges).

        OMEGA stores each memory as a row in the ``memories`` table and
        each typed relationship (``related``, ``supersedes``,
        ``contradicts``) as a row in an ``edges`` table. Schema names
        may drift between OMEGA versions; this method introspects
        ``sqlite_master`` and degrades gracefully when columns are
        absent rather than raising.
        """
        if not Path(self.db_path).exists():
            log.warning("omega db not found at %s; returning empty snapshot",
                        self.db_path)
            return [], []

        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True
            )
        except sqlite3.Error as e:
            log.warning("could not open omega db read-only: %s", e)
            return [], []

        try:
            entities = _read_omega_memories(conn)
            edges = _read_omega_edges(conn)
        finally:
            conn.close()

        return entities, edges

    # --- optional helpers ---------------------------------------------

    def reset(self) -> None:
        """Clear OMEGA's SQLite store. Deletes the file at ``db_path``.

        OMEGA has no documented ``reset()`` Python entry point; the
        ``omega clear_session`` CLI clears one session, not the whole
        store. For SME's purposes (clean re-ingest), removing the file
        and letting OMEGA recreate it on next ``store()`` is the
        simplest path. Refuses to operate on the default user-level db
        unless ``OMEGA_ALLOW_DEFAULT_RESET=1`` is set, to avoid
        accidentally wiping a user's real memories.
        """
        default = str(Path(os.path.expanduser(DEFAULT_DB_PATH)).resolve())
        if (
            self.db_path == default
            and os.environ.get("OMEGA_ALLOW_DEFAULT_RESET") != "1"
        ):
            raise RuntimeError(
                "refusing to reset the default OMEGA database "
                f"({self.db_path}). Set OMEGA_ALLOW_DEFAULT_RESET=1 or "
                "pass an explicit db_path to OmegaAdapter."
            )
        try:
            Path(self.db_path).unlink(missing_ok=True)
        except OSError as e:
            log.warning("omega reset failed: %s", e)

    def get_ontology_source(self) -> dict:
        return {
            "type": "readme",
            "schema": [
                {
                    "kind": "memory_types",
                    "values": [
                        "decision",
                        "lesson",
                        "error",
                        "summary",
                        "user_preference",
                    ],
                },
                {
                    "kind": "edge_types",
                    "values": ["related", "supersedes", "contradicts"],
                },
            ],
            "documentation": (
                "OMEGA stores memories in SQLite with a fixed type "
                "vocabulary (decision, lesson, error, summary, "
                "user_preference) and links them via typed edges "
                "(related, supersedes, contradicts). Embeddings are "
                "384-dim bge-small-en-v1.5 ONNX vectors stored in "
                "sqlite-vec. Auto-relate creates `related` edges when "
                "cosine similarity >= 0.45 between memories."
            ),
        }

    def close(self) -> None:
        if self._kg_conn is not None:
            try:
                self._kg_conn.close()
            except Exception:
                pass
            self._kg_conn = None


# --- helpers --------------------------------------------------------


def _normalise_query_hits(raw: Any) -> list[dict]:
    """Normalise OMEGA's query() return to a list[dict] of hits.

    OMEGA's documented surface is just ``results = query("text")``
    with no schema published in the README. We accept the most common
    shapes a Python memory library returns.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        inner = raw.get("results") or raw.get("memories") or raw.get("hits")
        if isinstance(inner, list):
            return [_coerce_hit(h) for h in inner]
        return []
    if isinstance(raw, list):
        return [_coerce_hit(h) for h in raw]
    return []


def _coerce_hit(h: Any) -> dict:
    if isinstance(h, dict):
        return h
    if isinstance(h, str):
        return {"content": h}
    # Try .__dict__ for dataclass-ish results
    d = getattr(h, "__dict__", None)
    if isinstance(d, dict):
        return d
    return {"content": str(h)}


_MEMORY_TABLE_CANDIDATES = ("memories", "memory")
_EDGE_TABLE_CANDIDATES = ("edges", "memory_edges", "links")


def _read_omega_memories(conn: sqlite3.Connection) -> list[Entity]:
    """Project OMEGA's memories table into Entity rows.

    Schema-tolerant: walks candidate table names, then probes columns.
    Returns an empty list if no recognised table exists.
    """
    table = _first_existing_table(conn, _MEMORY_TABLE_CANDIDATES)
    if table is None:
        return []
    cols = _table_columns(conn, table)
    id_col = _first_col(cols, ("id", "memory_id", "rowid"))
    content_col = _first_col(cols, ("content", "text", "memory"))
    type_col = _first_col(cols, ("type", "memory_type", "kind"))
    if id_col is None or content_col is None:
        log.warning(
            "omega memories table %r missing id/content columns; got %s",
            table,
            cols,
        )
        return []
    type_sql = f", {type_col}" if type_col else ""
    rows = conn.execute(
        f"SELECT {id_col}, {content_col}{type_sql} FROM {table}"
    ).fetchall()
    entities: list[Entity] = []
    for row in rows:
        mem_id = str(row[0])
        content = row[1] or ""
        mem_type = row[2] if type_col else "memory"
        entities.append(
            Entity(
                id=f"omega:{mem_id}",
                name=mem_id,
                entity_type=f"memory:{mem_type}",
                properties={
                    "_table": "omega_memory",
                    "type": mem_type,
                    "content_preview": str(content)[:200],
                },
            )
        )
    return entities


def _read_omega_edges(conn: sqlite3.Connection) -> list[Edge]:
    """Project OMEGA's edges table into Edge rows."""
    table = _first_existing_table(conn, _EDGE_TABLE_CANDIDATES)
    if table is None:
        return []
    cols = _table_columns(conn, table)
    src = _first_col(cols, ("source_id", "from_id", "src", "memory_a"))
    dst = _first_col(cols, ("target_id", "to_id", "dst", "memory_b"))
    etype = _first_col(cols, ("type", "edge_type", "relation"))
    if src is None or dst is None:
        log.warning(
            "omega edges table %r missing source/target columns; got %s",
            table,
            cols,
        )
        return []
    etype_sql = f", {etype}" if etype else ""
    rows = conn.execute(
        f"SELECT {src}, {dst}{etype_sql} FROM {table}"
    ).fetchall()
    edges: list[Edge] = []
    for row in rows:
        edges.append(
            Edge(
                source_id=f"omega:{row[0]}",
                target_id=f"omega:{row[1]}",
                edge_type=(row[2] if etype else "related"),
                properties={},
            )
        )
    return edges


def _first_existing_table(
    conn: sqlite3.Connection, candidates: tuple[str, ...]
) -> Optional[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    have = {r[0] for r in rows}
    for c in candidates:
        if c in have:
            return c
    return None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def _first_col(cols: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    have = set(cols)
    for c in candidates:
        if c in have:
            return c
    return None
