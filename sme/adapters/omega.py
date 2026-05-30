"""OMEGA adapter for SME.

OMEGA (https://pypi.org/project/omega-memory/, ``pip install
omega-memory``) is a local-only persistent memory system for AI coding
agents. It stores typed memories in a SQLite database under
``$OMEGA_HOME/omega.db`` (default ``~/.omega/omega.db``) and serves
semantic search via 384-dim ``bge-small-en-v1.5`` ONNX embeddings +
``sqlite-vec`` (with an FTS5 full-text fallback when the ONNX model
isn't loaded). Memories carry an ``event_type`` (``decision``,
``lesson``, ``error``, ``summary``, ``user_preference``, ...) and are
linked into a typed relationship graph (``edges`` table).

This adapter goes through OMEGA's Python library API (``from omega
import store, query_structured``) rather than its MCP server, so it
inherits OMEGA's actual storage + retrieval path with no extra hops.

Verified against omega-memory 1.4.15. The two facts that matter and are
*not* obvious from the README:

  * ``omega.query()`` returns a **formatted string** ("Results: N\\n## 1.
    ...") meant for an LLM to read. The machine-readable path is
    ``omega.query_structured(text, limit=...) -> list[dict]`` with
    ``id`` / ``content`` / ``event_type`` / ``relevance`` / ``strength``
    fields. SME uses ``query_structured``; it falls back to parsing the
    string form only if ``query_structured`` is unavailable (older
    OMEGA).
  * OMEGA resolves its store location from the ``OMEGA_HOME`` *directory*
    env var, **not** a db-file path. To keep a benchmark run from
    polluting the user's real ``~/.omega`` store, this adapter sets
    ``OMEGA_HOME`` to an isolated directory before importing ``omega``
    and restores the previous value on ``close()``.

``get_graph_snapshot()`` walks OMEGA's SQLite tables (``memories`` +
``edges``) to project memories + typed edges into SME's
``Entity``/``Edge`` shape — OMEGA's edges are exactly the structural
relationships SME is asking about.

Optional dependency. Construction raises ImportError with an install
hint if ``omega-memory`` isn't installed.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from sme.adapters.base import Edge, Entity, QueryResult, SMEAdapter

log = logging.getLogger(__name__)


DEFAULT_OMEGA_HOME = "~/.omega"
DB_FILENAME = "omega.db"


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
        omega_home: Directory OMEGA should use as its store root. OMEGA
            reads this from the ``OMEGA_HOME`` env var; this adapter sets
            it (and restores the previous value on ``close()``) so a
            benchmark run never writes into the user's real ``~/.omega``.
            Defaults to whatever ``OMEGA_HOME`` is already set to, else
            ``~/.omega``.
        db_path: Path to the OMEGA SQLite file. Accepted for CLI parity
            with the other adapters. When given, its **parent directory**
            becomes ``omega_home`` (OMEGA always names the file
            ``omega.db`` inside its home, so a custom basename is
            ignored — a warning is logged if it differs).
        default_memory_type: ``event_type`` applied to corpus rows that
            don't specify one. OMEGA's documented vocabulary is
            ``decision`` | ``lesson`` | ``error`` | ``summary`` |
            ``user_preference``. ``summary`` is the safest catch-all for
            arbitrary seeded content.
        n_results: Default top-K for ``query()``.
        read_only: Accepted for CLI parity. The diagnostic snapshot
            connection is always opened read-only regardless.
    """

    def __init__(
        self,
        *,
        omega_home: Optional[str] = None,
        db_path: Optional[str] = None,
        default_memory_type: str = "summary",
        n_results: int = 10,
        read_only: bool = True,
    ) -> None:
        # Resolve the store home BEFORE importing omega — omega reads
        # OMEGA_HOME at import time (module-level constant in omega.bridge).
        resolved_home = self._resolve_home(omega_home, db_path)
        self.omega_home = resolved_home
        self.db_path = str(Path(resolved_home) / DB_FILENAME)
        self.default_memory_type = default_memory_type
        self.n_results = n_results

        # Point omega at our isolated home, remembering the prior value so
        # close() can restore it. A sentinel distinguishes "was unset".
        self._prev_omega_home = os.environ.get("OMEGA_HOME", _UNSET)
        os.environ["OMEGA_HOME"] = resolved_home
        Path(resolved_home).mkdir(parents=True, exist_ok=True)

        self._omega = _require_omega()
        self._kg_conn: Optional[sqlite3.Connection] = None

        # OMEGA caches a SQLiteStore singleton keyed off the OMEGA_HOME it
        # saw at first construction. SQLiteStore re-reads os.environ
        # ["OMEGA_HOME"] each time it's built, so dropping the cached
        # singleton makes the next store()/query() re-target our isolated
        # home — essential for per-question isolation in the benchmark
        # runner and for not writing into the user's real ~/.omega.
        self._reset_omega_singleton()

    def _bind_store_to_home(self, attempts: int = 5) -> None:
        """Force OMEGA's store singleton to (re)bind to our OMEGA_HOME.

        Running OMEGA repeatedly in one process (the per-question
        benchmark pattern) is racy: a prior adapter's background
        daemon thread (auto-relate) can lazily rebuild the cached store
        singleton against a *stale* home between our env-set and the next
        write, so ``store()`` silently lands in the wrong db. We join
        outstanding background threads, drop the singleton, then build a
        fresh one and verify its ``db_path`` matches ours — retrying a
        few times because the offending thread may re-trip the singleton
        immediately after we reset it. Best-effort: if OMEGA's internals
        change, we simply proceed (writes still go through, just without
        the strict guarantee)."""
        bridge = getattr(self._omega, "bridge", None)
        get_store = getattr(bridge, "_get_store", None)
        for _ in range(max(1, attempts)):
            self._await_omega_background(timeout=5.0)
            self._reset_omega_singleton()
            os.environ["OMEGA_HOME"] = self.omega_home
            if not callable(get_store):
                return
            try:
                store = get_store()
            except Exception as e:  # noqa: BLE001
                log.debug("omega _get_store during bind failed: %s", e)
                return
            store_db = str(getattr(store, "db_path", ""))
            if os.path.realpath(store_db) == os.path.realpath(self.db_path):
                return
            log.debug(
                "omega store bound to %s, expected %s — retrying bind",
                store_db, self.db_path,
            )
        log.warning(
            "omega store did not bind to %s after %d attempts; writes may "
            "be racy under repeated in-process re-isolation",
            self.db_path, attempts,
        )

    def _await_omega_background(self, timeout: float = 5.0) -> None:
        """Join OMEGA's background daemon threads (auto-relate /
        entity-extraction) so deferred edge/relationship writes have
        landed before we read the SQLite file directly.

        OMEGA spawns these as ``daemon=True`` threads from ``store()`` and
        registers them on the store via ``register_background_thread``.
        Reading the db while they're mid-write yields a racy snapshot
        (sometimes zero rows). Reaching the registered thread list keeps
        us from guessing at sleep durations. Best-effort: degrade quietly
        if a future OMEGA changes the attribute name."""
        try:
            bridge = getattr(self._omega, "bridge", None)
            get_store = getattr(bridge, "_get_store", None)
            if not callable(get_store):
                return
            store = get_store()
        except Exception as e:  # noqa: BLE001
            log.debug("omega _get_store unavailable: %s", e)
            return
        lock = getattr(store, "_bg_threads_lock", None)
        threads = getattr(store, "_background_threads", None)
        if threads is None:
            return
        try:
            if lock is not None:
                with lock:
                    pending = list(threads)
            else:
                pending = list(threads)
            for t in pending:
                if t.is_alive():
                    t.join(timeout=timeout)
        except Exception as e:  # noqa: BLE001
            log.debug("omega background-thread join failed: %s", e)

    def _reset_omega_singleton(self) -> None:
        """Drop OMEGA's cached SQLiteStore so the next call re-resolves
        OMEGA_HOME. ``reset_memory`` lives in ``omega.bridge`` and is
        documented as "useful for testing"; degrade quietly if a future
        OMEGA renames it."""
        reset = getattr(self._omega, "reset_memory", None)
        if reset is None:
            bridge = getattr(self._omega, "bridge", None)
            reset = getattr(bridge, "reset_memory", None)
        if callable(reset):
            try:
                reset()
            except Exception as e:  # noqa: BLE001
                log.debug("omega reset_memory failed: %s", e)

    @staticmethod
    def _resolve_home(
        omega_home: Optional[str], db_path: Optional[str]
    ) -> str:
        if omega_home:
            base = omega_home
        elif db_path:
            p = Path(os.path.expanduser(db_path))
            if p.name and p.name != DB_FILENAME:
                log.warning(
                    "omega ignores the db basename %r — it always uses "
                    "%s inside OMEGA_HOME; using parent dir %s as home",
                    p.name, DB_FILENAME, p.parent,
                )
            base = str(p.parent)
        else:
            base = os.environ.get("OMEGA_HOME") or DEFAULT_OMEGA_HOME
        return str(Path(os.path.expanduser(base)).resolve())

    # --- SMEAdapter required ------------------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """Seed OMEGA from a corpus list.

        Each corpus row is a dict with at least ``content`` (the text to
        store). Optional fields:
          - ``type`` / ``event_type``: OMEGA ``event_type`` override
            (defaults to ``default_memory_type``)
          - ``metadata``: dict passed through to ``omega.store``
        """
        # Guarantee OMEGA's store is (re)bound to *our* OMEGA_HOME before
        # the first write. A prior adapter's close() leaves a daemon
        # background thread that can lazily rebuild the singleton against
        # a stale home; re-asserting the env + dropping the singleton here
        # makes the next store() land in this adapter's db deterministically.
        os.environ["OMEGA_HOME"] = self.omega_home
        self._bind_store_to_home()

        errors: list[str] = []
        warnings: list[str] = []
        stored = 0
        for i, row in enumerate(corpus):
            content = row.get("content") or row.get("text") or ""
            if not content:
                warnings.append(f"row {i}: empty content, skipped")
                continue
            mem_type = (
                row.get("type")
                or row.get("event_type")
                or self.default_memory_type
            )
            metadata = row.get("metadata")
            # Optional session_id passthrough. OMEGA stores it as a first-
            # class column and returns it on every query_structured hit, so a
            # benchmark runner can compute session-level hit@K (LongMemEval
            # R@K) against the question's expected session ids without
            # round-tripping through OMEGA's opaque mem ids.
            session_id = row.get("session_id")
            try:
                store_kwargs: dict[str, Any] = {}
                if metadata is not None:
                    store_kwargs["metadata"] = metadata
                if session_id is not None:
                    store_kwargs["session_id"] = session_id
                self._omega.store(content, mem_type, **store_kwargs)
                stored += 1
            except TypeError as e:
                # Signature mismatch with the installed omega-memory.
                errors.append(
                    f"row {i}: omega.store rejected its arguments ({e}) — "
                    "check the installed omega-memory version"
                )
                break
            except Exception as e:  # noqa: BLE001 — record and continue
                errors.append(f"row {i}: {e}")
        return {
            "entities_created": stored,
            # OMEGA creates `related` edges asynchronously after its
            # auto-relate pass; we can't count them from the store calls.
            "edges_created": 0,
            "errors": errors,
            "warnings": warnings,
        }

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        route: bool = False,  # accepted for CLI parity; OMEGA ranks itself
    ) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        try:
            hits = self._retrieve_hits(question, k)
        except Exception as e:  # noqa: BLE001
            return QueryResult(
                answer="", context_string="", error=f"INTERNAL: {e}"
            )

        if not hits:
            return QueryResult(answer="", context_string="", error="NO_RESULTS")

        hits = hits[:k]
        context_parts: list[str] = []
        retrieved: list[Entity] = []
        for i, hit in enumerate(hits):
            text = hit.get("content") or hit.get("memory") or hit.get("text", "")
            mem_type = (
                hit.get("event_type")
                or hit.get("type")
                or hit.get("memory_type")
                or "?"
            )
            # Use ``is not None`` (not ``or``) — a legitimate 0.0 score /
            # rank is falsy and would otherwise be dropped.
            score = next(
                (
                    hit.get(k)
                    for k in ("relevance", "strength", "score", "similarity", "rank")
                    if hit.get(k) is not None
                ),
                None,
            )
            mem_id = str(hit.get("id") or f"omega_hit:{i}")
            # OMEGA carries the ingest-time ``session_id`` through into
            # query_structured hits. Surfacing it on the Entity lets a
            # benchmark runner compute session-level hit@K (the LongMemEval
            # R@K metric) by matching against the question's expected
            # session ids — the OMEGA analogue of the daemon's
            # session→drawer_id map. None when the memory was stored without
            # a session_id (e.g. the smoke corpus).
            session_id = hit.get("session_id")
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
                        "session_id": session_id,
                    },
                )
            )
        context_string = "\n\n".join(context_parts)
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=retrieved,
            retrieval_path=[f"omega_query_structured:k={k}"],
        )

    def _retrieve_hits(self, question: str, k: int) -> list[dict]:
        """Return a list[dict] of hits from OMEGA's real retrieval path.

        Prefers ``query_structured`` (machine-readable list[dict]) and
        only falls back to parsing the human-readable ``query`` string if
        ``query_structured`` is missing on the installed OMEGA.
        """
        qs = getattr(self._omega, "query_structured", None)
        if callable(qs):
            try:
                raw = qs(question, limit=k)
            except TypeError:
                raw = qs(question)
            return _normalise_query_hits(raw)
        # Older OMEGA: only the string-returning query() exists.
        raw = self._omega.query(question)
        return _normalise_query_hits(raw)

    def get_graph_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        """Project OMEGA's SQLite store into SME (entities, edges).

        OMEGA stores each memory as a row in the ``memories`` table and
        each typed relationship as a row in an ``edges`` table. Schema
        names may drift between OMEGA versions; this method introspects
        ``sqlite_master`` and degrades gracefully when columns are absent
        rather than raising.
        """
        # OMEGA writes memory rows synchronously, but edge/relationship
        # rows land via background daemon threads (auto-relate). Join
        # those first so edges are present, then read.
        self._await_omega_background()

        if not Path(self.db_path).exists():
            log.warning("omega db not found at %s; returning empty snapshot",
                        self.db_path)
            return [], []

        # Read from our OWN fresh connection, NOT OMEGA's live store
        # connection: OMEGA's connection is shared with its auto-relate
        # daemon thread, and issuing our SELECTs on it concurrently with
        # that thread's writes yields a racy/empty view (and sqlite3
        # connections are thread-bound, so a cross-thread read would raise
        # ProgrammingError). We open read-WRITE (not mode=ro) so the
        # connection can map the WAL -shm index — a mode=ro connection that
        # can't map -shm reads only the main db file and may miss
        # WAL-resident rows. We only SELECT, so this is non-destructive. A
        # short bounded retry absorbs the window where OMEGA's background
        # thread briefly holds a write transaction (during which a SELECT
        # can transiently see zero synchronously-written rows).
        last_err: Optional[sqlite3.Error] = None
        for attempt in range(4):
            try:
                conn = sqlite3.connect(self.db_path, timeout=2.0)
            except sqlite3.Error as e:
                log.warning("could not open omega db: %s", e)
                return [], []
            try:
                entities = _read_omega_memories(conn)
                edges = _read_omega_edges(conn)
            except sqlite3.Error as e:
                # Corruption / locking / unexpected schema — degrade
                # gracefully (the adapter never raises from a read).
                last_err = e
                entities, edges = [], []
            finally:
                conn.close()
            # A non-empty memories read is authoritative. An empty read on
            # an early attempt may be the background-write window; retry
            # briefly before accepting "genuinely empty".
            if entities or attempt == 3:
                if not entities and last_err is not None:
                    log.warning("omega snapshot read error: %s", last_err)
                return entities, edges
            self._await_omega_background(timeout=0.5)
        return [], []

    # --- optional helpers ---------------------------------------------

    def reset(self) -> None:
        """Clear OMEGA's SQLite store for this adapter's home.

        Removes the db file (and WAL/SHM siblings) so OMEGA recreates a
        clean store on the next ``store()``. Refuses to touch the default
        user-level home unless ``OMEGA_ALLOW_DEFAULT_RESET=1`` is set, to
        avoid wiping a user's real memories.
        """
        default_home = str(
            Path(os.path.expanduser(DEFAULT_OMEGA_HOME)).resolve()
        )
        if (
            self.omega_home == default_home
            and os.environ.get("OMEGA_ALLOW_DEFAULT_RESET") != "1"
        ):
            raise RuntimeError(
                "refusing to reset the default OMEGA store "
                f"({self.db_path}). Set OMEGA_ALLOW_DEFAULT_RESET=1 or "
                "pass an explicit omega_home/db_path to OmegaAdapter."
            )
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(self.db_path + suffix).unlink(missing_ok=True)
            except OSError as e:
                log.warning("omega reset failed for %s%s: %s",
                            self.db_path, suffix, e)

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
                "OMEGA stores memories in SQLite with a typed event_type "
                "vocabulary (decision, lesson, error, summary, "
                "user_preference) and links them via typed edges in an "
                "`edges` table (edge_type: related, supersedes, "
                "contradicts). Embeddings are 384-dim bge-small-en-v1.5 "
                "ONNX vectors stored in sqlite-vec, with an FTS5 "
                "full-text fallback. Verified against omega-memory 1.4.15."
            ),
        }

    def close(self) -> None:
        if self._kg_conn is not None:
            try:
                self._kg_conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._kg_conn = None
        # Restore the user's prior OMEGA_HOME so we never leave the
        # process pointed at our isolated benchmark store, then drop the
        # cached store singleton so the next consumer rebuilds against the
        # restored home rather than our (now stale) benchmark one.
        if self._prev_omega_home is _UNSET:
            os.environ.pop("OMEGA_HOME", None)
        else:
            os.environ["OMEGA_HOME"] = self._prev_omega_home
        self._reset_omega_singleton()


# --- helpers --------------------------------------------------------


class _Unset:
    """Sentinel for 'OMEGA_HOME was not set before we touched it'."""

    __slots__ = ()


_UNSET = _Unset()


def _normalise_query_hits(raw: Any) -> list[dict]:
    """Normalise an OMEGA retrieval return into a list[dict] of hits.

    ``query_structured`` already returns a list[dict]; this also accepts
    the dict-envelope and list-of-strings shapes, plus the human-readable
    string returned by ``query()`` (parsed line-by-line) so the older
    string-only fallback still surfaces results.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return _parse_query_string(raw)
    if isinstance(raw, dict):
        inner = raw.get("results") or raw.get("memories") or raw.get("hits")
        if isinstance(inner, list):
            return [_coerce_hit(h) for h in inner]
        return []
    if isinstance(raw, list):
        return [_coerce_hit(h) for h in raw]
    return []


def _parse_query_string(text: str) -> list[dict]:
    """Parse OMEGA's human-readable query() output into hit dicts.

    The format (omega-memory 1.4.x) is::

        Results: N
        ## 1. [event_type] `mem-id` (str: 1.00)
        <content line(s)>
        *timestamp*

    We extract one hit per ``## n.`` header, pulling event_type, id, the
    leading score, and the content lines until the next header / blank.
    This is a best-effort fallback only; ``query_structured`` is the
    primary path on modern OMEGA.
    """
    import re

    hits: list[dict] = []
    header_re = re.compile(
        r"^##\s*\d+\.\s*\[(?P<etype>[^\]]+)\]\s*`?(?P<id>[^`\s]+)`?"
        r"(?:\s*\((?:[a-z]+:\s*)?(?P<score>[0-9.]+)\))?",
    )
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = header_re.match(lines[i].strip())
        if not m:
            i += 1
            continue
        etype = m.group("etype")
        mem_id = m.group("id")
        score = m.group("score")
        # Collect content lines until the next header or a metadata/blank line.
        content_lines: list[str] = []
        i += 1
        while i < len(lines):
            ln = lines[i].strip()
            if ln.startswith("##") or ln.startswith("**") or ln.startswith("*") or not ln:
                break
            content_lines.append(ln)
            i += 1
        hits.append(
            {
                "id": mem_id,
                "content": " ".join(content_lines).strip(),
                "event_type": etype,
                "relevance": float(score) if score else None,
            }
        )
    return hits


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
    The real type column on omega-memory 1.4.x is ``event_type``.
    Returns an empty list if no recognised table exists.
    """
    table = _first_existing_table(conn, _MEMORY_TABLE_CANDIDATES)
    if table is None:
        return []
    cols = _table_columns(conn, table)
    id_col = _first_col(cols, ("id", "memory_id", "rowid"))
    content_col = _first_col(cols, ("content", "text", "memory"))
    type_col = _first_col(cols, ("event_type", "type", "memory_type", "kind"))
    if id_col is None or content_col is None:
        log.warning(
            "omega memories table %r missing id/content columns; got %s",
            table, cols,
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
    etype = _first_col(cols, ("edge_type", "type", "relation"))
    if src is None or dst is None:
        log.warning(
            "omega edges table %r missing source/target columns; got %s",
            table, cols,
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
