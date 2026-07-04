"""SME adapter for engram — the TypeScript MCP memory server by 199-bio.

engram (``@199-bio/engram``, MIT) is a local-first MCP server: BM25 (SQLite
FTS5) + semantic embeddings (Transformers.js) + a knowledge graph, with
Ebbinghaus-decay and consolidation, all persisted in a single SQLite file.

This adapter measures engram the way it is actually used — as a local
stdio MCP server. It:

* **spawns** ``node <engram>/dist/index.js`` with ``ENGRAM_TRANSPORT=stdio``
  and a per-instance ``ENGRAM_DB_PATH``, and speaks MCP JSON-RPC over
  stdin/stdout (the ``remember`` / ``recall`` / ``forget`` tools);
* **owns the DB path**, which gives clean corpus isolation (a fresh temp
  DB per adapter, wiped on ``reset``) — engram's ``forget`` is only a
  soft-delete, so a fresh DB is the correct isolation primitive;
* reads engram's ``entities`` / ``relations`` / ``contradictions`` SQLite
  tables **directly** (read-only; WAL makes this concurrency-safe) for the
  structural categories — the same direct-DB precedent as the in-tree
  ``MemPalaceAdapter``. This yields a real knowledge-graph snapshot and
  real Cat 3 contradiction pairs rather than an empty stub.

The MCP transport is injectable (``transport=``) so the unit tests need no
Node runtime; the one live test is opt-in (``ENGRAM_LIVE=1``) and skips
cleanly when node / a built ``dist/`` is absent.

engram supports an HTTP transport too (``ENGRAM_TRANSPORT=http``), but
since the adapter must control the DB path for isolation it has to launch
the process regardless — and stdio (engram's default) avoids HTTP port
allocation, SSE framing, and the stateless-handshake nuance.
"""

from __future__ import annotations

import json
import logging
import os
import select
import shutil
import sqlite3
import subprocess
import tempfile
import time
from typing import Any, Optional

from sme.adapters.base import (
    ContradictionPair,
    Edge,
    Entity,
    HarnessDescriptor,
    ProbeResult,
    QueryResult,
    SMEAdapter,
)

log = logging.getLogger(__name__)

DEFAULT_NODE = "node"
DEFAULT_N_RESULTS = 5
DEFAULT_IMPORTANCE = 0.5
DEFAULT_STARTUP_TIMEOUT = 30.0
DEFAULT_CALL_TIMEOUT = 60.0
_MCP_PROTOCOL_VERSION = "2025-06-18"


class EngramTransportError(RuntimeError):
    """Raised when the engram subprocess can't be spawned or a JSON-RPC
    exchange fails (spawn error, handshake failure, timeout, MCP error,
    dead process). Adapter methods catch this and degrade gracefully —
    they never propagate it to the SME harness."""


class EngramStdioTransport:
    """A minimal MCP-over-stdio client for a spawned engram subprocess.

    Lazy: the process is spawned and the ``initialize`` handshake runs on
    the first ``call``. Messages are newline-delimited JSON-RPC (the MCP
    stdio framing); engram writes protocol to stdout and logs to stderr,
    so stdout stays clean. Non-JSON stdout lines (a stray banner) are
    skipped defensively.
    """

    def __init__(
        self,
        *,
        entry: str,
        db_dir: str,
        node_bin: str = DEFAULT_NODE,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
        env: Optional[dict] = None,
    ) -> None:
        self.entry = entry
        self.db_dir = db_dir
        self.node_bin = node_bin
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self._env_extra = env or {}
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 0
        self._started = False
        self._buf = b""  # unread stdout bytes (own line-buffer; see _next_message)

    def _spawn(self) -> None:
        env = dict(os.environ)
        env["ENGRAM_TRANSPORT"] = "stdio"
        env["ENGRAM_DB_PATH"] = self.db_dir
        env.update(self._env_extra)
        try:
            # Unbuffered BINARY pipes. We do our own newline framing over
            # os.read on the raw fd so select() is authoritative — a text-mode
            # buffered readline() would read ahead into a Python buffer that
            # select() can't see, deadlocking the response wait.
            self._proc = subprocess.Popen(  # noqa: S603 — argv list, no shell
                [self.node_bin, self.entry],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                bufsize=0,
            )
        except (OSError, ValueError) as e:
            raise EngramTransportError(
                f"failed to spawn engram ({self.node_bin} {self.entry}): {e}"
            ) from e

    def _ensure_started(self) -> None:
        if self._started:
            return
        if self._proc is None:
            self._spawn()
        init_id = self._new_id()
        self._send({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "sme-engram-adapter", "version": "1"},
            },
        })
        self._read_result(expected_id=init_id, timeout=self.startup_timeout)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._started = True

    def call(self, tool: str, arguments: dict) -> dict:
        """Invoke an MCP tool; return the parsed JSON object carried in
        ``result.content[0].text`` (engram's convention)."""
        self._ensure_started()
        rid = self._new_id()
        self._send({
            "jsonrpc": "2.0",
            "id": rid,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        })
        result = self._read_result(expected_id=rid, timeout=self.call_timeout)
        content = (result or {}).get("content") or []
        if not content:
            sc = (result or {}).get("structuredContent")
            if isinstance(sc, dict):
                return sc
            raise EngramTransportError(f"tool {tool!r} returned no content")
        text = content[0].get("text", "") if isinstance(content[0], dict) else ""
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, msg: dict) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise EngramTransportError("engram subprocess is not running")
        try:
            proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise EngramTransportError(f"failed to write to engram: {e}") from e

    def _read_result(self, *, expected_id: int, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            msg = self._next_message(deadline)
            if msg is None:
                continue  # skipped a blank / non-JSON-RPC line — keep reading
            if msg.get("id") != expected_id:
                continue  # a notification or an unrelated id
            if "error" in msg:
                raise EngramTransportError(f"engram MCP error: {msg['error']}")
            return msg.get("result") or {}

    def _next_message(self, deadline: float) -> Optional[dict]:
        """Return the next JSON-RPC message, or ``None`` when a blank /
        non-protocol line was consumed (caller loops). Raises on deadline,
        EOF, or subprocess exit. Uses os.read on the raw fd + an internal
        byte buffer so no data hides in a Python read-ahead buffer."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise EngramTransportError("engram subprocess has no stdout")
        fd = proc.stdout.fileno()
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                raw, self._buf = self._buf[:nl], self._buf[nl + 1:]
                line = raw.strip()
                if not line:
                    return None
                try:
                    obj = json.loads(line.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return None  # skip a non-protocol banner/log line
                return obj if isinstance(obj, dict) and obj.get("jsonrpc") == "2.0" else None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngramTransportError("timed out waiting for engram response")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                if proc.poll() is not None:
                    raise EngramTransportError(
                        f"engram exited (code {proc.returncode}) before responding"
                    )
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise EngramTransportError(f"engram closed stdout (exit {proc.poll()})")
            self._buf += chunk

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception:  # pragma: no cover - defensive teardown
            pass
        finally:
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # pragma: no cover
                    pass
            self._proc = None
            self._started = False


class EngramAdapter(SMEAdapter):
    """SMEAdapter for a spawned engram stdio MCP server.

    Construction never spawns and never raises for missing config — the
    transport is built lazily on first use. If ``engram_path`` is unset
    (and no ``transport`` is injected), ``query`` / ``ingest_corpus``
    degrade to error-bearing results and ``get_graph_snapshot`` returns
    ``([], [])`` — so the contract suite runs clean with no Node runtime.

    Args:
        engram_path: Path to the engram checkout (must contain
            ``dist/index.js`` — run ``npm install && npm run build`` once).
            Defaults to ``$ENGRAM_PATH``.
        node_bin: Node executable (default ``node``).
        db_path: ``ENGRAM_DB_PATH`` directory. ``None`` (default) creates a
            private temp dir the adapter owns and deletes on ``reset`` /
            ``close``; an explicit path is treated as caller-owned (``reset``
            wipes its rows in place — point it at a dedicated benchmark DB).
        n_results: Default top-K for ``query``.
        include_graph: Default for ``recall``'s graph-expansion flag.
        importance: Default importance score for ingested memories.
        reset_before_ingest: When True (default), wipe before each
            ``ingest_corpus`` — per-corpus isolation.
        startup_timeout / call_timeout: stdio handshake / per-call limits.
        transport: Inject a transport (tests); bypasses spawning.
        read_only: Accepted for CLI parity; ignored (ingest/reset mutate).
    """

    def __init__(
        self,
        *,
        engram_path: Optional[str] = None,
        node_bin: str = DEFAULT_NODE,
        db_path: Optional[str] = None,
        n_results: int = DEFAULT_N_RESULTS,
        include_graph: bool = True,
        importance: float = DEFAULT_IMPORTANCE,
        reset_before_ingest: bool = True,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
        transport: Optional[Any] = None,
        read_only: bool = False,
    ) -> None:
        self.engram_path = engram_path or os.environ.get("ENGRAM_PATH")
        self.node_bin = node_bin
        self.n_results = n_results
        self.include_graph = include_graph
        self.importance = importance
        self.reset_before_ingest = reset_before_ingest
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self._injected_transport = transport
        self._transport = transport
        self._id_map: dict[str, str] = {}
        self._node_version: Optional[str] = None

        if db_path:
            self.db_dir = os.path.abspath(os.path.expanduser(db_path))
            self._owns_db = False
        else:
            self.db_dir = tempfile.mkdtemp(prefix="engram-sme-")
            self._owns_db = True

        self._entry = (
            os.path.join(os.path.expanduser(self.engram_path), "dist", "index.js")
            if self.engram_path
            else None
        )
        self.engram_version = self._read_engram_version()

    # --- transport lifecycle ------------------------------------------

    def _get_transport(self):
        if self._transport is not None:
            return self._transport
        if not self._entry:
            raise EngramTransportError(
                "engram_path is not set — cannot spawn engram. Pass engram_path "
                "or set $ENGRAM_PATH (and build dist/ with npm run build)."
            )
        self._transport = EngramStdioTransport(
            entry=self._entry,
            db_dir=self.db_dir,
            node_bin=self.node_bin,
            startup_timeout=self.startup_timeout,
            call_timeout=self.call_timeout,
        )
        return self._transport

    def _close_transport(self) -> None:
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:  # pragma: no cover - defensive
                pass
            # Drop a self-built transport so the next call re-spawns on the
            # (possibly wiped) DB; keep an injected one for test reuse.
            if self._injected_transport is None:
                self._transport = None

    # --- SMEAdapter required methods ----------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        errors: list[str] = []
        warnings: list[str] = []

        if self.reset_before_ingest:
            try:
                removed = self.reset()
                warnings.append(f"reset before ingest: {removed} memories removed")
            except Exception as e:  # pragma: no cover - defensive
                warnings.append(f"reset before ingest failed: {e}")

        try:
            transport = self._get_transport()
        except EngramTransportError as e:
            return {
                "entities_created": 0,
                "edges_created": 0,
                "errors": [f"transport unavailable: {e}"],
                "warnings": warnings,
            }

        created = existed = edges = 0
        for row in corpus:
            content = row.get("document") or row.get("content") or row.get("text") or ""
            if not str(content).strip():
                continue
            args: dict[str, Any] = {
                "content": str(content),
                "importance": row.get("importance", self.importance),
            }
            ents = row.get("entities")
            if ents:
                args["entities"] = ents
            rels = row.get("relationships") or row.get("edges")
            if rels:
                args["relationships"] = rels

            try:
                resp = transport.call("remember", args)
            except Exception as e:
                errors.append(f"remember failed (id={row.get('id')}): {e}")
                continue
            if not isinstance(resp, dict):
                errors.append(f"remember returned non-dict (id={row.get('id')})")
                continue
            if resp.get("duplicate"):
                existed += 1
                continue
            mem_id = resp.get("memory_id")
            if resp.get("success") and mem_id:
                created += 1
                corpus_id = row.get("id")
                if corpus_id is not None:
                    self._id_map[str(mem_id)] = str(corpus_id)
                edges += len(resp.get("relationships_stored") or [])
            else:
                errors.append(f"remember unsuccessful (id={row.get('id')}): {resp}")

        if existed:
            warnings.append(f"{existed} duplicate memory(ies) skipped by engram")

        return {
            "entities_created": created,
            "edges_created": edges,
            "errors": errors,
            "warnings": warnings,
        }

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        include_graph: Optional[bool] = None,
    ) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        ig = include_graph if include_graph is not None else self.include_graph
        try:
            transport = self._get_transport()
            resp = transport.call(
                "recall", {"query": question, "limit": k, "include_graph": ig}
            )
        except Exception as e:
            return QueryResult(
                answer="", context_string="", error=f"{type(e).__name__}: {e}"
            )

        context = resp.get("context") if isinstance(resp, dict) else None
        ids = (resp.get("_ids") if isinstance(resp, dict) else None) or []
        retrieval_path = [f"recall:k={k}", f"include_graph={ig}", f"db={self.db_dir}"]
        if not context:
            return QueryResult(
                answer="", context_string="", error="NO_RESULTS",
                retrieval_path=retrieval_path,
            )

        context_string = "\n".join(str(c) for c in context)
        retrieved: list[Entity] = []
        for i, mem_id in enumerate(ids):
            mem_id = str(mem_id)
            corpus_id = self._id_map.get(mem_id, mem_id)
            retrieved.append(
                Entity(
                    id=mem_id,
                    name=corpus_id,
                    entity_type="engram:memory",
                    properties={
                        "_table": "engram_memory",
                        "rank": i + 1,
                        "source_id": corpus_id,
                    },
                )
            )
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=retrieved,
            retrieval_path=retrieval_path,
        )

    def get_graph_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        """Read engram's ``entities`` + ``relations`` tables directly.

        The adapter owns the SQLite file (via ``ENGRAM_DB_PATH``); WAL mode
        makes a concurrent read-only pass safe even while the subprocess is
        live. Returns ``([], [])`` when the DB or tables are absent."""
        con = self._open_db()
        if con is None:
            return [], []
        try:
            entities: list[Entity] = []
            edges: list[Edge] = []
            present: set[str] = set()
            try:
                rows = con.execute(
                    "SELECT id, name, type, metadata FROM entities"
                ).fetchall()
            except sqlite3.Error:
                return [], []
            for eid, name, etype, meta in rows:
                nid = f"entity:{eid}"
                present.add(nid)
                props: dict[str, Any] = {"_table": "engram_entity"}
                if meta:
                    try:
                        props["metadata"] = json.loads(meta)
                    except (ValueError, TypeError):
                        pass
                entities.append(
                    Entity(
                        id=nid,
                        name=name or str(eid),
                        entity_type=f"engram:{etype or 'unknown'}",
                        properties=props,
                    )
                )
            try:
                rel_rows = con.execute(
                    "SELECT from_entity, to_entity, type, properties FROM relations"
                ).fetchall()
            except sqlite3.Error:
                rel_rows = []
            for frm, to, rtype, _props in rel_rows:
                if frm is None or to is None:
                    continue
                edges.append(
                    Edge(
                        source_id=f"entity:{frm}",
                        target_id=f"entity:{to}",
                        edge_type=rtype or "related",
                        properties={"_table": "engram_relation"},
                    )
                )
            # Guarantee internal consistency: synthesize any edge endpoint
            # missing from the entity set (FK should prevent this, but a
            # partial/legacy DB shouldn't break the contract).
            for e in edges:
                for endpoint in (e.source_id, e.target_id):
                    if endpoint not in present:
                        present.add(endpoint)
                        entities.append(
                            Entity(
                                id=endpoint,
                                name=endpoint.split("entity:", 1)[-1],
                                entity_type="engram:entity",
                                properties={"_table": "engram_entity", "_endpoint_only": True},
                            )
                        )
            return entities, edges
        finally:
            con.close()

    # --- Optional SMEAdapter methods ----------------------------------

    def get_contradiction_pairs(self) -> list[ContradictionPair]:
        """Read engram's native ``contradictions`` table (Cat 3).

        Beats the base-class default (which derives pairs from
        ``contradicts`` edges) because engram tracks contradictions
        explicitly. Returns ``[]`` when the table/DB is absent."""
        con = self._open_db()
        if con is None:
            return []
        try:
            try:
                rows = con.execute(
                    "SELECT memory_id_a, memory_id_b, description FROM contradictions"
                ).fetchall()
            except sqlite3.Error:
                return []
            mem: dict[str, str] = {}
            try:
                for mid, content in con.execute(
                    "SELECT id, content FROM memories"
                ).fetchall():
                    mem[str(mid)] = content
            except sqlite3.Error:
                pass
            pairs: list[ContradictionPair] = []
            seen: set[frozenset] = set()
            for a, b, desc in rows:
                if a is None or b is None:
                    continue
                key = frozenset({str(a), str(b)})
                if key in seen:
                    continue
                seen.add(key)
                claim_a = mem.get(str(a)) or desc or str(a)
                claim_b = mem.get(str(b)) or str(b)
                pairs.append(
                    ContradictionPair(
                        claim_a=str(claim_a),
                        claim_b=str(claim_b),
                        source_a=str(a),
                        source_b=str(b),
                    )
                )
            return pairs
        finally:
            con.close()

    def get_flat_retrieval(self, question: str, k: int = 5) -> QueryResult:
        """Pure retrieval — engram's ``recall`` with graph expansion off."""
        return self.query(question, n_results=k, include_graph=False)

    def get_ontology_source(self) -> dict:
        return {
            "type": "declared",
            "schema": [
                {
                    "kind": "structural",
                    "entities": ["memory", "episode", "digest"],
                },
                {
                    "kind": "knowledge_graph",
                    "entities": ["entity", "relation", "observation", "contradiction"],
                },
                {
                    "kind": "entity_types",
                    "values": ["person", "place", "concept", "event", "organization"],
                },
            ],
            "documentation": (
                "engram (@199-bio/engram): local-first MCP memory server. Stores "
                "verbatim memories with hybrid retrieval (BM25 FTS5 + semantic "
                "embeddings), a knowledge graph of entities/relations/observations, "
                "explicit contradiction tracking, Ebbinghaus temporal decay, and "
                "memory consolidation into digests. Backed by SQLite; accessed here "
                "over the stdio MCP transport with a direct read of the graph tables."
            ),
        }

    def get_harness_manifest(self) -> list[HarnessDescriptor]:
        """engram's invocation surface is an MCP server (stdio). The probe
        does a real ``recall`` call-through (read-only) for Cat 9b. Version
        attribution (engram + node) rides in ``properties``."""
        versions = self._version_info()
        return [
            HarnessDescriptor(
                name="engram_mcp_stdio",
                kind="mcp_resource",
                probe_fn=self._probe_mcp,
                description="engram MCP server over stdio (spawned node subprocess).",
                properties={
                    "transport": "stdio",
                    "entry": self._entry or "(engram_path unset)",
                    "engram_version": versions["engram"],
                    "node_version": versions["node"],
                },
            )
        ]

    def _probe_mcp(self) -> ProbeResult:
        t0 = time.perf_counter()
        try:
            resp = self._get_transport().call("recall", {"query": "ping", "limit": 1})
            ok = isinstance(resp, dict)
        except Exception as e:
            return ProbeResult(
                success=False, latency_ms=(time.perf_counter() - t0) * 1000.0, error=str(e)
            )
        return ProbeResult(success=ok, latency_ms=(time.perf_counter() - t0) * 1000.0)

    # --- Isolation ----------------------------------------------------

    def reset(self) -> int:
        """Wipe the store to isolate the next corpus. Returns the number of
        memories that were present.

        Releases the subprocess first (so the DB file is free), then: for
        an adapter-owned temp DB, deletes the ``engram.db`` files; for a
        caller-provided DB, deletes all rows in place. Destructive by
        design — point ``db_path`` at a dedicated benchmark DB."""
        prior = self._count_memories()
        self._close_transport()
        self._id_map.clear()
        db_file = os.path.join(self.db_dir, "engram.db")
        if self._owns_db:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(db_file + suffix)
                except OSError:
                    pass
            try:
                os.remove(os.path.join(self.db_dir, "engram.pid"))
            except OSError:
                pass
        else:
            con = self._open_db(read_only=False)
            if con is not None:
                try:
                    for tbl in (
                        "digest_sources", "memory_connections", "contradictions",
                        "observations", "relations", "entities", "digests",
                        "episodes", "retrieval_logs", "memories",
                    ):
                        try:
                            con.execute(f"DELETE FROM {tbl}")
                        except sqlite3.Error:
                            pass
                    con.commit()
                finally:
                    con.close()
        return prior

    def close(self) -> None:
        self._close_transport()
        if self._owns_db and self.db_dir and os.path.isdir(self.db_dir):
            shutil.rmtree(self.db_dir, ignore_errors=True)

    # --- internals ----------------------------------------------------

    def _open_db(self, *, read_only: bool = True) -> Optional[sqlite3.Connection]:
        db_file = os.path.join(self.db_dir, "engram.db")
        if not os.path.exists(db_file):
            return None
        try:
            con = sqlite3.connect(db_file, timeout=5.0)
            con.execute("PRAGMA busy_timeout=3000")
            return con
        except sqlite3.Error:
            return None

    def _count_memories(self) -> int:
        con = self._open_db()
        if con is None:
            return 0
        try:
            try:
                return int(con.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            except sqlite3.Error:
                return 0
        finally:
            con.close()

    def _read_engram_version(self) -> Optional[str]:
        if not self.engram_path:
            return None
        pkg = os.path.join(os.path.expanduser(self.engram_path), "package.json")
        try:
            with open(pkg, encoding="utf-8") as f:
                return json.load(f).get("version")
        except (OSError, ValueError):
            return None

    def _detect_node_version(self) -> Optional[str]:
        try:
            out = subprocess.run(  # noqa: S603 — argv list, no shell
                [self.node_bin, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def _version_info(self) -> dict:
        if self._node_version is None:
            self._node_version = self._detect_node_version()
        return {"engram": self.engram_version, "node": self._node_version}
