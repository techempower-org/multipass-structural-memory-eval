"""Longhand adapter for SME.

Longhand (https://github.com/Wynelson94/longhand, by Nate Nelson) is a
persistent local memory server for Claude Code. It reads the raw session
JSONL that Claude Code already writes to
``~/.claude/projects/<project>/<session-id>.jsonl`` — every message, tool
call, thinking block, and file edit, verbatim — and indexes it locally
into SQLite (raw-JSON source of truth) plus ChromaDB (vector search) under
``~/.longhand/``. No network, no API calls, full local privacy.

This places Longhand squarely in the verbatim-first cohort: like
MemPalace it stores exact words rather than letting a model decide what
matters, and like the daemon it keeps a single writer over its own store.

**Daemon-strict design.** This adapter does NOT open Longhand's ChromaDB
or SQLite directly — that would mean a second process holding handles to
a store Longhand assumes it owns. Instead it shells out to the
``longhand`` CLI (``longhand search --json``), the same way
``mempalace_daemon`` talks HTTP instead of touching palace storage. The
CLI is the single supported read boundary that does not fight Longhand's
own writer.

Because Longhand ingests Claude Code session transcripts through its own
hooks rather than arbitrary seeded corpora, ``ingest_corpus`` is not
supported here — this is a diagnostic-only (Mode B) adapter, like
``mempalace_daemon``. Longhand exposes no knowledge graph, so
``get_graph_snapshot`` returns ``([], [])``; retrieval categories
(Cat 1/2c/3/6) and the harness-integration category (Cat 9) are the
meaningful readings for it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from sme.adapters.base import (
    Edge,
    Entity,
    HarnessDescriptor,
    ProbeResult,
    QueryResult,
    SMEAdapter,
)

log = logging.getLogger(__name__)


DEFAULT_BIN = "longhand"
DEFAULT_HOME = "~/.longhand"
DEFAULT_LIMIT = 5
DEFAULT_TIMEOUT = 30.0  # CLI search is vector-only; ~126ms typical, generous cap


class LonghandAdapter(SMEAdapter):
    """SMEAdapter that queries a local Longhand store via its CLI.

    Construction does not run anything — the binary is resolved eagerly so
    a misconfiguration surfaces at build time, but the first subprocess
    call happens in ``query()`` or a Cat 9 probe.

    Args:
        bin_path: Name or path of the Longhand CLI. Resolved against PATH.
            Defaults to ``"longhand"``. Raises ValueError if it cannot be
            found, so a typo fails loudly instead of every query erroring.
        home_dir: Longhand data dir, used only for a friendly
            not-initialised warning. Defaults to ``~/.longhand``.
        n_results: Default number of search hits per ``query()``.
        timeout_s: Per-subprocess timeout in seconds.
        project: Optional project filter passed to ``longhand search
            --project``. ``None`` searches across all indexed projects.
    """

    def __init__(
        self,
        *,
        bin_path: str = DEFAULT_BIN,
        home_dir: str = DEFAULT_HOME,
        n_results: int = DEFAULT_LIMIT,
        timeout_s: float = DEFAULT_TIMEOUT,
        project: Optional[str] = None,
        **_unused: Any,
    ) -> None:
        resolved = shutil.which(bin_path) or (
            bin_path if os.path.isabs(bin_path) and os.path.exists(bin_path) else None
        )
        if resolved is None:
            raise ValueError(
                f"Longhand CLI {bin_path!r} not found on PATH. Install it "
                "(pip install longhand) or pass bin_path=/abs/path/to/longhand."
            )
        self.bin_path = resolved
        self.home_dir = Path(os.path.expanduser(home_dir))
        self.n_results = n_results
        self.timeout_s = timeout_s
        self.project = project

        if not self.home_dir.exists():
            log.warning(
                "Longhand home %s does not exist — the store may be "
                "uninitialised; searches will return no results until "
                "Longhand has ingested at least one session.",
                self.home_dir,
            )

    # --- SMEAdapter required methods ---------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        raise NotImplementedError(
            "LonghandAdapter is diagnostic-only (Mode B). Longhand ingests "
            "Claude Code session JSONL through its own Stop/SessionEnd hooks "
            "(`longhand ingest-live` / `longhand ingest-session`), not "
            "arbitrary seeded corpora. To populate a store for testing, run "
            "Claude Code sessions with the Longhand hooks installed."
        )

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        project: Optional[str] = None,
    ) -> QueryResult:
        limit = n_results or self.n_results
        chosen_project = project if project is not None else self.project

        argv = [self.bin_path, "search", question, "--json", "--limit", str(limit)]
        if chosen_project:
            argv += ["--project", chosen_project]

        try:
            proc = subprocess.run(  # noqa: S603 — argv list, no shell
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            return QueryResult(answer="", error=f"BIN_NOT_FOUND: {exc}")
        except subprocess.TimeoutExpired:
            return QueryResult(
                answer="", error=f"TIMEOUT: longhand search exceeded {self.timeout_s}s"
            )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            return QueryResult(
                answer="", error=f"CLI_ERROR rc={proc.returncode}: {stderr[:500]}"
            )

        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            return QueryResult(
                answer="", error=f"BAD_JSON: {exc}: {(proc.stdout or '')[:200]}"
            )

        hits = self._extract_hits(payload)
        retrieval_path = [
            f"limit={limit}",
            f"project={chosen_project or 'all'}",
            f"hits={len(hits)}",
        ]

        if not hits:
            return QueryResult(
                answer="", context_string="", error="NO_RESULTS",
                retrieval_path=retrieval_path,
            )

        context_parts: list[str] = []
        retrieved: list[Entity] = []
        for i, hit in enumerate(hits):
            text = self._hit_text(hit)
            session = hit.get("session_id") or hit.get("session") or "?"
            project_name = hit.get("project") or "?"
            label = self._hit_label(hit, i)
            context_parts.append(f"[{i + 1}] [{project_name}/{session}] {label}\n{text}")
            retrieved.append(
                Entity(
                    id=f"longhand_hit:{i}",
                    name=label,
                    entity_type=f"event:{hit.get('event_type') or hit.get('type') or 'event'}",
                    properties={
                        "_table": "longhand_hit",
                        "project": project_name,
                        "session_id": session,
                        "score": hit.get("score") or hit.get("distance"),
                        "timestamp": hit.get("timestamp") or hit.get("ts"),
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
        # Longhand is a verbatim session archive, not a knowledge graph.
        # Structural categories (Cat 4/5/8) are not meaningful against it.
        return ([], [])

    # --- Cat 9 harness manifest --------------------------------------

    def get_harness_manifest(self) -> list[HarnessDescriptor]:
        """Declare Longhand's MCP search surface for Cat 9.

        Longhand installs an MCP server (``longhand mcp install``) exposing
        a ``search`` tool. This adapter probes the same retrieval through
        the CLI, which exercises the identical SQLite+Chroma read path the
        MCP ``search`` tool uses — a probe failure here predicts an MCP
        failure. It does not verify the MCP stdio/JSON-RPC layer itself
        (that is a future 9f concern that would spawn the server).
        """
        return [
            HarnessDescriptor(
                name="longhand_search",
                kind="mcp_resource",
                probe_fn=self._probe_search,
                description="MCP tool: semantic search over Claude Code session events",
                properties={
                    "tool_name": "search",
                    "underlying_call": "query() via `longhand search --json`",
                },
            ),
        ]

    def _probe_search(self) -> ProbeResult:
        start = time.perf_counter()
        try:
            result = self.query("probe query test")
        except Exception as exc:  # noqa: BLE001 — adapter probe, all errors captured
            return ProbeResult(
                success=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency = (time.perf_counter() - start) * 1000
        # NO_RESULTS is a successful call-through against an empty store —
        # that's a retrieval-quality (Cat 1) signal, not a Cat 9b failure.
        # A real CLI/bin error is a harness failure.
        if result.error and result.error != "NO_RESULTS":
            return ProbeResult(success=False, latency_ms=latency, error=result.error)
        return ProbeResult(
            success=True,
            latency_ms=latency,
            output=f"context_string length={len(result.context_string or '')}",
        )

    # --- helpers -----------------------------------------------------

    @staticmethod
    def _extract_hits(payload: Any) -> list[dict]:
        """Longhand's --json shape isn't a frozen contract across versions.

        Accept either a bare list of hits or a dict wrapping them under a
        ``results``/``hits``/``events`` key. Anything else → no hits.
        """
        if isinstance(payload, list):
            return [h for h in payload if isinstance(h, dict)]
        if isinstance(payload, dict):
            for key in ("results", "hits", "events", "matches"):
                val = payload.get(key)
                if isinstance(val, list):
                    return [h for h in val if isinstance(h, dict)]
        return []

    @staticmethod
    def _hit_text(hit: dict) -> str:
        for key in ("text", "content", "body", "message", "snippet"):
            val = hit.get(key)
            if isinstance(val, str) and val:
                return val
        return ""

    @staticmethod
    def _hit_label(hit: dict, i: int) -> str:
        for key in ("file", "file_path", "path", "title", "id", "event_id"):
            val = hit.get(key)
            if isinstance(val, str) and val:
                return Path(val).name or val
        return f"hit{i}"
