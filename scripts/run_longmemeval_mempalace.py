#!/usr/bin/env python3
"""Run MemPalace (and Familiar) through the LongMemEval E2E QA pipeline.

This script is the convenience entry point for issue #19 — produce the
first published LongMemEval score for the MemPalace fork. It builds on
top of two already-shipped pieces:

  1. ``scripts/cross_validate_longmemeval.py`` — the per-question loop,
     reader, judge, and dual-metric aggregator (issue #17).
  2. ``sme.adapters.mempalace_daemon.MemPalaceDaemonAdapter`` — the
     HTTP adapter into a running palace-daemon.

The piece this script adds is the **per-question ingest topology**:
LongMemEval gives every question its own haystack, but the daemon adapter
expects the corpus to already be present in the palace. We therefore POST
each question's sessions into ``/memory`` under a unique per-question
wing (``lme_<question_id>``) and scope the adapter's ``/search`` to that
wing so prior questions can't leak into the current question's retrieval.

For ``--adapter familiar`` the same per-question wing scoping applies via
the underlying palace-daemon — familiar reads the daemon's vault and
proxies ``/graph``.

CLI:

    run_longmemeval_mempalace.py
        --adapter mempalace-daemon | familiar
        --api-url URL              # daemon base URL (required for daemon)
        --api-key KEY              # X-API-Key (or PALACE_API_KEY env)
        --questions JSON           # longmemeval_oracle.json
        --max-questions N          # smoke-test cap (optional)
        --json PATH                # report JSON destination
        --dry-run                  # estimate cost without LLM/HTTP calls
        --skip-judge               # R@5-only pass (no reader, no judge)

The default reader/judge models for this script are Azure-friendly
(JP's homelab serves Azure-deployed OpenAI):

    reader: gpt-4o-mini
    judge:  gpt-4o

A dry-run estimate uses these as the assumed models for cost arithmetic
even though no LLM is called. The library-level defaults in
``sme.eval.answer_generator`` / ``sme.eval.longmemeval_judge`` keep the
gpt-4.1-mini / gpt-4o-2024-08-06 pair for direct-OpenAI users.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Make the repo importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# scripts/ also needs to be importable for the harness module.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cross_validate_longmemeval as harness  # noqa: E402
from sme.adapters.base import Entity, QueryResult, SMEAdapter  # noqa: E402
from sme.adapters.mempalace_daemon import MemPalaceDaemonAdapter  # noqa: E402
from sme.corpora.longmemeval import LMEQuestion, load_questions  # noqa: E402

log = logging.getLogger("run_longmemeval_mempalace")

# Per-question wing prefix used by the ingest+query path. Each
# LongMemEval question is loaded into wing=f"{LME_WING_PREFIX}{question_id}"
# and the adapter is told to scope its search to that wing.
LME_WING_PREFIX = "lme_"
LME_ROOM = "references"

# Default reader / judge models for this script's argparse. Azure-friendly
# (no date suffix on the judge; gpt-4.1-mini is not deployed on JP's Azure
# resource). The library-level DEFAULT_READER_MODEL / DEFAULT_JUDGE_MODEL
# in sme.eval.* are unchanged so direct-OpenAI users keep their defaults.
DEFAULT_READER_MODEL = "o4-mini"
DEFAULT_JUDGE_MODEL = "gpt-5.3-chat"

# Rough cost estimates per 1M tokens (USD), used only for --dry-run
# accounting. Update when pricing moves. The harness records actual usage
# at runtime; the dry-run number is a planning tool, not a billing source.
_MODEL_PRICING_USD_PER_M_TOKENS = {
    "gpt-4.1-mini":      {"input": 0.40,  "output": 1.60},
    "gpt-4.1":           {"input": 2.00,  "output": 8.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50,  "output": 10.00},
    "o4-mini":           {"input": 1.10,  "output": 4.40},
    "gpt-5.3-chat":      {"input": 2.00,  "output": 8.00},
}

# palace-daemon chunks each ingested drawer into ``<parent>_chunk_NNNNNN``
# sub-drawers. The ingest response returns the parent id; /search returns
# chunk ids. #98 — strip the suffix at compare time so drawer_hit_at_K
# reflects real overlap rather than exact-string equality against IDs
# the caller never saw.
_CHUNK_SUFFIX_RE = re.compile(r"_chunk_\d+$")


def _drawer_parent_id(drawer_id: Optional[str]) -> Optional[str]:
    """Strip the trailing ``_chunk_NNNNNN`` suffix from a daemon drawer_id.

    Returns the input unchanged if no suffix is present (or input is None /
    falsy). Idempotent — calling twice returns the same value.
    """
    if not drawer_id:
        return drawer_id
    return _CHUNK_SUFFIX_RE.sub("", str(drawer_id))


def _stratified_cap(questions: list, n: int, field: str) -> list:
    """Take ~n questions evenly across the values of ``field`` (round-robin).

    techempower-org/...#122: the oracle/S corpora are sorted by
    ``question_type``, so ``questions[:n]`` is a single-category slice. Drawing
    round-robin across each field value keeps a cap representative — e.g.
    ``n=150, field="question_type"`` over 6 types yields 25 of each. Falls back
    gracefully when a value has fewer than its share (the remainder is filled
    from whatever groups still have questions, preserving total ``n`` when
    possible). Deterministic: preserves within-group input order, sorts group
    keys, no RNG.
    """
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for q in questions:
        key = getattr(q, field, None)
        if key is None and isinstance(q, dict):
            key = q.get(field)
        groups[key].append(q)
    order = sorted(groups, key=lambda k: (k is None, str(k)))
    idx = {k: 0 for k in order}
    out: list = []
    while len(out) < n and any(idx[k] < len(groups[k]) for k in order):
        for k in order:
            if idx[k] < len(groups[k]):
                out.append(groups[k][idx[k]])
                idx[k] += 1
                if len(out) >= n:
                    break
    return out


# --- Daemon HTTP helpers ----------------------------------------------------

class DaemonIngestClient:
    """Thin POST /memory client for loading a LongMemEval haystack into
    palace-daemon. Pure HTTP — no ChromaDB or filesystem access. The
    daemon is the single writer per its design contract.

    Tests pass ``opener=fake_urlopen`` to bypass the network. In
    ``--dry-run`` mode the run script doesn't instantiate this class at
    all — it just counts what would have been posted.
    """

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        timeout_s: float = 30.0,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._opener = opener or urllib.request.urlopen

    def post_memory(
        self, *, content: str, wing: str, room: str = LME_ROOM,
    ) -> tuple[int, dict]:
        """POST /memory ``{content, wing, room}``. Returns (status, body)."""
        url = f"{self.api_url}/memory"
        body = json.dumps(
            {"content": content, "wing": wing, "room": room}
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                status = resp.status if hasattr(resp, "status") else resp.getcode()
                try:
                    parsed = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    parsed = {"_raw": raw[:200]}
                return status, parsed
        except urllib.error.HTTPError as e:
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            return e.code, {"_raw": raw[:200]}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return -1, {"_network_error": f"{type(e).__name__}: {e}"}

    def post_flush(self) -> tuple[int, dict]:
        """POST /flush — checkpoint memories to disk. No-op if the daemon
        manages flushing itself; harmless to call."""
        url = f"{self.api_url}/flush"
        req = urllib.request.Request(
            url,
            data=b"",
            headers={"X-API-Key": self.api_key},
            method="POST",
        )
        try:
            with self._opener(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                status = resp.status if hasattr(resp, "status") else resp.getcode()
                return status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            return e.code, {}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return -1, {"_network_error": f"{type(e).__name__}: {e}"}


def ingest_question_haystack(
    q: LMEQuestion,
    ingest_client: DaemonIngestClient,
    *,
    wing: Optional[str] = None,
    content_rules: str = "sme-rich",
) -> dict[str, Any]:
    """Load every session in ``q``'s haystack into the daemon under ``wing``.

    Returns a small report dict: ``{wing, posted, errors, session_to_drawer}``.
    Errors are appended on every non-2xx response; the loop continues so a
    single failing session doesn't kill the whole question.

    The ``session_to_drawer`` map (LongMemEval session_id → daemon drawer_id)
    is what makes rank-aware scoring possible. Without it, the SME substring
    matcher has to guess whether the daemon's retrieved drawers correspond
    to the expected sessions (see #58). With it, callers can compute
    drawer_id-based hit_at_K directly.

    ``content_rules`` controls the rendering shape per #54. ``sme-rich``
    (default) matches the existing rich rendering; ``upstream-exact``
    concatenates only user turns by newline, matching upstream's raw
    protocol and removing the -2.2pp loader-cost documented in #51.
    """
    target_wing = wing or f"{LME_WING_PREFIX}{q.question_id}"
    posted = 0
    errors: list[str] = []
    session_to_drawer: dict[str, str] = {}
    for s in q.haystack_sessions:
        if content_rules == "upstream-exact":
            text = "\n".join(t.content for t in s.turns if t.role == "user")
        else:
            # sme-rich: one drawer per session — concatenate the turns with
            # role headers + date. Same shape ``materialize_sme_corpus``
            # writes to disk in sme-rich mode.
            body_parts = [
                f"# Session {s.session_id}",
                f"_Date: {s.date}_",
                "",
            ]
            for t in s.turns:
                marker = "  <!-- evidence -->" if t.has_answer else ""
                body_parts.append(f"## {t.role}{marker}\n\n{t.content}")
            text = "\n".join(body_parts)

        status, body = ingest_client.post_memory(
            content=text, wing=target_wing, room=LME_ROOM,
        )
        if status != 200 and status != 201:
            errors.append(
                f"session {s.session_id}: HTTP {status} {body.get('_raw') or body!r}"
            )
        else:
            posted += 1
            # /memory returns the new drawer's id; remember it so the
            # scorer can match expected_session_ids → drawer_ids. Cast to
            # str so an integer PK from the daemon doesn't break the
            # later equality check against string Entity.id values.
            drawer_id = body.get("drawer_id") if isinstance(body, dict) else None
            if drawer_id is not None:
                session_to_drawer[s.session_id] = str(drawer_id)
    return {
        "wing": target_wing,
        "posted": posted,
        "errors": errors,
        "session_to_drawer": session_to_drawer,
    }


# --- Adapter factories ------------------------------------------------------

def _make_wing_scoped_daemon_adapter(
    *, api_url: str, api_key: str, wing: str, kind: Optional[str] = None,
    search_endpoint: str = "/search",
) -> SMEAdapter:
    """Build a MemPalaceDaemonAdapter that scopes /search to ``wing``.

    We subclass at runtime and override ``query()`` to build the
    ``/search`` URL with the ``wing`` query parameter included (the
    palace-daemon /search endpoint accepts ``wing`` per PR #22). The
    base adapter's ``query()`` declares ``wing`` as a reserved kwarg
    but does not yet pass it through, so this override is what makes
    per-question isolation work.

    The rest of the response handling (warnings, NO_RESULTS, context
    string assembly, retrieval_path) is inherited unchanged — we only
    swap out the URL construction.
    """
    inner = MemPalaceDaemonAdapter(
        api_url=api_url,
        api_key=api_key,
        kind=kind or "all",  # LongMemEval drawers aren't checkpoints
        search_endpoint=search_endpoint,
    )
    bound_wing = wing
    bound_endpoint = search_endpoint

    class _WingScoped(MemPalaceDaemonAdapter):
        def __init__(self) -> None:
            # Bypass __init__ — we copy state from `inner` so the new
            # instance shares its api_url / api_key without re-running
            # env-file resolution.
            self.api_url = inner.api_url
            self.api_key = inner.api_key
            self.kind = inner.kind
            self.api_timeout = inner.api_timeout
            self.prefer_graph_endpoint = inner.prefer_graph_endpoint
            self.search_endpoint = inner.search_endpoint
            self.candidate_strategy = inner.candidate_strategy

        def query(self, question: str, *, n_results: int = 5,
                  kind: Optional[str] = None, route: bool = False,
                  wing: Optional[str] = None, room: Optional[str] = None):
            import urllib.parse as _urlparse

            chosen_kind = kind or self.kind
            scope_wing = wing or bound_wing

            if bound_endpoint != "/search":
                # #45 — POST /search/age-fused — field name `query` (not
                # `q`), no `kind` filter (the daemon's age-fused path
                # doesn't accept it). Reuses the same body-parsing branch
                # below.
                payload: dict[str, Any] = {
                    "query": question, "limit": n_results,
                }
                if scope_wing:
                    payload["wing"] = scope_wing
                if room:
                    payload["room"] = room
                url = f"{self.api_url}{bound_endpoint}"
                body = self._http_post(url, payload)
            else:
                params: dict[str, Any] = {
                    "q": question, "limit": n_results, "kind": chosen_kind,
                }
                if scope_wing:
                    params["wing"] = scope_wing
                if room:
                    params["room"] = room
                url = f"{self.api_url}/search?{_urlparse.urlencode(params)}"
                body = self._http_get(url)
            if isinstance(body, QueryResult):
                return body

            results = body.get("results") or []
            warnings = body.get("warnings") or []
            total = body.get("total_before_filter")
            available = body.get("available_in_scope")
            retrieval_path = [
                f"kind={chosen_kind}",
                f"wing={scope_wing}" if scope_wing else "wing=*",
                f"available_in_scope={available}",
                f"total_before_filter={total}",
            ]
            if not results:
                err = (
                    f"WARN: {'; '.join(warnings)}" if warnings else "NO_RESULTS"
                )
                return QueryResult(
                    answer="", context_string="", error=err,
                    retrieval_path=retrieval_path,
                )

            context_parts: list[str] = []
            retrieved: list[Entity] = []
            for i, hit in enumerate(results):
                # #45 — /search nests fields under metadata; /search/age-fused
                # places them at top level. Tolerate both.
                meta = hit.get("metadata") or {}
                wing_name = meta.get("wing") or hit.get("wing", "?")
                room_name = meta.get("room") or hit.get("room", "?")
                source_file = (
                    meta.get("source_file") or hit.get("source_file") or f"hit{i}"
                )
                source_label = Path(source_file).name or source_file
                text = hit.get("text", "") or ""
                # Real drawer_id as Entity.id (#58). Synthetic fallback
                # only when the daemon response omits it. Cast to str so
                # an integer PK from the daemon matches the string Entity.id
                # type contract.
                raw_drawer_id = hit.get("drawer_id") or hit.get("id")
                drawer_id = str(raw_drawer_id) if raw_drawer_id is not None else f"drawer_hit:{i}"
                context_parts.append(
                    f"[{i + 1}] [{wing_name}/{room_name}] {source_label}\n{text}"
                )
                retrieved.append(
                    Entity(
                        id=drawer_id,
                        name=source_label,
                        entity_type=f"drawer:{room_name}",
                        properties={
                            "_table": "mempalace_daemon_hit",
                            "wing": wing_name,
                            "room": room_name,
                            "score": hit.get("score"),
                            "source_file": source_file,
                            "rank": i + 1,
                        },
                    )
                )
            context_string = "\n\n".join(context_parts)
            warn_err = f"WARN: {'; '.join(warnings)}" if warnings else None
            return QueryResult(
                answer=context_string,
                context_string=context_string,
                retrieved_entities=retrieved,
                retrieval_path=retrieval_path,
                error=warn_err,
            )

    return _WingScoped()


def _make_familiar_adapter_factory(
    *, base_url: str, mock_inference: bool = True,
) -> harness.AdapterFactory:
    """Familiar adapter factory. ``base_url`` points at the familiar
    HTTP service (which itself proxies palace-daemon). Familiar doesn't
    accept a wing scope on its /api/familiar/eval endpoint yet, so this
    factory currently ignores per-question wings — the harness records
    that limitation in the report metadata.

    When familiar adds wing scoping (tracked in familiar.realm.watch
    issue queue), update this to pass it through; the run script's
    wing-prefix convention is the same.
    """
    from sme.adapters.familiar import FamiliarAdapter

    def _factory(_per_q_vault: Path) -> SMEAdapter:
        return FamiliarAdapter(
            base_url=base_url, mock_inference=mock_inference,
        )

    return _factory


# --- Dry-run cost estimator -------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Very rough char→token approximation (~4 chars per token).

    Used only for --dry-run accounting. Real runs record actual usage
    via the OpenAI SDK and emit them in the report's
    ``judge_total_usage`` field.
    """
    return max(1, len(text) // 4)


def estimate_run_cost(
    questions: Iterable[LMEQuestion],
    *,
    reader_model: str,
    judge_model: str,
    avg_context_chars: int = 6000,
    avg_reader_output_chars: int = 200,
    avg_judge_prompt_chars: int = 1200,
    avg_judge_output_chars: int = 80,
) -> dict[str, Any]:
    """Estimate USD cost of running every question through reader+judge.

    The harness records actual usage at runtime; this estimate exists
    only to answer "is a full run $5 or $50?" before launching. The
    averages reflect a typical LongMemEval-oracle question: ~6000 chars
    of retrieved context, ~200 chars of generated answer, ~1200 chars
    of judge prompt, ~80 chars of judge JSON reply.

    Per-question and total tokens are reported alongside USD so callers
    can sanity-check the model-pricing table against current rates.
    """
    reader_price = _MODEL_PRICING_USD_PER_M_TOKENS.get(reader_model, {})
    judge_price = _MODEL_PRICING_USD_PER_M_TOKENS.get(judge_model, {})

    per_q_reader_input = _approx_tokens(
        "Answer the user's question using only the conversation history "
        "below.\n" + "x" * avg_context_chars + "\nQuestion: ?\nAnswer:"
    )
    per_q_reader_output = _approx_tokens("x" * avg_reader_output_chars)
    per_q_judge_input = _approx_tokens("x" * avg_judge_prompt_chars)
    per_q_judge_output = _approx_tokens("x" * avg_judge_output_chars)

    n = sum(1 for _ in questions)

    reader_input_tokens = per_q_reader_input * n
    reader_output_tokens = per_q_reader_output * n
    judge_input_tokens = per_q_judge_input * n
    judge_output_tokens = per_q_judge_output * n

    reader_cost = (
        reader_input_tokens / 1_000_000 * reader_price.get("input", 0.0)
        + reader_output_tokens / 1_000_000 * reader_price.get("output", 0.0)
    )
    judge_cost = (
        judge_input_tokens / 1_000_000 * judge_price.get("input", 0.0)
        + judge_output_tokens / 1_000_000 * judge_price.get("output", 0.0)
    )

    return {
        "n_questions": n,
        "reader_model": reader_model,
        "judge_model": judge_model,
        "reader_tokens": {
            "input": reader_input_tokens, "output": reader_output_tokens,
        },
        "judge_tokens": {
            "input": judge_input_tokens, "output": judge_output_tokens,
        },
        "reader_usd": round(reader_cost, 4),
        "judge_usd": round(judge_cost, 4),
        "total_usd": round(reader_cost + judge_cost, 4),
        "pricing_basis": "openai 2026-05 list prices; update _MODEL_PRICING_USD_PER_M_TOKENS",
    }


# --- Per-question loop wrapper ---------------------------------------------

def _resolve_api_key(args: argparse.Namespace) -> Optional[str]:
    """Pick the daemon API key from --api-key or PALACE_API_KEY env."""
    if args.api_key:
        return args.api_key
    return os.environ.get("PALACE_API_KEY")


def _build_factory(args: argparse.Namespace) -> Callable[[LMEQuestion, Path], SMEAdapter]:
    """Return a callable that builds a fresh per-question adapter.

    Differs from harness.AdapterFactory (Path -> SMEAdapter) because the
    daemon factory needs the LMEQuestion to derive the per-question wing.
    Wrapped into a (question, vault_path) -> adapter callable.
    """
    if args.adapter == "mempalace-daemon":
        api_key = _resolve_api_key(args)
        if not args.api_url:
            raise SystemExit("--api-url is required for --adapter mempalace-daemon")
        if not api_key:
            raise SystemExit(
                "API key required: pass --api-key or set PALACE_API_KEY"
            )

        def _factory(q: LMEQuestion, _vault: Path) -> SMEAdapter:
            return _make_wing_scoped_daemon_adapter(
                api_url=args.api_url, api_key=api_key,
                wing=f"{LME_WING_PREFIX}{q.question_id}",
                kind=args.kind,
                search_endpoint=getattr(args, "search_endpoint", "/search"),
            )
        return _factory

    if args.adapter == "familiar":
        if not args.familiar_url:
            raise SystemExit(
                "--familiar-url is required for --adapter familiar"
            )
        familiar_factory = _make_familiar_adapter_factory(
            base_url=args.familiar_url,
            mock_inference=not args.familiar_inference,
        )

        def _factory_fam(_q: LMEQuestion, vault: Path) -> SMEAdapter:
            return familiar_factory(vault)
        return _factory_fam

    raise SystemExit(
        f"unsupported adapter for issue #19 run script: {args.adapter!r}. "
        f"Use 'mempalace-daemon' or 'familiar', or fall back to the "
        f"sme-eval longmemeval subcommand for full-context / flat."
    )


def _run_questions(
    *,
    args: argparse.Namespace,
    questions: list[LMEQuestion],
    factory_fn: Callable[[LMEQuestion, Path], SMEAdapter],
    work_dir: Path,
    ingest_client: Optional[DaemonIngestClient],
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
) -> list[dict]:
    """Run each question through ingest → adapter → judge.

    Reuses ``harness.run_one_question`` for the adapter+judge half. The
    only piece this function adds is the per-question ingest step (for
    mempalace-daemon) and the adapter-from-question wiring.
    """
    records: list[dict] = []
    ingest_total = {"posted": 0, "errors": 0}

    for i, q in enumerate(questions):
        log.info(
            "[%d/%d] %s (%s / %s)",
            i + 1, len(questions), q.question_id,
            q.question_type, q.sme_category,
        )

        # 1. Ingest (daemon only — familiar reads existing palace contents)
        per_q_ingest: dict[str, Any] = {"posted": 0, "errors": [], "wing": None}
        if ingest_client is not None:
            ingest_report = ingest_question_haystack(
                q,
                ingest_client,
                content_rules=getattr(args, "content_rules", "sme-rich"),
            )
            ingest_total["posted"] += ingest_report["posted"]
            ingest_total["errors"] += len(ingest_report["errors"])
            per_q_ingest = {
                "posted": ingest_report["posted"],
                "errors": list(ingest_report["errors"]),
                "wing": ingest_report.get("wing"),
            }
            if ingest_report["errors"]:
                log.warning(
                    "ingest errors for %s: %d session(s) failed",
                    q.question_id, len(ingest_report["errors"]),
                )

        # 2. Build per-question adapter and run via the existing harness
        def adapter_factory_for_harness(vault, _q=q):
            return factory_fn(_q, vault)
        rec = harness.run_one_question(
            q,
            adapter_factory=adapter_factory_for_harness,
            work_dir=work_dir,
            skip_judge=args.skip_judge,
            skip_reader=args.skip_reader,
            reader_model=args.answer_model,
            judge_model=args.judge,
            reader_client=reader_client,
            judge_client=judge_client,
            capture_context=getattr(args, "pin_context", False),
        )

        # 3. #58 — rank-aware scoring on drawer_ids. If we have a
        # session→drawer map from the ingest step, use it to compute
        # drawer_hit_at_K against the expected_session_ids. The substring-
        # matcher recall (rec["sme_recall"]) stays in place unchanged,
        # so cross-system A/B against systems that don't expose drawer_ids
        # still works.
        #
        # #98 fix: the daemon chunks each drawer at ingest into
        # ``<parent>_chunk_NNNNNN`` sub-drawers, and /search returns the
        # chunk IDs. The ingest response gave us the *parent* drawer_id.
        # Strip the chunk suffix before comparing so hit_at_K reflects
        # real overlap, not exact-string equality against chunked IDs.
        session_to_drawer = (
            ingest_report.get("session_to_drawer") if ingest_client is not None else None
        ) or {}
        expected_drawer_ids = {
            session_to_drawer[sid]
            for sid in q.answer_session_ids
            if sid in session_to_drawer
        }
        retrieved_drawer_ids = list(rec.get("retrieved_entity_ids") or [])
        retrieved_parent_ids = [_drawer_parent_id(d) for d in retrieved_drawer_ids]
        rec["expected_drawer_ids"] = sorted(expected_drawer_ids)
        rec["retrieved_drawer_ids"] = retrieved_drawer_ids
        rec["retrieved_parent_ids"] = retrieved_parent_ids
        rec["drawer_hit_at_1"] = bool(
            expected_drawer_ids
            and retrieved_parent_ids
            and retrieved_parent_ids[0] in expected_drawer_ids
        )
        rec["drawer_hit_at_5"] = bool(
            expected_drawer_ids
            and any(d in expected_drawer_ids for d in retrieved_parent_ids[:5])
        )
        rec["drawer_hit_at_10"] = bool(
            expected_drawer_ids
            and any(d in expected_drawer_ids for d in retrieved_parent_ids[:10])
        )

        # #59 — attach per-question ingest report so the JSON output
        # surfaces which questions had incomplete haystacks.
        rec["ingest"] = per_q_ingest
        records.append(rec)

    if ingest_client is not None:
        log.info(
            "ingest summary: posted=%d errors=%d",
            ingest_total["posted"], ingest_total["errors"],
        )

    return records


# --- Public run() entry point ----------------------------------------------

def run(
    args: argparse.Namespace,
    *,
    questions: Optional[list[LMEQuestion]] = None,
    ingest_client: Optional[DaemonIngestClient] = None,
    factory_fn: Optional[Callable[[LMEQuestion, Path], SMEAdapter]] = None,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
) -> dict:
    """Programmatic entry point — used by tests and the CLI shim.

    Tests inject ``questions``, ``ingest_client``, ``factory_fn``,
    ``reader_client``, and ``judge_client`` so no live daemon or API
    key is needed.
    """
    # Load questions (unless tests already supplied them)
    if questions is None:
        questions = list(load_questions(args.questions))
        # techempower-org/...#122: the oracle/S corpora are sorted by
        # question_type, so a bare ``[:N]`` cap is a single-category slice.
        # --shuffle re-orders deterministically; --stratify-by draws an even
        # round-robin across the field's values so the cap stays representative.
        shuffle_seed = getattr(args, "shuffle", None)
        stratify_by = getattr(args, "stratify_by", None)
        if shuffle_seed is not None:
            import random
            random.Random(shuffle_seed).shuffle(questions)
        if args.max_questions is not None:
            if stratify_by:
                questions = _stratified_cap(
                    questions, args.max_questions, stratify_by
                )
            else:
                questions = questions[: args.max_questions]

    # Dry-run path — no HTTP, no LLM, just estimate
    if args.dry_run:
        cost = estimate_run_cost(
            questions,
            reader_model=args.answer_model,
            judge_model=args.judge,
        )
        return {
            "run_metadata": {
                "mode": "dry-run",
                "adapter": args.adapter,
                "questions": args.questions,
                "n_questions": cost["n_questions"],
                "answer_model": args.answer_model,
                "judge_model": args.judge,
                "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
            "cost_estimate": cost,
        }

    # Build per-question factory
    if factory_fn is None:
        factory_fn = _build_factory(args)

    # Build daemon ingest client for both daemon-direct and familiar
    # adapters (#46) — familiar wraps palace-daemon, so the per-question
    # haystack still needs to be loaded into the daemon's drawer store
    # for familiar's /api/familiar/eval to retrieve from. Without this
    # the familiar wing comes up empty and queries return whatever was
    # previously there (empty wing → empty results, low_confidence warning).
    if (
        ingest_client is None
        and args.adapter in ("mempalace-daemon", "familiar")
        and args.api_url is not None
    ):
        api_key = _resolve_api_key(args)
        if api_key is not None:
            ingest_client = DaemonIngestClient(
                api_url=args.api_url, api_key=api_key,
            )

    # Work dir
    work_dir_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    if args.work_dir is not None:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir_ctx = tempfile.TemporaryDirectory(prefix="sme_lme_mp_")
        work_dir = Path(work_dir_ctx.name)

    try:
        records = _run_questions(
            args=args, questions=questions, factory_fn=factory_fn,
            work_dir=work_dir, ingest_client=ingest_client,
            reader_client=reader_client, judge_client=judge_client,
        )
    finally:
        if work_dir_ctx is not None:
            work_dir_ctx.cleanup()

    summary = harness.aggregate(records)
    return {
        "run_metadata": {
            "mode": "live",
            "adapter": args.adapter,
            "questions": str(args.questions),
            "n_questions": len(records),
            "answer_model": (None if args.skip_judge else args.answer_model),
            "judge_model": (None if args.skip_judge else args.judge),
            "skip_judge": bool(args.skip_judge),
            "skip_reader": bool(args.skip_reader),
            "ingested_per_question": ingest_client is not None,
            "wing_prefix": LME_WING_PREFIX,
            # #83 — surface the actual --search-endpoint value so JSON
            # consumers (bench-results site, dashboards) can tell which
            # endpoint was queried. Without this the metadata previously
            # hardcoded "default" even when /search/age-fused had been
            # used, masking which endpoint produced a given reading.
            "search_endpoint": getattr(args, "search_endpoint", "/search"),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "summary": summary,
        "per_question": records,
    }


# --- CLI plumbing ----------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_longmemeval_mempalace",
        description=(
            "Run LongMemEval E2E QA through MemPalace (daemon) or Familiar. "
            "Builds on scripts/cross_validate_longmemeval.py's reader+judge "
            "pipeline; adds per-question wing-scoped ingest into the daemon."
        ),
    )
    p.add_argument("--adapter", required=True,
                   choices=["mempalace-daemon", "familiar"],
                   help="Which SME adapter to run.")
    p.add_argument("--questions", required=True, type=Path,
                   help="Path to longmemeval_oracle.json (or _s / _m).")
    p.add_argument("--api-url",
                   help="(mempalace-daemon) base URL, "
                        "e.g. http://your-daemon-host:8085.")
    p.add_argument("--api-key",
                   help="(mempalace-daemon) X-API-Key. "
                        "Defaults to PALACE_API_KEY env.")
    p.add_argument("--kind", default=None,
                   help="(mempalace-daemon) /search kind filter. "
                        "Default 'all' for LongMemEval (no checkpoints in "
                        "the ingested corpus).")
    p.add_argument("--familiar-url", default="http://familiar:8080",
                   help="(familiar) base URL. Default http://familiar:8080.")
    p.add_argument("--familiar-inference", action="store_true",
                   help="(familiar) run real LLM inference inside familiar "
                        "(default: mock_inference=True for "
                        "deterministic substring scoring).")
    p.add_argument("--answer-model", default=DEFAULT_READER_MODEL,
                   help=f"Reader model (default: {DEFAULT_READER_MODEL}).")
    p.add_argument("--judge", default=DEFAULT_JUDGE_MODEL,
                   help=f"Judge model (default: {DEFAULT_JUDGE_MODEL}).")
    p.add_argument("--max-questions", type=int, default=None,
                   help="Smoke-test cap on number of questions. NOTE: the oracle/S "
                        "corpora are question_type-sorted, so a bare cap is a single-"
                        "category slice — pair with --stratify-by or --shuffle (#122).")
    p.add_argument("--shuffle", type=int, default=None, metavar="SEED",
                   help="Deterministically shuffle questions (with SEED) before the "
                        "--max-questions cap, so the cap is not a single-category slice.")
    p.add_argument("--stratify-by", default=None, metavar="FIELD",
                   help="Stratify the --max-questions cap across this question field "
                        "(e.g. question_type) — even round-robin per category for "
                        "representative coverage (#122).")
    p.add_argument("--skip-judge", action="store_true",
                   help="R@5-only — no reader, no judge, no OPENAI_API_KEY.")
    p.add_argument("--skip-reader", action="store_true",
                   help="Feed raw retrieval text to the judge (diagnostic).")
    p.add_argument("--dry-run", action="store_true",
                   help="Estimate cost only — no HTTP, no LLM, no ingest.")
    p.add_argument("--json", type=Path, default=None,
                   help="Where to write the report JSON.")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Where per-question vaults materialize (default: "
                        "tmpdir).")
    p.add_argument("--content-rules", default="sme-rich",
                   choices=["sme-rich", "upstream-exact"],
                   help="How haystack sessions are rendered for embedding. "
                        "'sme-rich' (default) = frontmatter + role headers + "
                        "user + assistant turns. 'upstream-exact' = user "
                        "turns only, no metadata — matches upstream "
                        "longmemeval_bench.py raw protocol and removes the "
                        "documented -2.2pp loader-cost (#54 / #51).")
    p.add_argument("--search-endpoint", default="/search",
                   help="(mempalace-daemon) Daemon search endpoint. "
                        "Default '/search' (vector + BM25). Pass "
                        "'/search/age-fused' for the RRF fusion of "
                        "vector + BM25 + AGE-graph (#45).")
    p.add_argument("--pin-context", action="store_true",
                   help="(#116 Phase 1) Capture the full retrieved "
                        "context_string per question into the per-question "
                        "records, for offline reader-sweep replay. Pair with "
                        "--skip-reader --skip-judge to retrieve only.")
    p.add_argument("--pin-context-out", type=Path, default=None,
                   help="(#116 Phase 1) Write a pinned-context JSON "
                        "(question + gold + context_string per question) to "
                        "this path, tagged with the search-endpoint snippet "
                        "axis. Implies --pin-context.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_summary(report: dict) -> None:
    """Mirror the formatted summary the sme-eval longmemeval subcommand prints."""
    meta = report.get("run_metadata", {})
    if meta.get("mode") == "dry-run":
        c = report["cost_estimate"]
        print()
        print("=" * 78)
        print(f" DRY RUN  adapter={meta['adapter']}  n={c['n_questions']}")
        print("=" * 78)
        print(f"  reader: {c['reader_model']:25s} "
              f"tokens=in:{c['reader_tokens']['input']:>8} "
              f"out:{c['reader_tokens']['output']:>6}  "
              f"${c['reader_usd']:.2f}")
        print(f"  judge:  {c['judge_model']:25s} "
              f"tokens=in:{c['judge_tokens']['input']:>8} "
              f"out:{c['judge_tokens']['output']:>6}  "
              f"${c['judge_usd']:.2f}")
        print(f"  total:  ${c['total_usd']:.2f}")
        print(f"\n  ({c['pricing_basis']})")
        return

    summary = report["summary"]
    dual = summary.get("dual_metric", {})
    print()
    print("=" * 78)
    print(f" LongMemEval  adapter={meta['adapter']}  n={summary['total_questions']}")
    print("=" * 78)
    print(f"\n{'category':22s} {'n':>4} {'R@5':>8} {'QA-acc':>8} {'gap':>8}")
    for cat, slot in dual.get("per_category", {}).items():
        qa = slot.get("qa_accuracy")
        gap = slot.get("retrieval_qa_gap")
        qa_str = f"{qa:>7.2%}" if qa is not None else "    n/a"
        gap_str = f"{gap:+.3f}" if gap is not None else "  n/a"
        print(
            f"{cat:22s} {slot['n']:>4} "
            f"{slot['sme_recall_mean']:>7.2%} {qa_str} {gap_str:>8}"
        )
    overall = dual.get("overall", {})
    overall_qa = overall.get("qa_accuracy")
    overall_gap = overall.get("retrieval_qa_gap")
    overall_qa_str = (
        f"{overall_qa:>7.2%}" if overall_qa is not None else "    n/a"
    )
    overall_gap_str = (
        f"{overall_gap:+.3f}" if overall_gap is not None else "  n/a"
    )
    if overall.get("n"):
        print(
            f"\n{'overall':22s} {overall['n']:>4} "
            f"{overall['sme_recall_mean']:>7.2%} "
            f"{overall_qa_str} {overall_gap_str:>8}"
        )
    print(f"\n  disagreements: {len(summary.get('disagreements', []))}")


def _write_pinned_context(report: dict, args: argparse.Namespace) -> None:
    """#116 Phase 1 — extract a pinned-context JSON from a live report.

    Keeps only the fields the offline reader sweep needs, tagged with the
    daemon snippet-width axis (search_endpoint) so the sweep can group by it.
    """
    records = report.get("per_question", [])
    pinned = [
        {
            "question_id": r["question_id"],
            "question": r.get("question", ""),
            "gold_answer": r.get("gold_answer", ""),
            "question_type": r["question_type"],
            "sme_category": r.get("sme_category"),
            "is_abstention": r.get("is_abstention", False),
            "context_string": r.get("context_string", ""),
            "context_chars": r.get("context_chars", 0),
            # techempower-org/...#121: the live daemon retrieval-hit field is
            # ``drawer_hit_at_5`` (the #98 chunk-suffix matcher). ``hit_at_5`` is
            # the legacy substring metric, which ``--content-rules upstream-exact``
            # zeroes out, so reading it serialized null on every record. Prefer
            # the drawer metric; fall back to the legacy key for other adapters.
            "hit_at_5": r.get("drawer_hit_at_5", r.get("hit_at_5")),
        }
        for r in records
    ]
    endpoint = getattr(args, "search_endpoint", "/search")
    doc = {
        "run_metadata": {
            "diagnostic": "pinned_context",
            "issue": "techempower-org/multipass-structural-memory-eval#116",
            "adapter": args.adapter,
            "search_endpoint": endpoint,
            "snippet_width": endpoint,  # palace-daemon#150 axis label
            "n_questions": len(pinned),
            "source_questions": str(args.questions),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "pinned_context": pinned,
    }
    args.pin_context_out.parent.mkdir(parents=True, exist_ok=True)
    args.pin_context_out.write_text(json.dumps(doc, indent=2, default=str))
    print(f"Wrote pinned context → {args.pin_context_out} ({len(pinned)} questions)")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # --pin-context-out implies --pin-context.
    if getattr(args, "pin_context_out", None) is not None:
        args.pin_context = True
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    report = run(args)

    out_path = args.json
    if out_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d")
        suffix = "dryrun" if args.dry_run else "live"
        out_path = Path(
            f"longmemeval_{args.adapter.replace('-', '_')}_{ts}_{suffix}.json"
        )
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")

    if getattr(args, "pin_context_out", None) is not None and not args.dry_run:
        _write_pinned_context(report, args)

    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
