#!/usr/bin/env python3
"""Cross-validate SME's substring scorer against LongMemEval's GPT-4o judge.

Runs every question in a benchmark dataset through the chosen SME
adapter, scores the same retrieval with both:

  1. SME's substring matcher over `expected_sources` session ids
  2. LongMemEval's GPT-4o judge methodology (per-question-type prompts)

and reports per-SME-category disagreement. Per the KU / Cat 3 semantic-
divergence caveat documented in docs/related_work/longmemeval.md, the
report deliberately does NOT compute a single overall correlation —
each `sme_category` is reported separately so KU's silent-overwrite
reward doesn't drag a contradiction-flagging Cat 3 system's correlation
down.

Three corpora are wired (``--corpus``):

  - ``longmemeval`` (default): one haystack PER QUESTION; each question's
    sessions are materialized to a per-question vault, ingested, queried.
  - ``locomo``: one conversation PER SAMPLE shared by all of that
    sample's questions (LoCoMo's topology). The harness groups questions
    by ``sample_id``, materializes the full sample vault ONCE, ingests it,
    then queries every question in that sample against it. Adversarial
    (category-5) items are judged abstention-aware — the gold is a
    refusal, and the baited ``adversarial_answer`` is the wrong attractor.
  - ``beam``: one (very long) conversation PER conversation_id shared by
    its 20 probing questions (BEAM's topology, same per-conversation
    ingest as LoCoMo). Graded at a token ``--bucket`` (100K/500K/1M/10M);
    the bucket is recorded on every record because the same conversation
    at a different bucket is a different retrieval problem. Abstention
    items are judged abstention-aware.

CLI:

    cross_validate_longmemeval.py
        --dataset PATH                # dataset JSON (required)
        --corpus {longmemeval,locomo} # dataset shape (default longmemeval)
        --adapter NAME                # flat | mempalace | full-context
        --max-questions N             # smoke-test cap (optional)
        --reader-model MODEL          # default gpt-4o-mini
        --judge-model MODEL           # default gpt-4o-2024-08-06
        --skip-judge                  # SME-only pass when no API key
        --skip-reader                 # judge sees raw context_string
        --out PATH                    # report JSON destination

`OPENAI_API_KEY` controls reader and judge availability; when missing,
the harness still produces SME substring scores so partial readings are
useful.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

# Ensure repo root is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sme.adapters.base import QueryResult, SMEAdapter  # noqa: E402
from sme.corpora.longmemeval import (  # noqa: E402
    LMEQuestion,
    load_questions,
    materialize_sme_corpus,
)
from sme.eval.answer_generator import generate_answer  # noqa: E402
from sme.eval.dual_metric import aggregate_dual_metric  # noqa: E402
from sme.eval.longmemeval_judge import grade_answer  # noqa: E402

log = logging.getLogger("cross_validate_longmemeval")

# Disagreement is "interesting" when SME recall and judge correctness
# imply opposite verdicts. Concretely: SME recall >= 0.5 but judge
# INCORRECT/ABSTAIN-fail, OR SME recall < 0.5 but judge CORRECT.
_DISAGREE_THRESHOLD = 0.5


def _stratified_cap(questions: list, n: int, field: str) -> list:
    """Take ~n questions evenly across the values of ``field`` (round-robin).

    Mirrors ``scripts/run_longmemeval_mempalace._stratified_cap`` verbatim so
    a competitor adapter run (e.g. OMEGA via this script) lands on the SAME
    representative n=150 subset the mempalace-daemon baseline used — the
    oracle/S corpora are sorted by ``question_type``, so a bare ``[:n]`` cap is
    a single-category slice (techempower-org/...#122). Deterministic: preserves
    within-group input order, sorts group keys, no RNG. Identical inputs +
    field + n yield the identical question set, which is what makes the
    head-to-head apples-to-apples.
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


# --- Adapter construction ---------------------------------------------------

AdapterFactory = Callable[[Path], SMEAdapter]


def _make_full_context_adapter(per_q_vault: Path) -> SMEAdapter:
    from sme.conditions.full_context import FullContextAdapter

    return FullContextAdapter(per_q_vault)


def _make_flat_adapter(per_q_vault: Path) -> SMEAdapter:
    """Build a FlatBaselineAdapter from a per-question vault.

    FlatBaselineAdapter wants a ChromaDB persistence directory, not a
    raw markdown vault. For LongMemEval we ingest the per-question .md
    files into an ephemeral Chroma collection, then point the adapter
    at it. ChromaDB is an optional SME extra — if not installed, the
    harness raises a clear error so the user can pick a different
    adapter or pip-install it.
    """
    try:
        import chromadb  # noqa: F401
    except ImportError as e:  # pragma: no cover — env-dependent
        raise RuntimeError(
            "FlatChromaAdapter requires chromadb. Install with "
            "`pip install chromadb` or pass --adapter full-context."
        ) from e

    from sme.adapters.flat_baseline import FlatBaselineAdapter
    import chromadb as _chromadb  # noqa: WPS433 — late re-import for instance

    db_dir = per_q_vault / "_chroma"
    db_dir.mkdir(parents=True, exist_ok=True)
    client = _chromadb.PersistentClient(path=str(db_dir))
    coll = client.get_or_create_collection(name="lme_per_question")
    docs: list[str] = []
    ids: list[str] = []
    for md_file in sorted(per_q_vault.rglob("*.md")):
        if not md_file.is_file():
            continue
        text = md_file.read_text(encoding="utf-8", errors="replace")
        docs.append(text)
        ids.append(md_file.stem)
    if docs:
        coll.upsert(ids=ids, documents=docs)
    return FlatBaselineAdapter(
        db_path=str(db_dir),
        collection_name="lme_per_question",
        n_results=5,
    )


def _make_omega_adapter(per_q_vault: Path) -> SMEAdapter:
    """Build an OmegaAdapter from a per-question vault.

    OMEGA is a local SQLite memory store. We isolate it per question by
    pointing OMEGA_HOME at a ``_omega/`` dir inside the question's vault
    (the adapter sets the env var + drops OMEGA's cached store singleton,
    so no cross-question contamination and the user's real ~/.omega is
    never touched). Each .md session file becomes one ``omega.store``
    call; ``query_structured`` then drives retrieval.

    omega-memory is an optional SME extra — if not installed, raise a
    clear error so the user can pip-install it or pick another adapter.
    """
    try:
        import omega  # noqa: F401
    except ImportError as e:  # pragma: no cover — env-dependent
        raise RuntimeError(
            "OmegaAdapter requires omega-memory. Install with "
            "`pip install 'sme-eval[omega]'` or pass a different --adapter."
        ) from e

    from sme.adapters.omega import OmegaAdapter

    omega_home = per_q_vault / "_omega"
    adapter = OmegaAdapter(omega_home=str(omega_home), n_results=5)
    corpus: list[dict] = []
    for md_file in sorted(per_q_vault.rglob("*.md")):
        if not md_file.is_file():
            continue
        text = md_file.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            corpus.append({"content": text, "type": "summary"})
    if corpus:
        adapter.ingest_corpus(corpus)
    return adapter


def _make_hindsight_adapter(per_q_vault: Path) -> SMEAdapter:
    """Build a HindsightAdapter from a per-question vault.

    Hindsight is a Docker-hosted server that owns a single store; it
    isolates memories per ``bank_id``. We give each per-question vault its
    own unique bank (derived from the vault dir name) so prior questions
    can't leak into the current question's recall — the same role the
    daemon adapter's per-question wing plays. Each .md session file is a
    ``retain`` with ``document_id`` = the session id (file stem), so recall
    hits map back to the originating session for R@K.

    NOTE (extraction-based, #184): Hindsight stores LLM-EXTRACTED facts,
    not raw sessions. Retrieval is fact-level; session-level R@K is mediated
    by the extractor (a fact extracted *from* the evidence session ranking
    top-K), which is softer than a raw-chunk R@K. The QA number (reader +
    judge over recalled facts) is the cleaner apples-to-apples metric.

    Requires hindsight-client + a reachable server (HINDSIGHT_BASE_URL).
    """
    try:
        import hindsight_client  # noqa: F401
    except ImportError as e:  # pragma: no cover — env-dependent
        raise RuntimeError(
            "HindsightAdapter requires hindsight-client. Install with "
            "`pip install hindsight-client` and run a Hindsight server, "
            "or pass a different --adapter."
        ) from e

    from sme.adapters.hindsight import HindsightAdapter

    # Unique bank per question vault — the isolation primitive.
    bank_id = f"sme_{per_q_vault.name}"
    adapter = HindsightAdapter(bank_id=bank_id, n_results=5)
    corpus: list[dict] = []
    for md_file in sorted(per_q_vault.rglob("*.md")):
        if not md_file.is_file():
            continue
        text = md_file.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            # document_id = session id (file stem) → recall hits trace back
            # to the originating session for session-level R@K.
            corpus.append({"content": text, "document_id": md_file.stem})
    if corpus:
        adapter.ingest_corpus(corpus)
    return adapter


def _make_mempalace_adapter(per_q_vault: Path) -> SMEAdapter:  # pragma: no cover — heavy
    raise RuntimeError(
        "mempalace adapter not yet wired into cross_validate_longmemeval; "
        "use --adapter full-context for the no-retrieval baseline or "
        "--adapter flat for the chroma baseline."
    )


def _make_mempalace_daemon_adapter_factory(
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    kind: Optional[str] = None,
) -> AdapterFactory:  # pragma: no cover — network-dependent
    """Return a factory that builds a MemPalaceDaemonAdapter per question.

    Issue #19 will run the daemon against the full 500-question dataset;
    this factory is the connective tissue. The daemon adapter is HTTP-
    only — there's no per-question ingest step at construction time
    because the daemon owns the corpus. We rely on the caller (the CLI
    subcommand) to have ingested the LongMemEval haystack into the daemon
    once up-front, and we simply query through it for each question.

    Limitation: the daemon's wing/room scoping isn't per-question, so
    cross-question contamination is possible. For #17's E2E QA pipeline
    we accept this — the goal is to land the scoring pipeline; #19
    addresses the ingest topology.
    """
    from sme.adapters.mempalace_daemon import MemPalaceDaemonAdapter

    def _factory(_per_q_vault: Path) -> SMEAdapter:
        kwargs: dict[str, Any] = {}
        if api_url is not None:
            kwargs["api_url"] = api_url
        if api_key is not None:
            kwargs["api_key"] = api_key
        if kind is not None:
            kwargs["kind"] = kind
        return MemPalaceDaemonAdapter(**kwargs)

    return _factory


# Module-level singleton so the per-question/-conversation factory reuses one
# postgres connection + DDL across the whole run (the adapter is explicitly
# designed for this — close() is a no-op, shutdown() does the real teardown).
# ingest_corpus TRUNCATEs on every call, so each question's haystack is
# isolated exactly like the daemon's per-question wing / Hindsight's per-bank.
_PG_INGEST_SINGLETON: Optional[SMEAdapter] = None


def _make_postgres_adapter(per_q_vault: Path) -> SMEAdapter:
    """Build (once) a PostgresIngestAdapter and re-ingest the per-question vault.

    This is the postgres+pgvector twin of ``_make_flat_adapter``: same
    all-MiniLM-L6-v2 embedding (PostgresCollection reuses Chroma's default
    embedding function), the only swapped variable is the storage/retrieval
    backend (chroma -> postgres+pgvector). That makes the LoCoMo E2E QA row
    the "upstream MemPalace raw" ablation — mempalace's own verbatim postgres
    storage WITHOUT the palace graph on top.

    Requires SME_POSTGRES_DSN (no hardcoded DSN by design). Point it at an
    isolated throwaway instance — never the prod substrate.
    """
    global _PG_INGEST_SINGLETON

    import os as _os

    if not _os.environ.get("SME_POSTGRES_DSN"):
        raise RuntimeError(
            "PostgresIngestAdapter requires SME_POSTGRES_DSN pointing at an "
            "isolated throwaway postgres+pgvector instance (NOT prod). Pass a "
            "different --adapter to skip postgres."
        )

    from sme.adapters.postgres_ingest import PostgresIngestAdapter

    if _PG_INGEST_SINGLETON is None:
        _PG_INGEST_SINGLETON = PostgresIngestAdapter(n_results=5)

    corpus: list[dict] = []
    for md_file in sorted(per_q_vault.rglob("*.md")):
        if not md_file.is_file():
            continue
        text = md_file.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            # id = session id (file stem) so recall hits trace back to the
            # originating session for session-level R@K.
            corpus.append({"id": md_file.stem, "document": text})
    # TRUNCATE + upsert: prior question's haystack is wiped before this one.
    _PG_INGEST_SINGLETON.ingest_corpus(corpus)
    return _PG_INGEST_SINGLETON


def _make_karpathy_compiled_adapter(per_q_vault: Path) -> SMEAdapter:
    """Condition D2 wiring — per-question stub-compiled wiki.

    Compiles the per-question haystack into a sibling .compiled/ directory
    using the deterministic stub LLM client. This is a SMOKE-TEST wiring:
    the stub doesn't actually summarize, so the resulting D2 reading
    measures "concatenated stub text + stub index" rather than real
    LLM-compiled compression.

    For a real D2 measurement, run `sme-eval compile-wiki --llm-provider
    openai` once over a fixed corpus and point an offline run of the
    harness at the compiled output. That follow-up wiring is a separate
    PR (it requires per-question compilation to amortize across the
    LongMemEval haystack architecture).
    """
    from sme.eval.llm_clients import StubLLMClient
    from sme.conditions.karpathy_compiled import KarpathyCompiledAdapter
    from sme.conditions.wiki_compiler import compile_vault

    compiled_dir = per_q_vault.parent / f".compiled_{per_q_vault.name}"
    compile_vault(per_q_vault, compiled_dir, StubLLMClient())
    return KarpathyCompiledAdapter(compiled_dir)


def _make_mempalace_server_adapter(
    _per_q_vault: Path,
) -> SMEAdapter:  # pragma: no cover — network-dependent
    """Go MemPalace server (sefodo26) — per-question isolation via full wipe.

    The adapter's ``reset_before_ingest`` wipes the target server before each
    question's haystack loads, mirroring the reference benchmark's
    reset-store-per-question methodology. Point it at a DISPOSABLE eval
    instance (docker-compose default), never a store you care about.
    Config: MEMPALACE_SERVER_URL / MEMPALACE_SERVER_API_KEY env vars.
    """
    import os as _os

    from sme.adapters.mempalace_server_adapter import MemPalaceServerAdapter

    adapter = MemPalaceServerAdapter(
        api_url=_os.environ.get("MEMPALACE_SERVER_URL", "http://localhost:8000"),
        api_key=_os.environ.get("MEMPALACE_SERVER_API_KEY", "local-dev-key-change-me"),
        reset_before_ingest=True,
        read_only=False,
    )
    # Factory contract: the run loop only builds + queries, so the factory
    # ingests the per-question haystack itself (hindsight pattern).
    # source_file = session id (file stem) → the adapter surfaces it as
    # Entity.id, which is what the session-level R@K scorer matches.
    corpus: list[dict] = []
    for md_file in sorted(_per_q_vault.glob("*.md")):
        text = md_file.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            corpus.append({"content": text, "source_file": md_file.stem})
    if corpus:
        ing = adapter.ingest_corpus(corpus)
        if ing.get("errors"):
            raise RuntimeError(f"mempalace-server ingest errors: {ing['errors'][:3]}")
    return adapter


_ADAPTER_FACTORIES: dict[str, AdapterFactory] = {
    "full-context": _make_full_context_adapter,
    "flat": _make_flat_adapter,
    "karpathy-compiled": _make_karpathy_compiled_adapter,
    "mempalace": _make_mempalace_adapter,
    "omega": _make_omega_adapter,
    "hindsight": _make_hindsight_adapter,
    "postgres": _make_postgres_adapter,
    "mempalace-server": _make_mempalace_server_adapter,
}


# --- Scoring helpers --------------------------------------------------------


def sme_substring_recall(retrieved: str, expected: list[str]) -> tuple[float, list[str]]:
    """Substring-match the SME way: count how many expected sources
    appear as substrings of the retrieved context_string.

    Mirrors the scoring in sme.cli (lines ~1000) so the cross-validation
    numbers are comparable to SME's own retrieval-test reports.
    """
    if not expected:
        return 0.0, []
    matched = [s for s in expected if s and s in retrieved]
    return len(matched) / len(expected), matched


def judge_label_to_correct(label: str) -> Optional[bool]:
    """Convert an autoeval_label into a binary correctness signal.

    Returns True for CORRECT/ABSTAIN (ABSTAIN is "correctly refused"),
    False for INCORRECT/PARTIAL, None for ERROR (unknown — exclude
    from rate denominators).
    """
    if label == "CORRECT":
        return True
    if label == "ABSTAIN":
        return True  # for abstention questions this is the success state
    if label in ("INCORRECT", "PARTIAL"):
        return False
    return None


# LoCoMo question_type -> canonical LongMemEval judge template type. LoCoMo's
# native types are not in JUDGE_QUESTION_TYPES, so without this they all fall
# to the base correctness template. Only `temporal` needs a specialized
# template — the off-by-one-days tolerance the LoCoMo temporal questions assume
# (they ask "how many days/weeks since…"). single-hop / multi-hop / open-domain
# are plain factual correctness, which IS the base template, so they map to a
# base-template type. adversarial is handled separately via is_abstention.
_LOCOMO_TO_JUDGE_TYPE = {
    "single-hop": "single-session-user",  # base correctness template
    "multi-hop": "multi-session",  # base correctness template
    "open-domain": "multi-session",  # base correctness template
    "temporal": "temporal-reasoning",  # base + off-by-one-days tolerance
    # adversarial -> abstention is driven by is_abstention, not this map.
}


# --- Reader pass ------------------------------------------------------------


def generate_hypothesis(
    question: str,
    context_string: str,
    *,
    reader_model: str,
    client: Optional[Any] = None,
) -> str:
    """Back-compat shim — delegates to sme.eval.answer_generator.generate_answer.

    Kept so older scripts and tests calling ``harness.generate_hypothesis``
    keep working. New code should import ``generate_answer`` directly.
    """
    return generate_answer(
        question=question,
        context_string=context_string,
        reader_model=reader_model,
        client=client,
    )


# --- Per-question loop ------------------------------------------------------


def run_one_question(
    q: LMEQuestion,
    *,
    adapter_factory: AdapterFactory,
    work_dir: Path,
    skip_judge: bool,
    skip_reader: bool,
    reader_model: str,
    judge_model: str,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
    content_rules: str = "sme-rich",
    capture_context: bool = False,
) -> dict:
    """Materialize the question's per-question vault, build adapter, run
    SME + judge scorers, return one record.

    ``capture_context`` (#116 Phase 1): when True, the full retrieved
    ``context_string`` is stored on the record as ``context_string`` so a
    downstream reader sweep can replay it offline. Off by default — the
    daemon E2E benches only need ``context_chars``, not the full text.
    """
    # 1. Materialize ONLY this question to a per-question dir
    out_dir = work_dir / q.question_id
    materialize_sme_corpus([q], out_dir, max_questions=1, content_rules=content_rules)
    per_q_vault = out_dir / "vault" / q.question_id

    # 2. Build adapter
    adapter = adapter_factory(per_q_vault)

    # 3. Query
    try:
        try:
            result = adapter.query(q.question, n_results=5)
        except TypeError:
            result = adapter.query(q.question)
    except Exception as e:  # noqa: BLE001 — record but continue
        result = QueryResult(answer="", context_string="", error=str(e))
    finally:
        try:
            adapter.close()
        except Exception:  # noqa: BLE001
            pass

    return _score_and_judge(
        question=q.question,
        question_id=q.question_id,
        question_type=q.question_type,
        sme_category=q.sme_category,
        is_abstention=q.is_abstention,
        gold_answer=q.answer,
        expected=q.expected_sources_session_level(),
        result=result,
        skip_judge=skip_judge,
        skip_reader=skip_reader,
        reader_model=reader_model,
        judge_model=judge_model,
        reader_client=reader_client,
        judge_client=judge_client,
        capture_context=capture_context,
    )


def _score_and_judge(
    *,
    question: str,
    question_id: str,
    question_type: str,
    sme_category: str,
    is_abstention: bool,
    gold_answer: str,
    expected: list[str],
    result: QueryResult,
    skip_judge: bool,
    skip_reader: bool,
    reader_model: str,
    judge_model: str,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
    capture_context: bool = False,
    extra_fields: Optional[dict[str, Any]] = None,
    judge_question_type: Optional[str] = None,
) -> dict:
    """Score one retrieval (SME substring + rank-aware) and optionally run
    the reader + judge. Shared by the LongMemEval (per-question) and LoCoMo
    (per-sample) paths so both score identically.

    ``is_abstention`` drives abstention-aware judging: when True, the judge
    is invoked with ``question_type="abstention"`` regardless of the
    retrieval question_type, so a correct refusal scores as success and a
    fabricated answer scores INCORRECT. For LoCoMo this is wired from
    ``LoCoMoQuestion.is_adversarial``.

    ``judge_question_type`` lets a caller send the judge a *different* type
    string than the one recorded on the row. The recorded ``question_type``
    stays the corpus-native label (so per-category breakdowns read naturally),
    while the judge sees the canonical LongMemEval template type. LoCoMo uses
    this to map ``temporal`` -> ``temporal-reasoning`` (off-by-one tolerance);
    when None it defaults to ``question_type``.
    """
    ctx = result.context_string or ""

    # 4. SME substring score
    sme_recall, matched = sme_substring_recall(ctx, expected)

    # Rank-ordered entity IDs (#58 — surfaced so the run-script wrapper
    # can compute drawer_id-based rank-aware metrics) plus per-question
    # rank-aware diagnostics (#53 sub-task 2 — r1_misses + hit-at-K).
    # adaptmem's longmemeval_eval.py r1_misses framing: per-question
    # diagnostics on which question_types eat the recall budget. Adapter
    # retrieved_entities are in rank order; their .id is the session_id
    # for the per-session ingest topology. None-guard against adapters
    # that return retrieved_entities=None on error.
    retrieved_entity_ids = [e.id for e in (result.retrieved_entities or [])]
    expected_set = set(expected)

    # The flat baseline labels each retrieved entity ``chunk:<doc_id>``
    # (e.g. ``chunk:S0``), while ``expected`` carries bare session ids
    # (``S0``). Normalise a leading ``chunk:`` before the hit-at-K
    # comparison so per-session retrieval recall is real; the raw ids are
    # still stored verbatim on the record for provenance.
    def _norm_id(rid: str) -> str:
        return rid[len("chunk:") :] if rid.startswith("chunk:") else rid

    norm_ids = [_norm_id(rid) for rid in retrieved_entity_ids]
    rank_1 = norm_ids[0] if norm_ids else None
    hit_at_1 = rank_1 in expected_set if rank_1 is not None else False
    hit_at_5 = any(rid in expected_set for rid in norm_ids[:5])
    hit_at_10 = any(rid in expected_set for rid in norm_ids[:10])

    record: dict[str, Any] = {
        "question_id": question_id,
        "question_type": question_type,
        "sme_category": sme_category,
        "is_abstention": is_abstention,
        "expected_sources": expected,
        "matched_sources": matched,
        "sme_recall": sme_recall,
        "context_chars": len(ctx),
        "adapter_error": result.error,
        "retrieved_entity_ids": retrieved_entity_ids,
        "retrieved_rank_1": rank_1,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
    }
    if extra_fields:
        record.update(extra_fields)

    # #116 Phase 1 — persist the full retrieved context for offline reader
    # replay. Includes the fields the reader sweep needs as pinned records.
    if capture_context:
        record["context_string"] = ctx
        record["question"] = question
        record["gold_answer"] = gold_answer

    # 5. Optional reader → hypothesis
    if skip_judge:
        record["hypothesis"] = None
        record["judge"] = None
        return record

    qtype_for_judge = "abstention" if is_abstention else (judge_question_type or question_type)

    if skip_reader:
        # Hand the judge the raw retrieval (option (a) in the planning
        # doc — diagnostic-grade, not apples-to-apples with LongMemEval).
        hypothesis = ctx[:8000]  # cap to keep judge prompt manageable
    else:
        hypothesis = generate_hypothesis(
            question,
            ctx,
            reader_model=reader_model,
            client=reader_client,
        )
    record["hypothesis"] = hypothesis

    judge = grade_answer(
        question_type=qtype_for_judge,
        question=question,
        gold_answer=gold_answer,
        hypothesis=hypothesis,
        judge_model=judge_model,
        client=judge_client,
    )
    record["judge"] = judge
    return record


# --- LoCoMo per-sample loop -------------------------------------------------


def run_locomo_questions(
    questions: list[Any],  # list[LoCoMoQuestion]
    *,
    adapter_factory: AdapterFactory,
    work_dir: Path,
    skip_judge: bool,
    skip_reader: bool,
    reader_model: str,
    judge_model: str,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
    capture_context: bool = False,
    max_questions: Optional[int] = None,
) -> list[dict]:
    """Run a list of LoCoMoQuestions PER SAMPLE.

    LoCoMo shares one conversation across all of a sample's questions
    (unlike LongMemEval's per-question haystacks). For each ``sample_id``
    we materialize that sample's vault ONCE, build the adapter from it,
    then query every question in the sample against the same vault — the
    per-sample ingest topology from sme/corpora/locomo/README.md.

    The daemon adapter ignores the per-sample vault path (it owns the
    corpus and is queried over HTTP); the flat / full-context adapters
    ingest the vault, so per-sample materialization is what makes their
    LoCoMo retrieval correct (the whole conversation is in scope, not a
    single question's slice).

    Adversarial items carry ``is_adversarial=True`` → judged
    abstention-aware via ``_score_and_judge``.
    """
    from sme.corpora.locomo import materialize_sme_corpus as locomo_materialize

    # Preserve input order but group consecutive-or-not by sample_id so we
    # materialize + ingest each sample's vault exactly once.
    by_sample: dict[str, list[Any]] = {}
    order: list[str] = []
    n_capped = 0
    for q in questions:
        if max_questions is not None and n_capped >= max_questions:
            break
        n_capped += 1
        if q.sample_id not in by_sample:
            by_sample[q.sample_id] = []
            order.append(q.sample_id)
        by_sample[q.sample_id].append(q)

    records: list[dict] = []
    for sample_id in order:
        sample_qs = by_sample[sample_id]
        # Materialize this sample's full conversation ONCE. All of the
        # sample's questions share the same sessions, so any one carries
        # the full haystack — materialize just the first, capped to its
        # own sample (the loader writes the shared conversation once).
        out_dir = work_dir / sample_id
        locomo_materialize([sample_qs[0]], out_dir, max_questions=1)
        per_sample_vault = out_dir / "vault" / sample_id

        # Build the adapter once for this sample (ingests the sample vault
        # for flat / full-context; daemon ignores the path).
        adapter = adapter_factory(per_sample_vault)
        try:
            for q in sample_qs:
                try:
                    try:
                        result = adapter.query(q.question, n_results=5)
                    except TypeError:
                        result = adapter.query(q.question)
                except Exception as e:  # noqa: BLE001 — record but continue
                    result = QueryResult(answer="", context_string="", error=str(e))
                rec = _score_and_judge(
                    question=q.question,
                    question_id=q.question_id,
                    question_type=q.question_type,
                    sme_category=q.sme_category,
                    is_abstention=q.is_adversarial,
                    gold_answer=q.gold_answer,
                    expected=q.expected_sources_session_level(),
                    result=result,
                    skip_judge=skip_judge,
                    skip_reader=skip_reader,
                    reader_model=reader_model,
                    judge_model=judge_model,
                    reader_client=reader_client,
                    judge_client=judge_client,
                    capture_context=capture_context,
                    judge_question_type=_LOCOMO_TO_JUDGE_TYPE.get(q.question_type),
                    extra_fields={
                        "sample_id": q.sample_id,
                        "locomo_category": q.category,
                        "is_adversarial": q.is_adversarial,
                    },
                )
                records.append(rec)
                log.info(
                    "[%s] %s (%s / %s)",
                    sample_id,
                    q.question_id,
                    q.question_type,
                    q.sme_category,
                )
        finally:
            try:
                adapter.close()
            except Exception:  # noqa: BLE001
                pass
    return records


def run_beam_questions(
    questions: list[Any],  # list[BEAMQuestion]
    *,
    adapter_factory: AdapterFactory,
    work_dir: Path,
    skip_judge: bool,
    skip_reader: bool,
    reader_model: str,
    judge_model: str,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
    capture_context: bool = False,
    max_questions: Optional[int] = None,
    n_results: int = 5,
) -> list[dict]:
    """Run a list of BEAMQuestions PER CONVERSATION.

    BEAM shares one (very long) conversation across all of that
    conversation's 20 probing questions — the same per-conversation
    ingest topology as LoCoMo, not LongMemEval's per-question haystacks.
    For each ``conversation_id`` we materialize that conversation's vault
    ONCE, build the adapter from it, then query every question in the
    conversation against the same vault (sme/corpora/beam/README.md).

    The daemon adapter ignores the per-conversation vault path (it owns
    the corpus and is queried over HTTP); the flat / full-context
    adapters ingest the vault, so per-conversation materialization is
    what makes their BEAM retrieval correct.

    ``n_results`` is the per-query retrieval depth (top-K sessions). At
    the 100K bucket conversations have 3-5 sessions, so the default K=5
    returns the whole conversation (effectively full-context QA). At the
    500K/1M buckets conversations have ~10 sessions each totalling
    >500K/>1M tokens, far beyond any reader window - there K must be set
    low (2-3) so the retrieved top-K fits the reader. The bucket is
    recorded on every record, so a reading always states which regime it
    was taken in (sme/corpora/beam/README.md).

    Abstention items carry ``is_abstention=True``→ judged abstention-
    aware via ``_score_and_judge``.
    """
    from sme.corpora.beam import materialize_sme_corpus as beam_materialize

    # Preserve input order but group by conversation_id so we materialize
    # + ingest each conversation's vault exactly once.
    by_conv: dict[str, list[Any]] = {}
    order: list[str] = []
    n_capped = 0
    for q in questions:
        if max_questions is not None and n_capped >= max_questions:
            break
        n_capped += 1
        if q.conversation_id not in by_conv:
            by_conv[q.conversation_id] = []
            order.append(q.conversation_id)
        by_conv[q.conversation_id].append(q)

    records: list[dict] = []
    for conv_id in order:
        conv_qs = by_conv[conv_id]
        # Materialize this conversation's full chat ONCE. All of the
        # conversation's questions share the same sessions, so any one
        # carries the full haystack — materialize just the first, capped
        # to its own conversation.
        out_dir = work_dir / conv_id
        beam_materialize([conv_qs[0]], out_dir, max_questions=1)
        per_conv_vault = out_dir / "vault" / conv_id

        adapter = adapter_factory(per_conv_vault)
        try:
            for q in conv_qs:
                try:
                    try:
                        result = adapter.query(q.question, n_results=n_results)
                    except TypeError:
                        result = adapter.query(q.question)
                except Exception as e:  # noqa: BLE001 — record but continue
                    result = QueryResult(answer="", context_string="", error=str(e))
                rec = _score_and_judge(
                    question=q.question,
                    question_id=q.question_id,
                    question_type=q.ability_type,
                    sme_category=q.sme_category,
                    is_abstention=q.is_abstention,
                    gold_answer=q.gold_answer,
                    expected=q.expected_sources_session_level(),
                    result=result,
                    skip_judge=skip_judge,
                    skip_reader=skip_reader,
                    reader_model=reader_model,
                    judge_model=judge_model,
                    reader_client=reader_client,
                    judge_client=judge_client,
                    capture_context=capture_context,
                    extra_fields={
                        "conversation_id": q.conversation_id,
                        "bucket": q.bucket,
                        "beam_ability": q.ability_type,
                        "is_abstention": q.is_abstention,
                        "rubric_nuggets": q.ground_truth_nuggets,
                    },
                )
                records.append(rec)
                log.info(
                    "[%s] %s (%s / %s)",
                    conv_id,
                    q.question_id,
                    q.ability_type,
                    q.sme_category,
                )
        finally:
            try:
                adapter.close()
            except Exception:  # noqa: BLE001
                pass
    return records


# --- Aggregation ------------------------------------------------------------


def _empty_category_slot() -> dict[str, Any]:
    return {
        "n": 0,
        "sme_recall_sum": 0.0,
        "judge_correct": 0,
        "judge_incorrect": 0,
        "judge_partial": 0,
        "judge_abstain": 0,
        "judge_error": 0,
        "judge_skipped": 0,
    }


def _update_judge_label_count(slot: dict, label: str) -> None:
    key = f"judge_{label.lower()}"
    slot[key] = slot.get(key, 0) + 1


def _accumulate_usage(running: dict, fresh: dict) -> None:
    for k in running:
        running[k] += int(fresh.get(k, 0) or 0)


def _is_disagreement(sme_recall: float, judge_correct: Optional[bool]) -> bool:
    if judge_correct is None:
        return False
    sme_says_correct = sme_recall >= _DISAGREE_THRESHOLD
    return sme_says_correct != judge_correct


def _update_slot_for_record(slot: dict, record: dict) -> Optional[str]:
    """Apply one record's counts to its category slot.

    Returns the judge label string when there's a real judge verdict,
    None when the judge was skipped. Disagreement detection happens
    in the caller because it needs visibility into the disagreement
    list.
    """
    slot["n"] += 1
    slot["sme_recall_sum"] += record["sme_recall"]
    judge = record.get("judge")
    if judge is None:
        slot["judge_skipped"] += 1
        return None
    label = judge.get("autoeval_label", "ERROR")
    _update_judge_label_count(slot, label)
    return label


def aggregate(records: list[dict]) -> dict:
    """Compute per-category SME mean recall, judge correct-rate, and
    a disagreement set. Per the KU semantic-divergence caveat, NO
    overall single-number correlation is reported.
    """
    by_cat: dict[str, dict[str, Any]] = {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    disagreements: list[dict] = []

    for r in records:
        cat = r["sme_category"]
        if cat not in by_cat:
            by_cat[cat] = _empty_category_slot()
        label = _update_slot_for_record(by_cat[cat], r)
        if label is None:
            continue

        judge = r["judge"]
        _accumulate_usage(total_usage, judge.get("usage") or {})

        if _is_disagreement(r["sme_recall"], judge_label_to_correct(label)):
            disagreements.append(
                {
                    "question_id": r["question_id"],
                    "sme_category": cat,
                    "sme_recall": r["sme_recall"],
                    "judge_label": label,
                    "judge_rationale": judge.get("rationale", ""),
                }
            )

    per_cat = {}
    for cat, slot in sorted(by_cat.items()):
        n = slot["n"]
        judged = (
            slot["judge_correct"]
            + slot["judge_incorrect"]
            + slot["judge_partial"]
            + slot["judge_abstain"]
        )
        sme_recall_mean = slot["sme_recall_sum"] / n if n else 0.0
        judge_correct_rate = (
            (slot["judge_correct"] + slot["judge_abstain"]) / judged if judged else None
        )
        per_cat[cat] = {
            "n": n,
            "sme_recall_mean": round(sme_recall_mean, 4),
            "judge_correct_rate": (
                round(judge_correct_rate, 4) if judge_correct_rate is not None else None
            ),
            "judge_label_counts": {
                "CORRECT": slot["judge_correct"],
                "PARTIAL": slot["judge_partial"],
                "INCORRECT": slot["judge_incorrect"],
                "ABSTAIN": slot["judge_abstain"],
                "ERROR": slot["judge_error"],
                "skipped": slot["judge_skipped"],
            },
        }

    dual = aggregate_dual_metric(records)

    # Rank-aware diagnostics (#53 sub-task 2). r1_misses lists every
    # question whose rank-1 retrieval missed the expected session, with
    # whether top-5 or top-10 saved it. r1_miss_by_type is the
    # question_type histogram for triaging which categories eat the
    # recall budget — direct port from adaptmem's longmemeval_eval.py.
    r1_misses: list[dict] = []
    r1_miss_by_type: dict[str, int] = {}
    for r in records:
        if r.get("hit_at_1") is False:
            qtype = r.get("question_type", "unknown")
            r1_misses.append(
                {
                    "question_id": r["question_id"],
                    "question_type": qtype,
                    "retrieved_rank_1": r.get("retrieved_rank_1"),
                    "expected_sources": r.get("expected_sources", []),
                    "hit_at_5": r.get("hit_at_5"),
                    "hit_at_10": r.get("hit_at_10"),
                }
            )
            r1_miss_by_type[qtype] = r1_miss_by_type.get(qtype, 0) + 1

    return {
        "per_category": per_cat,
        "total_questions": len(records),
        "judge_total_usage": total_usage,
        "disagreements": disagreements,
        "dual_metric": dual,
        "r1_misses": r1_misses,
        "r1_miss_by_type": r1_miss_by_type,
        "ku_caveat": (
            "Per-category numbers are reported separately by design. "
            "KU (knowledge-update) and SME Cat 3 measure different "
            "primitives — KU rewards returning the new value, Cat 3 "
            "rewards flagging the contradiction. A single overall "
            "correlation would mislead. See docs/related_work/longmemeval.md."
        ),
    }


# --- Main -------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run LongMemEval questions through an SME adapter, score with "
            "both SME's substring matcher and LongMemEval's GPT-4o judge, "
            "report per-category disagreement."
        ),
    )
    p.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the dataset JSON. For --corpus longmemeval: "
        "longmemeval_oracle.json (or _s / _m). For --corpus "
        "locomo: locomo10.json. For --corpus beam: a cached "
        "per-bucket BEAM split (e.g. beam_100K.json) or the "
        "committed sample (sme/corpora/beam/sample/"
        "beam_100K_sample.json).",
    )
    p.add_argument(
        "--corpus",
        default="longmemeval",
        choices=["longmemeval", "locomo", "beam"],
        help="Dataset shape. 'longmemeval' (default) = one "
        "haystack per question. 'locomo' = one conversation "
        "per sample shared by that sample's questions; "
        "questions are grouped by sample_id and ingested "
        "per sample, and adversarial (cat-5) items are judged "
        "abstention-aware. 'beam' = one (very long) "
        "conversation per conversation_id shared by its 20 "
        "probing questions; grouped + ingested per "
        "conversation, graded at a token --bucket, and "
        "abstention items judged abstention-aware.",
    )
    p.add_argument(
        "--bucket",
        default="100K",
        choices=["100K", "500K", "1M", "10M"],
        help="BEAM token bucket (only used with --corpus beam). "
        "The same conversation at a different bucket is a "
        "different retrieval problem, so the bucket is "
        "recorded on every record.",
    )
    p.add_argument(
        "--beam-n-results",
        type=int,
        default=5,
        help="BEAM per-query retrieval depth (top-K sessions; "
        "only used with --corpus beam). Default 5 returns "
        "the whole conversation at the 100K bucket (3-5 "
        "sessions). At 500K/1M (~10 sessions, >500K/>1M "
        "tokens) set this low (2-3) so the retrieved top-K "
        "fits the reader window.",
    )
    p.add_argument(
        "--adapter", required=True, choices=sorted(_ADAPTER_FACTORIES), help="SME adapter to run."
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Smoke-test cap on number of questions. NOTE: the "
        "oracle/S corpora are question_type-sorted, so a bare "
        "cap is a single-category slice — pair with "
        "--stratify-by or --shuffle (#122). longmemeval only.",
    )
    p.add_argument(
        "--shuffle",
        type=int,
        default=None,
        metavar="SEED",
        help="Deterministically shuffle questions (with SEED) before "
        "the --max-questions cap, so the cap is not a single-"
        "category slice. longmemeval corpus only.",
    )
    p.add_argument(
        "--stratify-by",
        default=None,
        metavar="FIELD",
        help="Stratify the --max-questions cap across this question "
        "field (e.g. question_type) — even round-robin per "
        "category for representative coverage (#122). Matches the "
        "mempalace-daemon strat150 baseline. longmemeval only.",
    )
    p.add_argument(
        "--reader-model",
        default="gpt-4o-mini",
        help="Model used to turn retrieved context into an "
        "answer the judge can score. Default kept as "
        "gpt-4o-mini for back-compat with prior runs; "
        "the `sme-eval longmemeval` CLI defaults to "
        "gpt-4.1-mini per issue #17.",
    )
    p.add_argument("--judge-model", default="gpt-4o-2024-08-06", help="LongMemEval judge model.")
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Run SME-only — no reader, no judge, no API key required.",
    )
    p.add_argument(
        "--skip-reader",
        action="store_true",
        help="Skip the reader pass; feed raw retrieval to the judge (diagnostic mode).",
    )
    p.add_argument("--out", type=Path, default=None, help="Where to write the report JSON.")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Where per-question vaults are materialized (default: tmpdir).",
    )
    p.add_argument(
        "--content-rules",
        default="sme-rich",
        choices=["sme-rich", "upstream-exact"],
        help="Session rendering rules. 'sme-rich' (default) = "
        "frontmatter + role headers + user + assistant turns. "
        "'upstream-exact' = user turns only, no metadata — "
        "matches upstream protocol per #54 / #51.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run(
    args: argparse.Namespace,
    *,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
) -> dict:
    """Programmatic entry point — used by tests."""
    factory = _ADAPTER_FACTORIES[args.adapter]

    work_dir_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    if args.work_dir is not None:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir_ctx = tempfile.TemporaryDirectory(prefix="sme_xval_")
        work_dir = Path(work_dir_ctx.name)

    corpus = getattr(args, "corpus", "longmemeval")
    records: list[dict] = []
    try:
        if corpus == "locomo":
            from sme.corpora.locomo import load_questions as load_locomo

            records = run_locomo_questions(
                list(load_locomo(args.dataset)),
                adapter_factory=factory,
                work_dir=work_dir,
                skip_judge=args.skip_judge,
                skip_reader=args.skip_reader,
                reader_model=args.reader_model,
                judge_model=args.judge_model,
                reader_client=reader_client,
                judge_client=judge_client,
                max_questions=args.max_questions,
            )
        elif corpus == "beam":
            from sme.corpora.beam import load_questions as load_beam

            bucket = getattr(args, "bucket", "100K")
            records = run_beam_questions(
                list(load_beam(args.dataset, bucket=bucket)),
                adapter_factory=factory,
                work_dir=work_dir,
                skip_judge=args.skip_judge,
                skip_reader=args.skip_reader,
                reader_model=args.reader_model,
                judge_model=args.judge_model,
                reader_client=reader_client,
                judge_client=judge_client,
                max_questions=args.max_questions,
                n_results=getattr(args, "beam_n_results", 5),
            )
        else:
            questions = list(load_questions(args.dataset))
            # techempower-org/...#122: the oracle/S corpora are sorted by
            # question_type, so a bare ``[:N]`` cap is a single-category slice.
            # --shuffle re-orders deterministically; --stratify-by draws an even
            # round-robin across the field's values so the cap stays
            # representative. Mirrors run_longmemeval_mempalace so a competitor
            # adapter lands on the SAME stratified subset as the daemon baseline.
            shuffle_seed = getattr(args, "shuffle", None)
            stratify_by = getattr(args, "stratify_by", None)
            if shuffle_seed is not None:
                import random

                random.Random(shuffle_seed).shuffle(questions)
            if args.max_questions is not None:
                if stratify_by:
                    questions = _stratified_cap(questions, args.max_questions, stratify_by)
                else:
                    questions = questions[: args.max_questions]
            for i, q in enumerate(questions):
                log.info(
                    "[%d] %s (%s / %s)",
                    i,
                    q.question_id,
                    q.question_type,
                    q.sme_category,
                )
                rec = run_one_question(
                    q,
                    adapter_factory=factory,
                    work_dir=work_dir,
                    skip_judge=args.skip_judge,
                    skip_reader=args.skip_reader,
                    reader_model=args.reader_model,
                    judge_model=args.judge_model,
                    reader_client=reader_client,
                    judge_client=judge_client,
                    content_rules=getattr(args, "content_rules", "sme-rich"),
                )
                records.append(rec)
    finally:
        if work_dir_ctx is not None:
            work_dir_ctx.cleanup()

    summary = aggregate(records)
    report = {
        "run_metadata": {
            "dataset": str(args.dataset),
            "corpus": corpus,
            "bucket": (getattr(args, "bucket", None) if corpus == "beam" else None),
            "beam_n_results": (getattr(args, "beam_n_results", 5) if corpus == "beam" else None),
            "adapter": args.adapter,
            "reader_model": (None if args.skip_reader or args.skip_judge else args.reader_model),
            "judge_model": (None if args.skip_judge else args.judge_model),
            "skip_judge": bool(args.skip_judge),
            "skip_reader": bool(args.skip_reader),
            "max_questions": args.max_questions,
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "summary": summary,
        "per_question": records,
    }
    return report


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    report = run(args)

    out = args.out
    if out is None:
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        out = Path(f"cross_validation_{ts}.json")
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"Wrote {out}")
    summary = report["summary"]
    for cat, slot in summary["per_category"].items():
        print(
            f"  {cat:20s} n={slot['n']:4d}  "
            f"sme_recall={slot['sme_recall_mean']:.3f}  "
            f"judge_correct_rate={slot['judge_correct_rate']!s}"
        )
    print(f"  disagreements: {len(summary['disagreements'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
