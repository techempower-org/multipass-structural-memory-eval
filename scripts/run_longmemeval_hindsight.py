"""Run LongMemEval through Hindsight (MCP/HTTP memory) with session-level R@K
+ the canonical reader/judge QA pipeline — the SECOND independent competitor
run on SME's identical corpus+reader+judge (techempower-org/
multipass-structural-memory-eval#184), after OMEGA (#178/#183).

Sibling of ``run_longmemeval_omega.py``; same protocol (stratified subset,
per-question isolation, ``--content-rules``, reader, canonical type-specific
``gpt-5.3-chat`` judge), differing only where Hindsight differs:

  1. **Isolation = bank_id, not a tempdir.** Hindsight is a server that owns a
     single store and namespaces by ``bank_id``. Each question gets a unique
     bank, so prior questions can't leak into the current question's recall —
     the role the daemon's per-question wing and OMEGA's per-question OMEGA_HOME
     play.

  2. **EXTRACTION-BASED retrieval (the methodology caveat, #184).** Unlike
     mempalace (raw drawers) or OMEGA (raw memories), Hindsight stores
     LLM-EXTRACTED facts, not raw sessions. Recall returns fact-units. We tag
     each ingested session with ``document_id=<session_id>``; recall echoes that
     ``document_id`` back, so session-level R@K is a membership test against
     ``q.expected_sources_session_level()`` — BUT it measures "did a fact
     *extracted from* the evidence session rank top-K", which is softer and
     extraction-mediated vs a raw-chunk R@K. The QA number (reader + canonical
     judge over recalled facts) is the cleaner apples-to-apples metric.

  3. **Extraction LLM ≠ reader/judge LLM.** Hindsight's inline fact extraction
     runs on its own configured provider (here: a local ollama model). The SME
     reader + judge remain the canonical Azure ``gpt-5.3-chat``. A weak
     extractor silently empties the memory (#184) — verify the live smoke
     passes first.

Usage::

    HINDSIGHT_BASE_URL=http://localhost:8888 \
    AZURE_API_KEY=... AZURE_API_BASE=... \
    python scripts/run_longmemeval_hindsight.py \
      --questions sme/corpora/longmemeval/data/longmemeval_s_cleaned.json \
      --max-questions 150 --stratify-by question_type \
      --content-rules upstream-exact \
      --json baselines/longmemeval_hindsight_strat150_qa_<date>.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cross_validate_longmemeval as harness  # noqa: E402
from sme.adapters.base import QueryResult  # noqa: E402
from sme.corpora.longmemeval import LMEQuestion, load_questions  # noqa: E402

log = logging.getLogger("run_longmemeval_hindsight")

DEFAULT_READER_MODEL = "o4-mini"
DEFAULT_JUDGE_MODEL = "gpt-5.3-chat"


def _render_session(s: Any, content_rules: str) -> str:
    """Render one haystack session to text — byte-identical to the daemon /
    OMEGA ingest path so the only variable is the memory substrate."""
    if content_rules == "upstream-exact":
        return "\n".join(t.content for t in s.turns if t.role == "user")
    body_parts = [f"# Session {s.session_id}", f"_Date: {s.date}_", ""]
    for t in s.turns:
        marker = "  <!-- evidence -->" if t.has_answer else ""
        body_parts.append(f"## {t.role}{marker}\n\n{t.content}")
    return "\n".join(body_parts)


def _build_question_corpus(q: LMEQuestion, content_rules: str) -> list[dict[str, Any]]:
    """One corpus row per session, tagged with ``document_id=<session_id>``.

    document_id is what Hindsight echoes back on every recall hit, so
    session-level R@K is a membership test against the evidence sessions."""
    corpus: list[dict[str, Any]] = []
    for s in q.haystack_sessions:
        text = _render_session(s, content_rules)
        if not text.strip():
            continue
        corpus.append({"content": text, "document_id": s.session_id})
    return corpus


def _session_hits(result: QueryResult) -> list[Optional[str]]:
    """Rank-ordered session ids from a QueryResult's retrieved entities.

    Reads the ``document_id`` the Hindsight adapter surfaces in each Entity's
    properties (= the session id we supplied at retain time). None per hit when
    a recall result carries no document_id (e.g. a raw ``observation`` unit)."""
    out: list[Optional[str]] = []
    for e in result.retrieved_entities or []:
        props = e.properties or {}
        out.append(props.get("document_id"))
    return out


def run_one_question(
    q: LMEQuestion,
    *,
    base_url: str,
    content_rules: str,
    n_results: int,
    skip_judge: bool,
    skip_reader: bool,
    reader_model: str,
    judge_model: str,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
    adapter_factory: Optional[Any] = None,
) -> dict:
    """Ingest q's haystack into an isolated Hindsight bank, retrieve, score.

    ``adapter_factory(bank_id)`` is injectable for tests; defaults to a real
    HindsightAdapter against ``base_url``."""
    expected = q.expected_sources_session_level()
    expected_set = set(expected)

    bank_id = f"lme_{q.question_id}_{uuid.uuid4().hex[:6]}"
    if adapter_factory is not None:
        adapter = adapter_factory(bank_id)
    else:
        from sme.adapters.hindsight import HindsightAdapter
        adapter = HindsightAdapter(base_url=base_url, bank_id=bank_id, n_results=n_results)

    ingest_report: dict[str, Any] = {"entities_created": 0, "errors": []}
    try:
        corpus = _build_question_corpus(q, content_rules)
        ingest_report = adapter.ingest_corpus(corpus)
        try:
            result = adapter.query(q.question, n_results=n_results)
        except Exception as e:  # noqa: BLE001 — record but continue
            result = QueryResult(answer="", context_string="", error=str(e))
    finally:
        try:
            adapter.close()
        except Exception:  # noqa: BLE001
            pass

    # --- session-level R@K (extraction-mediated; the daemon drawer_hit_at_K
    #     analogue, via document_id) ---
    session_ranks = _session_hits(result)
    rank_1 = session_ranks[0] if session_ranks else None
    hit_at_1 = bool(rank_1 is not None and rank_1 in expected_set)
    hit_at_5 = bool(any(sid in expected_set for sid in session_ranks[:5]))
    hit_at_10 = bool(any(sid in expected_set for sid in session_ranks[:10]))
    recalled = len(expected_set & set(s for s in session_ranks if s))
    session_recall = recalled / len(expected_set) if expected_set else 0.0

    rec = harness._score_and_judge(
        question=q.question,
        question_id=q.question_id,
        question_type=q.question_type,
        sme_category=q.sme_category,
        is_abstention=q.is_abstention,
        gold_answer=q.answer,
        expected=[],  # session-level R is computed above; don't fight substring
        result=result,
        skip_judge=skip_judge,
        skip_reader=skip_reader,
        reader_model=reader_model,
        judge_model=judge_model,
        reader_client=reader_client,
        judge_client=judge_client,
    )

    rec["expected_sources"] = expected
    rec["retrieved_session_ids"] = session_ranks
    rec["hindsight_hit_at_1"] = hit_at_1
    rec["hindsight_hit_at_5"] = hit_at_5
    rec["hindsight_hit_at_10"] = hit_at_10
    rec["sme_recall"] = round(session_recall, 4)
    rec["ingest"] = {
        "sessions_stored": ingest_report.get("entities_created", 0),
        "errors": list(ingest_report.get("errors", [])),
    }
    return rec


def _group_by(records: list[dict], field: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r.get(field, "unknown"), []).append(r)
    return dict(sorted(out.items()))


def _aggregate(records: list[dict]) -> dict:
    summary = harness.aggregate(records)

    def _rate(rs: list[dict], key: str) -> Optional[float]:
        if not rs:
            return None
        return round(sum(1 for r in rs if r.get(key)) / len(rs), 4)

    summary["retrieval_session_level"] = {
        "overall": {
            "n": len(records),
            "r_at_1": _rate(records, "hindsight_hit_at_1"),
            "r_at_5": _rate(records, "hindsight_hit_at_5"),
            "r_at_10": _rate(records, "hindsight_hit_at_10"),
        },
        "per_category": {
            cat: {
                "n": len(rs),
                "r_at_1": _rate(rs, "hindsight_hit_at_1"),
                "r_at_5": _rate(rs, "hindsight_hit_at_5"),
                "r_at_10": _rate(rs, "hindsight_hit_at_10"),
            }
            for cat, rs in _group_by(records, "sme_category").items()
        },
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_longmemeval_hindsight",
        description="Run LongMemEval through Hindsight (extraction-based memory) "
                    "with session-level R@K + canonical reader/judge QA (#184).",
    )
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--base-url", default=None,
                   help="Hindsight server (default HINDSIGHT_BASE_URL or "
                        "http://localhost:8888).")
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--stratify-by", default=None, metavar="FIELD")
    p.add_argument("--shuffle", type=int, default=None, metavar="SEED")
    p.add_argument("--content-rules", default="upstream-exact",
                   choices=["sme-rich", "upstream-exact"])
    p.add_argument("--answer-model", default=DEFAULT_READER_MODEL)
    p.add_argument("--judge", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--n-results", type=int, default=5)
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--skip-reader", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--status", type=Path, default=None,
                   help="Write a one-line progress/STATUS file (for detached runs).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _select_questions(args: argparse.Namespace) -> list[LMEQuestion]:
    questions = list(load_questions(args.questions))
    if args.shuffle is not None:
        import random
        random.Random(args.shuffle).shuffle(questions)
    if args.max_questions is not None:
        if args.stratify_by:
            questions = harness._stratified_cap(
                questions, args.max_questions, args.stratify_by
            )
        else:
            questions = questions[: args.max_questions]
    return questions


def _resolve_base_url(args: argparse.Namespace) -> str:
    import os
    return args.base_url or os.environ.get("HINDSIGHT_BASE_URL") or "http://localhost:8888"


def run(
    args: argparse.Namespace,
    *,
    questions: Optional[list[LMEQuestion]] = None,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
    adapter_factory: Optional[Any] = None,
) -> dict:
    if questions is None:
        questions = _select_questions(args)
    base_url = _resolve_base_url(args)

    def _status(msg: str) -> None:
        if args.status is not None:
            Path(args.status).write_text(msg + "\n")

    records: list[dict] = []
    for i, q in enumerate(questions):
        log.info("[%d/%d] %s (%s / %s)", i + 1, len(questions),
                 q.question_id, q.question_type, q.sme_category)
        rec = run_one_question(
            q, base_url=base_url, content_rules=args.content_rules,
            n_results=args.n_results, skip_judge=args.skip_judge,
            skip_reader=args.skip_reader, reader_model=args.answer_model,
            judge_model=args.judge, reader_client=reader_client,
            judge_client=judge_client, adapter_factory=adapter_factory,
        )
        records.append(rec)
        if (i + 1) % 5 == 0:
            _status(f"RUNNING {i + 1}/{len(questions)} last={q.question_id}")

    summary = _aggregate(records)
    return {
        "run_metadata": {
            "mode": "live",
            "adapter": "hindsight",
            "hindsight_client_version": _hindsight_version(),
            "questions": str(args.questions),
            "n_questions": len(records),
            "stratify_by": args.stratify_by,
            "shuffle_seed": args.shuffle,
            "content_rules": args.content_rules,
            "n_results": args.n_results,
            "answer_model": (None if args.skip_judge else args.answer_model),
            "judge_model": (None if args.skip_judge else args.judge),
            "skip_judge": bool(args.skip_judge),
            "skip_reader": bool(args.skip_reader),
            "retrieval_metric": "hindsight session-level hit@K via document_id "
                                "(EXTRACTION-MEDIATED: a fact extracted from the "
                                "evidence session ranking top-K; softer than a "
                                "raw-chunk R@K — see #184). Comparable in spirit "
                                "to daemon drawer_hit_at_K but not identical.",
            "substrate": "Hindsight server (pgvector + biomimetic fact store); "
                         "local bge-small embeddings; cross-encoder rerank",
            "extraction_llm": "ollama (local) — separate from the SME reader/judge",
            "isolation": "per-question Hindsight bank_id; throwaway local server; "
                         "prod familiar/palace-daemon untouched",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "summary": summary,
        "per_question": records,
    }


def _hindsight_version() -> str:
    try:
        import hindsight_client
        return getattr(hindsight_client, "__version__", "?")
    except Exception:  # noqa: BLE001
        return "?"


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    report = run(args)
    out_path = args.json
    if out_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d")
        suffix = "r5" if args.skip_judge else "qa"
        out_path = Path(f"longmemeval_hindsight_{ts}_{suffix}.json")
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    rsl = report["summary"].get("retrieval_session_level", {}).get("overall", {})
    if args.status is not None:
        Path(args.status).write_text(
            f"DONE n={report['summary'].get('total_questions')} "
            f"R@5={rsl.get('r_at_5')} wrote {out_path}\n"
        )
    print(f"\nWrote {out_path}  R@5={rsl.get('r_at_5')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
