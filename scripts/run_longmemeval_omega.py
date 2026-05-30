#!/usr/bin/env python3
"""Run OMEGA through the LongMemEval E2E QA + R@K pipeline.

This is the OMEGA twin of ``scripts/run_longmemeval_mempalace.py`` — the
first **independent** competitor run on SME's identical corpus + reader +
canonical judge (techempower-org/multipass-structural-memory-eval#178). It
exists so the comparison matrix stops citing OMEGA's own self-reported
leaderboard number (95.4% QA, GPT-4.1 answer model, self-verified) and starts
measuring OMEGA under the **same conditions** as the mempalace-daemon: same
LongMemEval-S stratified subset, same per-question isolation topology, same
``--content-rules``, same reader (o4-mini), same canonical type-specific
``gpt-5.3-chat`` judge.

Why a dedicated runner rather than ``cross_validate_longmemeval --adapter
omega``? Two reasons, both about **scoring fidelity**:

  1. **Session-level R@K.** LongMemEval R@K asks "did retrieval surface the
     evidence *session*?" The mempalace-daemon answers this via a
     session_id→drawer_id map built at ingest (#58 / #98). OMEGA's analogue:
     ingest each session as one memory tagged with ``session_id=<the session
     id>``; OMEGA returns that ``session_id`` on every ``query_structured``
     hit, so hit@K is a direct set-membership test against the question's
     ``expected_sources``. The generic ``cross_validate`` path scores OMEGA
     with the substring matcher (``sme_recall``), which is **structurally 0**
     under ``--content-rules upstream-exact`` (the session ids never appear in
     the user-turn text) — that's a metric artifact, not an OMEGA deficiency.
  2. **One ONNX load per question, not per session.** A single reused adapter
     per question loads OMEGA's bge-small ONNX embedding model once and ingests
     all of that question's sessions through it.

Per-question isolation: each question gets its own ``OMEGA_HOME`` (a temp dir),
so prior questions can't leak into the current question's retrieval — the
OMEGA twin of the daemon's per-question wing scoping.

CLI mirrors run_longmemeval_mempalace where it makes sense:

    run_longmemeval_omega.py
        --questions JSON              # longmemeval_s_cleaned.json (or _oracle)
        --max-questions N             # cap (pair with --stratify-by)
        --stratify-by FIELD           # e.g. question_type — even round-robin (#122)
        --shuffle SEED                # deterministic shuffle before the cap
        --content-rules {sme-rich,upstream-exact}
        --answer-model MODEL          # reader (default o4-mini, Azure)
        --judge MODEL                 # judge (default gpt-5.3-chat, canonical prompts)
        --skip-judge                  # R@K-only (no reader, no judge, no Azure)
        --skip-reader                 # feed raw retrieval to the judge
        --n-results K                 # OMEGA top-K (default 5)
        --json PATH                   # report destination

LOCAL ONLY — OMEGA is a local SQLite store; this run never touches the prod
familiar / palace-daemon.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

# Make the repo importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cross_validate_longmemeval as harness  # noqa: E402
from sme.adapters.base import QueryResult  # noqa: E402
from sme.corpora.longmemeval import (  # noqa: E402
    LMEQuestion,
    load_questions,
)

log = logging.getLogger("run_longmemeval_omega")

# Azure-friendly defaults, identical to run_longmemeval_mempalace so the
# competitor row uses the same reader + judge as the mempalace row.
DEFAULT_READER_MODEL = "o4-mini"
DEFAULT_JUDGE_MODEL = "gpt-5.3-chat"


def _render_session(s: Any, content_rules: str) -> str:
    """Render one haystack session to text, matching the daemon ingest path.

    ``upstream-exact`` = user turns only joined by newline (upstream protocol,
    removes the documented sme-rich loader-cost); ``sme-rich`` = role headers +
    date + both roles. Mirrors ``ingest_question_haystack`` in
    run_longmemeval_mempalace so OMEGA ingests byte-identical content.
    """
    if content_rules == "upstream-exact":
        return "\n".join(t.content for t in s.turns if t.role == "user")
    body_parts = [f"# Session {s.session_id}", f"_Date: {s.date}_", ""]
    for t in s.turns:
        marker = "  <!-- evidence -->" if t.has_answer else ""
        body_parts.append(f"## {t.role}{marker}\n\n{t.content}")
    return "\n".join(body_parts)


def _build_question_corpus(
    q: LMEQuestion, content_rules: str
) -> list[dict[str, Any]]:
    """One corpus row per session, tagged with its ``session_id``.

    The session_id tag is what makes OMEGA's retrieval scorable at the session
    level — OMEGA returns it on every hit, so hit@K is a membership test
    against ``q.expected_sources_session_level()``.
    """
    corpus: list[dict[str, Any]] = []
    for s in q.haystack_sessions:
        text = _render_session(s, content_rules)
        if not text.strip():
            continue
        corpus.append(
            {"content": text, "type": "summary", "session_id": s.session_id}
        )
    return corpus


def _session_hits(result: QueryResult) -> list[Optional[str]]:
    """Rank-ordered session ids from a QueryResult's retrieved entities.

    Reads the ``session_id`` the OMEGA adapter surfaces in each Entity's
    properties (added for this benchmark). Falls back to None per hit when a
    memory was stored without a session_id.
    """
    out: list[Optional[str]] = []
    for e in result.retrieved_entities or []:
        props = e.properties or {}
        out.append(props.get("session_id"))
    return out


def run_one_question(
    q: LMEQuestion,
    *,
    content_rules: str,
    n_results: int,
    skip_judge: bool,
    skip_reader: bool,
    reader_model: str,
    judge_model: str,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
) -> dict:
    """Ingest q's haystack into an isolated OMEGA store, retrieve, score.

    Reuses ``harness._score_and_judge`` for the reader + judge half so QA is
    scored identically to the mempalace-daemon run; adds OMEGA session-level
    hit@K on top.
    """
    from sme.adapters.omega import OmegaAdapter

    expected = q.expected_sources_session_level()
    expected_set = set(expected)

    with tempfile.TemporaryDirectory(prefix="omega_lme_") as home:
        adapter = OmegaAdapter(omega_home=home, n_results=n_results)
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

    # --- OMEGA session-level R@K (the daemon drawer_hit_at_K analogue) ---
    session_ranks = _session_hits(result)
    rank_1 = session_ranks[0] if session_ranks else None
    hit_at_1 = bool(rank_1 is not None and rank_1 in expected_set)
    hit_at_5 = bool(any(sid in expected_set for sid in session_ranks[:5]))
    hit_at_10 = bool(any(sid in expected_set for sid in session_ranks[:10]))
    recalled = len(expected_set & set(s for s in session_ranks if s))
    session_recall = recalled / len(expected_set) if expected_set else 0.0

    # Reuse the shared scorer for the reader + judge half. Pass expected=[] so
    # the substring matcher (sme_recall) doesn't fight the session-level metric
    # we compute above; the judge half is what we want from it.
    rec = harness._score_and_judge(
        question=q.question,
        question_id=q.question_id,
        question_type=q.question_type,
        sme_category=q.sme_category,
        is_abstention=q.is_abstention,
        gold_answer=q.answer,
        expected=[],
        result=result,
        skip_judge=skip_judge,
        skip_reader=skip_reader,
        reader_model=reader_model,
        judge_model=judge_model,
        reader_client=reader_client,
        judge_client=judge_client,
    )

    # Overlay OMEGA's session-level retrieval metrics (the comparable R@K).
    rec["expected_sources"] = expected
    rec["retrieved_session_ids"] = session_ranks
    rec["omega_hit_at_1"] = hit_at_1
    rec["omega_hit_at_5"] = hit_at_5
    rec["omega_hit_at_10"] = hit_at_10
    rec["sme_recall"] = round(session_recall, 4)  # session-level R, not substring
    rec["ingest"] = {
        "sessions_stored": ingest_report.get("entities_created", 0),
        "errors": list(ingest_report.get("errors", [])),
    }
    return rec


def _aggregate(records: list[dict]) -> dict:
    """Per-category + overall R@K (session-level) and QA accuracy.

    Reuses ``harness.aggregate`` (which keys QA off the judge labels and the
    ``sme_recall`` field we set to the session-level recall) and adds explicit
    omega_hit_at_K rates so R@K is reported the OMEGA way alongside the daemon.
    """
    summary = harness.aggregate(records)

    # Session-level R@K rates, overall and per category — directly comparable
    # to the daemon's drawer_hit_at_K.
    def _rate(rs: list[dict], key: str) -> Optional[float]:
        if not rs:
            return None
        return round(sum(1 for r in rs if r.get(key)) / len(rs), 4)

    summary["retrieval_session_level"] = {
        "overall": {
            "n": len(records),
            "r_at_1": _rate(records, "omega_hit_at_1"),
            "r_at_5": _rate(records, "omega_hit_at_5"),
            "r_at_10": _rate(records, "omega_hit_at_10"),
        },
        "per_category": {
            cat: {
                "n": len(rs),
                "r_at_1": _rate(rs, "omega_hit_at_1"),
                "r_at_5": _rate(rs, "omega_hit_at_5"),
                "r_at_10": _rate(rs, "omega_hit_at_10"),
            }
            for cat, rs in _group_by(records, "sme_category").items()
        },
    }
    return summary


def _group_by(records: list[dict], field: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r.get(field, "unknown"), []).append(r)
    return dict(sorted(out.items()))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_longmemeval_omega",
        description=(
            "Run LongMemEval through OMEGA (local SQLite memory) with "
            "session-level R@K + the canonical reader/judge QA pipeline — the "
            "first independent competitor run on SME's identical "
            "corpus+reader+judge (#178)."
        ),
    )
    p.add_argument("--questions", required=True, type=Path,
                   help="Path to longmemeval_s_cleaned.json (or _oracle).")
    p.add_argument("--max-questions", type=int, default=None,
                   help="Cap. Pair with --stratify-by — the S corpus is "
                        "question_type-sorted, so a bare cap is single-category.")
    p.add_argument("--stratify-by", default=None, metavar="FIELD",
                   help="Even round-robin cap across this field (#122). Use "
                        "question_type to match the mempalace strat150 subset.")
    p.add_argument("--shuffle", type=int, default=None, metavar="SEED",
                   help="Deterministic shuffle before the cap.")
    p.add_argument("--content-rules", default="upstream-exact",
                   choices=["sme-rich", "upstream-exact"],
                   help="Session rendering. Default upstream-exact to match the "
                        "mempalace strat150 baseline.")
    p.add_argument("--answer-model", default=DEFAULT_READER_MODEL,
                   help=f"Reader model (default {DEFAULT_READER_MODEL}, Azure).")
    p.add_argument("--judge", default=DEFAULT_JUDGE_MODEL,
                   help=f"Judge model (default {DEFAULT_JUDGE_MODEL}, canonical "
                        "type-specific prompts).")
    p.add_argument("--n-results", type=int, default=5,
                   help="OMEGA top-K (default 5).")
    p.add_argument("--skip-judge", action="store_true",
                   help="R@K-only — no reader, no judge, no Azure.")
    p.add_argument("--skip-reader", action="store_true",
                   help="Feed raw retrieval to the judge (diagnostic).")
    p.add_argument("--json", type=Path, default=None,
                   help="Report JSON destination.")
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


def run(
    args: argparse.Namespace,
    *,
    questions: Optional[list[LMEQuestion]] = None,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
) -> dict:
    if questions is None:
        questions = _select_questions(args)

    records: list[dict] = []
    for i, q in enumerate(questions):
        log.info(
            "[%d/%d] %s (%s / %s)",
            i + 1, len(questions), q.question_id,
            q.question_type, q.sme_category,
        )
        rec = run_one_question(
            q,
            content_rules=args.content_rules,
            n_results=args.n_results,
            skip_judge=args.skip_judge,
            skip_reader=args.skip_reader,
            reader_model=args.answer_model,
            judge_model=args.judge,
            reader_client=reader_client,
            judge_client=judge_client,
        )
        records.append(rec)

    summary = _aggregate(records)
    return {
        "run_metadata": {
            "mode": "live",
            "adapter": "omega",
            "omega_version": _omega_version(),
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
            "retrieval_metric": "omega session-level hit@K (session_id-tagged "
                                "ingest); comparable to daemon drawer_hit_at_K",
            "embedding_model": "bge-small-en-v1.5 ONNX (semantic; not FTS5 "
                               "hash-fallback)",
            "isolation": "per-question OMEGA_HOME tempdir (local SQLite); prod "
                         "familiar/palace-daemon untouched",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "summary": summary,
        "per_question": records,
    }


def _omega_version() -> str:
    try:
        import omega
        return getattr(omega, "__version__", "?")
    except Exception:  # noqa: BLE001
        return "?"


def _print_summary(report: dict) -> None:
    summary = report["summary"]
    meta = report["run_metadata"]
    rsl = summary.get("retrieval_session_level", {})
    dual = summary.get("dual_metric", {})
    print()
    print("=" * 78)
    print(f" LongMemEval  adapter=omega  n={summary['total_questions']}  "
          f"reader={meta.get('answer_model')}  judge={meta.get('judge_model')}")
    print("=" * 78)
    print(f"\n{'category':22s} {'n':>4} {'R@5':>8} {'QA-acc':>8} {'gap':>8}")
    dual_cats = dual.get("per_category", {})
    rsl_cats = rsl.get("per_category", {})
    for cat in sorted(set(dual_cats) | set(rsl_cats)):
        slot = dual_cats.get(cat, {})
        r5 = (rsl_cats.get(cat, {}) or {}).get("r_at_5")
        qa = slot.get("qa_accuracy")
        gap = slot.get("retrieval_qa_gap")
        r5_str = f"{r5:>7.2%}" if r5 is not None else "    n/a"
        qa_str = f"{qa:>7.2%}" if qa is not None else "    n/a"
        gap_str = f"{gap:+.3f}" if gap is not None else "  n/a"
        n = slot.get("n") or (rsl_cats.get(cat, {}) or {}).get("n", 0)
        print(f"{cat:22s} {n:>4} {r5_str} {qa_str} {gap_str:>8}")
    overall = dual.get("overall", {})
    o_r5 = (rsl.get("overall", {}) or {}).get("r_at_5")
    o_qa = overall.get("qa_accuracy")
    o_gap = overall.get("retrieval_qa_gap")
    print(
        f"\n{'overall':22s} {summary['total_questions']:>4} "
        f"{(f'{o_r5:>7.2%}' if o_r5 is not None else '    n/a')} "
        f"{(f'{o_qa:>7.2%}' if o_qa is not None else '    n/a')} "
        f"{(f'{o_gap:+.3f}' if o_gap is not None else '  n/a'):>8}"
    )
    r1 = (rsl.get("overall", {}) or {}).get("r_at_1")
    print(f"\n  R@1={r1}  R@5={o_r5}  "
          f"R@10={(rsl.get('overall', {}) or {}).get('r_at_10')}")


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
        out_path = Path(f"longmemeval_omega_{ts}_{suffix}.json")
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
