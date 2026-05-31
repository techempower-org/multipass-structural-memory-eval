#!/usr/bin/env python3
"""Run a field HTTP-daemon adapter through the LongMemEval R@K pipeline.

The retrieval-cluster twin of ``scripts/run_longmemeval_omega.py``. Where the
OMEGA runner isolates per question with a temp ``OMEGA_HOME`` and ingests via a
Python library, this runner drives an SME ``SMEAdapter`` whose backend is a
**local HTTP daemon** (ai-memory, agentmemory, ...). Isolation is the adapter's
own per-call reset (ai-memory: ``forget`` the namespace then bulk-load;
agentmemory: rotate the project), so prior questions can't leak.

It uses the SAME session-tagged ingest + session-level hit@K topology as the
OMEGA runner: each haystack session becomes one corpus row tagged with its
``session_id``; the adapter surfaces ``session_id`` on every retrieved Entity;
hit@K is a set-membership test against the question's
``expected_sources_session_level()``. This makes the field row directly
comparable to the mempalace-daemon / OMEGA / postgres_ingest R@5 cells on the
identical strat150 subset.

R@K-only by default (``--skip-judge``) — these are retrieval-only systems whose
published headline is R@5; no reader/judge/cloud key is required.

    run_longmemeval_field.py
        --adapter ai-memory|agentmemory|...   # registry alias
        --api-url URL                         # daemon base URL
        --questions JSON                      # longmemeval_s_cleaned.json
        --max-questions N --stratify-by question_type   # strat150 subset
        --content-rules upstream-exact        # match the mempalace baseline
        --n-results K                         # top-K (default 5)
        --json PATH                           # report destination

LOCAL ONLY — talks to a localhost daemon; never touches prod familiar.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
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
from sme.cli import _load_adapter  # noqa: E402
from sme.corpora.longmemeval import LMEQuestion, load_questions  # noqa: E402

log = logging.getLogger("run_longmemeval_field")


def _render_session(s: Any, content_rules: str) -> str:
    """Render one haystack session to text — byte-identical to the OMEGA /
    mempalace ingest path so the field row is apples-to-apples."""
    if content_rules == "upstream-exact":
        return "\n".join(t.content for t in s.turns if t.role == "user")
    body_parts = [f"# Session {s.session_id}", f"_Date: {s.date}_", ""]
    for t in s.turns:
        marker = "  <!-- evidence -->" if t.has_answer else ""
        body_parts.append(f"## {t.role}{marker}\n\n{t.content}")
    return "\n".join(body_parts)


def _build_question_corpus(q: LMEQuestion, content_rules: str) -> list[dict]:
    """One corpus row per session, tagged with ``session_id`` so the adapter
    can surface it on each hit for session-level R@K."""
    corpus: list[dict] = []
    for s in q.haystack_sessions:
        text = _render_session(s, content_rules)
        if not text.strip():
            continue
        corpus.append(
            {"id": s.session_id, "document": text, "session_id": s.session_id}
        )
    return corpus


def _session_hits(result: QueryResult) -> list[Optional[str]]:
    out: list[Optional[str]] = []
    for e in result.retrieved_entities or []:
        props = e.properties or {}
        out.append(props.get("session_id"))
    return out


def run_one_question(
    q: LMEQuestion,
    *,
    adapter_factory,
    content_rules: str,
    n_results: int,
) -> dict:
    expected = q.expected_sources_session_level()
    expected_set = set(expected)

    adapter = adapter_factory()
    ingest_report: dict[str, Any] = {"entities_created": 0, "errors": []}
    try:
        corpus = _build_question_corpus(q, content_rules)
        ingest_report = adapter.ingest_corpus(corpus)
        try:
            result = adapter.query(q.question, n_results=n_results)
        except Exception as e:  # noqa: BLE001
            result = QueryResult(answer="", context_string="", error=str(e))
    finally:
        try:
            adapter.close()
        except Exception:  # noqa: BLE001
            pass

    session_ranks = _session_hits(result)
    rank_1 = session_ranks[0] if session_ranks else None
    hit_at_1 = bool(rank_1 is not None and rank_1 in expected_set)
    hit_at_5 = bool(any(sid in expected_set for sid in session_ranks[:5]))
    hit_at_10 = bool(any(sid in expected_set for sid in session_ranks[:10]))
    recalled = len(expected_set & {s for s in session_ranks if s})
    session_recall = recalled / len(expected_set) if expected_set else 0.0

    return {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "sme_category": q.sme_category,
        "is_abstention": q.is_abstention,
        "expected_sources": expected,
        "retrieved_session_ids": session_ranks,
        "field_hit_at_1": hit_at_1,
        "field_hit_at_5": hit_at_5,
        "field_hit_at_10": hit_at_10,
        "sme_recall": round(session_recall, 4),
        "context_chars": len(result.context_string or ""),
        "adapter_error": result.error,
        "ingest": {
            "sessions_stored": ingest_report.get("entities_created", 0),
            "errors": list(ingest_report.get("errors", []))[:5],
        },
    }


def _group_by(records: list[dict], field: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r.get(field, "unknown"), []).append(r)
    return dict(sorted(out.items()))


def _rate(rs: list[dict], key: str) -> Optional[float]:
    if not rs:
        return None
    return round(sum(1 for r in rs if r.get(key)) / len(rs), 4)


def _aggregate(records: list[dict]) -> dict:
    return {
        "total_questions": len(records),
        "retrieval_session_level": {
            "overall": {
                "n": len(records),
                "r_at_1": _rate(records, "field_hit_at_1"),
                "r_at_5": _rate(records, "field_hit_at_5"),
                "r_at_10": _rate(records, "field_hit_at_10"),
            },
            "per_category": {
                cat: {
                    "n": len(rs),
                    "r_at_1": _rate(rs, "field_hit_at_1"),
                    "r_at_5": _rate(rs, "field_hit_at_5"),
                    "r_at_10": _rate(rs, "field_hit_at_10"),
                }
                for cat, rs in _group_by(records, "sme_category").items()
            },
        },
        "errors": [r["adapter_error"] for r in records if r.get("adapter_error")][:20],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_longmemeval_field",
        description="LongMemEval session-level R@K for a field HTTP-daemon adapter.",
    )
    p.add_argument("--adapter", required=True,
                   help="Registry alias (ai-memory, agentmemory, ...).")
    p.add_argument("--api-url", default=None, help="Daemon base URL.")
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--stratify-by", default=None, metavar="FIELD",
                   help="Even round-robin cap (use question_type for strat150).")
    p.add_argument("--shuffle", type=int, default=None, metavar="SEED")
    p.add_argument("--content-rules", default="upstream-exact",
                   choices=["sme-rich", "upstream-exact"])
    p.add_argument("--n-results", type=int, default=5)
    p.add_argument("--api-timeout", type=float, default=None,
                   help="Per-request HTTP timeout (s). Low values let a hung "
                        "daemon call error-and-continue instead of blocking.")
    p.add_argument("--namespace", default=None, help="ai-memory namespace.")
    p.add_argument("--project", default=None, help="agentmemory base project.")
    p.add_argument("--json", type=Path, default=None)
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


def run(args: argparse.Namespace) -> dict:
    questions = _select_questions(args)

    def adapter_factory():
        kwargs: dict[str, Any] = {
            "api_url": args.api_url,
            "n_results": args.n_results,
            "namespace": args.namespace,
            "project": args.project,
            "api_timeout": args.api_timeout,
        }
        return _load_adapter(args.adapter, **kwargs)

    records: list[dict] = []
    for i, q in enumerate(questions):
        log.info("[%d/%d] %s (%s / %s)", i + 1, len(questions),
                 q.question_id, q.question_type, q.sme_category)
        records.append(
            run_one_question(
                q,
                adapter_factory=adapter_factory,
                content_rules=args.content_rules,
                n_results=args.n_results,
            )
        )

    summary = _aggregate(records)
    return {
        "run_metadata": {
            "mode": "live",
            "adapter": args.adapter,
            "api_url": args.api_url,
            "questions": str(args.questions),
            "n_questions": len(records),
            "stratify_by": args.stratify_by,
            "shuffle_seed": args.shuffle,
            "content_rules": args.content_rules,
            "n_results": args.n_results,
            "skip_judge": True,
            "retrieval_metric": "field session-level hit@K (session_id-tagged "
                                "ingest); comparable to daemon drawer_hit_at_K "
                                "and omega session-level hit@K",
            "isolation": "adapter per-question reset (ai-memory forget / "
                         "agentmemory project rotation); local daemon, prod "
                         "untouched",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "summary": summary,
        "per_question": records,
    }


def _print_summary(report: dict) -> None:
    summary = report["summary"]
    meta = report["run_metadata"]
    rsl = summary["retrieval_session_level"]
    print()
    print("=" * 72)
    print(f" LongMemEval  adapter={meta['adapter']}  n={summary['total_questions']}")
    print("=" * 72)
    print(f"\n{'category':22s} {'n':>4} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    for cat, slot in rsl["per_category"].items():
        def _f(v):
            return f"{v:>7.2%}" if v is not None else "    n/a"
        print(f"{cat:22s} {slot['n']:>4} {_f(slot['r_at_1'])} "
              f"{_f(slot['r_at_5'])} {_f(slot['r_at_10'])}")
    o = rsl["overall"]
    print(f"\n{'overall':22s} {o['n']:>4} "
          f"{o['r_at_1']:>7.2%} {o['r_at_5']:>7.2%} {o['r_at_10']:>7.2%}")
    if summary["errors"]:
        print(f"\n  {len(summary['errors'])} adapter error(s); first: "
              f"{summary['errors'][0]}")


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
        out_path = Path(f"longmemeval_{args.adapter}_{ts}_r5.json")
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
