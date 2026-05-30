#!/usr/bin/env python3
"""#117 — re-query the ALREADY-INGESTED strat150 lme_* wings at higher breadth.

The deployed-QA 0.610 was retrieval-limited at /search limit=5. The strat150
haystacks are already persisted in the prod familiar palace under
``lme_<question_id>`` wings (the original #116/#91 ingest never cleaned them up).
So we can re-query them at limit=20 / limit=50 WITHOUT re-ingesting anything —
pure read-only GET /search. This produces fresh pinned-context JSONs (same
schema as scripts/run_longmemeval_mempalace.py's --pin-context-out) that the
offline reader sweep then replays through the SAME reader + judge as 0.610.

Read-only: GET /search only. No POST /memory, no /flush, no writes to prod.

Usage:
    requery_deployed_breadth.py --limit 20 --out <pinned_limit20.json>
    requery_deployed_breadth.py --limit 50 --out <pinned_limit50.json>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

# Reuse the exact wing-scoped adapter from the deployed-QA path so the
# context_string assembly is byte-identical to the original run — the only
# variable we change is n_results (the retrieval limit).
from run_longmemeval_mempalace import _make_wing_scoped_daemon_adapter, LME_WING_PREFIX  # noqa: E402

# The strat150 subset definition (question_ids) lives in this pinned-context
# JSON, produced by the original #116 capture. We only read its question list
# (question_id / question / gold_answer / question_type) — the context_string
# inside it is the old limit=5 capture and is NOT used; we re-retrieve fresh.
_DEFAULT_SOURCE = Path(
    "/home/jp/.claude/projects/-home-jp-Projects-multipass-structural-memory-eval/"
    "scratch/sme-bench-2026-05-29/pinned_search-default_strat150.json"
)


def _api_key() -> str:
    import os
    if os.environ.get("PALACE_API_KEY"):
        return os.environ["PALACE_API_KEY"]
    env = Path.home() / ".config/palace-daemon/env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("PALACE_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no PALACE_API_KEY in env or ~/.config/palace-daemon/env")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "#117 read-only re-query of already-ingested lme_* wings at higher "
            "retrieval breadth. Reads the subset's question_ids from "
            "--source-pinned, re-retrieves each wing at --limit (GET /search "
            "only — no writes), and writes a fresh pinned-context JSON for the "
            "offline reader sweep to replay."
        )
    )
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-pinned", type=Path, default=_DEFAULT_SOURCE,
                   help="Pinned-context JSON defining the subset's question_ids "
                        "(default: the #116 strat150 capture). Only its question "
                        "list is read; its context_string is re-retrieved fresh.")
    p.add_argument("--from-oracle", type=Path, default=None, metavar="ORACLE_JSON",
                   help="Self-contained alternative to --source-pinned: derive the "
                        "strat150 subset directly from longmemeval_oracle.json via "
                        "the same stratified cap the deployed pipeline uses "
                        "(_stratified_cap, n=150, by question_type). Reproducible "
                        "from committed code + corpus — no scratch dependency.")
    p.add_argument("--api-url", default="http://familiar:8085")
    p.add_argument("--search-endpoint", default="/search")
    args = p.parse_args(argv)

    key = _api_key()
    if args.from_oracle is not None:
        # Derive the subset from the oracle corpus — identical question SET to
        # the #116 strat150 capture (the QA aggregator is order-independent).
        from run_longmemeval_mempalace import _stratified_cap
        from sme.corpora.longmemeval import load_questions
        qs = _stratified_cap(
            list(load_questions(str(args.from_oracle))), 150, "question_type"
        )
        questions = [
            {
                "question_id": q.question_id,
                "question": q.question,
                "gold_answer": q.answer,
                "question_type": q.question_type,
                "sme_category": q.sme_category,
                "is_abstention": q.is_abstention,
                "hit_at_5": None,
            }
            for q in qs
        ]
    else:
        src = json.loads(args.source_pinned.read_text())
        questions = src["pinned_context"]

    pinned = []
    n_empty = 0
    for i, q in enumerate(questions):
        qid = q["question_id"]
        wing = f"{LME_WING_PREFIX}{qid}"
        adapter = _make_wing_scoped_daemon_adapter(
            api_url=args.api_url, api_key=key, wing=wing, kind="all",
            search_endpoint=args.search_endpoint,
        )
        res = adapter.query(q["question"], n_results=args.limit, wing=wing)
        ctx = res.context_string or ""
        if not ctx:
            n_empty += 1
        pinned.append({
            "question_id": qid,
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "question_type": q["question_type"],
            "sme_category": q.get("sme_category"),
            "is_abstention": q.get("is_abstention", False),
            "context_string": ctx,
            "context_chars": len(ctx),
            # carry the original limit=5 hit flag forward only as reference;
            # the reader sweep doesn't depend on it for QA scoring.
            "hit_at_5": q.get("hit_at_5"),
            "n_hits": len(res.retrieved_entities or []),
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(questions)}] requeried at limit={args.limit}", flush=True)

    doc = {
        "run_metadata": {
            "diagnostic": "pinned_context",
            "issue": "techempower-org/multipass-structural-memory-eval#117",
            "adapter": "mempalace-daemon",
            "search_endpoint": args.search_endpoint,
            "retrieval_limit": args.limit,
            "subset": "stratified-25-per-type",
            "source_pinned": str(args.source_pinned),
            "note": (
                "Read-only re-query of already-ingested lme_* wings in prod "
                "familiar palace at higher breadth. No re-ingest, no writes."
            ),
            "n_questions": len(pinned),
            "n_empty_context": n_empty,
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "pinned_context": pinned,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, default=str))
    avg_chars = sum(p["context_chars"] for p in pinned) / max(1, len(pinned))
    avg_hits = sum(p["n_hits"] for p in pinned) / max(1, len(pinned))
    print(f"wrote {args.out}  n={len(pinned)}  empty={n_empty}  "
          f"avg_ctx_chars={avg_chars:.0f}  avg_hits={avg_hits:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
