#!/usr/bin/env python3
"""Hybrid convex scorer-weight sweep for #111.

Sweeps mempalace's hybrid convex-fusion weights (vector_weight, bm25_weight)
over the 12-query golden eval and reports R@5/R@10/MRR per weight point, to find
an operating point that keeps R@5 ≈ 1.000 without regressing MRR vs union/vector.

Two run modes:

  --mode in-process   (recommended; run ON the daemon host)
      Imports mempalace.searcher.search_memories and runs it directly against
      the configured postgres palace, setting PALACE_HYBRID_VECTOR_WEIGHT /
      PALACE_HYBRID_BM25_WEIGHT per weight point IN THIS PROCESS ONLY. The
      live daemon is never restarted or reconfigured. Requires the mempalace
      env (MEMPALACE_BACKEND=postgres, MEMPALACE_POSTGRES_DSN, ...) and
      PALACE_PATH. FlashRank should be disabled (PALACE_RERANK_ENABLED=false)
      to stay apples-to-apples with the candidate_strategy_eval /mcp path.

This is the harness behind baselines/hybrid-scorer-weight-tuning-2026-05-31.json
and docs/benchmarks/2026-05-31-hybrid-scorer-weight-tuning.md. The weight knob
lives in mempalace (techempower-org/mempalace) searcher._hybrid_weights().

CLI:
    hybrid_weight_sweep.py
        --queries PATH       # labeled JSON (rerank_eval_queries shape)
        --palace PATH        # postgres palace path (or PALACE_PATH env)
        --limit N            # candidate pool (default 20)
        --grid "0.6,0.4 0.85,0.15 ..."   # vw,bw points (default: the #111 grid)
        --json PATH          # report destination
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_GRID = [
    (0.6, 0.4),  # mempalace default
    (0.7, 0.3),
    (0.8, 0.2),
    (0.85, 0.15),  # #111 recommended operating point
    (0.9, 0.1),
    (0.5, 0.5),
    (0.4, 0.6),
]


def _is_relevant(hit: dict, predicate: dict) -> bool:
    meta = hit.get("metadata") or {}
    source_glob = predicate.get("source_glob")
    if source_glob:
        sf = meta.get("source_file") or hit.get("source_file") or ""
        if not (
            fnmatch.fnmatch(sf, source_glob) or fnmatch.fnmatch(os.path.basename(sf), source_glob)
        ):
            return False
    content_any = predicate.get("content_any") or []
    if not content_any:
        return True
    text = hit.get("text") or meta.get("text") or ""
    return any(sub in text for sub in content_any)


def _rank_of_first_relevant(results: list, predicate: dict) -> Optional[int]:
    for i, hit in enumerate(results, start=1):
        if _is_relevant(hit, predicate):
            return i
    return None


def run_point(search_fn, queries, palace, vw, bw, limit):
    os.environ["PALACE_HYBRID_VECTOR_WEIGHT"] = str(vw)
    os.environ["PALACE_HYBRID_BM25_WEIGHT"] = str(bw)
    n = len(queries)
    r5 = r10 = mrr = 0.0
    per_query = {}
    for q in queries:
        res = search_fn(
            query=q["query"],
            palace_path=palace,
            n_results=limit,
            candidate_strategy="hybrid",
            fusion_mode="convex",
        )
        results = (res.get("results") if isinstance(res, dict) else res) or []
        rk = _rank_of_first_relevant(results, q["relevant"])
        per_query[q["id"]] = rk
        if rk is not None:
            r5 += int(rk <= 5)
            r10 += int(rk <= 10)
            mrr += 1.0 / rk
    return {
        "vector_weight": vw,
        "bm25_weight": bw,
        "R@5": round(r5 / n, 4),
        "R@10": round(r10 / n, 4),
        "MRR": round(mrr / n, 4),
        "per_query": per_query,
    }


def parse_grid(spec: Optional[str]):
    if not spec:
        return DEFAULT_GRID
    out = []
    for pair in spec.split():
        vw, bw = pair.split(",")
        out.append((float(vw), float(bw)))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, type=Path)
    p.add_argument("--palace", default=os.environ.get("PALACE_PATH"))
    p.add_argument("--limit", type=int, default=20)
    p.add_argument(
        "--grid", default=None, help='Space-separated "vw,bw" points (default: the #111 grid)'
    )
    p.add_argument("--mode", choices=["in-process"], default="in-process")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.palace:
        raise SystemExit("--palace required (or set PALACE_PATH)")

    from mempalace.searcher import search_memories

    queries = json.loads(args.queries.read_text())["queries"]
    grid = parse_grid(args.grid)

    print(f"{'vw':>5} {'bw':>5}  {'R@5':>6} {'R@10':>6} {'MRR':>6}")
    rows = []
    for vw, bw in grid:
        row = run_point(search_memories, queries, args.palace, vw, bw, args.limit)
        rows.append(row)
        print(f"{vw:>5} {bw:>5}  {row['R@5']:>6.3f} {row['R@10']:>6.3f} {row['MRR']:>6.3f}")
        sys.stdout.flush()

    report = {
        "run_metadata": {
            "diagnostic": "hybrid-scorer-weight-tuning",
            "issue": "techempower-org/multipass-structural-memory-eval#111",
            "queries": str(args.queries),
            "n_questions": len(queries),
            "candidate_strategy": "hybrid",
            "fusion_mode": "convex",
            "search_limit": args.limit,
            "palace": args.palace,
            "flashrank": os.environ.get("PALACE_RERANK_ENABLED", "unset"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "grid": rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
