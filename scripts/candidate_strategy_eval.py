#!/usr/bin/env python3
"""Candidate-strategy A/B/C ablation for palace-daemon (closes #57).

Runs a labeled query set against palace-daemon's `mempalace_search` MCP
tool, varying `candidate_strategy` across {vector, union, hybrid}, and
reports per-strategy R@5/10, MRR, and latency p50/p95.

Why this exists:
  - palace-daemon now exposes three candidate_strategy options; after the
    canonical-predicate migration (techempower-org/palace-daemon#75)
    landed 2026-05-28, the hybrid graph leg can finally contribute.
  - JP's manual baseline (`baselines/candidate-strategy-2026-05-28.json`)
    showed +8.3pp R@5 (hybrid vs vector/union) at 15× latency cost.
    This script makes that diagnostic reproducible and SME-native.

Labeled query set format (matches palace-daemon's `rerank_eval_queries.json`):

    {
      "_about": "...",
      "queries": [
        {
          "id": "kill-cascade",
          "query": "kill-cascade incident systemd",
          "intent": "...",
          "relevant": {
            "source_glob": "optional glob",
            "content_any": ["substring1", "substring2"]
          }
        }
      ]
    }

A hit is relevant iff:
  - hit.source_file matches `source_glob` (if given), AND
  - at least one substring in `content_any` appears in hit.text

CLI:

    candidate_strategy_eval.py
        --queries PATH              # labeled JSON (palace-daemon rerank_eval_queries shape)
        --api-url URL               # daemon base URL (default: PALACE_DAEMON_URL env)
        --api-key KEY               # X-API-Key (default: PALACE_API_KEY env)
        --strategies vector union hybrid   # which to ablate (default: all 3)
        --limit N                   # per-query candidate pool (default: 20)
        --json PATH                 # report destination

Output JSON mirrors JP's manual baseline shape — drop-in replacement
for the ad-hoc script the original baseline was produced with.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional

log = logging.getLogger("candidate_strategy_eval")

DEFAULT_STRATEGIES = ("vector", "union", "hybrid")
DEFAULT_LIMIT = 20
DEFAULT_TIMEOUT = 60.0


def _resolve_url(arg: Optional[str]) -> str:
    if arg:
        return arg.rstrip("/")
    env = os.environ.get("PALACE_DAEMON_URL")
    if env:
        return env.rstrip("/")
    env_file = Path(os.path.expanduser("~/.config/palace-daemon/env"))
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("PALACE_DAEMON_URL="):
                return line.split("=", 1)[1].strip().strip('"').rstrip("/")
    raise SystemExit("--api-url required (or set PALACE_DAEMON_URL)")


def _resolve_key(arg: Optional[str]) -> str:
    if arg:
        return arg
    env = os.environ.get("PALACE_API_KEY")
    if env:
        return env
    env_file = Path(os.path.expanduser("~/.config/palace-daemon/env"))
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("PALACE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("--api-key required (or set PALACE_API_KEY)")


def _mcp_search(api_url: str, api_key: str, *, query: str, strategy: str,
                limit: int) -> tuple[dict, float]:
    """Call mempalace_search via MCP. Returns (parsed_result, latency_ms)."""
    url = f"{api_url}/mcp"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "mempalace_search",
            "arguments": {
                "query": query,
                "limit": limit,
                "candidate_strategy": strategy,
            },
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    d = json.loads(raw)
    if "error" in d:
        raise RuntimeError(f"MCP error: {d['error']}")
    text = d["result"]["content"][0]["text"]
    return json.loads(text), elapsed_ms


def _is_relevant(hit: dict, predicate: dict) -> bool:
    """A drawer is relevant iff source_glob matches AND any content_any matches.

    palace-daemon nests source_file under hit.metadata; older callers put it
    at the top level. Check both so the matcher works on either shape.
    """
    meta = hit.get("metadata") or {}
    source_glob = predicate.get("source_glob")
    if source_glob:
        sf = meta.get("source_file") or hit.get("source_file") or ""
        if not fnmatch.fnmatch(sf, source_glob):
            return False
    content_any = predicate.get("content_any") or []
    if not content_any:
        return True
    text = hit.get("text") or meta.get("text") or ""
    return any(sub in text for sub in content_any)


def _rank_of_first_relevant(results: list[dict], predicate: dict) -> Optional[int]:
    """Return 1-based rank of first relevant hit, or None if no relevant in list."""
    for i, hit in enumerate(results, start=1):
        if _is_relevant(hit, predicate):
            return i
    return None


def run_query(api_url: str, api_key: str, q: dict, strategy: str,
              limit: int) -> dict:
    """One labeled query × one strategy. Returns a per-q-per-strategy dict."""
    response, latency_ms = _mcp_search(
        api_url, api_key, query=q["query"], strategy=strategy, limit=limit,
    )
    results = response.get("results") or []
    rank = _rank_of_first_relevant(results, q["relevant"])
    return {
        "rank": rank if rank is not None else None,
        "r5": int(rank is not None and rank <= 5),
        "r10": int(rank is not None and rank <= 10),
        "rr": (1.0 / rank) if rank is not None else 0.0,
        "n_hits": len(results),
        "latency_ms": round(latency_ms, 1),
    }


def aggregate(per_query: dict[str, dict[str, dict]], strategies: list[str], n: int) -> dict:
    per_strategy: dict[str, dict] = {}
    for s in strategies:
        rows = [per_query[qid][s] for qid in per_query if s in per_query[qid]]
        if not rows:
            per_strategy[s] = {"n": 0, "n_ok": 0, "n_errors": 0}
            continue
        ok_rows = [r for r in rows if "error" not in r]
        n_errors = len(rows) - len(ok_rows)
        r5 = sum(r["r5"] for r in rows) / n
        r10 = sum(r["r10"] for r in rows) / n
        mrr = sum(r["rr"] for r in rows) / n
        if ok_rows:
            lat = sorted(r["latency_ms"] for r in ok_rows)
            p50 = median(lat)
            p95 = lat[int(len(lat) * 0.95)] if len(lat) >= 2 else lat[0]
            p50_out = round(p50, 1)
            p95_out = round(p95, 1)
        else:
            p50_out = None
            p95_out = None
        per_strategy[s] = {
            "n": n, "n_ok": len(ok_rows), "n_errors": n_errors,
            "R@5": round(r5, 4), "R@10": round(r10, 4),
            "MRR": round(mrr, 4),
            "p50_ms": p50_out, "p95_ms": p95_out,
        }
    headline = {}
    if per_strategy:
        best_r5_strategy = max(per_strategy, key=lambda s: per_strategy[s].get("R@5", 0))
        best_r5_val = per_strategy[best_r5_strategy].get("R@5", 0)
        headline["best_R@5"] = f"{best_r5_strategy} ({best_r5_val:.3f})"
    return {"per_strategy": per_strategy, "headline": headline}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, type=Path,
                   help="Labeled query JSON (palace-daemon rerank_eval_queries shape)")
    p.add_argument("--api-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES),
                   help="Strategies to ablate (default: vector union hybrid)")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help="Per-query candidate pool size")
    p.add_argument("--json", type=Path, default=None,
                   help="Output JSON path")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    api_url = _resolve_url(args.api_url)
    api_key = _resolve_key(args.api_key)

    payload = json.loads(args.queries.read_text())
    queries = payload.get("queries") or []
    if not queries:
        raise SystemExit(f"{args.queries}: no 'queries' array")

    per_query: dict[str, dict[str, dict]] = {}
    for i, q in enumerate(queries):
        log.info("[%d/%d] %s", i + 1, len(queries), q.get("id", "?"))
        per_query[q["id"]] = {}
        for s in args.strategies:
            try:
                per_query[q["id"]][s] = run_query(api_url, api_key, q, s, args.limit)
            except Exception as e:
                log.warning("query %s strategy %s failed: %s", q["id"], s, e)
                per_query[q["id"]][s] = {
                    "rank": None, "r5": 0, "r10": 0, "rr": 0.0,
                    "n_hits": 0, "latency_ms": 0.0, "error": str(e),
                }

    summary = aggregate(per_query, list(args.strategies), n=len(queries))
    report = {
        "run_metadata": {
            "mode": "candidate-strategy-ablation",
            "diagnostic": "candidate_strategy",
            "queries": str(args.queries),
            "n_questions": len(queries),
            "strategies": list(args.strategies),
            "search_limit": args.limit,
            "url": api_url,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "per_query": per_query,
    }

    print()
    print("=" * 60)
    print(f"  diagnostic: candidate_strategy  n: {len(queries)}  limit: {args.limit}")
    for s, blk in summary["per_strategy"].items():
        if blk.get("n", 0):
            print(f"  {s:10}  R@5={blk['R@5']:.3f}  R@10={blk['R@10']:.3f}  "
                  f"MRR={blk['MRR']:.3f}  p50={blk['p50_ms']:.0f}ms  p95={blk['p95_ms']:.0f}ms")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
