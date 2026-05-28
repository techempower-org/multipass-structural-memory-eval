"""Candidate-strategy A/B/C diagnostic (#57).

Drives palace-daemon's mempalace_search MCP across {vector, union, hybrid}
on a labeled query set and reports per-strategy R@5/10, MRR, latency
p50/p95, and a per-query strategy-flip diagnostic.

Used by both:
  - scripts/candidate_strategy_eval.py — standalone CLI tool (PR #65)
  - sme-eval candidate-strategy — built-in subcommand (this module)

Labeled query format matches palace-daemon's rerank_eval_queries.json:

    {
      "_about": "...",
      "queries": [
        {"id": "...", "query": "...", "intent": "...",
         "relevant": {"source_glob": "optional", "content_any": ["sub1"]}}
      ]
    }
"""
from __future__ import annotations

import fnmatch
import json
import time
import urllib.error
import urllib.request
from statistics import median
from typing import Optional

DEFAULT_STRATEGIES = ("vector", "union", "hybrid")
DEFAULT_LIMIT = 20
DEFAULT_TIMEOUT = 60.0


def mcp_search(
    api_url: str, api_key: str, *, query: str, strategy: str, limit: int,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[dict, float]:
    """Call mempalace_search via the daemon's /mcp endpoint.

    Returns ``(parsed_result, latency_ms)``. The parsed result is the
    daemon's standard search-response shape: {results: [...], ...}.
    """
    url = f"{api_url.rstrip('/')}/mcp"
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    d = json.loads(raw)
    if "error" in d:
        raise RuntimeError(f"MCP error: {d['error']}")
    text = d["result"]["content"][0]["text"]
    return json.loads(text), elapsed_ms


def is_relevant(hit: dict, predicate: dict) -> bool:
    """A drawer is relevant iff source_glob matches AND any content_any does."""
    source_glob = predicate.get("source_glob")
    if source_glob:
        sf = hit.get("source_file") or ""
        if not fnmatch.fnmatch(sf, source_glob):
            return False
    content_any = predicate.get("content_any") or []
    if not content_any:
        # Predicate has no content_any (and source_glob matched if given).
        return True
    text = hit.get("text") or ""
    return any(sub in text for sub in content_any)


def rank_of_first_relevant(results: list[dict], predicate: dict) -> Optional[int]:
    """1-based rank of first relevant hit, or None if no relevant hit."""
    for i, hit in enumerate(results, start=1):
        if is_relevant(hit, predicate):
            return i
    return None


def run_one(api_url: str, api_key: str, q: dict, strategy: str,
            limit: int, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """One labeled query × one strategy."""
    response, latency_ms = mcp_search(
        api_url, api_key, query=q["query"], strategy=strategy, limit=limit,
        timeout=timeout,
    )
    results = response.get("results") or []
    rank = rank_of_first_relevant(results, q["relevant"])
    return {
        "rank": rank if rank is not None else None,
        "r5": int(rank is not None and rank <= 5),
        "r10": int(rank is not None and rank <= 10),
        "rr": (1.0 / rank) if rank is not None else 0.0,
        "n_hits": len(results),
        "latency_ms": round(latency_ms, 1),
    }


def aggregate(per_query: dict[str, dict[str, dict]], strategies: list[str],
              n: int) -> dict:
    """Per-strategy + headline + strategy-flip diagnostic.

    The flip table identifies queries whose rank moved between strategies
    — per #57's "the most actionable signal, not just aggregates." For
    each pair of strategies (A,B), counts queries where:
      - moved up: B's rank < A's rank (better with B)
      - moved down: B's rank > A's rank (worse with B)
      - new hit: A had no relevant in top-K but B did
      - lost hit: A had a relevant but B didn't
    """
    per_strategy: dict[str, dict] = {}
    for s in strategies:
        rows = [per_query[qid][s] for qid in per_query if s in per_query[qid]]
        if not rows:
            per_strategy[s] = {"n": 0}
            continue
        r5 = sum(r["r5"] for r in rows) / n
        r10 = sum(r["r10"] for r in rows) / n
        mrr = sum(r["rr"] for r in rows) / n
        lat = sorted(r["latency_ms"] for r in rows)
        p50 = median(lat)
        p95 = lat[int(len(lat) * 0.95)] if len(lat) >= 2 else lat[0]
        per_strategy[s] = {
            "n": n, "R@5": round(r5, 4), "R@10": round(r10, 4),
            "MRR": round(mrr, 4),
            "p50_ms": round(p50, 1), "p95_ms": round(p95, 1),
        }

    flips: dict[str, dict] = {}
    for i, a in enumerate(strategies):
        for b in strategies[i + 1:]:
            ab_key = f"{a}_to_{b}"
            moved_up: list[dict] = []
            moved_down: list[dict] = []
            new_hits: list[str] = []
            lost_hits: list[str] = []
            same: list[str] = []
            for qid, rows in per_query.items():
                ra = rows.get(a, {}).get("rank")
                rb = rows.get(b, {}).get("rank")
                if ra is None and rb is None:
                    continue
                if ra is None and rb is not None:
                    new_hits.append(qid)
                elif ra is not None and rb is None:
                    lost_hits.append(qid)
                elif ra == rb:
                    same.append(qid)
                elif rb < ra:
                    moved_up.append({"qid": qid, "from_rank": ra, "to_rank": rb})
                else:
                    moved_down.append({"qid": qid, "from_rank": ra, "to_rank": rb})
            flips[ab_key] = {
                "moved_up": moved_up, "moved_down": moved_down,
                "new_hits": new_hits, "lost_hits": lost_hits,
                "unchanged_count": len(same),
            }

    headline = {}
    if per_strategy:
        best_r5 = max(per_strategy, key=lambda s: per_strategy[s].get("R@5", 0))
        headline["best_R@5"] = f"{best_r5} ({per_strategy[best_r5].get('R@5', 0):.3f})"

    return {
        "per_strategy": per_strategy,
        "headline": headline,
        "strategy_flips": flips,
    }


def run_eval(
    *,
    api_url: str,
    api_key: str,
    queries: list[dict],
    strategies: list[str] = list(DEFAULT_STRATEGIES),
    limit: int = DEFAULT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
    progress: Optional[callable] = None,
) -> dict:
    """End-to-end candidate-strategy A/B over a labeled query set.

    Returns the full report dict (run_metadata + summary + per_query) so
    callers (both the script and the CLI subcommand) can render or persist
    however they like.
    """
    per_query: dict[str, dict[str, dict]] = {}
    for i, q in enumerate(queries):
        if progress:
            progress(i + 1, len(queries), q.get("id", "?"))
        per_query[q["id"]] = {}
        for s in strategies:
            try:
                per_query[q["id"]][s] = run_one(api_url, api_key, q, s, limit, timeout)
            except Exception as e:  # noqa: BLE001 — log + continue
                per_query[q["id"]][s] = {
                    "rank": None, "r5": 0, "r10": 0, "rr": 0.0,
                    "n_hits": 0, "latency_ms": 0.0, "error": str(e),
                }
    summary = aggregate(per_query, list(strategies), n=len(queries))
    return {"summary": summary, "per_query": per_query}
