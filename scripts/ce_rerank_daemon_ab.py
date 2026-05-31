#!/usr/bin/env python3
"""ce_rerank_daemon_ab.py — cross-encoder rerank A/B against the live daemon.

Issue #103 (routed from techempower-org/mempalace#301). Measures whether the
optional cross-encoder rerank stage improves retrieval quality vs the baseline
hybrid pipeline (vector + BM25 + AGE fusion) on the 200-probe git-derived set.

Why this script and not ``mempalace/scripts/eval_cross_encoder_rerank.py``
--------------------------------------------------------------------------
The shipped mempalace harness runs *in-process* against a local ChromaDB
palace (``mempalace.searcher.search_memories(query, palace_path)``). That path
is dead for this corpus: the local chroma palace was retired 2026-05-14 at the
pgvector cutover, and the only surviving local snapshot holds 2/200 of the
probe targets (1% coverage). Production now lives behind the daemon
(Postgres/pgvector + Apache AGE, ~385K drawers) — which is the corpus the
git-derived probes actually target.

The daemon exposes the rerank toggle as a *per-request* flag on
``POST /search/hybrid`` (``rerank: bool``), alongside ``fusion_mode`` and
``candidate_strategy``. So the A/B is pure read-only HTTP: same query, same
fusion, ``rerank:false`` vs ``rerank:true``. No daemon env flip, no writes,
no prod contamination.

Scoring contract is identical to ``mempalace/scripts/eval_fusion_ab.py`` so
the numbers are directly comparable to the #162 fusion A/B:
  * rank = 1-indexed position of the first hit whose ``source_file`` basename
    matches the probe's ``expected`` basename (None = miss).
  * MRR averages ``1/rank`` (0 for misses) over all probes.
  * Recall@k = fraction of probes whose expected doc ranked ``<= k``.

Probe-set format: ``probes_v2_git_derived.json`` is a dict
``{"_meta": {...}, "probes": [{"query", "expected", "why"}, ...]}`` — NOT the
``[query, expected, why]`` list-of-lists the in-process harness expects. This
script reads the dict shape directly.

Usage::

    python scripts/ce_rerank_daemon_ab.py \\
        --probes /home/jp/Projects/memorypalace/scripts/probes_v2_git_derived.json \\
        --api-url http://familiar:8085 \\
        --candidate-strategy hybrid \\
        --fusion-mode rrf \\
        --n-results 10 \\
        --i-know-the-corpus-is-stable \\
        --out baselines/ce_rerank_ab_2026-05-30.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

DEFAULT_TIMEOUT = 120  # rerank-on can be slow on a cold model load


# ── auth / url resolution (mirrors scripts/candidate_strategy_eval.py) ────────


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


# ── HTTP search ───────────────────────────────────────────────────────────────


def _hybrid_search(
    api_url: str,
    api_key: str,
    *,
    query: str,
    fusion_mode: str,
    candidate_strategy: str,
    limit: int,
    rerank: bool,
) -> tuple[list[dict], float]:
    """POST /search/hybrid. Returns (results, latency_ms)."""
    url = f"{api_url}/search/hybrid"
    payload = {
        "query": query,
        "limit": limit,
        "fusion_mode": fusion_mode,
        "candidate_strategy": candidate_strategy,
        "rerank": rerank,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    d = json.loads(raw)
    results = d.get("results") or d.get("hits") or []
    return results, elapsed_ms


def _source_file(hit: dict) -> str:
    """Extract source_file regardless of nesting (top-level or .metadata)."""
    meta = hit.get("metadata") or {}
    return hit.get("source_file") or meta.get("source_file") or ""


# ── scoring (identical contract to mempalace eval_fusion_ab.py) ───────────────


def rank_of_target(ranked_source_files: Sequence[str], target: str) -> Optional[int]:
    """1-indexed rank of the first hit whose basename matches ``target``."""
    target_name = Path(target).name
    for i, sf in enumerate(ranked_source_files, start=1):
        if Path((sf or "").strip()).name == target_name:
            return i
    return None


def evaluate_ranking(ranks: Sequence[Optional[int]]) -> dict[str, Any]:
    """MRR / Recall@5 / Recall@10 from per-probe ranks."""
    n = len(ranks)
    if n == 0:
        return {"n_probes": 0, "mrr": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "found": 0}
    rr_sum = 0.0
    r5 = r10 = found = 0
    for rank in ranks:
        if rank is None:
            continue
        found += 1
        rr_sum += 1.0 / rank
        if rank <= 5:
            r5 += 1
        if rank <= 10:
            r10 += 1
    return {
        "n_probes": n,
        "mrr": round(rr_sum / n, 4),
        "recall_at_5": round(r5 / n, 4),
        "recall_at_10": round(r10 / n, 4),
        "found": found,
    }


def _rank_sort_key(rank: Optional[int]) -> int:
    return rank if rank is not None else 1_000_000


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (nearest-rank-ish; small N tolerant)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _latency_summary(lat_ms: Sequence[float]) -> dict[str, float]:
    return {
        "n": len(lat_ms),
        "mean_ms": round(sum(lat_ms) / len(lat_ms), 2) if lat_ms else 0.0,
        "p50_ms": round(_percentile(lat_ms, 0.50), 2),
        "p95_ms": round(_percentile(lat_ms, 0.95), 2),
        "max_ms": round(max(lat_ms), 2) if lat_ms else 0.0,
    }


# ── probe loading ─────────────────────────────────────────────────────────────


def load_probes(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Dict shape: {"_meta": ..., "probes": [{query, expected, why}, ...]}
    if isinstance(raw, dict) and "probes" in raw:
        probes = raw["probes"]
    elif isinstance(raw, list):
        # Legacy list-of-lists [query, expected, why] — normalize to dicts.
        probes = [
            {"query": p[0], "expected": p[1], "why": (p[2] if len(p) > 2 else "")} for p in raw
        ]
    else:
        raise ValueError(f"probe file {path} has unrecognized shape")
    if not probes:
        raise ValueError(f"probe file {path} contains no probes")
    return probes


# ── A/B driver ─────────────────────────────────────────────────────────────────


def run_leg(
    api_url: str,
    api_key: str,
    probes: Sequence[dict],
    *,
    fusion_mode: str,
    candidate_strategy: str,
    n_results: int,
    rerank: bool,
    label: str,
) -> tuple[list[Optional[int]], list[float]]:
    """Run every probe under one rerank toggle. Returns (ranks, latencies_ms)."""
    ranks: list[Optional[int]] = []
    lat_ms: list[float] = []
    for i, p in enumerate(probes):
        query, expected = p["query"], p["expected"]
        try:
            results, elapsed = _hybrid_search(
                api_url,
                api_key,
                query=query,
                fusion_mode=fusion_mode,
                candidate_strategy=candidate_strategy,
                limit=n_results,
                rerank=rerank,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"[{label}] probe {i} FAILED: {e}", file=sys.stderr)
            ranks.append(None)
            continue
        source_files = [_source_file(h) for h in results]
        ranks.append(rank_of_target(source_files, expected))
        lat_ms.append(elapsed)
        if (i + 1) % 25 == 0:
            print(f"[{label}] {i + 1}/{len(probes)} done", file=sys.stderr)
    return ranks, lat_ms


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--candidate-strategy", default="hybrid")
    parser.add_argument("--fusion-mode", default="rrf")
    parser.add_argument("--n-results", type=int, default=10)
    parser.add_argument(
        "--top-n", type=int, default=25, help="rerank ceiling (daemon-side, for the record)."
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap probes (0 = all).")
    parser.add_argument("--out", default="")
    parser.add_argument("--writethrough-note", default="evening PDT, low writethrough")
    parser.add_argument(
        "--i-know-the-corpus-is-stable",
        action="store_true",
        help="Required ack: this hits the live daemon /search (read-only).",
    )
    args = parser.parse_args(argv)

    if not args.i_know_the_corpus_is_stable:
        print(
            "Refusing to run: this A/B hits the live daemon. Re-run with "
            "--i-know-the-corpus-is-stable once the corpus is steady.",
            file=sys.stderr,
        )
        return 2

    api_url = _resolve_url(args.api_url)
    api_key = _resolve_key(args.api_key)
    probes = load_probes(args.probes)
    if args.limit > 0:
        probes = probes[: args.limit]

    print(f"probes: {len(probes)}  target: {api_url}/search/hybrid", file=sys.stderr)

    # Leg A — baseline (rerank OFF).
    ranks_off, lat_off = run_leg(
        api_url,
        api_key,
        probes,
        fusion_mode=args.fusion_mode,
        candidate_strategy=args.candidate_strategy,
        n_results=args.n_results,
        rerank=False,
        label="rerank-off",
    )
    # Leg B — cross-encoder rerank ON (daemon's pinned ms-marco-MiniLM-L-6).
    ranks_on, lat_on = run_leg(
        api_url,
        api_key,
        probes,
        fusion_mode=args.fusion_mode,
        candidate_strategy=args.candidate_strategy,
        n_results=args.n_results,
        rerank=True,
        label="rerank-on",
    )

    metrics_off = evaluate_ranking(ranks_off)
    metrics_on = evaluate_ranking(ranks_on)

    queries = [p["query"] for p in probes]
    improved, regressed = [], []
    for q, ra, rb in zip(queries, ranks_off, ranks_on):
        ka, kb = _rank_sort_key(ra), _rank_sort_key(rb)
        if kb < ka:
            improved.append({"query": q, "rank_off": ra, "rank_on": rb})
        elif kb > ka:
            regressed.append({"query": q, "rank_off": ra, "rank_on": rb})

    lat_off_s = _latency_summary(lat_off)
    lat_on_s = _latency_summary(lat_on)
    overhead_ms = round(lat_on_s["mean_ms"] - lat_off_s["mean_ms"], 2)

    report = {
        "issue": "techempower-org/multipass-structural-memory-eval#103",
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "daemon /search/hybrid per-request rerank flag (read-only)",
        "api_url": api_url,
        "probe_set": os.path.basename(args.probes),
        "n_probes": len(probes),
        "fusion_mode": args.fusion_mode,
        "candidate_strategy": args.candidate_strategy,
        "n_results": args.n_results,
        "rerank_top_n": args.top_n,
        "rerank_model": "daemon-pinned cross-encoder/ms-marco-MiniLM-L-6-v2 (flashrank path)",
        "writethrough_note": args.writethrough_note,
        "label_a": "rerank-off",
        "label_b": "rerank-on",
        "metrics_a": metrics_off,
        "metrics_b": metrics_on,
        "delta": {
            "mrr": round(metrics_on["mrr"] - metrics_off["mrr"], 4),
            "recall_at_5": round(metrics_on["recall_at_5"] - metrics_off["recall_at_5"], 4),
            "recall_at_10": round(metrics_on["recall_at_10"] - metrics_off["recall_at_10"], 4),
        },
        "latency": {
            "rerank_off": lat_off_s,
            "rerank_on": lat_on_s,
            "overhead_mean_ms": overhead_ms,
        },
        "n_improved": len(improved),
        "n_regressed": len(regressed),
        "improved": improved,
        "regressed": regressed,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
