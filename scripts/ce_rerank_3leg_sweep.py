#!/usr/bin/env python3
"""Cross-encoder rerank A/B — 3-leg MODEL sweep over a corpus-seeded daemon.

Issue techempower-org/multipass-structural-memory-eval#103. Companion to
``scripts/ce_rerank_daemon_ab.py`` (Nyx, #225): that runner does a single
read-only pass with two legs (rerank off vs the env-pinned model) and is the
right tool when you only need on/off on whatever model the daemon already pins.

This driver runs THREE legs that differ by rerank MODEL — the off / small-CE /
large-CE sweep #103 asks for:

  * Leg A — rerank OFF                         (baseline hybrid: vector+BM25+AGE)
  * Leg B — rerank ON, small cross-encoder     (default ms-marco-TinyBERT-L-2-v2)
  * Leg C — rerank ON, large cross-encoder     (default ms-marco-MiniLM-L-12-v2)

Why a driver and not flags on the shipped runner: palace-daemon pins the rerank
model at startup (``PALACE_RERANK_MODEL``, read at import in ``rerank.py`` and
cached in a module-global ``_ranker``). The per-request ``rerank`` flag only
toggles the pinned model on/off — it cannot switch models. So a model sweep
needs the daemon RESTARTED with a different ``PALACE_RERANK_MODEL`` per leg.

This script delegates the restart to a caller-supplied hook (``--restart-cmd``,
invoked as ``<cmd> <model> <enabled>``) so it carries no environment-specific
assumptions: point it at whatever brings your daemon up with a given rerank
model. It REUSES ``ce_rerank_daemon_ab``'s scoring verbatim
(``run_leg`` / ``evaluate_ranking`` / ``load_probes`` / ``_latency_summary``),
so the numbers are directly comparable to the #225 2-leg pass and the #162
fusion A/B.

Run-time isolation guard (the INVERSE of the ingest guard): refuses unless the
daemon URL is localhost AND the palace is POPULATED — a model sweep over an
empty palace would re-floor recall exactly like the #225 BLOCKED run.

Usage::

    python scripts/ce_rerank_3leg_sweep.py \\
        --probes /path/to/probes_v2_git_derived.json \\
        --api-url http://localhost:8086 --api-key "$KEY" \\
        --restart-cmd /path/to/launch_daemon.sh \\
        --small-model ms-marco-TinyBERT-L-2-v2 \\
        --large-model ms-marco-MiniLM-L-12-v2 \\
        --out baselines/ce_rerank_corpus_seeded_<date>.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ce_rerank_daemon_ab as ab  # noqa: E402  (scoring + leg machinery, single source of truth)


def isolation_guard(api_url: str, api_key: str) -> int:
    """Refuse unless localhost AND the palace is populated (seeded)."""
    if "localhost" not in api_url and "127.0.0.1" not in api_url:
        raise SystemExit(f"ISOLATION GUARD: {api_url} is not localhost — refusing.")
    req = urllib.request.Request(f"{api_url}/list?limit=1", headers={"X-API-Key": api_key})
    with urllib.request.urlopen(req, timeout=15) as r:
        n = int(json.loads(r.read().decode()).get("total") or 0)
    if n <= 0:
        raise SystemExit(
            f"GUARD: palace at {api_url} is EMPTY ({n}). A model sweep over an empty "
            "palace re-floors recall (the #225 BLOCKED failure mode). Seed it first."
        )
    return n


def restart_daemon(
    restart_cmd: str, model: str | None, enabled: bool, api_url: str, api_key: str
) -> None:
    subprocess.run(
        ["bash", restart_cmd, model or "", "true" if enabled else "false"],
        check=True,
        capture_output=True,
        text=True,
    )
    for _ in range(90):  # rerank-on cold-loads a flashrank model (large CE first time)
        try:
            req = urllib.request.Request(f"{api_url}/health", headers={"X-API-Key": api_key})
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise SystemExit(f"daemon not healthy after restart (model={model})")


def warm_rerank(api_url: str, api_key: str, fusion_mode: str, strategy: str) -> None:
    """Fire one throwaway rerank-on query so the CE model loads BEFORE the timed
    leg — keeps the cold-load out of the latency distribution."""
    try:
        ab._hybrid_search(
            api_url,
            api_key,
            query="palace daemon postgres pgvector migration runbook",
            fusion_mode=fusion_mode,
            candidate_strategy=strategy,
            limit=10,
            rerank=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  warm_rerank note: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--api-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument(
        "--restart-cmd",
        required=True,
        help="Hook invoked as `<cmd> <model> <enabled>` to relaunch the daemon per leg.",
    )
    ap.add_argument("--small-model", default="ms-marco-TinyBERT-L-2-v2")
    ap.add_argument("--large-model", default="ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--fusion-mode", default="rrf")
    ap.add_argument("--candidate-strategy", default="hybrid")
    ap.add_argument("--n-results", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="Cap probes (0 = all).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    probes = ab.load_probes(args.probes)
    if args.limit > 0:
        probes = probes[: args.limit]

    legs = [
        ("A_rerank_off", None, False, "(none — rerank off)"),
        ("B_rerank_small", args.small_model, True, args.small_model),
        ("C_rerank_large", args.large_model, True, args.large_model),
    ]

    results: dict = {}
    for leg_key, model, enabled, model_label in legs:
        print(f"\n=== LEG {leg_key}: model={model_label} enabled={enabled} ===", file=sys.stderr)
        restart_daemon(args.restart_cmd, model, enabled, args.api_url, args.api_key)
        n_drawers = isolation_guard(args.api_url, args.api_key)
        print(f"  guard OK — seeded palace ({n_drawers} drawers)", file=sys.stderr)
        if enabled:
            warm_rerank(args.api_url, args.api_key, args.fusion_mode, args.candidate_strategy)
        ranks, lat = ab.run_leg(
            args.api_url,
            args.api_key,
            probes,
            fusion_mode=args.fusion_mode,
            candidate_strategy=args.candidate_strategy,
            n_results=args.n_results,
            rerank=enabled,
            label=leg_key,
        )
        results[leg_key] = {
            "label": leg_key,
            "rerank_enabled": enabled,
            "rerank_model": model_label,
            "metrics": ab.evaluate_ranking(ranks),
            "latency": ab._latency_summary(lat),
        }
        m = results[leg_key]["metrics"]
        print(
            f"  {leg_key}: MRR={m['mrr']} R@5={m['recall_at_5']} R@10={m['recall_at_10']} "
            f"found={m['found']}/{m['n_probes']} p50={results[leg_key]['latency']['p50_ms']}ms",
            file=sys.stderr,
        )

    base = results["A_rerank_off"]["metrics"]
    base_lat = results["A_rerank_off"]["latency"]
    deltas = {}
    for k in ("B_rerank_small", "C_rerank_large"):
        mk, lk = results[k]["metrics"], results[k]["latency"]
        deltas[f"{k}_vs_A"] = {
            "mrr": round(mk["mrr"] - base["mrr"], 4),
            "recall_at_5": round(mk["recall_at_5"] - base["recall_at_5"], 4),
            "recall_at_10": round(mk["recall_at_10"] - base["recall_at_10"], 4),
            "latency_overhead_p50_ms": round(lk["p50_ms"] - base_lat["p50_ms"], 2),
        }

    report = {
        "issue": "techempower-org/multipass-structural-memory-eval#103",
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "corpus-seeded scratch daemon /search/hybrid, 3-leg model sweep (daemon restart per model)",
        "api_url": args.api_url,
        "probe_set": os.path.basename(args.probes),
        "n_probes": len(probes),
        "fusion_mode": args.fusion_mode,
        "candidate_strategy": args.candidate_strategy,
        "n_results": args.n_results,
        "legs": results,
        "deltas_vs_baseline": deltas,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)
    print(json.dumps({k: v["metrics"] for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
