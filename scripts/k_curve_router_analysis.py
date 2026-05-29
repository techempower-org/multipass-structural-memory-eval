#!/usr/bin/env python3
"""K-curve sweep + confidence-router analysis (part of #85).

Ports AdaptMem's
``benchmarks/structural_memory_eval/k_curve_and_router_analysis.py``
(2026-05-28 cut) into SME idioms. The upstream script hardcoded macmini
paths and ``/tmp`` inputs and printed to stdout; this port takes its
inputs as CLI args, is unit-testable, and emits a structured JSON report.

It answers two diagnostic questions under controlled conditions:

  1. K-CURVE — for a per-strategy chunking-ablation result (probes with a
     gold ``rank``), how does R@K and MRR@K move across K in {1,3,5,10}?
     Useful for reading whether a strategy's advantage is a rank-1 effect
     or only shows up deeper in the list. Optionally compares two encoders
     (e.g. a baseline ONNX MiniLM vs a domain-tuned FT-300) on the same
     probe set, reporting B-A / C-A strategy deltas per encoder.

  2. ROUTER — given two per-question retrieval result files (e.g. the
     regex+IDF entity-graph baseline vs an encoder run), does a MAX router
     (best-of-two top-1) beat either single strategy, and does the
     entity-graph's own top-1 score predict when it wins? If the score
     separates wins from losses, it is a runtime routing signal — directly
     relevant to the hybrid-router findings in #44 / #45.

Diagnostic posture: every number here is a delta under a fixed corpus and
fixed candidate set, not a leaderboard score. A positive MAX-router lift is
an *upper bound* (oracle best-of-two), not an achievable online number.

Per-question input schema (auto-detected; either naming works):
  - ``regex_idf_baseline`` style: ``per_q`` list of
    {question_id|qid, question_type|qtype, top1, top1_score, hit_at_1|hit@1,
     gold}
  - ``encoder_swap`` style: ``per_question`` list of
    {question_id, question_type, retrieved_rank_1, expected_sources,
     hit_at_1}

Usage:
    # router only (two SME per-q result files):
    venv/bin/python scripts/k_curve_router_analysis.py \\
        --graph-results baselines/regex_idf_baseline_oracle_2026-05-28.json \\
        --encoder-results baselines/longmemeval_encoder_swap_default_2026-05-28.json \\
        --score-system graph \\
        --out baselines/k_curve_router_2026-05-29.json

    # k-curve only (one or two chunk-ablation probe-result files):
    venv/bin/python scripts/k_curve_router_analysis.py \\
        --kcurve-baseline path/to/chunk_strategy_ablation_baseline_result.json \\
        --kcurve-ft path/to/chunk_strategy_ablation_ft300_result.json \\
        --out baselines/k_curve_2026-05-29.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("k_curve_router")

K_VALUES = (1, 3, 5, 10)


# --------------------------------------------------------------------------
# K-curve over chunk-ablation probe results
# --------------------------------------------------------------------------
def r_at_k(probes: list[dict], k: int) -> float:
    """Recall@K (%) over probes carrying a gold ``rank`` (None == miss)."""
    if not probes:
        return 0.0
    n = sum(1 for p in probes if p.get("rank") is not None and p["rank"] <= k)
    return 100.0 * n / len(probes)


def mrr_at_k(probes: list[dict], k: int) -> float:
    """Mean reciprocal rank, truncated at K."""
    if not probes:
        return 0.0
    s = 0.0
    for p in probes:
        rank = p.get("rank")
        if rank is not None and rank <= k:
            s += 1.0 / rank
    return s / len(probes)


def k_curve_for_source(source: dict, strategies: list[str]) -> dict:
    """Per-strategy R@K / MRR@K across K for one encoder's ablation result.

    Returns {strategy: {"R@K": {k: pct}, "MRR@K": {k: val}}} plus the
    pairwise B-A / C-A deltas (relative to the first strategy) keyed by K.
    """
    out: dict[str, dict] = {}
    strat_probes = {s: source["strategies"][s]["probes"] for s in strategies}
    for s in strategies:
        out[s] = {
            "R@K": {str(k): round(r_at_k(strat_probes[s], k), 1) for k in K_VALUES},
            "MRR@K": {str(k): round(mrr_at_k(strat_probes[s], k), 4) for k in K_VALUES},
        }
    # deltas vs first strategy (the "A" reference)
    ref = strategies[0]
    deltas: dict[str, dict] = {}
    for s in strategies[1:]:
        deltas[f"{s}_minus_{ref}"] = {
            str(k): round(
                r_at_k(strat_probes[s], k) - r_at_k(strat_probes[ref], k), 1
            )
            for k in K_VALUES
        }
    out["_deltas_R@K"] = deltas
    return out


def build_k_curve(
    baseline: dict | None, ft: dict | None
) -> dict | None:
    """Assemble the K-curve block from one or two ablation result files."""
    sources = [(label, src) for label, src in
               (("baseline", baseline), ("FT-300", ft)) if src is not None]
    if not sources:
        return None
    # Strategy list is taken from whichever source is present first; both
    # ablation files share the same strategy keys by construction.
    strategies = list(sources[0][1]["strategies"].keys())
    block: dict = {"strategies": strategies, "by_encoder": {}}
    for label, src in sources:
        block["by_encoder"][label] = k_curve_for_source(src, strategies)
    return block


# --------------------------------------------------------------------------
# Per-question result loading (schema-flexible)
# --------------------------------------------------------------------------
def _norm_record(rec: dict) -> dict:
    """Normalise one per-question record to a common shape.

    Common shape: {qid, qtype, top1, top1_score, gold(set), hit1(int|None)}.
    ``hit1`` may be None when the source provides a top-1 + gold but no
    pre-computed hit flag (we recompute in that case). ``top1_score`` may be
    None when the source carries no confidence signal.
    """
    qid = rec.get("question_id") or rec.get("qid")
    qtype = rec.get("question_type") or rec.get("qtype") or "unknown"

    top1 = rec.get("top1")
    if top1 is None:
        top1 = rec.get("retrieved_rank_1")
    if top1 is None:
        ranked = rec.get("ranked_top10")
        if ranked:
            top1 = ranked[0]

    gold_raw = rec.get("gold")
    if gold_raw is None:
        gold_raw = rec.get("expected_sources")
    if gold_raw is None:
        gold_raw = rec.get("answer_session_ids")
    gold = set(gold_raw or [])

    hit1 = rec.get("hit_at_1")
    if hit1 is None:
        hit1 = rec.get("hit@1")
    if hit1 is not None:
        hit1 = int(hit1)
    elif top1 is not None and gold:
        hit1 = int(top1 in gold)

    score = rec.get("top1_score")

    return {
        "qid": qid,
        "qtype": qtype,
        "top1": top1,
        "top1_score": score,
        "gold": gold,
        "hit1": hit1,
    }


def load_per_q(path: Path) -> dict[str, dict]:
    """Load a per-question result file -> {qid: normalised record}.

    Accepts top-level ``per_q`` or ``per_question`` lists (or a bare list).
    """
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        records = raw
    else:
        records = raw.get("per_q") or raw.get("per_question") or []
    out: dict[str, dict] = {}
    for rec in records:
        norm = _norm_record(rec)
        if norm["qid"] is not None:
            out[norm["qid"]] = norm
    return out


# --------------------------------------------------------------------------
# Router cross-table (best-of-two MAX router)
# --------------------------------------------------------------------------
def _hit(rec: dict) -> bool:
    """Did this system's top-1 land on a gold session for this question?"""
    if rec["hit1"] is not None:
        return bool(rec["hit1"])
    return bool(rec["gold"]) and rec["top1"] in rec["gold"]


def router_cross_table(
    graph: dict[str, dict], encoder: dict[str, dict]
) -> dict:
    """Cross-tabulate per-question top-1 hits for two systems.

    Counts both_hit / only_graph / only_encoder / neither, and (within
    both_hit) same_pick vs different_pick — the latter being the only
    region where a *union* router could beat a *max* router on multi-gold
    questions. Reports MAX / graph-alone / encoder-alone R@1.
    """
    shared = [qid for qid in graph if qid in encoder]
    n = len(shared)
    both = only_g = only_e = neither = 0
    same_pick = diff_pick = 0
    for qid in shared:
        g_hit = _hit(graph[qid])
        e_hit = _hit(encoder[qid])
        if g_hit and e_hit:
            both += 1
            if graph[qid]["top1"] == encoder[qid]["top1"]:
                same_pick += 1
            else:
                diff_pick += 1
        elif g_hit:
            only_g += 1
        elif e_hit:
            only_e += 1
        else:
            neither += 1
    denom = n or 1
    return {
        "n_shared": n,
        "both_hit": both,
        "both_same_pick": same_pick,
        "both_different_pick": diff_pick,
        "only_graph": only_g,
        "only_encoder": only_e,
        "neither": neither,
        "R@1_max_router": round((both + only_g + only_e) / denom, 4),
        "R@1_graph_alone": round((both + only_g) / denom, 4),
        "R@1_encoder_alone": round((both + only_e) / denom, 4),
        "max_lift_over_encoder": round(only_g / denom, 4),
        "max_lift_over_graph": round(only_e / denom, 4),
    }


def confidence_routing_signal(
    score_system: dict[str, dict], other: dict[str, dict]
) -> dict:
    """Does ``score_system``'s top-1 score predict when it uniquely wins?

    Compares the score distribution for questions where the scoring system
    hit but the other missed (a "unique win") against questions where the
    scoring system missed. A higher win-median than lose-median means the
    score is a usable runtime router gate.
    """
    wins: list[float] = []
    loses: list[float] = []
    for qid, rec in score_system.items():
        if qid not in other:
            continue
        if rec["top1_score"] is None:
            continue
        s_hit = _hit(rec)
        o_hit = _hit(other[qid])
        if s_hit and not o_hit:
            wins.append(float(rec["top1_score"]))
        elif not s_hit:
            loses.append(float(rec["top1_score"]))
    return {
        "n_unique_wins": len(wins),
        "n_losses": len(loses),
        "win_median_score": round(statistics.median(wins), 4) if wins else None,
        "win_mean_score": round(sum(wins) / len(wins), 4) if wins else None,
        "lose_median_score": round(statistics.median(loses), 4) if loses else None,
        "lose_mean_score": round(sum(loses) / len(loses), 4) if loses else None,
        "separates": (
            bool(wins) and bool(loses)
            and statistics.median(wins) > statistics.median(loses)
        ),
    }


def per_category_router_gain(
    graph: dict[str, dict], encoder: dict[str, dict]
) -> dict:
    """Per-question-type MAX-router R@1 vs encoder-alone R@1 lift."""
    by_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "enc_hit": 0, "max_hit": 0, "graph_hit": 0}
    )
    for qid in graph:
        if qid not in encoder:
            continue
        cat = encoder[qid]["qtype"] or graph[qid]["qtype"]
        b = by_cat[cat]
        b["n"] += 1
        g_hit = _hit(graph[qid])
        e_hit = _hit(encoder[qid])
        if e_hit:
            b["enc_hit"] += 1
        if g_hit:
            b["graph_hit"] += 1
        if g_hit or e_hit:
            b["max_hit"] += 1
    out: dict[str, dict] = {}
    for cat, b in sorted(by_cat.items()):
        enc_r = b["enc_hit"] / b["n"]
        max_r = b["max_hit"] / b["n"]
        out[cat] = {
            "n": b["n"],
            "R@1_encoder_alone": round(enc_r, 4),
            "R@1_graph_alone": round(b["graph_hit"] / b["n"], 4),
            "R@1_max_router": round(max_r, 4),
            "lift": round(max_r - enc_r, 4),
        }
    return out


def build_router(
    graph: dict[str, dict],
    encoder: dict[str, dict],
    score_system: str,
) -> dict:
    """Assemble the full router block from two normalised per-q maps."""
    cross = router_cross_table(graph, encoder)
    if score_system == "graph":
        scorer, other = graph, encoder
    else:
        scorer, other = encoder, graph
    return {
        "cross_table": cross,
        "score_system": score_system,
        "confidence_signal": confidence_routing_signal(scorer, other),
        "by_category": per_category_router_gain(graph, encoder),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # router inputs
    p.add_argument("--graph-results", type=Path,
                   help="Per-question result JSON for the entity-graph/baseline system.")
    p.add_argument("--encoder-results", type=Path,
                   help="Per-question result JSON for the encoder system.")
    p.add_argument("--score-system", choices=("graph", "encoder"), default="graph",
                   help="Which system's top-1 score to test as a router gate "
                        "(default: graph).")
    # k-curve inputs
    p.add_argument("--kcurve-baseline", type=Path,
                   help="Chunk-ablation probe-result JSON (baseline encoder).")
    p.add_argument("--kcurve-ft", type=Path,
                   help="Chunk-ablation probe-result JSON (FT/swapped encoder).")
    p.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    report: dict = {
        "experiment": "K-curve sweep + confidence-router analysis",
        "posture": (
            "controlled-condition deltas; MAX-router R@1 is an oracle "
            "best-of-two upper bound, not an online-achievable score"
        ),
    }

    did_something = False

    if args.kcurve_baseline or args.kcurve_ft:
        baseline = (json.loads(args.kcurve_baseline.read_text())
                    if args.kcurve_baseline else None)
        ft = json.loads(args.kcurve_ft.read_text()) if args.kcurve_ft else None
        kc = build_k_curve(baseline, ft)
        if kc is not None:
            report["k_curve"] = kc
            did_something = True
            log.info("k-curve: %d strategies, encoders=%s",
                     len(kc["strategies"]), list(kc["by_encoder"].keys()))

    if args.graph_results and args.encoder_results:
        graph = load_per_q(args.graph_results)
        encoder = load_per_q(args.encoder_results)
        log.info("router: graph=%d q, encoder=%d q", len(graph), len(encoder))
        router = build_router(graph, encoder, args.score_system)
        report["router"] = router
        did_something = True
        ct = router["cross_table"]
        log.info(
            "router R@1: max=%.4f  graph=%.4f  encoder=%.4f  (n=%d)",
            ct["R@1_max_router"], ct["R@1_graph_alone"],
            ct["R@1_encoder_alone"], ct["n_shared"],
        )
        sig = router["confidence_signal"]
        log.info(
            "%s score gate: win_median=%s  lose_median=%s  separates=%s",
            args.score_system, sig["win_median_score"],
            sig["lose_median_score"], sig["separates"],
        )
    elif args.graph_results or args.encoder_results:
        log.warning("router needs BOTH --graph-results and --encoder-results; "
                    "skipping router block")

    if not did_something:
        p.error("provide router inputs (--graph-results + --encoder-results) "
                "and/or k-curve inputs (--kcurve-baseline / --kcurve-ft)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
