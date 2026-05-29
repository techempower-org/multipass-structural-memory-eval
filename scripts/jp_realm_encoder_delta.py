#!/usr/bin/env python3
"""Compute the A→B encoder-swap delta on jp-realm-v0.1 (#84).

Takes two jp_realm_encoder_swap.py output JSONs (leg A = baseline encoder,
leg B = FT-300) and reports the per-question + aggregate Recall@{1,5,10}
deltas, plus the verdict against the Tau2 +30-33pp prediction.

Usage:
    jp_realm_encoder_delta.py \
        --a baselines/jp_realm_encoder_swap_default_<date>.json \
        --b baselines/jp_realm_encoder_swap_ft300_<date>.json \
        --json baselines/jp_realm_encoder_delta_<date>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREDICTION_PP = (30, 33)  # Tau2-predicted jp-realm recall gap, percentage points


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path, help="leg A (baseline) JSON")
    ap.add_argument("--b", required=True, type=Path, help="leg B (FT-300) JSON")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    a = json.loads(args.a.read_text())
    b = json.loads(args.b.read_text())
    sa, sb = a["summary"], b["summary"]

    ks = [k for k in (1, 5, 10) if f"R@{k}" in sa["overall"]]

    def _delta(blk_a: dict, blk_b: dict) -> dict:
        return {f"R@{k}": round(blk_b[f"R@{k}"] - blk_a[f"R@{k}"], 4) for k in ks}

    overall_delta = _delta(sa["overall"], sb["overall"])
    covered_delta = _delta(sa["covered_only"], sb["covered_only"])

    # per-question R@10 delta (the headline K)
    pa = {q["id"]: q for q in a["per_question"]}
    pq_delta = []
    for q in b["per_question"]:
        qa = pa.get(q["id"])
        if not qa:
            continue
        d10 = q["recall_at_k"].get("10", 0.0) - qa["recall_at_k"].get("10", 0.0)
        pq_delta.append({
            "id": q["id"],
            "snapshot_uncovered": q.get("snapshot_uncovered", False),
            "R@10_a": qa["recall_at_k"].get("10", 0.0),
            "R@10_b": q["recall_at_k"].get("10", 0.0),
            "delta_R@10": round(d10, 4),
        })

    # Verdict: does any covered-only RecallK delta reach the predicted band?
    best_pp = max(covered_delta.values()) * 100 if covered_delta else 0.0
    lo, hi = PREDICTION_PP
    if best_pp >= lo:
        verdict = f"VALIDATES Tau2 (+{best_pp:.1f}pp ≥ +{lo}pp predicted)"
    elif best_pp <= 0:
        verdict = (f"REFUTES Tau2: FT-300 did NOT help jp-realm "
                   f"(best Δ {best_pp:+.1f}pp vs +{lo}-{hi}pp predicted). "
                   f"LongMemEval/code-FT does not generalise to JP's KB.")
    else:
        verdict = (f"PARTIAL: best Δ {best_pp:+.1f}pp, below the +{lo}-{hi}pp "
                   f"predicted band — FT-300 helps less than Tau2 implied.")

    report = {
        "leg_a": {"model": sa["model"], "n_drawers": sa["n_drawers"]},
        "leg_b": {"model": sb["model"], "n_drawers": sb["n_drawers"]},
        "k_cutoffs": ks,
        "overall": {"a": {f"R@{k}": sa["overall"][f"R@{k}"] for k in ks},
                    "b": {f"R@{k}": sb["overall"][f"R@{k}"] for k in ks},
                    "delta_pp": {k: round(v * 100, 2) for k, v in overall_delta.items()}},
        "covered_only": {"a": {f"R@{k}": sa["covered_only"][f"R@{k}"] for k in ks},
                         "b": {f"R@{k}": sb["covered_only"][f"R@{k}"] for k in ks},
                         "delta_pp": {k: round(v * 100, 2) for k, v in covered_delta.items()},
                         "n": sa["covered_only"]["n"]},
        "snapshot_uncovered_ids": sa.get("snapshot_uncovered_ids", []),
        "prediction_pp": list(PREDICTION_PP),
        "verdict": verdict,
        "per_question_delta_R@10": sorted(pq_delta, key=lambda r: r["delta_R@10"]),
    }

    print("=" * 70)
    print(f"  A (baseline): {sa['model']}")
    print(f"  B (FT-300):   {sb['model']}")
    print(f"  drawers: {sa['n_drawers']}  (A) / {sb['n_drawers']} (B)")
    print()
    print(f"  {'cut':>6}  {'A':>8}  {'B':>8}  {'Δpp':>8}   (overall, n={sa['overall']['n']})")
    for k in ks:
        print(f"  {'R@'+str(k):>6}  {sa['overall'][f'R@{k}']:>8.4f}  "
              f"{sb['overall'][f'R@{k}']:>8.4f}  {overall_delta[f'R@{k}']*100:>+7.2f}")
    print()
    print(f"  {'cut':>6}  {'A':>8}  {'B':>8}  {'Δpp':>8}   "
          f"(covered-only, n={sa['covered_only']['n']})")
    for k in ks:
        print(f"  {'R@'+str(k):>6}  {sa['covered_only'][f'R@{k}']:>8.4f}  "
              f"{sb['covered_only'][f'R@{k}']:>8.4f}  {covered_delta[f'R@{k}']*100:>+7.2f}")
    print()
    print(f"  Tau2 predicted: +{PREDICTION_PP[0]}-{PREDICTION_PP[1]}pp on jp-realm")
    print(f"  VERDICT: {verdict}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
