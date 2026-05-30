#!/usr/bin/env python3
"""Multi-encoder RRF validation gate — significance analysis (#106 / mempalace#82).

mempalace#82's RFC says any new candidate strategy lands behind a benchmark
gate: **n>=200 git-derived probes showing measurable lift vs single-encoder
hybrid before flipping the default.** The point estimate already exists in
``baselines/rrf_multi_encoder_age_2026-05-29.json``: 3-way RRF (default ONNX +
ft-code-1000 + ft-code-5000) vs the *current* age-fused hybrid on n=200
git-derived probes — +0.0299 MRR, +2pp R@5, +3pp R@10. This script asks the
question the point estimate can't answer alone: **is the lift statistically
distinguishable from noise, or is it 22-improved/8-worsened coin-flips?**

It runs three paired tests on the per-probe records (no network):

  1. Paired bootstrap CI on the MRR delta (does the 95% CI exclude 0?).
  2. McNemar's exact test on R@5 hit/miss flips (the discordant-pairs test
     for paired binary outcomes — the right test for "did fusion flip more
     probes into the top-5 than out of it?").
  3. Sign test on the non-zero per-probe RR diffs (distribution-free check
     that improvements outnumber regressions beyond chance).

Posture (CLAUDE.md): diagnostic delta under a controlled condition. The verdict
is whether the *measured* lift clears a significance bar on *this* probe set —
not a universal claim. The natural-query leg and the 2-vs-3 encoder
decomposition are NOT in the committed artifact (they need a fresh
multi-encoder retrieval run over the local FT-Code checkpoints); this script
reports what the committed git-derived data supports and flags the rest as
open, per the no-fake-results rule.

Usage:
    venv/bin/python scripts/rrf_gate_significance.py \\
        --artifact baselines/rrf_multi_encoder_age_2026-05-29.json \\
        --out baselines/rrf_gate_significance_2026-05-29.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def paired_probes(artifact: dict) -> tuple[list[dict], list[dict]]:
    """Extract aligned (baseline, multi) per-probe records.

    Aligns **positionally** (both lists are written in the same probe order
    by the generator), not by a query->record dict: the probe set contains
    duplicate query strings (e.g. "Reconfigure stdio to UTF-8 on Windows"
    appears twice), and a dict keyed on query silently collapses duplicates,
    mispairing one of them. A query-equality assert guards against the two
    lists drifting out of order.
    """
    b = artifact["baseline"]["per_probe"]
    m = artifact["multi_encoder_rrf"]["per_probe"]
    if len(b) != len(m):
        raise ValueError(f"per_probe length mismatch: {len(b)} vs {len(m)}")
    for i, (br, mr) in enumerate(zip(b, m)):
        if br["query"] != mr["query"]:
            raise ValueError(
                f"probe order mismatch at index {i}: "
                f"{br['query']!r} vs {mr['query']!r}"
            )
    return list(b), list(m)


def paired_rr(artifact: dict) -> tuple[list[float], list[float]]:
    """Extract aligned (baseline_rr, multi_rr) per-probe reciprocal ranks."""
    base, multi = paired_probes(artifact)
    return ([float(r["rr"]) for r in base], [float(r["rr"]) for r in multi])


def bootstrap_mrr_delta_ci(
    base_rr: list[float],
    multi_rr: list[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for the paired MRR delta (multi - base)."""
    rng = random.Random(seed)
    n = len(base_rr)
    diffs = [m - b for b, m in zip(base_rr, multi_rr)]
    point = sum(diffs) / n
    boot = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boot.append(s / n)
    boot.sort()
    lo = boot[int((alpha / 2) * n_boot)]
    hi = boot[int((1 - alpha / 2) * n_boot)]
    return {
        "delta_mrr_point": point,
        "ci_95": [lo, hi],
        "excludes_zero": (lo > 0) or (hi < 0),
        "n_boot": n_boot,
    }


def mcnemar_exact(base_hit: list[int], multi_hit: list[int]) -> dict:
    """Exact McNemar test on paired binary hit/miss (here: R@k hit).

    b = baseline hit & multi miss (fusion lost a hit).
    c = baseline miss & multi hit (fusion gained a hit).
    Exact two-sided p from the binomial with p=0.5 over the b+c discordant
    pairs. The right test for paired binary outcomes on the same probes.
    """
    b = sum(1 for x, y in zip(base_hit, multi_hit) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(base_hit, multi_hit) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return {"b_lost": b, "c_gained": c, "n_discordant": 0, "p_value": 1.0}
    k = min(b, c)
    # two-sided exact binomial p
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return {"b_lost": b, "c_gained": c, "n_discordant": n, "p_value": p}


def sign_test(diffs: list[float]) -> dict:
    """Two-sided sign test on non-zero paired diffs."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"n_pos": 0, "n_neg": 0, "n_nonzero": 0, "p_value": 1.0}
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return {"n_pos": pos, "n_neg": neg, "n_nonzero": n, "p_value": p}


def hit_at_k(probe: dict, k: int) -> int:
    """Hit@k from the explicit ``rank`` field (None == miss).

    The artifact stores the gold rank directly; reconstructing it from the
    rounded ``rr`` float misclassifies tie-affected ranks, so read ``rank``.
    """
    rank = probe.get("rank")
    return int(rank is not None and rank <= k)


def analyze(artifact: dict, *, n_boot: int = 10000, seed: int = 42) -> dict:
    base, multi = paired_probes(artifact)
    base_rr = [float(r["rr"]) for r in base]
    multi_rr = [float(r["rr"]) for r in multi]
    n = len(base_rr)
    diffs = [m - b for b, m in zip(base_rr, multi_rr)]

    boot = bootstrap_mrr_delta_ci(base_rr, multi_rr, n_boot=n_boot, seed=seed)
    sign = sign_test(diffs)

    # R@5 and R@10 McNemar from the explicit per-probe rank field.
    base_h5 = [hit_at_k(r, 5) for r in base]
    multi_h5 = [hit_at_k(r, 5) for r in multi]
    base_h10 = [hit_at_k(r, 10) for r in base]
    multi_h10 = [hit_at_k(r, 10) for r in multi]
    mcnemar5 = mcnemar_exact(base_h5, multi_h5)
    mcnemar10 = mcnemar_exact(base_h10, multi_h10)

    base_mrr = sum(base_rr) / n
    multi_mrr = sum(multi_rr) / n

    return {
        "n_probes": n,
        "leg": "git-derived (commit-subject-shaped probes)",
        "baseline": "current age-fused single-encoder hybrid",
        "treatment": f"RRF over encoders {artifact.get('encoders')}",
        "mrr": {"baseline": base_mrr, "multi": multi_mrr,
                "delta": multi_mrr - base_mrr},
        "recall_at_5_pct": {
            "baseline": 100 * sum(base_h5) / n,
            "multi": 100 * sum(multi_h5) / n,
            "delta_pp": 100 * (sum(multi_h5) - sum(base_h5)) / n,
        },
        "recall_at_10_pct": {
            "baseline": 100 * sum(base_h10) / n,
            "multi": 100 * sum(multi_h10) / n,
            "delta_pp": 100 * (sum(multi_h10) - sum(base_h10)) / n,
        },
        "bootstrap_mrr_delta": boot,
        "mcnemar_r_at_5": mcnemar5,
        "mcnemar_r_at_10": mcnemar10,
        "sign_test_rr": sign,
        "n_improved": sum(1 for d in diffs if d > 0),
        "n_worsened": sum(1 for d in diffs if d < 0),
        "n_unchanged": sum(1 for d in diffs if d == 0),
    }


def verdict(result: dict, *, p_thresh: float = 0.05) -> dict:
    """Render the gate verdict + what is still open (natural-query, N choice)."""
    boot_sig = result["bootstrap_mrr_delta"]["excludes_zero"]
    mc5 = result["mcnemar_r_at_5"]["p_value"]
    sign_p = result["sign_test_rr"]["p_value"]
    positive = result["mrr"]["delta"] > 0
    clears = bool(positive and (boot_sig or mc5 < p_thresh or sign_p < p_thresh))
    return {
        "clears_gate_on_git_derived": clears,
        "rationale": (
            f"delta MRR {result['mrr']['delta']:+.4f}; "
            f"bootstrap 95% CI {result['bootstrap_mrr_delta']['ci_95']} "
            f"({'excludes' if boot_sig else 'includes'} 0); "
            f"McNemar R@5 p={mc5:.4f}; sign-test p={sign_p:.4f}"
        ),
        "open_questions": [
            "Natural-language probe leg NOT in committed artifact — needs a "
            "fresh multi-encoder retrieval run over user-style queries to test "
            "whether the git-derived lift translates off commit-subject probes.",
            "2-vs-3 encoder decomposition NOT in committed artifact — the solo "
            "and 2-encoder MRRs need a fresh run over the local FT-Code "
            "checkpoints (/home/jp/Downloads/ft1000/model, "
            "/home/jp/Projects/adaptmem-cache/model).",
        ],
        "recommendation": (
            "Hand the git-derived significance verdict to mempalace#82. The "
            "lift is small (+0.03 MRR) and the current age-fused hybrid already "
            "absorbs most of the 2026-05-15 +0.0841 measured against the old "
            "single-encoder hybrid. Recommend NOT flipping the default on this "
            "evidence alone: the productization cost (multi-encoder ingest + "
            "schema) is high and the natural-query leg — the leg that matters "
            "for real usage — is unmeasured."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--artifact", type=Path, required=True,
                   help="RRF baseline JSON with baseline + multi_encoder_rrf per_probe.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    artifact = json.loads(args.artifact.read_text())
    result = analyze(artifact, n_boot=args.n_boot, seed=args.seed)
    report = {
        "experiment": "multi-encoder RRF validation gate — significance (#106 / mempalace#82)",
        "posture": "paired significance tests on committed git-derived per-probe data; no network",
        "source_artifact": str(args.artifact),
        **result,
        "verdict": verdict(result),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    v = report["verdict"]
    print(f"n={result['n_probes']}  ΔMRR={result['mrr']['delta']:+.4f}  "
          f"R@5 Δ={result['recall_at_5_pct']['delta_pp']:+.1f}pp  "
          f"R@10 Δ={result['recall_at_10_pct']['delta_pp']:+.1f}pp")
    print(f"bootstrap 95% CI: {result['bootstrap_mrr_delta']['ci_95']} "
          f"(excludes 0: {result['bootstrap_mrr_delta']['excludes_zero']})")
    print(f"McNemar R@5 p={result['mcnemar_r_at_5']['p_value']:.4f} "
          f"(gained={result['mcnemar_r_at_5']['c_gained']} "
          f"lost={result['mcnemar_r_at_5']['b_lost']})")
    print(f"sign-test p={result['sign_test_rr']['p_value']:.4f} "
          f"(+{result['sign_test_rr']['n_pos']}/-{result['sign_test_rr']['n_neg']})")
    print(f"clears gate on git-derived: {v['clears_gate_on_git_derived']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
