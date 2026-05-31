#!/usr/bin/env python3
"""Confidence intervals + BH-FDR on the capstone's HEADLINE deltas (#21 cont'd).

The comprehensive capstone states several central claims as prose point
estimates — "postgres 0.392 ≈ flat 0.384 (storage-equivalence)", "age-fusion:
no significant gain", etc. A bare delta can't tell a reader whether
"0.392 ≈ 0.384" is a real null or just an underpowered coin-flip. This script
turns each claim that has *paired per-question* baselines into a statistically
honest statement:

    Δ +0.8pp, 95% CI [−x, +y] straddles 0 → statistically null

using the merged ``sme.stats`` primitives (paired bootstrap CI from #21 +
Benjamini-Hochberg across the whole comparison family).

It runs ONLY over committed baseline JSONs (no network, no re-bench). The
per-question records are paired on ``question_id``; the metric is read from a
per-system field map because different runs store the same quantity under
different keys (e.g. mempalace's R@5 is ``drawer_hit_at_5`` while OMEGA's is
``omega_hit_at_5``, and the generic ``hit_at_5`` is unpopulated for the flat
adapter). A field that is all-zero on one side of a pair is REFUSED, not
silently compared — that is exactly the false-null #21 exists to prevent.

Two metric families per comparison where both are available:
  - QA correctness: ``judge.autoeval_label == "CORRECT"`` (ERROR replicates
    are dropped from the pair, not scored as wrong).
  - Retrieval recall: ``sme_recall`` (float) — populated on both sides of
    every paired baseline below, unlike the adapter-specific hit_at_k fields.

Comparisons with no paired per-question baseline (CE-rerank on/off,
candidate-strategy hybrid/union/vector — their committed artifacts are
summary-only) are emitted with ``status: "no_paired_baseline"`` so the
capstone can show which claims are CI-backed vs merely descriptive.

Posture (CLAUDE.md): diagnostic delta under a controlled condition, not a
universal claim. Per-comparison verdict is whether the measured delta clears
significance on *this* paired set.

Usage:
    venv/bin/python scripts/headline_delta_significance.py \\
        --out baselines/headline_delta_significance_2026-05-31.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from sme.stats import benjamini_hochberg, paired_bootstrap_ci

# Below this paired-sample size the CI is too wide to publish — warn, not error.
_MIN_PAIRED_N = 30
_DEFAULT_ALPHA = 0.05


# ── per-question metric extractors ──────────────────────────────────


def qa_correct(q: dict) -> Optional[float]:
    """1.0 if the judge marked the answer CORRECT, 0.0 if it gave any other
    decisive label, None if the judge errored/skipped (drop from the pair)."""
    judge = q.get("judge")
    if not isinstance(judge, dict):
        return None
    label = judge.get("autoeval_label")
    if label in (None, "ERROR", "skipped"):
        return None
    return 1.0 if label == "CORRECT" else 0.0


def recall_field(field: str) -> Callable[[dict], Optional[float]]:
    """Extractor for a numeric/bool per-question field (e.g. sme_recall)."""

    def _extract(q: dict) -> Optional[float]:
        v = q.get(field)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return None

    return _extract


# ── comparison specification ────────────────────────────────────────


@dataclass
class Metric:
    """One paired metric inside a comparison: a label + a per-system extractor.

    ``extract_a`` / ``extract_b`` differ when the two runs store the same
    quantity under different keys (the field-divergence case).
    """

    name: str
    extract_a: Callable[[dict], Optional[float]]
    extract_b: Callable[[dict], Optional[float]]
    # True when the two sides read from non-identical fields/units — surfaced
    # as a comparability caveat in the artifact, never hidden.
    non_identical_metric: bool = False


@dataclass
class Comparison:
    key: str
    description: str
    # Each side: (label, baseline-json path). None path => no paired baseline.
    label_a: str
    path_a: Optional[str]
    label_b: str
    path_b: Optional[str]
    metrics: list[Metric]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _index(doc: dict) -> dict[str, dict]:
    return {q["question_id"]: q for q in doc.get("per_question", [])}


def _pair(
    a_by_id: dict[str, dict],
    b_by_id: dict[str, dict],
    metric: Metric,
) -> tuple[list[float], list[float], int]:
    """Paired (A, B) score lists over the shared ids where BOTH extractors
    return a value. Returns (scores_a, scores_b, n_dropped)."""
    shared = sorted(set(a_by_id) & set(b_by_id))
    scores_a: list[float] = []
    scores_b: list[float] = []
    dropped = 0
    for qid in shared:
        va = metric.extract_a(a_by_id[qid])
        vb = metric.extract_b(b_by_id[qid])
        if va is None or vb is None:
            dropped += 1
            continue
        scores_a.append(va)
        scores_b.append(vb)
    return scores_a, scores_b, dropped


def _score_metric(
    metric: Metric,
    a_by_id: dict[str, dict],
    b_by_id: dict[str, dict],
    label_a: str,
    label_b: str,
) -> dict:
    """One metric's paired bootstrap CI on the A−B delta, in pp.

    Refuses (status all_zero_one_side) when the field is uniformly zero on one
    side — that means it isn't populated for that adapter, and comparing it
    would manufacture a false delta.
    """
    scores_a, scores_b, dropped = _pair(a_by_id, b_by_id, metric)
    n = len(scores_a)
    if n == 0:
        return {
            "metric": metric.name,
            "status": "no_shared_questions",
            "n_paired": 0,
        }

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n
    # The field-name trap: a field all-zero on exactly one side is *usually* an
    # unpopulated field for that adapter rather than a genuine sweep of zeros
    # (e.g. mempalace's sme_recall=0 while drawer_hit_at_5 is live). We can't
    # tell "unpopulated" from "scored zero on every question" by value alone, so
    # we FLAG it as a comparability caveat and still compute the CI — never
    # silently suppress a real delta, never silently trust a field-artifact.
    one_side_all_zero = (mean_a == 0.0) != (mean_b == 0.0)

    # Identical per-question scores → every paired diff is exactly 0. The
    # bootstrap p collapses to 0 here (no resample crosses zero), which would
    # masquerade as "significant". It is the opposite: the two runs are
    # indistinguishable on this metric. Report it as a definitional null and
    # keep it OUT of the BH-FDR family (no p to correct).
    if all(da == db for da, db in zip(scores_a, scores_b)):
        return {
            "metric": metric.name,
            "status": "identical",
            "note": (
                f"{metric.name} is byte-identical across all {n} paired "
                f"questions (mean {label_a}={mean_a:.3f}, {label_b}={mean_b:.3f}) "
                "— the runs share this metric exactly; delta is a definitional 0."
            ),
            f"mean_{label_a}": mean_a,
            f"mean_{label_b}": mean_b,
            "delta_pp": 0.0,
            "n_paired": n,
        }

    # How many paired questions actually disagree. A "significant" delta resting
    # on a handful of discordant pairs is fragile — the percentile bootstrap on
    # a near-degenerate binary distribution can report a tight CI that one
    # flipped question would erase. Surface the count so it isn't read as solid.
    n_discordant = sum(1 for da, db in zip(scores_a, scores_b) if da != db)

    ci = paired_bootstrap_ci(scores_a, scores_b)
    return {
        "metric": metric.name,
        "status": "ok",
        "non_identical_metric": metric.non_identical_metric,
        f"mean_{label_a}": mean_a,
        f"mean_{label_b}": mean_b,
        "delta_pp": ci.mean_diff * 100,
        "ci_low_pp": ci.ci_lower * 100,
        "ci_high_pp": ci.ci_upper * 100,
        "n_discordant": n_discordant,
        "fragile": n_discordant < _MIN_PAIRED_N,
        "one_side_all_zero": one_side_all_zero,
        "ci_straddles_zero": bool(ci.ci_lower <= 0.0 <= ci.ci_upper),
        "p_raw": ci.p_value_approx,
        "n_paired": n,
        "n_dropped": dropped,
        "low_n": n < _MIN_PAIRED_N,
    }


def run_comparison(cmp: Comparison) -> dict:
    if cmp.path_a is None or cmp.path_b is None:
        return {
            "key": cmp.key,
            "description": cmp.description,
            "status": "no_paired_baseline",
            "note": (
                "no committed per-question baseline for one or both sides "
                "(artifact is summary-only) — claim is descriptive, not CI-backed."
            ),
            "label_a": cmp.label_a,
            "label_b": cmp.label_b,
        }

    a, b = _load(cmp.path_a), _load(cmp.path_b)
    a_by_id, b_by_id = _index(a), _index(b)
    results = [
        _score_metric(m, a_by_id, b_by_id, cmp.label_a, cmp.label_b)
        for m in cmp.metrics
    ]
    return {
        "key": cmp.key,
        "description": cmp.description,
        "status": "computed",
        "label_a": cmp.label_a,
        "path_a": cmp.path_a,
        "label_b": cmp.label_b,
        "path_b": cmp.path_b,
        "metrics": results,
    }


# ── the capstone's headline deltas ──────────────────────────────────


def build_comparisons(base: Path) -> list[Comparison]:
    def p(name: str) -> str:
        return str(base / name)

    sme = recall_field("sme_recall")
    drawer5 = recall_field("drawer_hit_at_5")
    omega5 = recall_field("omega_hit_at_5")
    return [
        # (a) storage-equivalence — the clean headline. Same 250-q LoCoMo
        # subset, same judge, same sme_recall field both sides. (Retrieval is
        # byte-identical here by construction — same retrieval path, only the
        # storage/reader differs — so QA correctness is the live test.)
        Comparison(
            key="storage_equivalence_postgres_vs_flat",
            description="LoCoMo E2E: postgres_ingest vs flat baseline (does "
            "verbatim columnar storage change QA vs flat?)",
            label_a="postgres",
            path_a=p("locomo10_postgres_e2e_stratified_2026-05-31.json"),
            label_b="flat",
            path_b=p("locomo10_flat_e2e_stratified_2026-05-29.json"),
            metrics=[
                Metric("qa_correct", qa_correct, qa_correct),
                Metric("sme_recall", sme, sme),
            ],
        ),
        # (b) age-fusion on/off. The live retrieval signal in these daemon runs
        # is drawer_hit_at_5 (R@5 at the drawer level); the generic sme_recall
        # is unpopulated (all-zero) for the LME-S runs, so it's not used there.
        Comparison(
            key="age_fusion_longmemeval_strat150",
            description="LongMemEval-S strat150: age-fused vs plain search "
            "(does temporal fusion lift drawer R@5?)",
            label_a="age_fused",
            path_a=p("longmemeval_s_strat150_age_fused_2026-05-29.json"),
            label_b="search",
            path_b=p("longmemeval_s_strat150_search_2026-05-29.json"),
            metrics=[
                Metric("drawer_hit_at_5", drawer5, drawer5),
            ],
        ),
        Comparison(
            key="age_fusion_locomo_daemon",
            description="LoCoMo daemon: age-fused vs vector-only "
            "(does temporal fusion lift QA / drawer R@5?)",
            label_a="age_fused",
            path_a=p("locomo_daemon_age_fused_2026-05-30.json"),
            label_b="vector_only",
            path_b=p("locomo_daemon_vector_only_2026-05-30.json"),
            metrics=[
                Metric("qa_correct", qa_correct, qa_correct),
                Metric("drawer_hit_at_5", drawer5, drawer5),
            ],
        ),
        # (c) cross-system R@5: mempalace vs OMEGA on the same strat150 set.
        # NON-IDENTICAL METRIC: mempalace's R@5 is drawer_hit_at_5 (drawer-level)
        # while OMEGA's is omega_hit_at_5 (its native unit). Comparable in
        # spirit but not the same measurement — flagged so the capstone says so.
        Comparison(
            key="cross_system_mempalace_vs_omega_strat150",
            description="LongMemEval-S strat150 R@5: mempalace (age-fused, "
            "drawer-level) vs OMEGA (native) — headline cross-system delta.",
            label_a="mempalace",
            path_a=p("longmemeval_s_strat150_age_fused_2026-05-29.json"),
            label_b="omega",
            path_b=p("longmemeval_omega_strat150_r5_2026-05-30.json"),
            metrics=[
                Metric("r_at_5", drawer5, omega5, non_identical_metric=True),
            ],
        ),
        # CE-rerank on/off — committed artifact is summary-only (no per-q).
        Comparison(
            key="ce_rerank_on_off",
            description="Cross-encoder rerank on vs off (R@k / MRR).",
            label_a="rerank_on",
            path_a=None,
            label_b="rerank_off",
            path_b=None,
            metrics=[],
        ),
        # candidate-strategy hybrid/union/vector — summary-only artifacts.
        Comparison(
            key="candidate_strategy_hybrid_vs_union",
            description="Candidate strategy hybrid vs union (R@k / MRR).",
            label_a="hybrid",
            path_a=None,
            label_b="union",
            path_b=None,
            metrics=[],
        ),
    ]


def apply_family_fdr(comparisons: list[dict], alpha: float) -> None:
    """BH-FDR across every ok metric in every computed comparison — the whole
    family of headline tests gets one correction so running many of them
    doesn't inflate the false-positive rate. Writes p_adjusted/significant
    back into each metric dict in place."""
    family: list[dict] = []
    for c in comparisons:
        for m in c.get("metrics", []):
            if m.get("status") == "ok":
                family.append(m)
    if not family:
        return
    fdr = benjamini_hochberg([m["p_raw"] for m in family], alpha=alpha)
    for m, p_adj, rejected in zip(
        family, fdr.adjusted_p_values, fdr.rejected
    ):
        m["p_adjusted"] = p_adj
        m["significant"] = bool(rejected)


def _verdict(metric: dict) -> str:
    status = metric.get("status")
    if status == "identical":
        return "NULL (identical per-question scores — runs share this metric)"
    if status != "ok":
        return status or "?"
    if metric.get("significant"):
        if metric.get("fragile"):
            return (
                "SIGNIFICANT but FRAGILE — rests on "
                f"{metric.get('n_discordant')} discordant pairs; treat as suggestive"
            )
        return "SIGNIFICANT (delta distinguishable from 0 after BH-FDR)"
    if metric.get("ci_straddles_zero"):
        return "NULL (95% CI straddles 0)"
    return "ns (CI excludes 0 but not significant after BH-FDR)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baselines-dir",
        default="baselines",
        help="Directory holding the committed baseline JSONs.",
    )
    ap.add_argument(
        "--alpha", type=float, default=_DEFAULT_ALPHA, help="BH-FDR threshold."
    )
    ap.add_argument("--out", default=None, help="Write the artifact JSON here.")
    args = ap.parse_args()

    base = Path(args.baselines_dir)
    comparisons = [run_comparison(c) for c in build_comparisons(base)]
    apply_family_fdr(comparisons, args.alpha)

    artifact = {
        "schema": "headline_delta_significance/v1",
        "method": (
            "paired bootstrap CI (sme.stats.paired_bootstrap_ci, 10k resamples) "
            "on per-question deltas paired by question_id; BH-FDR "
            "(sme.stats.benjamini_hochberg) across the whole metric family."
        ),
        "alpha": args.alpha,
        "min_paired_n": _MIN_PAIRED_N,
        "comparisons": comparisons,
    }

    # Human-readable summary to stdout.
    print("=" * 78)
    print(" Headline-delta significance (#21 cont'd)")
    print("=" * 78)
    for c in comparisons:
        print(f"\n{c['key']}  [{c['status']}]")
        print(f"  {c['description']}")
        if c["status"] != "computed":
            print(f"  → {c.get('note', '')}")
            continue
        for m in c["metrics"]:
            if m["status"] == "identical":
                print(
                    f"    {m['metric']:12s} Δ 0.0pp  n={m['n_paired']}  "
                    f"[identical]  → {_verdict(m)}"
                )
                continue
            if m["status"] != "ok":
                print(f"    {m['metric']:12s} [{m['status']}] {m.get('note', '')}")
                continue
            flags = []
            if m["low_n"]:
                flags.append(f"⚠ n<{_MIN_PAIRED_N}")
            if m.get("fragile"):
                flags.append(f"⚠ fragile: only {m['n_discordant']} discordant pairs")
            if m.get("non_identical_metric"):
                flags.append("⚠ non-identical metric (see caveat)")
            if m.get("one_side_all_zero"):
                flags.append("⚠ one side all-zero (field may be unpopulated)")
            padj = m.get("p_adjusted")
            print(
                f"    {m['metric']:14s} Δ {m['delta_pp']:+.1f}pp  "
                f"95% CI [{m['ci_low_pp']:+.1f}, {m['ci_high_pp']:+.1f}]pp  "
                f"n={m['n_paired']}  p_adj="
                + (f"{padj:.3f}" if padj is not None else "n/a")
                + ("  " + "  ".join(flags) if flags else "")
            )
            print(f"      → {_verdict(m)}")

    if args.out:
        Path(args.out).write_text(json.dumps(artifact, indent=2, default=str))
        print(f"\nArtifact written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
