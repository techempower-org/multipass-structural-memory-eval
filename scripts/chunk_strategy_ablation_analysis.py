#!/usr/bin/env python3
"""Chunk-strategy ablation analysis + encoder-swap hypothesis test (#85).

Ports the *analysis half* of AdaptMem's
``benchmarks/structural_memory_eval/chunk_strategy_ablation_ft300.py``
(2026-05-28 cut) into SME idioms.

The upstream script was a thin monkey-patch that injected an FT-300
SentenceTransformer embedding function into mempalace, re-ran a chunking
A/B/C ablation, and dumped a probe-result JSON. The *mining* half is
mempalace-internal and not portable here; the *diagnostic* half — reading
two such probe-result files and asking whether the encoder swap changes
the chunking-axis story — is what this port reconstructs.

The hypothesis under test (jpheinden's #1384): chunking-axis sensitivity is
downstream of encoder calibration. If it holds, the markdown-aware lift
(strategy B over strategy A) should *compress* when you swap a general
ONNX MiniLM for a domain-tuned FT-300 encoder. This script computes that
compression directly:

    compression(B-A) = (B-A on baseline encoder) - (B-A on FT-300 encoder)

A positive compression means the FT encoder narrowed the chunking-strategy
gap — evidence the gap was an encoder-calibration artifact rather than an
intrinsic chunking effect.

To actually *produce* the input probe-result JSONs, run mempalace's own
chunk_strategy_ablation under each encoder (out of band — see
``sme/adapters/adaptmem_adapter.py`` for driving the FT encoder under the
SME contract instead of monkey-patching mempalace). This script is the
read-only analysis layer over those artifacts.

Diagnostic posture: every figure is a controlled-condition delta on a
fixed probe set, not a benchmark score.

Probe-result schema (per encoder file):
  {"strategies": {STRAT_KEY: {"mrr": float, "recall_at_5_pct": float,
                              "probes": [{"expected": str, "rank": int|None,
                                          "rr": float}, ...]}}}

Usage:
    venv/bin/python scripts/chunk_strategy_ablation_analysis.py \\
        --baseline path/to/chunk_strategy_ablation_baseline_result.json \\
        --ft path/to/chunk_strategy_ablation_ft300_result.json \\
        --out baselines/chunk_ablation_encoder_swap_2026-05-29.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("chunk_ablation_analysis")


def mrr(probes: list[dict]) -> float:
    """Mean reciprocal rank over a probe list (rank None == 0 contribution)."""
    if not probes:
        return 0.0
    return sum(
        (1.0 / p["rank"]) if p.get("rank") else 0.0 for p in probes
    ) / len(probes)


def recall_at_k(probes: list[dict], k: int) -> float:
    """Recall@K (fraction in [0,1]) over probes carrying a gold ``rank``."""
    if not probes:
        return 0.0
    n = sum(1 for p in probes if p.get("rank") is not None and p["rank"] <= k)
    return n / len(probes)


def filter_probes(probes: list[dict], suffix: str | None) -> list[dict]:
    """Optionally restrict to probes whose ``expected`` ends with ``suffix``.

    The upstream script carved out a markdown-only sub-slice (``.md``
    expected files) because the B (heading-aware-markdown) strategy can
    only help on markdown targets. ``suffix=None`` returns all probes.
    """
    if not suffix:
        return probes
    return [p for p in probes if str(p.get("expected", "")).endswith(suffix)]


def strategy_metrics(source: dict, strat: str, suffix: str | None = None) -> dict:
    """MRR + R@5 + R@10 for one strategy in one encoder's result file."""
    probes = filter_probes(source["strategies"][strat]["probes"], suffix)
    return {
        "n_probes": len(probes),
        "mrr": round(mrr(probes), 4),
        "recall_at_5": round(recall_at_k(probes, 5), 4),
        "recall_at_10": round(recall_at_k(probes, 10), 4),
    }


def ablation_deltas(source: dict, strategies: list[str],
                    suffix: str | None = None) -> dict:
    """Per-strategy metrics + B-A / C-A MRR deltas for one encoder."""
    metrics = {s: strategy_metrics(source, s, suffix) for s in strategies}
    ref = strategies[0]
    deltas = {
        f"{s}_minus_{ref}_mrr": round(
            metrics[s]["mrr"] - metrics[ref]["mrr"], 4
        )
        for s in strategies[1:]
    }
    return {"metrics": metrics, "deltas": deltas}


def encoder_swap_compression(
    baseline: dict, ft: dict, strategies: list[str],
    suffix: str | None = None,
) -> dict:
    """Compression of each B-A / C-A MRR delta under the FT encoder swap.

    compression = baseline_delta - ft_delta. Positive => the FT encoder
    narrowed the chunking-strategy gap (supports the #1384 hypothesis that
    the gap is an encoder-calibration artifact).
    """
    base = ablation_deltas(baseline, strategies, suffix)["deltas"]
    ftd = ablation_deltas(ft, strategies, suffix)["deltas"]
    comp: dict[str, dict] = {}
    for key in base:
        comp[key] = {
            "baseline_delta": base[key],
            "ft_delta": ftd.get(key),
            "compression": (
                round(base[key] - ftd[key], 4) if key in ftd else None
            ),
        }
    return comp


def build_report(baseline: dict, ft: dict | None) -> dict:
    """Assemble the full ablation-analysis report from one or two files."""
    strategies = list(baseline["strategies"].keys())
    report: dict = {
        "experiment": "chunk-strategy ablation analysis + encoder-swap test",
        "hypothesis": (
            "#1384: chunking-axis sensitivity is downstream of encoder "
            "calibration; FT-300 swap should compress the B-A markdown lift"
        ),
        "posture": "controlled-condition deltas on a fixed probe set",
        "strategies": strategies,
        "by_encoder": {
            "baseline": {
                "all": ablation_deltas(baseline, strategies),
                "markdown_only": ablation_deltas(baseline, strategies, ".md"),
            }
        },
    }
    if ft is not None:
        report["by_encoder"]["FT-300"] = {
            "all": ablation_deltas(ft, strategies),
            "markdown_only": ablation_deltas(ft, strategies, ".md"),
        }
        report["encoder_swap_compression"] = {
            "all": encoder_swap_compression(baseline, ft, strategies),
            "markdown_only": encoder_swap_compression(
                baseline, ft, strategies, ".md"
            ),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", required=True, type=Path,
                   help="Chunk-ablation probe-result JSON (baseline encoder).")
    p.add_argument("--ft", type=Path,
                   help="Chunk-ablation probe-result JSON (FT/swapped encoder). "
                        "Omit to report baseline-only (no compression test).")
    p.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    baseline = json.loads(args.baseline.read_text())
    ft = json.loads(args.ft.read_text()) if args.ft else None
    report = build_report(baseline, ft)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    log.info("strategies: %s", report["strategies"])
    if ft is not None:
        comp = report["encoder_swap_compression"]["all"]
        for key, c in comp.items():
            log.info("  %-28s baseline=%+.4f  ft=%+.4f  compression=%+.4f",
                     key, c["baseline_delta"], c["ft_delta"] or 0.0,
                     c["compression"] or 0.0)
    else:
        log.info("baseline-only (no --ft): reported deltas without compression")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
