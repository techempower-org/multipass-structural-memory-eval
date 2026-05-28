#!/usr/bin/env python3
"""aggregate_probe_sweep.py — consolidate the 4 per-config probe results
from `sweep_probe_47.sh` into a single sweep JSON and a markdown table.

Usage::

    python3 aggregate_probe_sweep.py \
        --date 2026-05-28 \
        --in-dir tests/eval \
        --out-json tests/eval/probe-results-2026-05-28-sweep.json \
        --out-md   tests/eval/probe-results-2026-05-28-sweep.md

Inputs (one per config): probe-results-{date}-{config}.json from
run_paraphrase_probe.py. Output JSON pivots recall@5 and MRR across
configs so a reader can see ΔRecall and ΔMRR vs. baseline at a glance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONFIGS = ["baseline", "modality_only", "flashrank_only", "both"]

CONFIG_LABELS = {
    "baseline": "Baseline (rerank=off, mod=off)",
    "modality_only": "Modality only (rerank=off, mod=on)",
    "flashrank_only": "FlashRank only (rerank=on, mod=off)",
    "both": "Both ON (rerank=on, mod=on)",
}


def _load(path: Path) -> dict:
    """Read a single per-config probe JSON; return {} if missing so the
    aggregator can still summarize partial sweeps and show holes."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.stderr.write(f"WARN: {path}: {e}\n")
        return {}


def _recall_by_shape(data: dict, arm: str) -> dict[str, dict]:
    """Per-shape recall@5 (cnt/n + pct) for the chosen HyDE arm
    ('no_hyde'|'yes_hyde'). Iterates over the per-row arm rather than
    re-deriving from `summary` so missing-shape gaps stay visible."""
    by_shape: dict[str, dict] = {}
    for r in data.get("rows", []):
        s = r.get("shape", "unknown")
        slot = by_shape.setdefault(s, {"n": 0, "hits": 0})
        slot["n"] += 1
        if r.get(arm, {}).get("matched"):
            slot["hits"] += 1
    for s, slot in by_shape.items():
        slot["recall"] = round(slot["hits"] / slot["n"], 4) if slot["n"] else 0.0
    return by_shape


def _overall_recall(data: dict, arm: str) -> dict:
    rows = data.get("rows", [])
    n = len(rows)
    hits = sum(1 for r in rows if r.get(arm, {}).get("matched"))
    return {"n": n, "hits": hits, "recall": round(hits / n, 4) if n else 0.0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, help="Date stamp matching input filenames")
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    in_dir = Path(args.in_dir)
    loaded: dict[str, dict] = {}
    for cfg in CONFIGS:
        path = in_dir / f"probe-results-{args.date}-{cfg}.json"
        loaded[cfg] = _load(path)
        if not loaded[cfg]:
            sys.stderr.write(f"WARN: config '{cfg}' missing or empty at {path}\n")

    # Build pivot: for each config × arm, capture overall + per-shape recall + MRR.
    pivot: dict[str, dict] = {}
    for cfg in CONFIGS:
        d = loaded[cfg]
        if not d:
            pivot[cfg] = {"missing": True}
            continue
        pivot[cfg] = {
            "no_hyde": {
                "overall": _overall_recall(d, "no_hyde"),
                "by_shape": _recall_by_shape(d, "no_hyde"),
                "mrr": d.get("mrr", {}).get("overall", {}).get("no_hyde", 0.0),
            },
            "yes_hyde": {
                "overall": _overall_recall(d, "yes_hyde"),
                "by_shape": _recall_by_shape(d, "yes_hyde"),
                "mrr": d.get("mrr", {}).get("overall", {}).get("yes_hyde", 0.0),
            },
            "latency": d.get("latency", {}),
            "state_summary": d.get("summary", {}),
        }

    # Compute deltas vs. baseline (HyDE-off arm dominates the comparison
    # because that's where retrieval changes show up cleanly; HyDE-on
    # mixes in generator variance).
    base = pivot.get("baseline", {})
    deltas: dict[str, dict] = {}
    if not base.get("missing"):
        base_recall_no = base["no_hyde"]["overall"]["recall"]
        base_recall_yes = base["yes_hyde"]["overall"]["recall"]
        base_mrr_no = base["no_hyde"]["mrr"]
        base_mrr_yes = base["yes_hyde"]["mrr"]
        for cfg in CONFIGS:
            if pivot[cfg].get("missing"):
                deltas[cfg] = {"missing": True}
                continue
            deltas[cfg] = {
                "no_hyde": {
                    "recall_delta_pp": round(
                        (pivot[cfg]["no_hyde"]["overall"]["recall"] - base_recall_no) * 100, 2
                    ),
                    "mrr_delta": round(pivot[cfg]["no_hyde"]["mrr"] - base_mrr_no, 4),
                },
                "yes_hyde": {
                    "recall_delta_pp": round(
                        (pivot[cfg]["yes_hyde"]["overall"]["recall"] - base_recall_yes) * 100, 2
                    ),
                    "mrr_delta": round(pivot[cfg]["yes_hyde"]["mrr"] - base_mrr_yes, 4),
                },
            }

    out = {
        "date": args.date,
        "configs": CONFIGS,
        "pivot": pivot,
        "deltas_vs_baseline": deltas,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ── Markdown ───────────────────────────────────────────────────────
    md = []
    md.append(f"# Eval probe sweep — {args.date}\n")
    md.append("4-config matrix over (PALACE_RERANK_ENABLED × PALACE_MODALITY_WEIGHT).\n")
    md.append("## Overall recall@5 + MRR\n")
    md.append("| Config | HyDE off recall | HyDE on recall | MRR off | MRR on |")
    md.append("|---|---|---|---|---|")
    for cfg in CONFIGS:
        slot = pivot[cfg]
        if slot.get("missing"):
            md.append(f"| {CONFIG_LABELS[cfg]} | — | — | — | — |")
            continue
        md.append(
            f"| {CONFIG_LABELS[cfg]} | "
            f"{slot['no_hyde']['overall']['recall']*100:.1f}% "
            f"({slot['no_hyde']['overall']['hits']}/{slot['no_hyde']['overall']['n']}) | "
            f"{slot['yes_hyde']['overall']['recall']*100:.1f}% "
            f"({slot['yes_hyde']['overall']['hits']}/{slot['yes_hyde']['overall']['n']}) | "
            f"{slot['no_hyde']['mrr']:.3f} | "
            f"{slot['yes_hyde']['mrr']:.3f} |"
        )
    md.append("\n## Δ vs. baseline (percentage points)\n")
    md.append("| Config | ΔRecall HyDE off | ΔRecall HyDE on | ΔMRR off | ΔMRR on |")
    md.append("|---|---|---|---|---|")
    for cfg in CONFIGS:
        slot = deltas.get(cfg, {})
        if slot.get("missing"):
            md.append(f"| {CONFIG_LABELS[cfg]} | — | — | — | — |")
            continue
        md.append(
            f"| {CONFIG_LABELS[cfg]} | "
            f"{slot['no_hyde']['recall_delta_pp']:+.2f}pp | "
            f"{slot['yes_hyde']['recall_delta_pp']:+.2f}pp | "
            f"{slot['no_hyde']['mrr_delta']:+.3f} | "
            f"{slot['yes_hyde']['mrr_delta']:+.3f} |"
        )

    # Per-shape pivot for HyDE-off arm (cleanest comparison: no generator
    # variance, retrieval changes are the only moving part).
    md.append("\n## Per-shape recall (HyDE off)\n")
    all_shapes: set[str] = set()
    for cfg in CONFIGS:
        s = pivot[cfg]
        if not s.get("missing"):
            all_shapes.update(s["no_hyde"]["by_shape"].keys())
    shape_list = sorted(all_shapes)
    md.append("| Shape | " + " | ".join(CONFIG_LABELS[c] for c in CONFIGS) + " |")
    md.append("|---" * (len(CONFIGS) + 1) + "|")
    for shape in shape_list:
        cells = [shape]
        for cfg in CONFIGS:
            s = pivot[cfg]
            if s.get("missing") or shape not in s["no_hyde"]["by_shape"]:
                cells.append("—")
                continue
            slot = s["no_hyde"]["by_shape"][shape]
            cells.append(f"{slot['recall']*100:.0f}% ({slot['hits']}/{slot['n']})")
        md.append("| " + " | ".join(cells) + " |")

    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
