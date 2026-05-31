#!/usr/bin/env python3
"""LoCoMo-10 end-to-end QA for the postgres+pgvector verbatim substrate.

The postgres twin of the flat LoCoMo E2E run (baselines/
locomo10_flat_e2e_stratified_2026-05-29.json). Same stratified subset
(50/question_type, n=250 — read VERBATIM from the flat baseline's per_question
records so the two runs land on the IDENTICAL questions), same reader = judge =
gpt-5.3-chat (Azure Foundry), same canonical LongMemEval type-specific judge.
The only swapped variable vs flat is the storage/retrieval backend
(chroma -> postgres+pgvector) — this is the "upstream MemPalace raw" ablation:
mempalace's own verbatim postgres storage WITHOUT the palace graph on top.

Requires SME_POSTGRES_DSN (isolated throwaway instance) and AZURE_API_KEY /
AZURE_API_BASE (gpt-5.3-chat). Aggregation mirrors the flat baseline:
qa_by_locomo_type (per question_type), qa_overall, and qa_overall_weighted
(proportion-weighted by natural LoCoMo-10 category shares).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Natural LoCoMo-10 category shares (full 1986-question set) — for the
# proportion-weighted overall, identical to the flat baseline's denominator.
NATURAL_CATEGORY_N = {
    "adversarial": 446,
    "multi-hop": 282,
    "open-domain": 841,
    "single-hop": 321,
    "temporal": 96,
}
NATURAL_TOTAL = sum(NATURAL_CATEGORY_N.values())

FLAT_BASELINE = REPO / "baselines/locomo10_flat_e2e_stratified_2026-05-29.json"
DATASET = REPO / "sme/corpora/locomo/data/locomo10.json"
OUT = REPO / "baselines/locomo10_postgres_e2e_stratified_2026-05-31.json"
READER_MODEL = "gpt-5.3-chat"
JUDGE_MODEL = "gpt-5.3-chat"


def _label_correct(label: str, is_adversarial: bool) -> bool | None:
    """CORRECT == right answer; for adversarial items ABSTAIN == success."""
    if label in ("ERROR", "skipped", None):
        return None
    if is_adversarial:
        return label == "ABSTAIN"
    return label == "CORRECT"


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)

    import os

    if not os.environ.get("SME_POSTGRES_DSN"):
        raise SystemExit("SME_POSTGRES_DSN not set (isolated throwaway pg).")

    # Exact subset replication: pull the 250 question_ids the flat run used.
    flat = json.loads(FLAT_BASELINE.read_text())
    subset_qids = {r["question_id"] for r in flat["per_question"]}
    print(f"[{time.time():.0f}] flat subset: {len(subset_qids)} question_ids")

    from sme.corpora.locomo import load_questions
    from scripts.cross_validate_longmemeval import run_locomo_questions
    from sme.eval.answer_generator import _default_client

    all_qs = list(load_questions(str(DATASET)))
    qs = [q for q in all_qs if q.question_id in subset_qids]
    print(f"[{time.time():.0f}] matched {len(qs)}/{len(subset_qids)} subset questions")
    if len(qs) != len(subset_qids):
        missing = subset_qids - {q.question_id for q in qs}
        print(f"  WARNING: {len(missing)} subset qids not found: {list(missing)[:5]}")

    client = _default_client()
    if client is None:
        raise SystemExit("No reader/judge client (set AZURE_API_KEY/AZURE_API_BASE).")

    import tempfile

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="sme_pg_locomo_") as work:
        records = run_locomo_questions(
            qs,
            adapter_factory=__import__(
                "scripts.cross_validate_longmemeval", fromlist=["_ADAPTER_FACTORIES"]
            )._ADAPTER_FACTORIES["postgres"],
            work_dir=Path(work),
            skip_judge=False,
            skip_reader=False,
            reader_model=READER_MODEL,
            judge_model=JUDGE_MODEL,
            reader_client=client,
            judge_client=client,
        )
    elapsed = time.time() - t0
    print(f"[{time.time():.0f}] {len(records)} records in {elapsed:.0f}s")

    # --- Aggregate by question_type (mirrors flat baseline shape) ----------
    by_type: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "n_scored": 0, "n_error": 0, "correct": 0,
                 "label_counts": Counter()}
    )
    total_usage = Counter()
    for r in records:
        qt = r["question_type"]
        slot = by_type[qt]
        slot["n"] += 1
        judge = r.get("judge") or {}
        label = judge.get("autoeval_label")
        is_adv = r.get("is_adversarial", False)
        total_usage.update(judge.get("usage") or {})
        if label in (None, "ERROR", "skipped"):
            slot["n_error"] += 1
            if label:
                slot["label_counts"][label] += 1
            continue
        slot["n_scored"] += 1
        slot["label_counts"][label] += 1
        if _label_correct(label, is_adv):
            slot["correct"] += 1

    qa_by_locomo_type = {}
    for qt in sorted(by_type):
        s = by_type[qt]
        qa_by_locomo_type[qt] = {
            "n": s["n"],
            "n_scored": s["n_scored"],
            "n_error": s["n_error"],
            "qa_accuracy": round(s["correct"] / s["n_scored"], 4) if s["n_scored"] else None,
            "label_counts": dict(s["label_counts"]),
        }

    n_total = sum(s["n"] for s in by_type.values())
    n_scored = sum(s["n_scored"] for s in by_type.values())
    n_error = sum(s["n_error"] for s in by_type.values())
    n_correct = sum(s["correct"] for s in by_type.values())
    qa_overall = {
        "n": n_total,
        "n_scored": n_scored,
        "n_error": n_error,
        "qa_accuracy": round(n_correct / n_scored, 4) if n_scored else None,
    }

    # Proportion-weighted overall by natural LoCoMo-10 shares.
    weighted_num = 0.0
    weighted_den = 0.0
    for qt, share_n in NATURAL_CATEGORY_N.items():
        acc = qa_by_locomo_type.get(qt, {}).get("qa_accuracy")
        if acc is not None:
            weighted_num += acc * share_n
            weighted_den += share_n
    qa_overall_weighted = {
        "method": "proportion-weighted by natural LoCoMo-10 category shares (estimate of full-set overall)",
        "natural_category_n": NATURAL_CATEGORY_N,
        "natural_total": NATURAL_TOTAL,
        "qa_accuracy_weighted": round(weighted_num / weighted_den, 4) if weighted_den else None,
        "qa_accuracy_unweighted": qa_overall["qa_accuracy"],
    }

    report = {
        "run_metadata": {
            "corpus": "locomo",
            "subset": "locomo10",
            "subset_qa_count_full": NATURAL_TOTAL,
            "adversarial_included": True,
            "sampling": "exact replication of the flat baseline's stratified subset (read from per_question question_ids)",
            "per_type_cap": 50,
            "seed": 1729,
            "n_run": n_total,
            "per_type_counts": {qt: by_type[qt]["n"] for qt in sorted(by_type)},
            "adapter": "postgres",
            "adapter_note": "postgres+pgvector verbatim ingest (upstream MemPalace raw ablation; same all-MiniLM-L6-v2 embedding as flat, backend swapped chroma->postgres). Isolated throwaway docker instance.",
            "reader_model": READER_MODEL,
            "judge_model": JUDGE_MODEL,
            "judge_prompts": "canonical LongMemEval type-specific (temporal->off-by-one; adversarial->abstention)",
            "retrieval_n_results": 5,
            "elapsed_sec": round(elapsed, 1),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "comparison_anchor": "baselines/locomo10_flat_e2e_stratified_2026-05-29.json (flat, identical subset/reader/judge)",
        },
        "qa_by_locomo_type": qa_by_locomo_type,
        "qa_overall": qa_overall,
        "qa_overall_weighted": qa_overall_weighted,
        "judge_total_usage": dict(total_usage),
        "per_question": records,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))

    print("=" * 60)
    print("  adapter:           postgres (pg+pgvector verbatim)")
    print(f"  n_run:             {n_total}  (n_error={n_error})")
    print(f"  QA overall:        {qa_overall['qa_accuracy']} (unweighted)")
    print(f"  QA weighted:       {qa_overall_weighted['qa_accuracy_weighted']}")
    print("  by question_type:")
    for qt in sorted(qa_by_locomo_type):
        s = qa_by_locomo_type[qt]
        print(f"    {qt:14s}  n={s['n']:3d}  acc={s['qa_accuracy']}")
    print(f"  Written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
