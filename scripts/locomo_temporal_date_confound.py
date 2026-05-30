"""Date-confound diagnostic for the flat LoCoMo temporal slice (#108).

Re-runs ONLY the 50 temporal questions from the stratified subset
(seed=1729, 50/type — reselected identically, temporal bucket kept) with
``capture_context=True``, so we can read EXACTLY what the reader saw and
answer two questions:

  1. Did the reader's retrieved context contain session DATES? The OMEGA
     date-confound is: dates stripped -> temporal QA collapses (cat_6 0.04;
     restored to 0.36 once dates were present). If LoCoMo's flat reader
     context was date-starved, temporal 0.26 would be a fixable artifact.
  2. For the wrong answers, classify each: DATE_STRIPPED (the confound),
     RETRIEVAL_MISS (evidence session not retrieved), IDK_DESPITE_EVIDENCE
     (refused though content present), or GENUINE_REASONING_FAIL (dates +
     content present, still wrong).

Result (2026-05-30, gpt-5.3-chat reader+judge): temporal 0.26 reproduced,
ctx_with_date 50/50 = 100%, DATE_STRIPPED=0. Confound ruled out; failures are
RETRIEVAL_MISS 24 / GENUINE_REASONING_FAIL 8 / IDK_DESPITE_EVIDENCE 5. See
docs/benchmarks/2026-05-30-locomo-date-confound-ruled-out.md.

Usage:
    python scripts/locomo_temporal_date_confound.py \
        --dataset sme/corpora/locomo/data/locomo10.json \
        --out baselines/locomo10_temporal_date_diagnostic_<date>.json \
        --work-dir <tmp> --reader-model gpt-5.3-chat --judge-model gpt-5.3-chat

This is read-only diagnosis + 50 reader+judge calls (no writes to any palace).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
import sys
import tempfile
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cross_validate_longmemeval as harness  # noqa: E402
from sme.corpora.locomo import load_questions  # noqa: E402

PER_TYPE = 50
SEED = 1729

# A date in the rendered context: "_Date: 1:56 pm on 8 May, 2023_" or the
# frontmatter "date: '...'"; bare month names are a weak secondary signal.
_DATE_RE = re.compile(
    r"_Date:|^date:|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b",
    re.M,
)
_IDK_RE = re.compile(r"\bi don'?t know\b", re.I)


def stratified_temporal(questions, per_type, seed):
    """Replay the exact stratified shuffle the full run used (iterate all
    types in sorted order so the RNG state matches), keep only temporal."""
    by = collections.defaultdict(list)
    for q in questions:
        by[q.question_type].append(q)
    rng = random.Random(seed)
    picked = {}
    for t in sorted(by):
        pool = sorted(by[t], key=lambda q: q.question_id)
        rng.shuffle(pool)
        picked[t] = pool[:per_type]
    return sorted(picked["temporal"], key=lambda q: (q.sample_id, q.question_id))


def classify(row_correct, ctx_has_date, said_idk, evidence_present):
    if row_correct:
        return None
    if not ctx_has_date:
        return "DATE_STRIPPED"          # the OMEGA confound
    if said_idk and evidence_present:
        return "IDK_DESPITE_EVIDENCE"
    if not evidence_present:
        return "RETRIEVAL_MISS"
    return "GENUINE_REASONING_FAIL"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=pathlib.Path,
                   default=pathlib.Path("sme/corpora/locomo/data/locomo10.json"))
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("locomo10_temporal_date_diagnostic.json"))
    p.add_argument("--work-dir", type=pathlib.Path, default=None)
    p.add_argument("--reader-model", default="gpt-5.3-chat")
    p.add_argument("--judge-model", default="gpt-5.3-chat")
    args = p.parse_args(argv)

    work_dir = args.work_dir or pathlib.Path(tempfile.mkdtemp(prefix="locomo_diag_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    qs = list(load_questions(args.dataset))
    temporal = stratified_temporal(qs, PER_TYPE, SEED)
    print(f"running temporal diagnostic n={len(temporal)} seed={SEED}", flush=True)

    records = harness.run_locomo_questions(
        temporal,
        adapter_factory=harness._ADAPTER_FACTORIES["flat"],
        work_dir=work_dir,
        skip_judge=False,
        skip_reader=False,
        reader_model=args.reader_model,
        judge_model=args.judge_model,
        capture_context=True,
        max_questions=None,
    )
    dt = time.time() - t0

    rows = []
    n_ctx_with_date = n_correct = 0
    for r in records:
        ctx = r.get("context_string", "") or ""
        hyp = r.get("hypothesis", "") or ""
        label = (r.get("judge") or {}).get("autoeval_label", "ERROR")
        ctx_has_date = bool(_DATE_RE.search(ctx))
        n_ctx_with_date += int(ctx_has_date)
        is_correct = harness.judge_label_to_correct(label) is True
        n_correct += int(is_correct)
        klass = classify(
            is_correct, ctx_has_date,
            said_idk=bool(_IDK_RE.search(hyp)),
            evidence_present=r.get("sme_recall", 0) >= 0.5,
        )
        rows.append({
            "question_id": r["question_id"], "label": label,
            "correct": is_correct, "ctx_has_date": ctx_has_date,
            "sme_recall": r.get("sme_recall"), "hit_at_5": r.get("hit_at_5"),
            "failure_class": klass, "question": r.get("question", ""),
            "gold_answer": r.get("gold_answer", ""), "hypothesis": hyp[:400],
        })

    fail_hist = collections.Counter(
        x["failure_class"] for x in rows if x["failure_class"])
    report = {
        "run_metadata": {
            "diagnostic": "LoCoMo temporal date-confound (#108)",
            "n": len(records), "seed": SEED, "adapter": "flat",
            "reader_model": args.reader_model, "judge_model": args.judge_model,
            "capture_context": True, "elapsed_sec": round(dt, 1),
            "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        },
        "headline": {
            "temporal_accuracy": round(n_correct / len(records), 4),
            "ctx_with_date_pct": round(n_ctx_with_date / len(records), 4),
            "ctx_with_date_n": n_ctx_with_date, "ctx_total": len(records),
            "failure_class_histogram": dict(fail_hist),
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"DONE in {dt:.1f}s | temporal_acc={report['headline']['temporal_accuracy']} "
          f"| ctx_with_date={n_ctx_with_date}/{len(records)} "
          f"| failures={dict(fail_hist)} | wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
