#!/usr/bin/env python3
"""Re-judge the Pass B reader-sweep answers with the Opus judge (#116).

Pass B (baselines/reader_sweep_passB_opus_strat150_search-default_2026-05-29.json)
ran three readers — claude-opus-4-8, o4-mini, gpt-5.3-chat — on a stratified-150
pinned context set, all judged by ``gpt-5.3-chat``. Opus scored *worst* (0.393),
which we diagnosed as a prompt/JUDGE confound: gpt-5.3-chat penalizes Opus's
thorough, cautious answers.

This script isolates the JUDGE variable. It re-grades the *exact same* Pass B
hypotheses (no re-generation) with ``claude-opus-4-8`` as the judge, recomputes
QA-acc per reader and per category, and writes a side-by-side comparison:

    gpt-5.3-chat-judge QA   vs   opus-judge QA       (per reader, per category)

Key question: does the Opus judge RESCUE Opus's penalized answers — especially
the single-session-preference 0.00 collapse and the knowledge-update PARTIALs?

Scoring is kept identical to the original sweep by importing
``sme.eval.reader_sweep.aggregate_labels`` (CORRECT counts; ABSTAIN counts only
on abstention questions). Judge labels come from
``sme.eval.longmemeval_judge.grade_answer(..., judge_model="claude-opus-4-8")``,
which routes through the AnthropicBedrock shim.

Output: baselines/reader_sweep_passB_opus_REJUDGED_2026-05-29.json

Usage (nohup-detached — makes ~450 Bedrock judge calls):
    nohup ./venv/bin/python scripts/rejudge_passB_opus.py \
        > scratch/.../rejudge.out 2>&1 &
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

# Repo root (worktree) on sys.path so ``sme`` imports work when run as a script.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Baselines (Pass B input + rejudged output) live in the *main* checkout's
# baselines/ dir, which is shared across worktrees and is where the parent
# agent expects the output. The worktree's own baselines/ is empty.
_BASELINES = Path(
    "/home/jp/Projects/multipass-structural-memory-eval/baselines"
)

from sme.eval.longmemeval_judge import grade_answer  # noqa: E402
from sme.eval.reader_sweep import aggregate_labels  # noqa: E402

PASS_B = (
    _BASELINES
    / "reader_sweep_passB_opus_strat150_search-default_2026-05-29.json"
)
PINNED = Path(
    "/home/jp/.claude/projects/-home-jp-Projects-multipass-structural-memory-eval"
    "/scratch/sme-bench-2026-05-29/pinned_search-default_strat150.json"
)
OUT = (
    _BASELINES
    / "reader_sweep_passB_opus_REJUDGED_2026-05-29.json"
)
STATUS = Path(
    "/home/jp/.claude/projects/-home-jp-Projects-multipass-structural-memory-eval"
    "/scratch/sme-bench-2026-05-29/rejudge.status"
)

OPUS_JUDGE = "claude-opus-4-8"
ORIG_JUDGE = "gpt-5.3-chat"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _write_status(msg: str) -> None:
    try:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(f"{_now()}  {msg}\n")
    except OSError:
        pass


def load_gold(pinned_path: Path) -> dict[str, dict]:
    """question_id -> {question, gold_answer, question_type}."""
    data = json.loads(pinned_path.read_text())
    gold: dict[str, dict] = {}
    for rec in data["pinned_context"]:
        gold[rec["question_id"]] = {
            "question": rec["question"],
            "gold_answer": rec["gold_answer"],
            "question_type": rec["question_type"],
            "is_abstention": rec.get("is_abstention", False),
        }
    return gold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--limit", type=int, default=0,
        help="If >0, only re-judge the first N questions per config (smoke).",
    )
    args = ap.parse_args()

    pass_b = json.loads(PASS_B.read_text())
    gold = load_gold(PINNED)

    configs = pass_b["configs"]
    total_calls = sum(
        min(len(c["per_question"]), args.limit) if args.limit else
        len(c["per_question"])
        for c in configs
    )
    _write_status(f"START rejudge judge={OPUS_JUDGE} total_calls={total_calls}")
    print(f"[rejudge] {len(configs)} configs, ~{total_calls} Opus judge calls",
          flush=True)

    out_configs = []
    done = 0
    t0 = time.time()

    for cfg in configs:
        label = cfg["config"]                      # e.g. 'claude-opus-4-8|baseline|ctx=full'
        reader_model = label.split("|", 1)[0]
        rows = cfg["per_question"]
        if args.limit:
            rows = rows[: args.limit]

        rejudged_rows = []
        for row in rows:
            qid = row["question_id"]
            g = gold.get(qid)
            if g is None:
                # Should not happen — pinned set and Pass B share the 150 ids.
                rejudged_rows.append({
                    "question_id": qid,
                    "question_type": row["question_type"],
                    "sme_category": row.get("sme_category"),
                    "hypothesis": row.get("hypothesis", ""),
                    "orig_label": row.get("autoeval_label"),
                    "opus_label": "ERROR",
                    "opus_rationale": "no gold for question_id",
                })
                done += 1
                continue

            verdict = grade_answer(
                question_type=g["question_type"],
                question=g["question"],
                gold_answer=g["gold_answer"],
                hypothesis=row.get("hypothesis", ""),
                judge_model=OPUS_JUDGE,
            )
            rejudged_rows.append({
                "question_id": qid,
                # aggregate_labels keys on these two fields:
                "question_type": g["question_type"],
                "autoeval_label": verdict["autoeval_label"],
                "sme_category": row.get("sme_category"),
                "hypothesis": row.get("hypothesis", ""),
                "orig_label": row.get("autoeval_label"),
                "opus_label": verdict["autoeval_label"],
                "opus_rationale": verdict.get("rationale", ""),
            })
            done += 1
            if done % 25 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                _write_status(
                    f"PROGRESS {done}/{total_calls} "
                    f"({rate:.2f} calls/s) cfg={label}"
                )
                print(f"[rejudge] {done}/{total_calls} ({rate:.2f}/s) {label}",
                      flush=True)

        # Recompute QA-acc with the Opus labels, identical scoring to the sweep.
        opus_summary = aggregate_labels(rejudged_rows)
        out_configs.append({
            "config": label,
            "reader_model": reader_model,
            "orig_judge_summary": cfg["summary"],   # gpt-5.3-chat judge (Pass B)
            "opus_judge_summary": opus_summary,      # claude-opus-4-8 judge
            "per_question": rejudged_rows,
        })
        print(f"[rejudge] DONE config {label}: "
              f"orig={cfg['summary']['overall']['qa_acc']:.4f} "
              f"opus={opus_summary['overall']['qa_acc']:.4f}", flush=True)

    # Build the side-by-side comparison table (overall + per category).
    comparison = _build_comparison(out_configs)

    result = {
        "run_metadata": {
            "diagnostic": "reader_sweep_rejudge",
            "issue": "techempower-org/multipass-structural-memory-eval#116",
            "purpose": (
                "Isolate the JUDGE variable: re-grade the exact Pass B "
                "hypotheses with the Opus judge to test whether a stronger "
                "judge rescues Opus's penalized answers."
            ),
            "source_passB": str(PASS_B),
            "pinned_context": str(PINNED),
            "orig_judge": ORIG_JUDGE,
            "rejudge_judge": OPUS_JUDGE,
            "n_questions": pass_b["n_questions"],
            "n_configs": len(out_configs),
            "limit": args.limit or None,
            "timestamp_utc": _now(),
        },
        "comparison": comparison,
        "configs": out_configs,
    }
    OUT.write_text(json.dumps(result, indent=2))
    _write_status(f"DONE wrote {OUT} ({done} judge calls)")
    print(f"[rejudge] wrote {OUT}", flush=True)
    _print_comparison(comparison)
    return 0


def _build_comparison(out_configs: list[dict]) -> dict:
    """Per-reader overall + per-category: orig-judge QA vs opus-judge QA + delta."""
    table = {}
    for c in out_configs:
        reader = c["reader_model"]
        orig = c["orig_judge_summary"]
        opus = c["opus_judge_summary"]
        cats = {}
        orig_by = orig["by_question_type"]
        opus_by = opus["by_question_type"]
        for qt in sorted(set(orig_by) | set(opus_by)):
            o = orig_by.get(qt, {}).get("qa_acc", 0.0)
            n = opus_by.get(qt, {}).get("qa_acc", 0.0)
            cats[qt] = {
                "orig_judge_qa": o,
                "opus_judge_qa": n,
                "delta": round(n - o, 4),
                "orig_labels": orig_by.get(qt, {}).get("labels", {}),
                "opus_labels": opus_by.get(qt, {}).get("labels", {}),
            }
        oa = orig["overall"]["qa_acc"]
        na = opus["overall"]["qa_acc"]
        table[reader] = {
            "overall": {
                "orig_judge_qa": oa,
                "opus_judge_qa": na,
                "delta": round(na - oa, 4),
                "orig_labels": orig["overall"]["labels"],
                "opus_labels": opus["overall"]["labels"],
            },
            "by_category": cats,
        }
    return table


def _print_comparison(comparison: dict) -> None:
    print("\n================ JUDGE-AXIS COMPARISON ================", flush=True)
    print(f"{'reader':<20} {'category':<28} {'gpt5.3-J':>9} "
          f"{'opus-J':>8} {'delta':>8}", flush=True)
    for reader, blk in comparison.items():
        ov = blk["overall"]
        print(f"{reader:<20} {'OVERALL':<28} {ov['orig_judge_qa']:>9.3f} "
              f"{ov['opus_judge_qa']:>8.3f} {ov['delta']:>+8.3f}", flush=True)
        for qt, row in blk["by_category"].items():
            print(f"{'':<20} {qt:<28} {row['orig_judge_qa']:>9.3f} "
                  f"{row['opus_judge_qa']:>8.3f} {row['delta']:>+8.3f}",
                  flush=True)
        print("", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — surface to status + nohup log
        _write_status(f"ERROR {type(e).__name__}: {e}")
        raise
