#!/usr/bin/env python3
"""Cat 9a invocation-rate cross-validation across a Tau2 orchestrator ladder
(issue techempower-org/multipass-structural-memory-eval#194).

Runs the jp-realm-v0.1 question set through one or more real-model
orchestrators wired with the ``mempalace_search`` tool (palace-daemon,
READ-ONLY) and measures, per orchestrator:

  - invocation rate  — fraction of questions on which the model issued
    ≥1 mempalace_search tool call (the load-bearing 9a signal)
  - substring recall  — comparable to the Cat-1 / ``retrieve`` numbers
  - each orchestrator's published Tau2 (tool-agent) score, recorded
    alongside the reading per reference_tau2_predicts_cat9a

Hypothesis (#194): a higher-Tau2 orchestrator raises the invocation
rate. The documented prior (reference_tau2_predicts_cat9a): a +37.7pp
Tau2 gap between gemma4:e4b and qwen3.5:4b predicted a +30-33pp Cat-9a
recall gap on this corpus. This run extends the ladder to the frontier
tier (claude-opus-4-8, a Tau2 leader) via Bedrock.

Usage (nohup-detach for the full run — Bedrock leg makes ~30 model calls):

    ./venv/bin/python scripts/cat9a_invocation_rate.py \
        --api-url http://familiar:8085 \
        --models gemma4:e4b qwen3.5:4b claude-opus-4-8 \
        --out-prefix baselines/cat9a_tau2_ladder_2026-05-30

Each model writes ``<out-prefix>__<safe-model>.json``; a combined
``<out-prefix>__matrix.json`` carries the Tau2-vs-invocation table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the repo root without installing the package path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from sme.categories.harness_integration import (  # noqa: E402
    Cat9aResult,
    format_cat9a_report,
    run_cat9a,
)
from sme.eval.cat9a_orchestrators import DaemonSearch, make_orchestrator  # noqa: E402

# Published Tau2 (tool-agent) scores for the orchestrator ladder. Sources
# recorded so every reading is auditable (reference_tau2_predicts_cat9a
# methodology). gemma4/qwen3.5 4B comparison: maniac.ai 4B blog
# (qwen3.5 leads gemma4 by +37.7pp). Opus telecom: taubench.com /
# sierra-research tau2-bench leaderboard.
TAU2_SCORES = {
    "gemma4:e4b": (
        42.2,
        "maniac.ai 4B comparison (tau2-bench): gemma4 E4B 42.2, 37.7pp below qwen3.5",
    ),
    "qwen3.5:4b": (
        79.9,
        "maniac.ai 4B comparison (tau2-bench): qwen3.5-4B 79.9, +37.7pp over gemma4:e4b",
    ),
    "claude-opus-4-8": (
        99.3,
        "tau2-bench telecom — Opus 4.x frontier tier (Opus 4.6 99.3% telecom / 91.9% retail)",
    ),
}

DEFAULT_QUESTIONS = "sme/corpora/jp_realm_v0_1/questions.yaml"


def _safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _result_to_dict(result: Cat9aResult, *, corpus_version: str, elapsed_s: float) -> dict:
    return {
        "subtest": "9a",
        "category": "harness_integration",
        "orchestrator": result.orchestrator,
        "tau2": result.tau2,
        "tau2_note": result.tau2_note,
        "corpus_version": corpus_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed_s, 1),
        "summary": {
            "total_questions": result.total_questions,
            "invoked_questions": result.invoked_questions,
            "errored_questions": result.errored_questions,
            "invocation_rate": result.invocation_rate,
            "mean_recall": result.mean_recall,
            "hit_rate": result.hit_rate,
            "band": result.band,
        },
        "questions": [
            {
                "id": r.question_id,
                "text": r.text,
                "expected_sources": r.expected_sources,
                "tool_calls": r.outcome.tool_calls,
                "invoked": r.outcome.invoked,
                "matched_sources": r.matched_sources,
                "recall": r.recall,
                "hit": r.hit,
                "error": r.outcome.error,
            }
            for r in result.readings
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-url", default=os.environ.get("PALACE_DAEMON_URL", "http://familiar:8085"))
    ap.add_argument("--api-key", default=os.environ.get("PALACE_API_KEY", ""))
    ap.add_argument("--kind", default="content")
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["gemma4:e4b", "qwen3.5:4b", "claude-opus-4-8"],
        help="Orchestrator model ids, low-Tau2 first.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap questions (smoke runs).")
    ap.add_argument("--out-prefix", default="baselines/cat9a_tau2_ladder")
    args = ap.parse_args()

    with open(args.questions) as f:
        qdoc = yaml.safe_load(f)
    questions = qdoc.get("questions", [])
    if args.limit:
        questions = questions[: args.limit]
    corpus_version = qdoc.get("version", "?")
    print(f"Cat 9a invocation-rate ladder — corpus={corpus_version} n={len(questions)}")
    print(f"daemon={args.api_url} (READ-ONLY /search) models={args.models}\n")

    matrix_rows = []
    for model in args.models:
        backend = DaemonSearch(args.api_url, args.api_key, kind=args.kind)
        try:
            driver = make_orchestrator(model, backend)
        except Exception as e:  # noqa: BLE001
            print(f"[{model}] driver init FAILED: {type(e).__name__}: {e}\n")
            continue
        tau2, note = TAU2_SCORES.get(model, (None, ""))

        def _progress(reading):
            mark = "✓" if reading.outcome.invoked else "·"
            print(
                f"  {mark} {reading.question_id:28} calls={reading.outcome.tool_calls} "
                f"recall={reading.recall:.2f}"
                + (f"  ERR={reading.outcome.error}" if reading.outcome.error else "")
            )

        print(f"=== {model} (Tau2 {tau2}) ===")
        t0 = time.time()
        result = run_cat9a(
            questions, driver, orchestrator=model, tau2=tau2, tau2_note=note,
            on_question=_progress,
        )
        elapsed = time.time() - t0
        print()
        print(format_cat9a_report(result, source_label=corpus_version))
        # Audit: prove read-only — record how many searches were issued.
        print(f"  [audit] {len(backend.queries)} daemon /search GETs, 0 writes\n")

        out_path = Path(f"{args.out_prefix}__{_safe(model)}.json")
        out_path.write_text(json.dumps(_result_to_dict(result, corpus_version=corpus_version, elapsed_s=elapsed), indent=2))
        print(f"  wrote {out_path}\n")

        matrix_rows.append(
            {
                "orchestrator": model,
                "tau2": tau2,
                "tau2_note": note,
                "invocation_rate": result.invocation_rate,
                "mean_recall": result.mean_recall,
                "hit_rate": result.hit_rate,
                "invoked": result.invoked_questions,
                "total": result.total_questions,
                "errored": result.errored_questions,
            }
        )

    matrix = {
        "subtest": "9a",
        "corpus_version": corpus_version,
        "n_questions": len(questions),
        "daemon": args.api_url,
        "read_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ladder": matrix_rows,
    }
    matrix_path = Path(f"{args.out_prefix}__matrix.json")
    matrix_path.write_text(json.dumps(matrix, indent=2))

    print("=" * 70)
    print("Tau2 → invocation-rate ladder")
    print("=" * 70)
    print(f"{'orchestrator':22} {'Tau2':>6} {'invoke%':>9} {'recall':>8} {'hit%':>7}")
    for r in matrix_rows:
        ir = (r["invocation_rate"] or 0) * 100
        print(
            f"{r['orchestrator']:22} {(r['tau2'] or 0):>6.1f} {ir:>8.1f}% "
            f"{r['mean_recall']:>7.1%} {r['hit_rate']:>6.1%}"
        )
    print(f"\nmatrix written to {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
