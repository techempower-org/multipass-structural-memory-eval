#!/usr/bin/env python3
"""#116 orchestrator sweep — vary the reader, hold retrieval fixed.

The unified finding: on LongMemEval oracle, retrieval is ~97% R@5 but
end-to-end QA-acc is ~38pp lower — the reader is the bottleneck, and the
encoder swap (#84) was a null. This harness isolates the reader.

Two phases (keeps daemon load minimal and serial — the orchestrator runs
Phase 1; Phase 2 is offline):

  pin-context   PHASE 1 (daemon). Retrieve once per oracle question and dump
                the full context_string per question to a pinned-context JSON.
                Run one per daemon snippet-width setting (palace-daemon#150).
                Reuses run_longmemeval_mempalace.py's retrieval path with the
                reader+judge disabled (--skip-reader --skip-judge), then
                captures the context the harness otherwise discards.

  reader-sweep  PHASE 2 (offline, no daemon). Replay the pinned context through
                the reader matrix (model × prompt × context-width), judge each,
                report QA-acc per config vs the fixed retrieval ceiling.

  dry-run       Size the sweep (config count, LLM-call count) without running.

Two operating modes (JP greenlit phi4-default for exploration):

  DEFAULT (exploratory)  reader + judge are LOCAL (phi4 via ollama). No
                         Azure/Bedrock calls → no 429s, no cost, unlimited
                         concurrency. The right mode for iterating on prompts /
                         context-widths / sweep shape.
  --headline             opt into the rate-limited Azure/Bedrock readers + the
                         canonical LongMemEval judge, for publishable numbers.

Explicit --reader-models / --judge ALWAYS win over either mode's default.

Usage:

    # Exploratory sizing (default = local phi4, no LLM hit in dry-run):
    reader_sweep_eval.py dry-run \
        --pinned baselines/pinned_context_search-default.json \
        --prompts baseline cot extractive \
        --context-widths 2000 6000 full

    # Exploratory run (default = phi4 reader + phi4 judge, all local/free):
    reader_sweep_eval.py reader-sweep \
        --pinned baselines/pinned_context_search-default.json \
        --prompts baseline cot \
        --json baselines/reader_sweep_<date>.json

    # Headline run (Azure/Bedrock readers + canonical gpt-4o judge):
    reader_sweep_eval.py reader-sweep --headline \
        --pinned baselines/pinned_context_search-default.json \
        --json baselines/reader_sweep_headline_<date>.json

    # Explicit models always win (here: a local sweep on a bigger ollama model):
    reader_sweep_eval.py reader-sweep \
        --pinned baselines/pinned_context_search-default.json \
        --reader-models qwen2.5:14b-instruct-q4_K_M --judge phi4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sme.eval.reader_sweep import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    PROMPT_VARIANTS,
    SweepMatrix,
    estimate_sweep_calls,
    load_pinned_context,
    run_sweep,
)

log = logging.getLogger("reader_sweep_eval")

# Two operating modes (JP greenlit phi4-default for exploration):
#   - DEFAULT (exploratory): reader + judge are LOCAL (phi4 via ollama). No
#     Azure/Bedrock calls, so no 429s, no cost, and concurrency is unlimited —
#     the right mode for iterating on prompts / context-widths / sweep shape.
#   - --headline: opt into the rate-limited Azure/Bedrock readers + the
#     canonical LongMemEval judge, for publishable numbers.
# Explicit --reader-models / --judge ALWAYS win over either mode's default.
EXPLORATORY_READERS = ["phi4"]
EXPLORATORY_JUDGE = "phi4"
HEADLINE_READERS = ["gpt-4.1-mini", "gpt-4o"]
HEADLINE_JUDGE = "gpt-4o"

# Back-compat alias — some callers/imports reference DEFAULT_JUDGE.
DEFAULT_JUDGE = EXPLORATORY_JUDGE


def _parse_widths(raw: list[str]) -> list[Optional[int]]:
    """'full' → None; integers → char caps."""
    out: list[Optional[int]] = []
    for w in raw:
        out.append(None if w.lower() == "full" else int(w))
    return out


def _resolve_models(args: argparse.Namespace) -> None:
    """Fill in reader/judge defaults from the operating mode, in place.

    Explicit ``--reader-models`` / ``--judge`` always win (back-compat). When a
    value is omitted it comes from ``--headline`` (Azure/Bedrock + canonical
    judge) or, by default, the local exploratory lane (phi4 via ollama).
    ``--judge`` defaults to the argparse sentinel ``None`` so we can tell an
    explicit value apart from an unset one. ``dry-run`` has no ``--judge``.
    """
    headline = getattr(args, "headline", False)
    if args.reader_models is None:
        args.reader_models = HEADLINE_READERS if headline else EXPLORATORY_READERS
    # dry-run has no --judge; only the reader-sweep subcommand sets it.
    if hasattr(args, "judge") and args.judge is None:
        args.judge = HEADLINE_JUDGE if headline else EXPLORATORY_JUDGE


def _build_matrix(args: argparse.Namespace) -> SweepMatrix:
    return SweepMatrix(
        reader_models=list(args.reader_models),
        prompts=list(args.prompts),
        context_widths=_parse_widths(args.context_widths),
    )


def cmd_dry_run(args: argparse.Namespace) -> int:
    _resolve_models(args)
    _, records = load_pinned_context(args.pinned)
    matrix = _build_matrix(args)
    est = estimate_sweep_calls(len(records), matrix)
    print("=" * 64)
    print(f"  pinned: {args.pinned}  ({est['n_questions']} questions)")
    print(f"  configs: {est['n_configs']}  reader_calls: {est['reader_calls']}  "
          f"judge_calls: {est['judge_calls']}  total_llm_calls: {est['total_llm_calls']}")
    print("  config matrix:")
    for c in est["configs"]:
        print(f"    {c}")
    if args.json:
        args.json.write_text(json.dumps(est, indent=2))
        print(f"\n  wrote {args.json}")
    return 0


def cmd_reader_sweep(args: argparse.Namespace) -> int:
    _resolve_models(args)
    meta, records = load_pinned_context(args.pinned)
    matrix = _build_matrix(args)

    def _progress(i, n, qid):
        log.info("[%d/%d] %s", i, n, qid)

    report_core = run_sweep(
        records=records, matrix=matrix, judge_model=args.judge,
        progress=_progress if args.verbose else None,
        concurrency=args.concurrency,
    )
    report = {
        "run_metadata": {
            "diagnostic": "reader_sweep",
            "issue": "techempower-org/multipass-structural-memory-eval#116",
            "pinned_context": str(args.pinned),
            "pinned_metadata": meta,
            "mode": "headline" if args.headline else "exploratory",
            "judge_model": args.judge,
            "reader_models": list(args.reader_models),
            "prompts": list(args.prompts),
            "context_widths": list(args.context_widths),
            "concurrency": args.concurrency,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        **report_core,
    }
    print("=" * 64)
    print(f"  #116 reader sweep — {report_core['n_questions']} questions, "
          f"{report_core['n_configs']} configs")
    for c in report_core["configs"]:
        o = c["summary"]["overall"]
        print(f"  {c['config']:48}  QA-acc={o.get('qa_acc', 0):.4f}  (n={o.get('n', 0)})")
    if report_core["best"]:
        b = report_core["best"]
        print(f"\n  best: {b['config']}  QA-acc={b['qa_acc']:.4f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser. Exposed so tests drive the real parser rather
    than a hand-rolled copy that could drift from production."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_sweep_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--pinned", required=True, type=Path,
                        help="Phase-1 pinned-context JSON")
        sp.add_argument("--reader-models", nargs="+", default=None,
                        help="Reader model ids to sweep. Omit to use the mode "
                             "default: local phi4 (exploratory) or "
                             f"{HEADLINE_READERS} (--headline). Explicit values "
                             "always win.")
        sp.add_argument("--headline", action="store_true",
                        help="Opt into the rate-limited Azure/Bedrock readers + "
                             "the canonical LongMemEval judge for publishable "
                             "numbers. Default is the local exploratory lane "
                             "(phi4 via ollama — no 429s, free, unlimited "
                             "concurrency).")
        sp.add_argument("--prompts", nargs="+", default=["baseline"],
                        choices=sorted(PROMPT_VARIANTS),
                        help=f"Prompt variants (default: baseline). "
                             f"Known: {sorted(PROMPT_VARIANTS)}")
        sp.add_argument("--context-widths", nargs="+", default=["full"],
                        help="Char caps or 'full' (default: full)")

    sp_dry = sub.add_parser("dry-run", help="Size the sweep, no LLM calls")
    _add_sweep_args(sp_dry)
    sp_dry.add_argument("--json", type=Path, default=None)
    sp_dry.set_defaults(func=cmd_dry_run)

    sp_run = sub.add_parser("reader-sweep", help="Run the offline reader sweep")
    _add_sweep_args(sp_run)
    sp_run.add_argument(
        "--judge", default=None,
        help=f"Judge model id. Omit to use the mode default: local phi4 "
             f"(exploratory) or {HEADLINE_JUDGE!r} (--headline). Explicit "
             f"value always wins.")
    sp_run.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Parallel reader/judge calls per config via a thread pool "
             f"(default {DEFAULT_CONCURRENCY}; 1 = serial). The calls are "
             f"network-I/O-bound, so K>1 is a near-linear speed-up. Results are "
             f"identical to serial. Raising K too high risks provider 429 "
             f"rate-limit errors.",
    )
    sp_run.add_argument("--json", type=Path, default=None)
    sp_run.set_defaults(func=cmd_reader_sweep)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
