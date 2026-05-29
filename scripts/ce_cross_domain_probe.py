#!/usr/bin/env python3
"""Cross-encoder cross-domain robustness probe (#85).

Ports AdaptMem's
``benchmarks/structural_memory_eval/sprint5_ce_cross_domain.py``
(2026-05-28 cut) into SME idioms. The upstream script hardcoded macmini
paths and a specific code-trained CE checkpoint; this port takes the run
file, gold file, and CE checkpoint as CLI args, isolates the trust-gated
rerank logic as unit-testable functions, and skips gracefully (with a
clear message) when ``sentence-transformers`` or the checkpoint are
unavailable.

Diagnostic question: does a cross-encoder reranker inherit the
domain-mismatch curve seen on the bi-encoder axis? We take an
out-of-domain CE (e.g. a CodeSearchNet-trained checkpoint) and slot it
into a *trust-gated* rerank step over a conversational run (LongMemEval),
then compare R@1 before/after the rerank.

Trust gate (the load-bearing methodology): the CE only overrides the
bi-encoder's top-1 when the CE's best candidate beats the bi-encoder's
top-1 by at least ``--margin``. This isolates "CE confidently disagrees"
from "CE marginally reshuffles", so the delta measures domain-robustness
of *confident* CE decisions, not rerank noise.

Three outcomes (controlled-condition deltas, not scores):
  H1 (CE inherits mismatch): R@1_final drops toward the un-reranked floor.
  H2 (CE more domain-sensitive than bi-encoder): R@1_final drops below it.
  H3 (CE domain-robust): R@1_final holds near the in-domain rerank number.

Run-file schema (JSONL, one record per question):
  {"question_id": str, "question_type": str, "question": str,
   "retrieval_results": {"ranked_items": [{"corpus_id": str, "text": str},
                                           ...]}}

Gold-file schema (JSON list):
  [{"question_id": str, "answer_session_ids": [str, ...]}, ...]

Usage:
    venv/bin/python scripts/ce_cross_domain_probe.py \\
        --run path/to/run6_hybrid_v4.jsonl \\
        --gold path/to/longmemeval_s.json \\
        --ce-checkpoint path/to/codecrossenc-v2 \\
        --margin 1.0 --top-k 20 \\
        --out baselines/ce_cross_domain_2026-05-29.json

    # dry-run without loading any model (scoring/aggregation only):
    venv/bin/python scripts/ce_cross_domain_probe.py \\
        --run ... --gold ... --out ...        # no --ce-checkpoint -> baseline-only
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("ce_cross_domain")

# A scorer maps (question, [candidate_texts]) -> [score per candidate].
Scorer = Callable[[str, list[str]], list[float]]


def load_gold(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text())
    return {
        q["question_id"]: set(q.get("answer_session_ids") or [])
        for q in data
    }


def load_runs(path: Path) -> list[dict]:
    """Load a JSONL run file (one question record per line)."""
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def trust_gated_rerank(
    question: str,
    items: list[dict],
    scorer: Scorer | None,
    *,
    margin: float,
    text_clip: int = 2000,
) -> tuple[str, bool]:
    """Apply one trust-gated CE override to a candidate list.

    Returns (final_top1_corpus_id, override_applied).

    The CE scores every candidate; it only overrides the bi-encoder top-1
    (index 0) when its best candidate is a *different* index AND beats
    index 0 by >= ``margin``. With no scorer (or < 2 candidates) the
    bi-encoder top-1 is kept unchanged.
    """
    bi_top1 = items[0]["corpus_id"]
    if scorer is None or len(items) < 2:
        return bi_top1, False
    texts = [str(it.get("text", ""))[:text_clip] for it in items]
    scores = scorer(question, texts)
    ce_idx = max(range(len(scores)), key=lambda j: scores[j])
    margin_val = float(scores[ce_idx]) - float(scores[0])
    if ce_idx == 0 or margin_val < margin:
        return bi_top1, False
    return items[ce_idx]["corpus_id"], True


def evaluate(
    runs: list[dict],
    gold: dict[str, set[str]],
    scorer: Scorer | None,
    *,
    margin: float,
    top_k: int,
) -> dict:
    """Run the trust-gated rerank over every question; aggregate by type.

    Pure given a scorer (or None) — the test suite injects a deterministic
    scorer so no model is needed.
    """
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "baseline": 0, "final": 0, "overrides": 0,
                 "override_helped": 0, "override_hurt": 0}
    )
    for rec in runs:
        qid = rec["question_id"]
        qtype = rec.get("question_type", "unknown")
        items = rec["retrieval_results"]["ranked_items"][:top_k]
        g = gold.get(qid, set())
        if not items:
            continue

        baseline_top1 = items[0]["corpus_id"]
        baseline_hit = int(bool(g) and baseline_top1 in g)

        final_top1, override = trust_gated_rerank(
            rec.get("question", ""), items, scorer, margin=margin
        )
        final_hit = int(bool(g) and final_top1 in g)

        b = by_type[qtype]
        b["n"] += 1
        b["baseline"] += baseline_hit
        b["final"] += final_hit
        if override:
            b["overrides"] += 1
            if baseline_hit == 0 and final_hit == 1:
                b["override_helped"] += 1
            if baseline_hit == 1 and final_hit == 0:
                b["override_hurt"] += 1

    n = sum(b["n"] for b in by_type.values())
    if n == 0:
        return {"n": 0, "R@1_baseline": 0.0, "R@1_final": 0.0, "by_type": {}}
    return {
        "n": n,
        "R@1_baseline": round(sum(b["baseline"] for b in by_type.values()) / n, 4),
        "R@1_final": round(sum(b["final"] for b in by_type.values()) / n, 4),
        "total_overrides": sum(b["overrides"] for b in by_type.values()),
        "total_helped": sum(b["override_helped"] for b in by_type.values()),
        "total_hurt": sum(b["override_hurt"] for b in by_type.values()),
        "by_type": {
            t: {
                "n": b["n"],
                "R@1_baseline": round(b["baseline"] / b["n"], 4),
                "R@1_final": round(b["final"] / b["n"], 4),
                "overrides": b["overrides"],
                "override_helped": b["override_helped"],
                "override_hurt": b["override_hurt"],
            }
            for t, b in sorted(by_type.items())
        },
    }


def build_cross_encoder_scorer(
    checkpoint: str, *, batch_size: int = 64, max_length: int = 384
) -> Optional[Scorer]:
    """Build a Scorer backed by a sentence-transformers CrossEncoder.

    Returns None (with a logged reason) if sentence-transformers is missing
    or the checkpoint can't be loaded, so the caller can fall back to a
    baseline-only report rather than crash.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        log.warning(
            "sentence-transformers not installed; skipping CE rerank "
            "(baseline-only report). `pip install sentence-transformers` "
            "to enable."
        )
        return None
    if not Path(checkpoint).exists():
        log.warning("CE checkpoint not found at %s; skipping CE rerank "
                    "(baseline-only report).", checkpoint)
        return None

    t0 = time.time()
    log.info("loading CrossEncoder from %s", checkpoint)
    ce = CrossEncoder(checkpoint, max_length=max_length)
    log.info("  loaded in %.1fs", time.time() - t0)

    def _scorer(question: str, texts: list[str]) -> list[float]:
        pairs = [(question, t) for t in texts]
        scores = ce.predict(pairs, batch_size=batch_size,
                            show_progress_bar=False)
        return [float(s) for s in scores]

    return _scorer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run", required=True, type=Path,
                   help="JSONL run file with ranked_items per question.")
    p.add_argument("--gold", required=True, type=Path,
                   help="Gold JSON (list of {question_id, answer_session_ids}).")
    p.add_argument("--ce-checkpoint", type=str, default=None,
                   help="Cross-encoder checkpoint path. Omit to emit a "
                        "baseline-only report (no rerank).")
    p.add_argument("--margin", type=float, default=1.0,
                   help="Trust-gate margin: CE must beat bi-top1 by this much "
                        "to override (default: 1.0).")
    p.add_argument("--top-k", type=int, default=20,
                   help="Candidates the CE reranks per question (default: 20).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap questions processed (0 = all).")
    p.add_argument("--ce-training-domain", type=str, default="unspecified",
                   help="Label for the CE's training domain (for the report).")
    p.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    gold = load_gold(args.gold)
    runs = load_runs(args.run)
    if args.limit:
        runs = runs[: args.limit]
    log.info("loaded %d gold questions, %d run records", len(gold), len(runs))

    scorer = (build_cross_encoder_scorer(args.ce_checkpoint,
                                         batch_size=args.batch_size)
              if args.ce_checkpoint else None)

    t0 = time.time()
    result = evaluate(runs, gold, scorer, margin=args.margin, top_k=args.top_k)
    report = {
        "experiment": "cross-domain CE rerank robustness probe",
        "posture": "controlled-condition delta on a fixed bi-encoder run",
        "ce_checkpoint": args.ce_checkpoint,
        "ce_training_domain": args.ce_training_domain,
        "ce_active": scorer is not None,
        "margin": args.margin,
        "top_k": args.top_k,
        "runtime_s": round(time.time() - t0, 1),
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    log.info("R@1 baseline=%.4f  final=%.4f  (overrides=%s helped=%s hurt=%s)",
             report["R@1_baseline"], report["R@1_final"],
             report.get("total_overrides"), report.get("total_helped"),
             report.get("total_hurt"))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
