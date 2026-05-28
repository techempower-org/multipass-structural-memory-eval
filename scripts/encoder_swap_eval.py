#!/usr/bin/env python3
"""Encoder-swap A/B harness for LongMemEval.

#53 sub-task 1 — port adaptmem's `mempal_bench_with_ft.py` shape into
SME so any SentenceTransformer-compatible encoder can be A/B'd against
the upstream-default baseline through SME's scorer.

The shape:
  - Per question, build a throwaway in-memory ChromaDB collection from
    the question's haystack sessions (matching upstream's per-question
    isolation).
  - Embed each session via the chosen SentenceTransformer encoder.
  - Query with the question text; return top-K hits.
  - Score recall@K against `answer_session_ids` (the LongMemEval gold).

Why this script exists vs the daemon-adapter runner:
  - Independent of palace-daemon (per #61 blocker).
  - Matches upstream's bench protocol exactly — every per-question
    palace is fresh, deterministic, no cross-question leak.
  - Standalone: one .py file, no daemon, no API key, no judge cost.

CLI:

    encoder_swap_eval.py
        --questions PATH         # longmemeval_oracle.json (or _s_cleaned.json)
        --model MODEL_NAME       # sentence-transformers checkpoint
        --max-questions N        # smoke-test cap
        --content-rules sme-rich | upstream-exact   # per #54
        --json PATH              # output report

Examples:
    # Default encoder (matches upstream raw R@5=0.966 reference)
    encoder_swap_eval.py --questions longmemeval_oracle.json --json out.json

    # FT-300 encoder swap
    encoder_swap_eval.py --questions longmemeval_oracle.json \\
        --model /path/to/minilm-lme-ft-300 --json ft300.json

    # Larger encoder
    encoder_swap_eval.py --questions longmemeval_oracle.json \\
        --model BAAI/bge-large-en-v1.5 --json bge.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# Repo importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sme.corpora.longmemeval import LMEQuestion, load_questions  # noqa: E402

log = logging.getLogger("encoder_swap_eval")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 50  # upstream's default; lets us compute R@1/5/10/30/50 from one query


def _render_session(s: Any, content_rules: str) -> str:
    """Match the same content rules as the loader (#54)."""
    if content_rules == "upstream-exact":
        return "\n".join(t.content for t in s.turns if t.role == "user")
    # sme-rich: include role-tagged content (no frontmatter for retrieval-only)
    parts = []
    for t in s.turns:
        parts.append(f"## {t.role}\n\n{t.content}")
    return "\n".join(parts)


def run_one_question(q: LMEQuestion, encoder: Any, content_rules: str, top_k: int) -> dict:
    """Build a fresh ChromaDB, embed the haystack, query, score."""
    import chromadb

    client = chromadb.EphemeralClient()
    coll = client.create_collection(
        name=f"lme_{q.question_id}",
        embedding_function=encoder,
    )

    # ~7% of LongMemEval-S questions have duplicate session_ids within
    # their haystack. ChromaDB requires unique ids per collection, so we
    # prefix with the haystack index. Map back to the bare session_id
    # for scoring (the LongMemEval gold uses the session_id form).
    docs = [_render_session(s, content_rules) for s in q.haystack_sessions]
    unique_ids = [f"{i}:{s.session_id}" for i, s in enumerate(q.haystack_sessions)]
    coll.add(ids=unique_ids, documents=docs)

    k = min(top_k, len(unique_ids))
    result = coll.query(query_texts=[q.question], n_results=k)
    raw_ids = result["ids"][0]
    # Strip the "<idx>:" prefix to recover bare session_ids.
    retrieved_ids = [rid.split(":", 1)[-1] for rid in raw_ids]

    expected_set = set(q.answer_session_ids)
    # hit_at_K — always include all K thresholds. For small haystacks
    # (len < K), retrieved_ids[:K] just returns the full list, which is
    # the correct semantic: "did we find it in the top-K candidates
    # we returned?"
    hit_at = {
        n: any(rid in expected_set for rid in retrieved_ids[:n])
        for n in (1, 3, 5, 10, 30, 50)
    }
    recall_at_5 = sum(1 for rid in retrieved_ids[:5] if rid in expected_set) / max(1, len(expected_set))

    return {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "sme_category": q.sme_category,
        "n_haystack": len(unique_ids),
        "retrieved_rank_1": retrieved_ids[0] if retrieved_ids else None,
        "expected_sources": list(q.answer_session_ids),
        "hit_at_1": hit_at.get(1, False),
        "hit_at_5": hit_at.get(5, False),
        "hit_at_10": hit_at.get(10, False),
        "hit_at_30": hit_at.get(30, False),
        "hit_at_50": hit_at.get(50, False),
        "recall_at_5": recall_at_5,
    }


def aggregate(records: list[dict]) -> dict:
    """Per-category + overall R@K + r1_miss_by_type histogram."""
    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r["sme_category"], []).append(r)

    def _block(rs: list[dict]) -> dict:
        n = len(rs)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "R@1": round(sum(r["hit_at_1"] for r in rs) / n, 4),
            "R@5": round(sum(r["hit_at_5"] for r in rs) / n, 4),
            "R@10": round(sum(r["hit_at_10"] for r in rs) / n, 4),
            "R@30": round(sum(r["hit_at_30"] for r in rs) / n, 4),
            "R@50": round(sum(r["hit_at_50"] for r in rs) / n, 4),
            "recall_at_5_mean": round(sum(r["recall_at_5"] for r in rs) / n, 4),
        }

    r1_misses = [r for r in records if not r["hit_at_1"]]
    r1_miss_by_type = Counter(r["question_type"] for r in r1_misses)

    return {
        "overall": _block(records),
        "per_category": {cat: _block(rs) for cat, rs in sorted(by_cat.items())},
        "r1_misses": [
            {"question_id": r["question_id"], "question_type": r["question_type"],
             "retrieved_rank_1": r["retrieved_rank_1"],
             "expected_sources": r["expected_sources"],
             "hit_at_5": r["hit_at_5"], "hit_at_10": r["hit_at_10"]}
            for r in r1_misses
        ],
        "r1_miss_by_type": dict(r1_miss_by_type),
    }


def _build_encoder(model_name: str) -> Any:
    """Load a SentenceTransformer-compatible embedding function.

    Accepts a HuggingFace model id or a local checkpoint path.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    log.info("loading encoder: %s", model_name)
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True, type=Path,
                   help="Path to longmemeval_oracle.json / _s_cleaned.json")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="SentenceTransformer model id or local path "
                        f"(default: {DEFAULT_MODEL}, matches upstream raw baseline)")
    p.add_argument("--content-rules", default="upstream-exact",
                   choices=["sme-rich", "upstream-exact"],
                   help="Session rendering (default upstream-exact for "
                        "matched-protocol comparison)")
    p.add_argument("--max-questions", type=int, default=None,
                   help="Smoke-test cap")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help=f"Per-question retrieval depth (default {DEFAULT_TOP_K})")
    p.add_argument("--json", type=Path, default=None, help="Output JSON path")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    encoder = _build_encoder(args.model)

    questions = list(load_questions(args.questions))
    if args.max_questions is not None:
        questions = questions[: args.max_questions]

    records: list[dict] = []
    t0 = time.time()
    for i, q in enumerate(questions):
        log.info("[%d/%d] %s (%s)", i + 1, len(questions), q.question_id, q.question_type)
        records.append(run_one_question(q, encoder, args.content_rules, args.top_k))

    elapsed = time.time() - t0
    summary = aggregate(records)
    summary["wall_clock_seconds"] = round(elapsed, 1)
    summary["per_question_seconds"] = round(elapsed / max(1, len(records)), 2)
    summary["model"] = args.model
    summary["content_rules"] = args.content_rules
    summary["top_k"] = args.top_k

    report = {
        "summary": summary,
        "per_question": records,
    }

    print()
    print("=" * 60)
    print(f"  encoder: {args.model}")
    print(f"  content_rules: {args.content_rules}  top_k: {args.top_k}")
    print(f"  n: {len(records)}  wall: {elapsed:.1f}s  per-q: {elapsed/max(1,len(records)):.2f}s")
    o = summary["overall"]
    print(f"  R@1: {o.get('R@1','-'):.4f}  R@5: {o.get('R@5','-'):.4f}  "
          f"R@10: {o.get('R@10','-'):.4f}  R@30: {o.get('R@30','-'):.4f}  R@50: {o.get('R@50','-'):.4f}")
    print()
    print("  by question_type (R@5):")
    for cat, blk in summary["per_category"].items():
        print(f"    {cat:25}  n={blk['n']:3}  R@1={blk['R@1']:.3f}  R@5={blk['R@5']:.3f}  R@10={blk['R@10']:.3f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
