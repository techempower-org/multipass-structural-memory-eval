#!/usr/bin/env python3
"""LongMemEval R@5 using adaptmem-tuned MiniLM-L6-v2 as the encoder, brute-force
cosine over per-question vault. No vector DB — fully isolates the encoder
swap from any backend variable.

Comparison:
  - upstream chromadb + MiniLM-L6-v2 (sentence-transformers default):  R@5 = 0.9660
  - our postgres+pgvector + MiniLM-L6-v2:                              R@5 = 0.9660 (parity)
  - this run: same MiniLM-L6-v2 *fine-tuned by adaptmem*:              TBD

The adaptmem model lives at ~/Projects/adaptmem-cache/model/ (90MB,
fine-tuned on a 5000-pair dataset with MultipleNegativesRankingLoss).
Same arch + dim as the default MiniLM — pure weight swap. Brute-force
cosine over the ~50-200 sessions per question removes any retrieval-
algorithm variable.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

ADAPTMEM_MODEL = "/home/jp/Projects/adaptmem-cache/model"
DATA_PATH = Path("/home/jp/Projects/multipass-structural-memory-eval/sme/corpora/longmemeval/data/longmemeval_s_cleaned.json")
OUT_PATH = Path("/home/jp/Projects/multipass-structural-memory-eval/baselines/lme_substrate_adaptmem_2026-05-17.json")


def main() -> int:
    print(f"[{time.time():.0f}] loading sentence-transformers + adaptmem model from {ADAPTMEM_MODEL}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ADAPTMEM_MODEL)
    print(f"[{time.time():.0f}] model loaded, output dim = {model.get_sentence_embedding_dimension()}")

    print(f"[{time.time():.0f}] loading dataset")
    data = json.loads(DATA_PATH.read_text())
    print(f"[{time.time():.0f}] loaded {len(data)} questions")

    per_q: list[dict] = []
    cat_counts: Counter[str] = Counter()
    cat_hits: Counter[str] = Counter()
    t_start = time.time()

    for idx, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        answer_ids = set(entry["answer_session_ids"])

        # Build session docs: user-turns-only, no metadata (upstream protocol)
        session_ids: list[str] = []
        session_docs: list[str] = []
        for session, sess_id in zip(entry["haystack_sessions"], entry["haystack_session_ids"]):
            user_turns = [t["content"] for t in session if t["role"] == "user"]
            if not user_turns:
                continue
            session_ids.append(sess_id)
            session_docs.append("\n".join(user_turns))

        if not session_docs:
            per_q.append({"question_id": qid, "question_type": qtype, "expected": list(answer_ids), "retrieved": [], "hit": 0})
            cat_counts[qtype] += 1
            continue

        # Embed corpus and query, brute-force cosine top-5
        corpus_emb = model.encode(
            session_docs,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_emb = model.encode(
            [question],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        sims = corpus_emb @ query_emb  # normalized → dot = cosine
        top5_idx = np.argsort(-sims)[:5]
        retrieved_ids = [session_ids[i] for i in top5_idx]
        hit = 1 if any(aid in retrieved_ids for aid in answer_ids) else 0

        cat_counts[qtype] += 1
        cat_hits[qtype] += hit
        per_q.append({
            "question_id": qid,
            "question_type": qtype,
            "expected": list(answer_ids),
            "retrieved": retrieved_ids,
            "scores": [float(sims[i]) for i in top5_idx],
            "hit": hit,
        })

        if idx % 25 == 0 or idx == len(data) - 1:
            elapsed = time.time() - t_start
            rate = (idx + 1) / max(elapsed, 1)
            cum = sum(cat_hits.values()) / max(sum(cat_counts.values()), 1)
            print(f"[{idx+1:4d}/{len(data)}] {qid[:20]:20s} qtype={qtype:25s} hit={hit}  cum_R@5={cum:.4f}  rate={rate:.2f}q/s")

    summary = {
        "encoder": ADAPTMEM_MODEL,
        "encoder_dim": model.get_sentence_embedding_dimension(),
        "total": len(per_q),
        "mean_recall_any_at_5": sum(q["hit"] for q in per_q) / max(len(per_q), 1),
        "by_qtype": {
            qt: {"n": cat_counts[qt], "recall_any_at_5": cat_hits[qt] / cat_counts[qt]}
            for qt in sorted(cat_counts)
        },
    }
    OUT_PATH.write_text(json.dumps({"summary": summary, "per_question": per_q}, indent=2))

    print()
    print("=" * 60)
    print("  ENCODER: adaptmem (fine-tuned MiniLM-L6-v2)")
    print(f"  TOTAL: {summary['total']}")
    print(f"  R@5 (recall_any@5): {summary['mean_recall_any_at_5']:.4f}")
    print(f"  Time: {time.time() - t_start:.1f}s")
    print()
    print("  By question_type:")
    for qt, stats in summary["by_qtype"].items():
        print(f"    {qt:30s}  n={stats['n']:3d}  R@5={stats['recall_any_at_5']:.4f}")
    print()
    print(f"  Written: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
