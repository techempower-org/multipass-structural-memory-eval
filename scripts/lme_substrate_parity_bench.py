#!/usr/bin/env python3
"""LongMemEval R@5 against our postgres+pgvector backend, ingest mirroring
upstream's raw-mode protocol exactly: one doc per session = user-turns
concatenated by newline, no markdown, no frontmatter, no assistant turns.

Comparison vs:
  - upstream raw (chromadb default):                R@5 = 0.966 (just reproduced)
  - SME loader rendering (full markdown):            R@5 = 0.944 (just measured)
  - this run (upstream content, our backend):        TBD

If TBD ≈ 0.966, the SME loader's markdown rendering accounts for the gap.
If TBD < 0.966, the gap is real backend swap regression on our fork.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

# Force line-buffered stdout so progress shows immediately when piped through tee.
sys.stdout.reconfigure(line_buffering=True)

from sme.adapters.postgres_ingest import PostgresIngestAdapter  # noqa: E402


def main() -> int:
    print(f"[{time.time():.0f}] start, importing modules")
    data_path = "/home/jp/Projects/multipass-structural-memory-eval/sme/corpora/longmemeval/data/longmemeval_s_cleaned.json"
    print(f"[{time.time():.0f}] loading dataset {data_path}")
    data = json.loads(Path(data_path).read_text())
    print(f"[{time.time():.0f}] loaded {len(data)} questions")

    print(f"[{time.time():.0f}] creating PostgresIngestAdapter")
    adapter = PostgresIngestAdapter(n_results=5)
    print(f"[{time.time():.0f}] adapter ready, starting bench")
    t_start = time.time()

    per_q: list[dict] = []
    cat_counts = Counter()
    cat_hits = Counter()

    for idx, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        answer_ids = set(entry["answer_session_ids"])

        # Build ingest exactly like upstream raw-mode session granularity:
        # one doc per session = concatenated user turns
        corpus: list[dict] = []
        for session, sess_id in zip(entry["haystack_sessions"], entry["haystack_session_ids"]):
            user_turns = [t["content"] for t in session if t["role"] == "user"]
            if not user_turns:
                continue
            corpus.append({"id": sess_id, "document": "\n".join(user_turns)})

        adapter.ingest_corpus(corpus)
        result = adapter.query(question, n_results=5)

        retrieved_ids = [
            (e.id.replace("chunk:", "") if e.id.startswith("chunk:") else e.id)
            for e in (result.retrieved_entities or [])
        ]
        top5 = set(retrieved_ids[:5])
        hit = 1 if any(aid in top5 for aid in answer_ids) else 0

        cat_counts[qtype] += 1
        cat_hits[qtype] += hit

        per_q.append({
            "question_id": qid,
            "question_type": qtype,
            "expected": list(answer_ids),
            "retrieved": retrieved_ids[:5],
            "hit": hit,
        })

        if idx % 25 == 0 or idx == len(data) - 1:
            elapsed = time.time() - t_start
            rate = (idx + 1) / max(elapsed, 1)
            cum = sum(cat_hits.values()) / max(sum(cat_counts.values()), 1)
            print(f"[{idx+1:4d}/{len(data)}] {qid[:20]:20s} qtype={qtype:25s} hit={hit}  cum_R@5={cum:.4f}  rate={rate:.2f}q/s")

    out_path = "/tmp/sme_pg_lme_upstream_parity.json"
    summary = {
        "total": len(per_q),
        "mean_recall_any_at_5": sum(q["hit"] for q in per_q) / len(per_q),
        "by_qtype": {
            qt: {"n": cat_counts[qt], "recall_any_at_5": cat_hits[qt] / cat_counts[qt]}
            for qt in sorted(cat_counts)
        },
    }
    Path(out_path).write_text(json.dumps({"summary": summary, "per_question": per_q}, indent=2))

    print()
    print("=" * 60)
    print(f"  TOTAL: {summary['total']}")
    print(f"  R@5 (recall_any@5): {summary['mean_recall_any_at_5']:.4f}")
    print(f"  Time: {time.time() - t_start:.1f}s")
    print()
    print("  By question_type:")
    for qt, stats in summary["by_qtype"].items():
        print(f"    {qt:30s}  n={stats['n']:3d}  R@5={stats['recall_any_at_5']:.4f}")
    print()
    print(f"  Written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
