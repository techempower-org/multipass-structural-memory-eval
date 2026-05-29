#!/usr/bin/env python3
"""Daemon-independent encoder-swap A/B on the jp-realm-v0.1 corpus.

#84 — tests the Tau2 prediction of a +30-33pp recall gap on jp-realm when
swapping the retrieval encoder. The standard jp-realm path (`sme-eval retrieve`)
runs retrieval SERVER-SIDE inside the palace-daemon at familiar:8085, so the
encoder cannot be swapped locally. This script instead reconstructs a fully
LOCAL haystack from a ChromaDB palace BACKUP (no daemon, no network), embeds it
with a chosen SentenceTransformer, and scores the 30 jp-realm questions exactly
the way `cmd_retrieve` does.

Methodology (matches the repo's brute-force-cosine encoder-swap convention,
cf. scripts/lme_substrate_adaptmem_bench.py):
  1. Load drawer documents from the local ChromaDB backup (chroma.sqlite3).
  2. Embed the whole drawer corpus ONCE with the chosen encoder (GPU).
  3. For each question, cosine-rank drawers, take top-K, concatenate their text
     into a `context_string`.
  4. Score: recall = |expected_sources ∩ context_string| / |expected_sources|,
     hit = recall > 0 — identical to sme/cli.py cmd_retrieve.

Run the same way with two encoders to get the A/B:
    # baseline (upstream default MiniLM)
    jp_realm_encoder_swap.py --model all-MiniLM-L6-v2 \
        --json baselines/jp_realm_encoder_swap_default_<date>.json
    # FT-300-approx (LongMemEval fine-tune — see train_ft300_approx.py)
    jp_realm_encoder_swap.py --model baselines/ft300_approx_model \
        --json baselines/jp_realm_encoder_swap_ft300_<date>.json

Snapshot caveat: the backup is frozen at its creation date; topics newer than
the snapshot (e.g. palace-daemon) won't be retrievable regardless of encoder.
Such questions are flagged `snapshot_uncovered` (no drawer in the corpus
contains any expected_source) so they can be excluded from the encoder delta.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.stdout.reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = _REPO_ROOT / "sme" / "corpora" / "jp_realm_v0_1" / "questions.yaml"
DEFAULT_BACKUP = (
    Path.home() / ".mempalace" / "palace-backup-20260416-110359" / "chroma.sqlite3"
)
DEFAULT_TOP_K = 10  # context window of retrieved drawers, matching n_results default


def load_drawers(backup: Path, limit: int | None) -> tuple[list[str], list[str]]:
    """Return (documents, drawer_labels) from a ChromaDB backup sqlite."""
    con = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    cur = con.cursor()
    # chroma:document holds drawer text; join wing/room for a readable label.
    cur.execute(
        """
        SELECT em_doc.string_value AS document,
               em_wing.string_value AS wing,
               em_room.string_value AS room
        FROM embedding_metadata em_doc
        LEFT JOIN embedding_metadata em_wing
               ON em_wing.id = em_doc.id AND em_wing.key = 'wing'
        LEFT JOIN embedding_metadata em_room
               ON em_room.id = em_doc.id AND em_room.key = 'room'
        WHERE em_doc.key = 'chroma:document'
          AND em_doc.string_value IS NOT NULL
        """
        + (f" LIMIT {int(limit)}" if limit else "")
    )
    docs: list[str] = []
    labels: list[str] = []
    for document, wing, room in cur.fetchall():
        if not document:
            continue
        docs.append(document)
        labels.append(f"{wing or '?'}/{room or '?'}")
    con.close()
    return docs, labels


def load_questions() -> list[dict]:
    doc = yaml.safe_load(QUESTIONS.read_text())
    return doc.get("questions", [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="SentenceTransformer id or local path")
    ap.add_argument("--backup", type=Path, default=DEFAULT_BACKUP,
                    help="ChromaDB palace backup sqlite (local, no daemon)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help="Drawers concatenated into the context window")
    ap.add_argument("--limit-drawers", type=int, default=None,
                    help="Cap corpus size for a smoke run")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.backup.exists():
        print(f"ERROR: backup not found: {args.backup}", file=sys.stderr)
        return 2

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.time():.0f}] loading encoder {args.model} on {device}")
    model = SentenceTransformer(args.model, device=device)

    print(f"[{time.time():.0f}] loading drawers from {args.backup}")
    docs, labels = load_drawers(args.backup, args.limit_drawers)
    print(f"[{time.time():.0f}] {len(docs)} drawers loaded")

    questions = load_questions()
    print(f"[{time.time():.0f}] {len(questions)} jp-realm questions")

    t0 = time.time()
    corpus_emb = model.encode(
        docs, batch_size=args.batch, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    print(f"[{time.time():.0f}] corpus embedded in {time.time()-t0:.1f}s "
          f"({corpus_emb.shape})")

    q_texts = [q.get("text", "") for q in questions]
    q_emb = model.encode(
        q_texts, batch_size=args.batch, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    k_cutoffs = sorted({1, 5, 10, args.top_k})
    per_question: list[dict] = []
    for q, qe in zip(questions, q_emb):
        expected = q.get("expected_sources", []) or []
        # Is ANY expected_source present anywhere in the corpus? If not, this
        # question is uncovered by the snapshot — encoder can't be blamed.
        snapshot_uncovered = not any(
            any(src in d for src in expected) for d in docs
        )
        sims = corpus_emb @ qe
        ranked = np.argsort(-sims)
        # expected_sources recall at each K cutoff: fraction of expected
        # substrings present in the concatenated top-K drawer context
        # (same semantics as sme/cli.py cmd_retrieve, evaluated per K).
        recall_at: dict[int, float] = {}
        matched_at: dict[int, list[str]] = {}
        for k in k_cutoffs:
            top_idx = ranked[:k]
            ctx = "\n".join(docs[i] for i in top_idx)
            m = [src for src in expected if src in ctx]
            recall_at[k] = len(m) / len(expected) if expected else 0.0
            matched_at[k] = m
        top_idx = ranked[: args.top_k]
        per_question.append({
            "id": q.get("id", "?"),
            "text": q.get("text", ""),
            "min_hops": q.get("min_hops", 0),
            "expected_sources": expected,
            "matched_sources": matched_at[args.top_k],
            "recall_at_k": {str(k): round(recall_at[k], 4) for k in k_cutoffs},
            "recall": recall_at[args.top_k],
            "hit": recall_at[args.top_k] > 0,
            "snapshot_uncovered": snapshot_uncovered,
            "top_drawers": [labels[i] for i in top_idx],
            "top_scores": [round(float(sims[i]), 4) for i in top_idx],
        })

    # Aggregates — overall and covered-only (excludes snapshot_uncovered).
    def _agg(rs: list[dict]) -> dict:
        n = len(rs)
        if n == 0:
            return {"n": 0}
        recall_at = {
            f"R@{k}": round(sum(r["recall_at_k"][str(k)] for r in rs) / n, 4)
            for k in k_cutoffs
        }
        return {
            "n": n,
            **recall_at,
            "mean_recall": round(sum(r["recall"] for r in rs) / n, 4),
            "hit_rate": round(sum(r["hit"] for r in rs) / n, 4),
            "full_recall": sum(1 for r in rs if r["recall"] >= 1.0),
        }

    covered = [r for r in per_question if not r["snapshot_uncovered"]]
    by_hop: dict[int, list[dict]] = {}
    for r in per_question:
        by_hop.setdefault(r["min_hops"], []).append(r)

    summary = {
        "model": args.model,
        "encoder_dim": int(corpus_emb.shape[1]),
        "n_drawers": len(docs),
        "top_k": args.top_k,
        "overall": _agg(per_question),
        "covered_only": _agg(covered),
        "snapshot_uncovered_ids": [r["id"] for r in per_question if r["snapshot_uncovered"]],
        "by_hop": {str(h): _agg(rs) for h, rs in sorted(by_hop.items())},
        "wall_clock_seconds": round(time.time() - t0, 1),
    }

    report = {"summary": summary, "per_question": per_question}

    print()
    print("=" * 64)
    print(f"  encoder: {args.model}  (dim={summary['encoder_dim']})")
    print(f"  drawers: {len(docs)}  top_k: {args.top_k}")
    o, c = summary["overall"], summary["covered_only"]
    _rk = lambda blk: "  ".join(f"R@{k}={blk[f'R@{k}']:.4f}" for k in k_cutoffs)  # noqa: E731
    print(f"  overall:      n={o['n']:2}  {_rk(o)}  "
          f"hit_rate={o['hit_rate']:.4f}  full={o['full_recall']}")
    print(f"  covered-only: n={c['n']:2}  {_rk(c)}  "
          f"hit_rate={c['hit_rate']:.4f}  full={c['full_recall']}")
    if summary["snapshot_uncovered_ids"]:
        print(f"  snapshot-uncovered ({len(summary['snapshot_uncovered_ids'])}): "
              f"{summary['snapshot_uncovered_ids']}")
    miss = Counter(r["id"] for r in covered if not r["hit"])
    if miss:
        print(f"  covered misses: {list(miss)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
