#!/usr/bin/env python3
"""Reconstruct the AdaptMem "FT-300" LongMemEval encoder locally (approximate).

#84 — the published `nakata-app/minilm-lme-ft-300` artifact is a PRIVATE HF
repo (API 401) and its weights are NOT on this workstation; only the
config/tokenizer skeleton ships in the adaptmem repo. This script rebuilds
a faithful *approximation* of FT-300 from the local source files so the
encoder swap on jp-realm (and LongMemEval) can be run without the Mac mini
that produced the original.

What FT-300 is
--------------
A MultipleNegativesRankingLoss fine-tune of `sentence-transformers/all-MiniLM-L6-v2`
on LongMemEval-S *train* queries. Pairs are (question, gold_session_text). The
encoder has NEVER seen jp-realm — that is exactly what #84 tests (does a
LongMemEval-trained encoder generalise to JP's personal palace?).

Why this is APPROXIMATE, not bit-exact
---------------------------------------
The original `ft-300-base` model card reports `dataset_size: 565`. Locally we can
rebuild:
  - 203 base pairs (100 LongMemEval-S train queries × multi-gold sessions)
  - 264 synthetic-preference paraphrases (`s2_syn_preferences.jsonl`)
  = 467 pairs
The remaining ~98 pairs came from `s2_syn_all.jsonl` (5448 Sprint-2 synthetic
paraphrases), which was never committed and is absent on this workstation. The
reconstructed encoder is therefore labelled **FT-300-approx (467 pairs)** in all
outputs and MUST NOT be presented as the published artifact.

Recipe (from adaptmem `train_biencoder_ftv4.ipynb` + `s3_build_biencoder_pairs.py`):
  base model = all-MiniLM-L6-v2, MNR loss, batch 64, lr 2e-5, 2 epochs, seed 42.

Inputs (all present on this workstation):
  - split:  ~/Projects/adaptmem/benchmarks/data/split_ids_100_400.json
  - run5:   ~/Projects/adaptmem/benchmarks/v335/run5_v335_hybrid_v4_ft300.jsonl
  - gold:   sme/corpora/longmemeval/data/longmemeval_s_cleaned.json
  - synpref:~/Projects/adaptmem/results/sprint_0p99/s2_syn_preferences.jsonl

Output: a SentenceTransformer dir (default ./baselines/ft300_approx_model/),
gitignored by size — the bench scripts load it by path.

Usage:
    ./venv/bin/python scripts/train_ft300_approx.py \
        --out baselines/ft300_approx_model \
        --epochs 2 --batch 64 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTMEM = Path.home() / "Projects" / "adaptmem"
SPLIT = ADAPTMEM / "benchmarks" / "data" / "split_ids_100_400.json"
RUN5 = ADAPTMEM / "benchmarks" / "v335" / "run5_v335_hybrid_v4_ft300.jsonl"
SYN_PREF = ADAPTMEM / "results" / "sprint_0p99" / "s2_syn_preferences.jsonl"

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_DOC_CHARS = 1800

# longmemeval_s_cleaned.json (277MB) is gitignored — it lives once in the
# canonical checkout, not in worktrees. Fall back to it when the local copy
# is absent so the builder works from any worktree.
_GOLD_REL = Path("sme/corpora/longmemeval/data/longmemeval_s_cleaned.json")
_GOLD_CANONICAL = Path.home() / "Projects" / "multipass-structural-memory-eval" / _GOLD_REL


def _resolve_gold() -> Path:
    local = _REPO_ROOT / _GOLD_REL
    if local.exists():
        return local
    if _GOLD_CANONICAL.exists():
        return _GOLD_CANONICAL
    return local  # let build_pairs() raise the clear FileNotFoundError


def build_pairs(seed: int) -> tuple[list[dict], dict]:
    """Reconstruct the FT-300 base + syn-preference training pairs.

    Mirrors adaptmem's s3_build_biencoder_pairs.py, minus the missing
    s2_syn_all.jsonl source. Returns (pairs, provenance).
    """
    gold_path = _resolve_gold()
    for p in (SPLIT, RUN5, gold_path):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")

    split = json.loads(SPLIT.read_text())
    train_qids = set(split["train_question_ids"])
    gold_map = {r["question_id"]: r for r in json.loads(gold_path.read_text())}

    run5: dict[str, dict] = {}
    with RUN5.open() as f:
        for line in f:
            r = json.loads(line)
            run5[r["question_id"]] = r

    # ---- Source 1: 100 LongMemEval-S train queries → ~203 positive pairs ----
    base: list[dict] = []
    for qid in train_qids:
        g = gold_map.get(qid)
        if not g:
            continue
        ans = g.get("answer_session_ids") or []
        if not ans:
            continue
        items_by_id = (
            {it["corpus_id"]: it for it in run5[qid]["retrieval_results"]["ranked_items"]}
            if qid in run5 else {}
        )
        haystack = dict(zip(g.get("haystack_session_ids", []), g.get("haystack_sessions", []) or []))
        for gid in ans:
            text = None
            if gid in items_by_id:
                text = items_by_id[gid]["text"]
            elif gid in haystack:
                s = haystack[gid]
                if isinstance(s, list):
                    text = "\n".join(
                        (t.get("content", "") if isinstance(t, dict) else str(t)) for t in s
                    )
                else:
                    text = str(s)
            if text:
                base.append({
                    "q": g["question"], "doc": text[:MAX_DOC_CHARS],
                    "orig_qid": qid, "qtype": g["question_type"], "source": "base",
                })

    # ---- Source 2: synthetic preference paraphrases (present locally) ----
    synpref: list[dict] = []
    if SYN_PREF.exists():
        for line in SYN_PREF.open():
            r = json.loads(line)
            synpref.append({
                "q": r["syn_q"], "doc": r["gold_text"][:MAX_DOC_CHARS],
                "orig_qid": r["orig_qid"], "qtype": "single-session-preference",
                "source": "syn_pref",
            })

    pairs = base + synpref
    rng = random.Random(seed)
    rng.shuffle(pairs)

    provenance = {
        "base_pairs": len(base),
        "syn_pref_pairs": len(synpref),
        "total_pairs": len(pairs),
        "missing_source": "s2_syn_all.jsonl (Sprint-2 synthetic, not on workstation)",
        "original_card_size": 565,
        "label": "FT-300-approx",
        "by_qtype": dict(Counter(p["qtype"] for p in pairs)),
    }
    return pairs, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "baselines" / "ft300_approx_model")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build + report pairs only; skip training.")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    pairs, prov = build_pairs(args.seed)
    print(f"[pairs] base={prov['base_pairs']} syn_pref={prov['syn_pref_pairs']} "
          f"total={prov['total_pairs']} (original card: {prov['original_card_size']})")
    print(f"[pairs] by qtype: {prov['by_qtype']}")
    print(f"[pairs] label: {prov['label']}  missing: {prov['missing_source']}")
    if args.dry_run:
        print("[dry-run] skipping training.")
        return 0

    import numpy as np
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    examples = [InputExample(texts=[p["q"], p["doc"]]) for p in pairs]
    model = SentenceTransformer(BASE_MODEL, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(0.1 * len(loader) * args.epochs)
    print(f"[train] examples={len(examples)} epochs={args.epochs} batch={args.batch} "
          f"lr={args.lr} steps/epoch={len(loader)} warmup={warmup}")

    t0 = time.time()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    elapsed = time.time() - t0
    (args.out / "ft300_approx_provenance.json").write_text(
        json.dumps({**prov, "epochs": args.epochs, "batch": args.batch,
                    "lr": args.lr, "seed": args.seed, "base_model": BASE_MODEL,
                    "train_seconds": round(elapsed, 1)}, indent=2)
    )
    print(f"[train] saved → {args.out}  ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
