#!/usr/bin/env python
"""HotpotQA retrieval smoke — Cat 2c multi-hop recall at the smallest slice.

Runs a dependency-free lexical (token-overlap / IDF) retrieval over the
smallest slice of the HotpotQA dev distractor split and reports multi-hop
recall: for each question, did the top-K retrieved paragraphs include BOTH
annotated gold paragraphs? This is the same shape the SME Cat 2c reading
takes, exercised end-to-end through the loader without needing chromadb or an
embedding model — a true retrieval smoke, not just a parse check.

The full dev split (hotpot_dev_distractor_v1.json, ~44 MB, CC BY-SA 4.0) is
gitignored; download per sme/corpora/hotpotqa/README.md. This script reads
that file if present and otherwise falls back to the inline test fixture so
it always runs.

Usage:
    python scripts/hotpotqa_retrieval_smoke.py            # auto-detect data
    python scripts/hotpotqa_retrieval_smoke.py --n 50     # first 50 questions
    python scripts/hotpotqa_retrieval_smoke.py --k 5      # top-5 retrieval
    python scripts/hotpotqa_retrieval_smoke.py --data path/to/hotpot.json
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

from sme.corpora.hotpotqa import HotpotQuestion, load_questions

_WORD_RE = re.compile(r"[a-z0-9]+")

# Default location the README's download instructions write to.
_DEFAULT_DATA = (
    Path(__file__).resolve().parent.parent
    / "sme/corpora/hotpotqa/data/hotpot_dev_distractor_v1.json"
)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _idf_retrieve(question: HotpotQuestion, k: int) -> list[str]:
    """Rank the question's context paragraphs by IDF-weighted token overlap
    with the question, returning the top-k paragraph titles.

    A minimal lexical retriever — enough to demonstrate the loader feeds a
    retriever correctly and that multi-hop recall is measurable. The real
    Cat 2c run uses the daemon/flat adapter; this is the smoke.
    """
    paragraphs = question.paragraphs
    n_docs = len(paragraphs) or 1
    # document frequency per token across this question's paragraphs
    df: Counter[str] = Counter()
    para_tokens: list[set[str]] = []
    for p in paragraphs:
        toks = set(_tokens(p.title + " " + p.text))
        para_tokens.append(toks)
        for t in toks:
            df[t] += 1

    q_tokens = _tokens(question.question)
    scored: list[tuple[float, str]] = []
    for p, toks in zip(paragraphs, para_tokens):
        score = 0.0
        for t in q_tokens:
            if t in toks:
                idf = math.log((n_docs + 1) / (df[t] + 0.5))
                score += idf
        scored.append((score, p.title))
    scored.sort(key=lambda s: -s[0])
    return [title for _, title in scored[:k]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--n", type=int, default=20, help="questions to score")
    ap.add_argument("--k", type=int, default=5, help="top-k retrieval")
    args = ap.parse_args()

    data_path = args.data or (_DEFAULT_DATA if _DEFAULT_DATA.exists() else None)
    if data_path is None:
        print(
            "no dev split found at "
            f"{_DEFAULT_DATA} — download per sme/corpora/hotpotqa/README.md; "
            "falling back to the inline fixture for a parse-only smoke."
        )
        from tests.test_hotpotqa_loader import FIXTURE  # type: ignore
        import json
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "fixture.json"
        tmp.write_text(json.dumps(FIXTURE))
        data_path = tmp

    questions = []
    for q in load_questions(data_path):
        questions.append(q)
        if len(questions) >= args.n:
            break

    print(f"HotpotQA retrieval smoke — {data_path}")
    print(f"scoring {len(questions)} questions, top-{args.k} lexical retrieval\n")

    full_recall = 0      # both gold paragraphs in top-k
    partial_recall = 0   # at least one gold paragraph in top-k
    by_type: Counter[str] = Counter()
    by_type_full: Counter[str] = Counter()
    for q in questions:
        retrieved = set(_idf_retrieve(q, args.k))
        gold = set(q.gold_titles)
        hit = gold & retrieved
        by_type[q.qtype] += 1
        if gold and gold <= retrieved:
            full_recall += 1
            by_type_full[q.qtype] += 1
        if hit:
            partial_recall += 1

    n = len(questions)
    print(f"  multi-hop recall (both gold paragraphs in top-{args.k}): "
          f"{full_recall}/{n} = {full_recall / n:.1%}")
    print(f"  partial recall (>=1 gold paragraph):                    "
          f"{partial_recall}/{n} = {partial_recall / n:.1%}")
    print("  by type:")
    for t in sorted(by_type):
        print(f"    {t:12} {by_type_full[t]}/{by_type[t]} full multi-hop recall")

    # Smoke gate: lexical retrieval must surface at least one gold paragraph
    # for the overwhelming majority — if it can't, the loader is feeding the
    # retriever garbage.
    ok = n > 0 and partial_recall / n >= 0.9
    print(f"\n  smoke: {'PASS' if ok else 'FAIL'} "
          f"(partial recall >= 90%)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
