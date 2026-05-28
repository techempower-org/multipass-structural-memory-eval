#!/usr/bin/env python3
"""Regex + IDF entity-graph retrieval baseline on LongMemEval (closes #82).

Ports the ML-free baseline from
techempower-org/adaptmem's ``benchmarks/structural_memory_eval/entity_graph_baseline.py``
(2026-05-28 cut, R@5=0.406 on LongMemEval-500). Sets the floor any
KG-based retrieval system must beat. No encoder, no LLM, no
fine-tuning — just regex entity extraction + per-question IDF.

Substrate:
  - For each question, extract entities from query + every haystack session
  - Build inverted index: entity -> {session_ids that contain it}
  - Score each session by IDF-weighted shared-entity count with the query
  - Compare ranked top-K against the question's gold answer_session_ids

Entity extraction (deliberately simple — heuristics over LLM):
  - Capitalised proper noun phrases ("Apache Kafka", "New York", "Susan")
  - Dates (YYYY-MM-DD, YYYY, "March 5", "5 days ago")
  - Numbers with units (5km, 10 years, $50)
  - URLs and emails
  - CamelCase / snake_case / kebab-case technical tokens
  - Quoted strings (2-6 words)

Scoring: per-query IDF computed over the haystack of THIS question only,
because LongMemEval is per-question independent. Session score =
sum(log((N+1)/df(e))) over shared entities e. Ties broken by raw shared
entity count, then session_id (for determinism).

Usage:
    sme-eval/venv/bin/python scripts/regex_idf_baseline.py \\
        --questions sme/corpora/longmemeval/data/longmemeval_oracle.json \\
        --out baselines/regex_idf_baseline_oracle_2026-05-28.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger("regex_idf_baseline")

STOPWORDS: frozenset[str] = frozenset({
    "The", "This", "That", "These", "Those", "There", "Their", "They",
    "What", "When", "Where", "Why", "How", "Who", "Which",
    "Hello", "Hi", "Hey", "Thanks", "Thank", "Please", "Sure", "Okay",
    "Yes", "No", "Maybe", "Some", "Any", "All", "Each", "Every",
    "User", "Assistant", "Bot", "AI", "Human",
    "Could", "Would", "Should", "Will", "Shall", "May", "Might",
    "Have", "Has", "Had", "Was", "Were", "Been", "Being",
    "Get", "Got", "Make", "Made", "Take", "Took", "Give", "Gave",
    "Know", "Knew", "Think", "Thought", "See", "Saw", "Say", "Said",
    "I", "You", "He", "She", "It", "We", "Me", "Him", "Her", "Us", "Them",
    "Mr", "Ms", "Mrs", "Dr",
    "Like", "Want", "Need", "Use", "Used", "Try", "Tried",
})

PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]+){0,3})\b")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*|"
    r"\d+\s+(?:year|month|week|day|hour|minute)s?\s+ago)\b",
    re.IGNORECASE,
)
NUM_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?\s*(?:km|mi|kg|lb|cm|mm|ft|in|m|g|mg|"
    r"hour|min|sec|year|month|day|week|"
    r"%|usd|eur|gbp|tl|jpy|inr|cny))\b",
    re.IGNORECASE,
)
DOLLAR_RE = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d+)?")
URL_RE = re.compile(r"https?://\S+|\b\w+@\w+\.\w+")
TECH_TOKEN_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|"
    r"[a-z]+_[a-z]+(?:_[a-z]+)*|"
    r"[a-z]+-[a-z]+(?:-[a-z]+)*)\b"
)
QUOTED_RE = re.compile(r'"([^"]{3,80})"|\'([^\']{3,80})\'')


def extract_entities(text: str) -> set[str]:
    """Heuristic entity extraction — proper nouns, dates, num+units, etc."""
    ents: set[str] = set()
    for m in PROPER_NOUN_RE.findall(text):
        m = m.strip()
        if m and m not in STOPWORDS and len(m) >= 3:
            ents.add(m.lower())
    for m in DATE_RE.findall(text):
        ents.add(m.lower())
    for m in NUM_UNIT_RE.findall(text):
        ents.add(m.lower().replace(" ", ""))
    for m in DOLLAR_RE.findall(text):
        ents.add(m.lower())
    for m in URL_RE.findall(text):
        ents.add(m.lower())
    for m in TECH_TOKEN_RE.findall(text):
        if len(m) >= 4:
            ents.add(m.lower())
    for a, b in QUOTED_RE.findall(text):
        s = (a or b).strip().lower()
        if 8 <= len(s) <= 80:
            ents.add(s)
    return ents


def session_text(session_turns: list[dict]) -> str:
    """Concatenate all turns of a session into a single string."""
    return "\n".join(
        f"{t.get('role', '')}: {t.get('content', '')}" for t in session_turns
    )


def score_one_question(q: dict) -> dict:
    """Score one LongMemEval question; returns the per-q record."""
    query = q["question"]
    gold = set(q.get("answer_session_ids") or [])
    session_ids: list[str] = q["haystack_session_ids"]
    sessions: list[list[dict]] = q["haystack_sessions"]
    n_sessions = len(sessions)

    sess_ents = [extract_entities(session_text(s)) for s in sessions]
    df: Counter[str] = Counter()
    for ents in sess_ents:
        for e in ents:
            df[e] += 1

    q_ents = extract_entities(query)

    scores: list[tuple[str, float, int]] = []
    for sid, ents in zip(session_ids, sess_ents):
        shared = q_ents & ents
        if not shared:
            scores.append((sid, 0.0, 0))
            continue
        idf_sum = sum(math.log((n_sessions + 1) / df[e]) for e in shared)
        scores.append((sid, idf_sum, len(shared)))

    scores.sort(key=lambda x: (-x[1], -x[2], x[0]))
    ranked_ids = [s[0] for s in scores]

    hit_at: dict[int, int] = {}
    for k in (1, 5, 10):
        topk = set(ranked_ids[:k])
        hit_at[k] = int(bool(gold) and bool(gold & topk))

    return {
        "question_id": q["question_id"],
        "question_type": q["question_type"],
        "n_query_entities": len(q_ents),
        "gold": list(gold),
        "ranked_top10": ranked_ids[:10],
        "top1_score": round(scores[0][1], 4) if scores else 0.0,
        "hit_at_1": hit_at[1],
        "hit_at_5": hit_at[5],
        "hit_at_10": hit_at[10],
    }


def aggregate(per_q: list[dict]) -> dict:
    """Aggregate per-question records into a summary block."""
    n = len(per_q)
    if n == 0:
        return {"n": 0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "by_type": {}}
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "h1": 0, "h5": 0, "h10": 0}
    )
    for r in per_q:
        b = by_type[r["question_type"]]
        b["n"] += 1
        b["h1"] += r["hit_at_1"]
        b["h5"] += r["hit_at_5"]
        b["h10"] += r["hit_at_10"]
    return {
        "n": n,
        "R@1": round(sum(r["hit_at_1"] for r in per_q) / n, 4),
        "R@5": round(sum(r["hit_at_5"] for r in per_q) / n, 4),
        "R@10": round(sum(r["hit_at_10"] for r in per_q) / n, 4),
        "by_type": {
            t: {
                "n": b["n"],
                "R@1": round(b["h1"] / b["n"], 4),
                "R@5": round(b["h5"] / b["n"], 4),
                "R@10": round(b["h10"] / b["n"], 4),
            }
            for t, b in sorted(by_type.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--questions", required=True, type=Path,
        help="LongMemEval questions JSON (oracle / s / m split).",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Output JSON path.",
    )
    p.add_argument(
        "--progress-every", type=int, default=50,
        help="Log progress every N questions (default: 50).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log.info("loading questions from %s", args.questions)
    data = json.loads(args.questions.read_text())
    log.info("  %d questions loaded", len(data))

    t0 = time.time()
    per_q: list[dict] = []
    for i, q in enumerate(data):
        per_q.append(score_one_question(q))
        if (i + 1) % args.progress_every == 0:
            el = time.time() - t0
            eta = el * (len(data) - i - 1) / (i + 1)
            log.info("  q%d/%d  el=%.1fs  eta=%.0fs", i + 1, len(data), el, eta)

    summary = aggregate(per_q)
    report = {
        "experiment": "regex+IDF entity-graph baseline (ML-free retrieval floor)",
        "method": (
            "regex entity extraction (proper nouns, dates, num+unit, "
            "tech tokens, quoted) + per-question IDF over haystack"
        ),
        "source": str(args.questions),
        "runtime_s": round(time.time() - t0, 1),
        **summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": report, "per_q": per_q}, indent=2))

    log.info("==========  R@1=%.4f  R@5=%.4f  R@10=%.4f  (n=%d) ==========",
             report["R@1"], report["R@5"], report["R@10"], report["n"])
    for t, b in sorted(report["by_type"].items()):
        log.info("  %-22s n=%-4d  R@1=%.4f  R@5=%.4f  R@10=%.4f",
                 t, b["n"], b["R@1"], b["R@5"], b["R@10"])
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
