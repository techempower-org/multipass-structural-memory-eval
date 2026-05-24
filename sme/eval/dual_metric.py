"""Dual-metric reporting: R@K retrieval recall AND E2E QA accuracy.

Bridges the gap between SME's substring-based retrieval recall (R@K)
and LongMemEval's judge-scored QA accuracy. The two numbers measure
different things:

- **R@K** — "did the right session appear in the top-K retrieved chunks?"
  Cheap, deterministic, no LLM required. SME's existing scoring path.
- **QA accuracy** — "given the retrieved context, can a reader produce a
  correct answer that the judge accepts?" Requires reader + judge calls.

A system can score high R@K (right session retrieved) but low QA
accuracy (reader couldn't piece together the answer from the chunks).
The **gap** between the two is a structural finding in its own right —
it isolates retrieval quality from answer-generation quality.

The aggregator emits per-category and overall numbers, plus the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def judge_label_to_qa_correct(label: str) -> Optional[bool]:
    """Map an ``autoeval_label`` to a binary QA-correctness signal.

    - ``CORRECT`` / ``ABSTAIN`` → True (ABSTAIN is the success state on
      abstention questions).
    - ``INCORRECT`` / ``PARTIAL`` → False (PARTIAL is graded as wrong
      under LongMemEval's reported accuracy metric).
    - ``ERROR`` / unknown → None (exclude from denominators).
    """
    if label in ("CORRECT", "ABSTAIN"):
        return True
    if label in ("INCORRECT", "PARTIAL"):
        return False
    return None


@dataclass
class _CategorySlot:
    n: int = 0
    sme_recall_sum: float = 0.0
    judge_correct: int = 0
    judge_incorrect: int = 0
    judge_partial: int = 0
    judge_abstain: int = 0
    judge_error: int = 0
    judge_skipped: int = 0

    @property
    def judged(self) -> int:
        return (self.judge_correct + self.judge_incorrect
                + self.judge_partial + self.judge_abstain)

    @property
    def qa_correct(self) -> int:
        return self.judge_correct + self.judge_abstain


@dataclass
class DualMetricRecord:
    """One per-question reading combining retrieval + QA grading.

    Mirrors the shape the cross_validation harness emits so the
    aggregator can consume harness records directly.
    """
    question_id: str
    sme_category: str
    sme_recall: float
    judge_label: Optional[str] = None  # None when the judge was skipped


def _slot_to_summary(slot: _CategorySlot) -> dict[str, Any]:
    n = slot.n
    judged = slot.judged
    sme_recall_mean = slot.sme_recall_sum / n if n else 0.0
    qa_accuracy = slot.qa_correct / judged if judged else None
    gap = (
        round(sme_recall_mean - qa_accuracy, 4)
        if qa_accuracy is not None else None
    )
    return {
        "n": n,
        "n_judged": judged,
        "sme_recall_mean": round(sme_recall_mean, 4),
        "qa_accuracy": round(qa_accuracy, 4) if qa_accuracy is not None else None,
        "retrieval_qa_gap": gap,
        "judge_label_counts": {
            "CORRECT": slot.judge_correct,
            "PARTIAL": slot.judge_partial,
            "INCORRECT": slot.judge_incorrect,
            "ABSTAIN": slot.judge_abstain,
            "ERROR": slot.judge_error,
            "skipped": slot.judge_skipped,
        },
    }


def aggregate_dual_metric(records: list[dict]) -> dict[str, Any]:
    """Compute per-category and overall dual-metric summary.

    Args:
        records: Per-question dicts with at least:
            - ``sme_category``: str
            - ``sme_recall``: float in [0, 1]
            - ``judge``: dict with ``autoeval_label`` key, OR None when
              the judge was skipped.

    Returns:
        Dict with ``per_category``, ``overall``, and ``retrieval_qa_gap``
        (the overall gap: ``sme_recall_mean - qa_accuracy``). Categories
        are reported separately by design — see the KU vs Cat 3 caveat
        in docs/related_work/longmemeval.md.
    """
    by_cat: dict[str, _CategorySlot] = {}
    overall = _CategorySlot()

    for r in records:
        cat = r.get("sme_category", "unmapped")
        slot = by_cat.setdefault(cat, _CategorySlot())

        recall = float(r.get("sme_recall", 0.0) or 0.0)
        slot.n += 1
        slot.sme_recall_sum += recall
        overall.n += 1
        overall.sme_recall_sum += recall

        judge = r.get("judge")
        if judge is None:
            slot.judge_skipped += 1
            overall.judge_skipped += 1
            continue

        label = judge.get("autoeval_label", "ERROR")
        if label == "CORRECT":
            slot.judge_correct += 1
            overall.judge_correct += 1
        elif label == "PARTIAL":
            slot.judge_partial += 1
            overall.judge_partial += 1
        elif label == "INCORRECT":
            slot.judge_incorrect += 1
            overall.judge_incorrect += 1
        elif label == "ABSTAIN":
            slot.judge_abstain += 1
            overall.judge_abstain += 1
        else:
            slot.judge_error += 1
            overall.judge_error += 1

    per_cat = {
        cat: _slot_to_summary(slot)
        for cat, slot in sorted(by_cat.items())
    }
    return {
        "per_category": per_cat,
        "overall": _slot_to_summary(overall),
    }
