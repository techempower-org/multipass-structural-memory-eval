"""Tests for sme.eval.dual_metric.

Pure aggregation — no LLM, no adapter. Verifies the per-category and
overall numbers, the retrieval/QA gap, and the judge-skipped path.
"""
from __future__ import annotations

import pytest

from sme.eval.dual_metric import (
    aggregate_dual_metric,
    judge_label_to_qa_correct,
)


def _rec(qid, cat, recall, label=None):
    """Build a minimal record matching the harness shape."""
    rec = {"question_id": qid, "sme_category": cat, "sme_recall": recall}
    if label is None:
        rec["judge"] = None
    else:
        rec["judge"] = {"autoeval_label": label}
    return rec


# --- label mapping ---------------------------------------------------------

def test_judge_label_to_qa_correct_mapping():
    assert judge_label_to_qa_correct("CORRECT") is True
    assert judge_label_to_qa_correct("ABSTAIN") is True
    assert judge_label_to_qa_correct("INCORRECT") is False
    assert judge_label_to_qa_correct("PARTIAL") is False
    assert judge_label_to_qa_correct("ERROR") is None
    assert judge_label_to_qa_correct("MAYBE") is None


# --- aggregation basics ----------------------------------------------------

def test_aggregate_empty_records():
    out = aggregate_dual_metric([])
    assert out["per_category"] == {}
    assert out["overall"]["n"] == 0
    assert out["overall"]["sme_recall_mean"] == 0.0
    assert out["overall"]["qa_accuracy"] is None
    assert out["overall"]["retrieval_qa_gap"] is None


def test_aggregate_single_correct_record():
    out = aggregate_dual_metric([_rec("q1", "cat_1", 1.0, "CORRECT")])
    cat = out["per_category"]["cat_1"]
    assert cat["n"] == 1
    assert cat["n_judged"] == 1
    assert cat["sme_recall_mean"] == 1.0
    assert cat["qa_accuracy"] == 1.0
    assert cat["retrieval_qa_gap"] == 0.0
    assert cat["judge_label_counts"]["CORRECT"] == 1


def test_aggregate_gap_positive_when_recall_high_qa_low():
    """SME retrieves the right thing but the reader can't answer — the
    classic retrieval/QA gap that motivates the dual-metric pipeline."""
    records = [
        _rec("q1", "cat_1", 1.0, "INCORRECT"),
        _rec("q2", "cat_1", 1.0, "INCORRECT"),
        _rec("q3", "cat_1", 1.0, "CORRECT"),
        _rec("q4", "cat_1", 1.0, "CORRECT"),
    ]
    out = aggregate_dual_metric(records)
    cat = out["per_category"]["cat_1"]
    assert cat["sme_recall_mean"] == 1.0
    assert cat["qa_accuracy"] == 0.5
    assert cat["retrieval_qa_gap"] == pytest.approx(0.5)


def test_aggregate_gap_negative_when_recall_low_qa_high():
    """Reader rescued the answer despite poor retrieval — gap goes negative."""
    records = [
        _rec("q1", "cat_2c", 0.0, "CORRECT"),
        _rec("q2", "cat_2c", 0.0, "CORRECT"),
    ]
    out = aggregate_dual_metric(records)
    cat = out["per_category"]["cat_2c"]
    assert cat["sme_recall_mean"] == 0.0
    assert cat["qa_accuracy"] == 1.0
    assert cat["retrieval_qa_gap"] == pytest.approx(-1.0)


def test_aggregate_partial_counted_as_qa_incorrect():
    """LongMemEval reports accuracy = CORRECT / judged — PARTIAL is wrong."""
    records = [
        _rec("q1", "cat_1", 0.5, "PARTIAL"),
        _rec("q2", "cat_1", 0.5, "PARTIAL"),
        _rec("q3", "cat_1", 0.5, "CORRECT"),
    ]
    out = aggregate_dual_metric(records)
    cat = out["per_category"]["cat_1"]
    assert cat["judge_label_counts"]["PARTIAL"] == 2
    assert cat["qa_accuracy"] == pytest.approx(1 / 3, abs=1e-3)


def test_aggregate_abstain_counted_as_qa_correct():
    records = [
        _rec("q_abs1", "cat_1_negative", 0.0, "ABSTAIN"),
        _rec("q_abs2", "cat_1_negative", 0.0, "ABSTAIN"),
        _rec("q_abs3", "cat_1_negative", 0.0, "INCORRECT"),
    ]
    out = aggregate_dual_metric(records)
    cat = out["per_category"]["cat_1_negative"]
    assert cat["qa_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert cat["judge_label_counts"]["ABSTAIN"] == 2


def test_aggregate_error_excluded_from_qa_denominator():
    """ERROR is judge call failure, not a real verdict — drop from rate."""
    records = [
        _rec("q1", "cat_1", 1.0, "CORRECT"),
        _rec("q2", "cat_1", 1.0, "ERROR"),
    ]
    out = aggregate_dual_metric(records)
    cat = out["per_category"]["cat_1"]
    assert cat["n"] == 2
    assert cat["n_judged"] == 1  # error excluded
    assert cat["qa_accuracy"] == 1.0
    assert cat["judge_label_counts"]["ERROR"] == 1


def test_aggregate_skipped_judge_yields_none_qa_accuracy():
    """When skip_judge=True the harness emits judge=None for every record."""
    records = [
        _rec("q1", "cat_1", 1.0),  # judge=None
        _rec("q2", "cat_1", 0.5),
    ]
    out = aggregate_dual_metric(records)
    cat = out["per_category"]["cat_1"]
    assert cat["sme_recall_mean"] == 0.75
    assert cat["qa_accuracy"] is None
    assert cat["retrieval_qa_gap"] is None
    assert cat["judge_label_counts"]["skipped"] == 2


def test_aggregate_multiple_categories_kept_separate():
    """Per the KU caveat, categories never collapse into one number."""
    records = [
        _rec("q1", "cat_1", 1.0, "CORRECT"),
        _rec("q2", "cat_6", 0.0, "INCORRECT"),
    ]
    out = aggregate_dual_metric(records)
    assert set(out["per_category"]) == {"cat_1", "cat_6"}
    assert out["per_category"]["cat_1"]["qa_accuracy"] == 1.0
    assert out["per_category"]["cat_6"]["qa_accuracy"] == 0.0


def test_aggregate_overall_combines_all_categories():
    records = [
        _rec("q1", "cat_1", 1.0, "CORRECT"),
        _rec("q2", "cat_6", 0.0, "INCORRECT"),
    ]
    out = aggregate_dual_metric(records)
    overall = out["overall"]
    assert overall["n"] == 2
    assert overall["sme_recall_mean"] == 0.5
    assert overall["qa_accuracy"] == 0.5
    assert overall["retrieval_qa_gap"] == 0.0
