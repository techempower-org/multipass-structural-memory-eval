"""Unit tests for the headline-delta significance script (#21 cont'd).

Covers the guards that keep the artifact honest — these are the load-bearing
logic, since the CI/FDR math is already tested in tests/test_stats.py:

- qa_correct extraction from the judge verdict dict (CORRECT=1, decisive
  other=0, ERROR/skipped/None dropped).
- all-zero-on-one-side refusal (the field-name trap: a metric unpopulated for
  one adapter must not be compared).
- identical-scores -> definitional null, kept OUT of the BH-FDR family.
- fragility flag when few paired questions actually disagree.
- non-identical-metric flag carried through.
- no-paired-baseline status when a side has no committed per-question data.
- BH-FDR applied across the whole computed family.

No network, no real baseline files.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "headline_delta_significance.py"
)
_spec = importlib.util.spec_from_file_location("headline_delta_significance", _SPEC_PATH)
hds = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
# Register before exec: the module defines @dataclass with string annotations,
# and dataclasses resolves them via sys.modules[cls.__module__].
sys.modules["headline_delta_significance"] = hds
_spec.loader.exec_module(hds)


# ── helpers ─────────────────────────────────────────────────────────


def _judge(label):
    return {"autoeval_label": label, "judge_model": "test"}


def _doc(per_question: list[dict]) -> dict:
    return {"per_question": per_question}


def _write(tmp_path, name, per_question):
    p = tmp_path / name
    p.write_text(json.dumps(_doc(per_question)))
    return str(p)


# ── qa_correct extraction ───────────────────────────────────────────


def test_qa_correct_maps_labels():
    assert hds.qa_correct({"judge": _judge("CORRECT")}) == 1.0
    assert hds.qa_correct({"judge": _judge("INCORRECT")}) == 0.0
    assert hds.qa_correct({"judge": _judge("PARTIAL")}) == 0.0
    assert hds.qa_correct({"judge": _judge("ABSTAIN")}) == 0.0


def test_qa_correct_drops_error_and_missing():
    assert hds.qa_correct({"judge": _judge("ERROR")}) is None
    assert hds.qa_correct({"judge": _judge("skipped")}) is None
    assert hds.qa_correct({"judge": None}) is None
    assert hds.qa_correct({}) is None


def test_recall_field_handles_bool_and_float():
    f = hds.recall_field("x")
    assert f({"x": True}) == 1.0
    assert f({"x": False}) == 0.0
    assert f({"x": 0.5}) == 0.5
    assert f({"x": None}) is None
    assert f({}) is None


# ── _score_metric guards ────────────────────────────────────────────


def _ab(a_vals, b_vals, extractor=None):
    """Build two by-id dicts over shared ids q0..qn with the given per-q field."""
    extractor = extractor or hds.recall_field("m")
    a = {f"q{i}": {"m": v} for i, v in enumerate(a_vals)}
    b = {f"q{i}": {"m": v} for i, v in enumerate(b_vals)}
    metric = hds.Metric("m", extractor, extractor)
    return metric, a, b


def test_all_zero_one_side_is_flagged_not_refused():
    """A field all-zero on exactly one side gets a comparability flag but is
    still computed — we can't tell 'unpopulated field' from 'scored zero on
    every question' by value alone, so flag rather than silently suppress."""
    metric, a, b = _ab([0.0] * 40, [1.0] * 40)
    out = hds._score_metric(metric, a, b, "A", "B")
    assert out["status"] == "ok"
    assert out["one_side_all_zero"] is True
    assert out["delta_pp"] == pytest.approx(-100.0)


def test_both_sides_nonzero_not_flagged():
    metric, a, b = _ab([0.6] * 40, [0.4] * 40)
    out = hds._score_metric(metric, a, b, "A", "B")
    assert out["status"] == "ok"
    assert out["one_side_all_zero"] is False


def test_identical_scores_are_definitional_null():
    """Byte-identical per-question scores -> definitional null, not significant,
    and excluded from the BH family (no p_raw)."""
    metric, a, b = _ab([0.4] * 50, [0.4] * 50)
    out = hds._score_metric(metric, a, b, "A", "B")
    assert out["status"] == "identical"
    assert out["delta_pp"] == 0.0
    assert "p_raw" not in out


def test_fragile_flag_when_few_discordant():
    """A delta resting on a handful of discordant pairs is flagged fragile.
    Both sides have non-zero mean so the all-zero flag doesn't confound it."""
    # 5 questions flip A=1/B=0; the rest agree at 1.0 → both means non-zero.
    a = [1.0] * 5 + [1.0] * 45
    b = [0.0] * 5 + [1.0] * 45
    metric, ad, bd = _ab(a, b)
    out = hds._score_metric(metric, ad, bd, "A", "B")
    assert out["status"] == "ok"
    assert out["n_discordant"] == 5
    assert out["fragile"] is True
    assert out["one_side_all_zero"] is False


def test_not_fragile_when_many_discordant():
    a = [1.0] * 40 + [1.0] * 10
    b = [0.0] * 40 + [1.0] * 10
    metric, ad, bd = _ab(a, b)
    out = hds._score_metric(metric, ad, bd, "A", "B")
    assert out["n_discordant"] == 40
    assert out["fragile"] is False


def test_no_shared_questions():
    metric = hds.Metric("m", hds.recall_field("m"), hds.recall_field("m"))
    out = hds._score_metric(metric, {"q1": {"m": 1.0}}, {"q2": {"m": 1.0}}, "A", "B")
    assert out["status"] == "no_shared_questions"


def test_non_identical_metric_uses_per_side_extractors_and_flags():
    """When A and B read different fields, both are used and the result is
    flagged non_identical_metric."""
    a = {f"q{i}": {"fa": 1.0} for i in range(40)}
    b = {f"q{i}": {"fb": 1.0 if i < 20 else 0.0} for i in range(40)}
    metric = hds.Metric(
        "r", hds.recall_field("fa"), hds.recall_field("fb"),
        non_identical_metric=True,
    )
    out = hds._score_metric(metric, a, b, "A", "B")
    assert out["status"] == "ok"
    assert out["non_identical_metric"] is True
    # A=1.0 everywhere, B=0.5 -> delta +50pp
    assert out["delta_pp"] == pytest.approx(50.0, abs=1.0)


# ── run_comparison + family FDR ─────────────────────────────────────


def test_no_paired_baseline_when_path_missing():
    cmp = hds.Comparison(
        key="k", description="d",
        label_a="A", path_a=None, label_b="B", path_b=None, metrics=[],
    )
    out = hds.run_comparison(cmp)
    assert out["status"] == "no_paired_baseline"


def test_descriptive_only_emits_per_side_means_no_ci(tmp_path):
    """comparable=False -> per-side point estimates, NO CI, with the reason.
    A paired bootstrap on incomparable per-question metrics would launder a
    methodology error into a rigorous-looking number, so we refuse the CI."""
    a = _write(tmp_path, "a.json",
               [{"question_id": f"q{i}", "fa": 1.0} for i in range(40)])
    b = _write(tmp_path, "b.json",
               [{"question_id": f"q{i}", "fb": 0.5} for i in range(40)])
    cmp = hds.Comparison(
        key="x", description="d",
        label_a="A", path_a=a, label_b="B", path_b=b,
        metrics=[hds.Metric("r", hds.recall_field("fa"), hds.recall_field("fb"))],
        comparable=False,
        not_comparable_reason="different hit semantics",
    )
    out = hds.run_comparison(cmp)
    assert out["status"] == "descriptive_only"
    assert out["not_comparable_reason"] == "different hit semantics"
    m = out["metrics"][0]
    assert m["status"] == "descriptive_only"
    assert m["mean_A"] == pytest.approx(1.0)
    assert m["mean_B"] == pytest.approx(0.5)
    # No CI / delta / p — the whole point.
    assert "delta_pp" not in m
    assert "ci_low_pp" not in m
    assert "p_raw" not in m


def test_descriptive_only_excluded_from_bh_family():
    """descriptive_only metrics carry no p_raw and must not enter BH-FDR."""
    ok = {"status": "ok", "p_raw": 0.02}
    desc = {"status": "descriptive_only", "mean_A": 0.9, "mean_B": 0.9}
    comparisons = [{"metrics": [ok, desc]}]
    hds.apply_family_fdr(comparisons, alpha=0.05)
    assert "p_adjusted" in ok
    assert "p_adjusted" not in desc


def test_run_comparison_computes_over_files(tmp_path):
    # A clearly beats B on QA over 40 paired questions.
    a = _write(tmp_path, "a.json",
               [{"question_id": f"q{i}", "judge": _judge("CORRECT")} for i in range(40)])
    b = _write(tmp_path, "b.json",
               [{"question_id": f"q{i}", "judge": _judge("INCORRECT")} for i in range(40)])
    cmp = hds.Comparison(
        key="k", description="d",
        label_a="A", path_a=a, label_b="B", path_b=b,
        metrics=[hds.Metric("qa_correct", hds.qa_correct, hds.qa_correct)],
    )
    out = hds.run_comparison(cmp)
    assert out["status"] == "computed"
    m = out["metrics"][0]
    assert m["status"] == "ok"
    assert m["delta_pp"] == pytest.approx(100.0)
    assert m["ci_straddles_zero"] is False


def test_family_fdr_only_corrects_ok_metrics():
    """identical/refused/no-shared metrics carry no p_raw and must not enter
    the BH family; ok metrics get p_adjusted written back."""
    ok = {"status": "ok", "p_raw": 0.01}
    ident = {"status": "identical"}
    refused = {"status": "all_zero_one_side"}
    comparisons = [{"metrics": [ok, ident, refused]}]
    hds.apply_family_fdr(comparisons, alpha=0.05)
    assert "p_adjusted" in ok and "significant" in ok
    assert "p_adjusted" not in ident
    assert "p_adjusted" not in refused


def test_family_fdr_empty_is_noop():
    comparisons = [{"metrics": [{"status": "identical"}]}]
    hds.apply_family_fdr(comparisons, alpha=0.05)  # must not raise
    assert "p_adjusted" not in comparisons[0]["metrics"][0]


def test_verdict_strings():
    assert "NULL" in hds._verdict({"status": "identical"})
    assert "NULL" in hds._verdict(
        {"status": "ok", "significant": False, "ci_straddles_zero": True}
    )
    assert "SIGNIFICANT" in hds._verdict(
        {"status": "ok", "significant": True, "fragile": False}
    )
    assert "FRAGILE" in hds._verdict(
        {"status": "ok", "significant": True, "fragile": True, "n_discordant": 3}
    )
