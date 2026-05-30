"""Unit tests for the isotonic confidence calibrator (#105).

Covers the pure math (PAV fit, interpolation, ECE, Brier) and the relevance
labeling rule. No daemon / network touched.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the script as a module (it lives in scripts/, not the importable pkg).
_SPEC_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fit_confidence_calibrator.py"
_spec = importlib.util.spec_from_file_location("fit_confidence_calibrator", _SPEC_PATH)
cal = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(cal)


# --------------------------------------------------------------------------- #
# PAV monotonicity + correctness
# --------------------------------------------------------------------------- #
def test_pav_monotone_non_decreasing():
    # Deliberately non-monotone label pattern; PAV must pool to monotone.
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    labels = [0, 1, 0, 1, 1]
    x, y = cal.pav_fit(scores, labels)
    assert all(y[i] <= y[i + 1] + 1e-9 for i in range(len(y) - 1)), y


def test_pav_already_monotone_passthrough():
    # Perfectly separable: low scores all 0, high scores all 1.
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    x, y = cal.pav_fit(scores, labels)
    # apply should give ~0 on the low end, ~1 on the high end
    assert cal.apply_pav(x, y, 0.15) < 0.5
    assert cal.apply_pav(x, y, 0.85) > 0.5


def test_pav_pools_violation_to_mean():
    # Two points at the same effective block that violate: [1, 0] pools to 0.5.
    scores = [0.4, 0.5]
    labels = [1, 0]
    x, y = cal.pav_fit(scores, labels)
    # The pooled block should have probability 0.5.
    assert pytest.approx(y[0], abs=1e-9) == 0.5


def test_apply_pav_clamps_outside_knots():
    x, y = [0.2, 0.8], [0.1, 0.9]
    assert cal.apply_pav(x, y, 0.0) == 0.1   # below first knot -> first y
    assert cal.apply_pav(x, y, 1.0) == 0.9   # above last knot -> last y


def test_apply_pav_interpolates_between_knots():
    x, y = [0.0, 1.0], [0.0, 1.0]
    assert pytest.approx(cal.apply_pav(x, y, 0.5), abs=1e-9) == 0.5


def test_apply_pav_empty_knots_passthrough():
    assert cal.apply_pav([], [], 0.42) == 0.42


def test_pav_empty_input():
    assert cal.pav_fit([], []) == ([], [])


# --------------------------------------------------------------------------- #
# Brier + ECE
# --------------------------------------------------------------------------- #
def test_brier_perfect_is_zero():
    assert cal.brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0


def test_brier_worst_is_one():
    assert cal.brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_brier_half_confidence():
    # all p=0.5 -> each term 0.25
    assert pytest.approx(cal.brier_score([0.5, 0.5], [1, 0]), abs=1e-9) == 0.25


def test_ece_perfectly_calibrated_is_zero():
    # In each bin, mean confidence == accuracy.
    probs = [0.05, 0.95, 0.05, 0.95]
    labels = [0, 1, 0, 1]
    assert cal.expected_calibration_error(probs, labels, n_bins=10) < 0.06


def test_ece_overconfident_is_large():
    # Claim p=0.9 but only half are relevant -> bin gap ~0.4.
    probs = [0.9, 0.9, 0.9, 0.9]
    labels = [1, 0, 1, 0]
    ece = cal.expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.4, abs=1e-9)


def test_ece_last_bin_includes_one():
    # p == 1.0 must land in the last bin, not be dropped.
    probs = [1.0, 1.0]
    labels = [1, 0]
    ece = cal.expected_calibration_error(probs, labels, n_bins=10)
    # bin confidence 1.0, accuracy 0.5 -> gap 0.5
    assert ece == pytest.approx(0.5, abs=1e-9)


def test_ece_empty():
    assert cal.expected_calibration_error([], []) == 0.0


# --------------------------------------------------------------------------- #
# Relevance labeling
# --------------------------------------------------------------------------- #
def test_hit_relevant_flat_shape():
    hit = {"source_file": "/home/jp/x/postgres.py"}
    assert cal.hit_is_relevant(hit, "postgres.py")
    assert not cal.hit_is_relevant(hit, "searcher.py")


def test_hit_relevant_nested_metadata_shape():
    hit = {"metadata": {"source_file": "a/b/cli.py"}}
    assert cal.hit_is_relevant(hit, "cli.py")


def test_hit_relevant_missing_source():
    assert not cal.hit_is_relevant({}, "anything.py")


# --------------------------------------------------------------------------- #
# Probe loading
# --------------------------------------------------------------------------- #
def test_load_probes_from_rrf_artifact(tmp_path):
    import json
    artifact = {
        "baseline": {
            "per_probe": [
                {"query": "Q1", "expected": "a.py", "rank": 1},
                {"query": "Q2", "expected": "b.py", "rank": None},
                {"query": "", "expected": "c.py"},          # dropped: no query
                {"query": "Q4", "expected": ""},            # dropped: no expected
            ]
        }
    }
    f = tmp_path / "rrf.json"
    f.write_text(json.dumps(artifact))
    probes = cal.load_probes(f)
    assert len(probes) == 2
    assert probes[0] == {"query": "Q1", "expected": "a.py"}


# --------------------------------------------------------------------------- #
# End-to-end fit_and_report on a synthetic, deliberately-miscalibrated set
# --------------------------------------------------------------------------- #
def test_fit_and_report_improves_calibration():
    # Build a set where raw similarity is systematically overconfident:
    # similarity ~0.9 but only ~50% relevant. PAV should pull the mapped
    # probability down toward the empirical base rate, lowering ECE.
    rows = []
    for i in range(100):
        rel = i % 2
        rows.append({"score": 0.9, "relevant": rel, "query": f"q{i}", "expected": "x"})
    result = cal.fit_and_report(rows, n_bins=10)
    assert result["n_pairs"] == 100
    # raw 0.9 vs 0.5 accuracy -> ECE ~0.4 before (headline = cross-validated)
    assert result["before_calibration"]["ece"] > 0.3
    # PAV maps the pooled block to ~0.5 -> ECE far lower after, even held-out
    assert result["after_calibration"]["ece"] < result["before_calibration"]["ece"]
    assert result["delta_ece"] < 0  # improvement


def test_in_sample_vs_cross_validated_present():
    # The report must carry BOTH the optimistic in-sample and the honest CV
    # numbers, so a reader can see the gap. The headline must equal the CV one.
    rows = [{"score": 0.1 + 0.8 * (i / 200), "relevant": int(i % 3 == 0),
             "query": f"q{i}", "expected": "x"} for i in range(200)]
    result = cal.fit_and_report(rows, n_bins=10)
    assert "in_sample" in result and "cross_validated" in result
    cv = result["cross_validated"]
    assert cv["n_folds"] == 5 and cv["n_oos"] == 200
    # headline before/after are the CV numbers
    assert result["before_calibration"] == cv["before_calibration"]
    assert result["after_calibration"] == cv["after_calibration"]
    # in-sample after-ECE should be <= CV after-ECE (in-sample is optimistic)
    assert result["in_sample"]["after_calibration"]["ece"] <= cv["after_calibration"]["ece"] + 1e-9


def test_cross_validated_eval_skips_tiny_sets():
    rows = [{"score": 0.5, "relevant": 1, "query": "q", "expected": "x"}]
    cv = cal.cross_validated_eval(rows, n_folds=5)
    assert "skipped" in cv
