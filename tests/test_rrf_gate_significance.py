"""Unit tests for the multi-encoder RRF significance gate (#106).

Covers the paired statistical tests (bootstrap CI, McNemar, sign test), the
hit@k derivation from the explicit rank field, and the positional alignment
that must survive duplicate query strings. No network.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC_PATH = Path(__file__).resolve().parent.parent / "scripts" / "rrf_gate_significance.py"
_spec = importlib.util.spec_from_file_location("rrf_gate_significance", _SPEC_PATH)
rgs = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(rgs)


def _artifact(base_probes, multi_probes, encoders=("a", "b")):
    return {
        "encoders": list(encoders),
        "baseline": {"per_probe": base_probes},
        "multi_encoder_rrf": {"per_probe": multi_probes},
    }


# --------------------------------------------------------------------------- #
# hit_at_k
# --------------------------------------------------------------------------- #
def test_hit_at_k_from_rank():
    assert rgs.hit_at_k({"rank": 1}, 5) == 1
    assert rgs.hit_at_k({"rank": 5}, 5) == 1
    assert rgs.hit_at_k({"rank": 6}, 5) == 0
    assert rgs.hit_at_k({"rank": None}, 5) == 0
    assert rgs.hit_at_k({}, 5) == 0


# --------------------------------------------------------------------------- #
# Positional alignment survives duplicate queries
# --------------------------------------------------------------------------- #
def test_paired_probes_positional_with_duplicates():
    # "dup" appears twice with DIFFERENT ranks in each list; a dict keyed on
    # query would collapse them. Positional alignment must keep both.
    base = [
        {"query": "dup", "rank": 1, "rr": 1.0},
        {"query": "x", "rank": None, "rr": 0.0},
        {"query": "dup", "rank": 8, "rr": 0.125},
    ]
    multi = [
        {"query": "dup", "rank": 2, "rr": 0.5},
        {"query": "x", "rank": 1, "rr": 1.0},
        {"query": "dup", "rank": None, "rr": 0.0},
    ]
    b, m = rgs.paired_probes(_artifact(base, multi))
    assert [r["rank"] for r in b] == [1, None, 8]
    assert [r["rank"] for r in m] == [2, 1, None]


def test_paired_probes_order_mismatch_raises():
    base = [{"query": "a", "rank": 1, "rr": 1.0}]
    multi = [{"query": "b", "rank": 1, "rr": 1.0}]
    with pytest.raises(ValueError, match="order mismatch"):
        rgs.paired_probes(_artifact(base, multi))


# --------------------------------------------------------------------------- #
# McNemar exact
# --------------------------------------------------------------------------- #
def test_mcnemar_no_discordant_pairs():
    r = rgs.mcnemar_exact([1, 0, 1], [1, 0, 1])
    assert r["n_discordant"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_all_gains_significant():
    # 10 probes flipped miss->hit, none the other way: p should be tiny.
    base = [0] * 10
    multi = [1] * 10
    r = rgs.mcnemar_exact(base, multi)
    assert r["c_gained"] == 10
    assert r["b_lost"] == 0
    assert r["p_value"] < 0.01


def test_mcnemar_balanced_not_significant():
    base = [1, 0]
    multi = [0, 1]
    r = rgs.mcnemar_exact(base, multi)
    assert r["b_lost"] == 1 and r["c_gained"] == 1
    assert r["p_value"] == 1.0


# --------------------------------------------------------------------------- #
# Sign test
# --------------------------------------------------------------------------- #
def test_sign_test_ignores_zero_diffs():
    r = rgs.sign_test([0.0, 0.0, 0.5, 0.5, 0.5])
    assert r["n_nonzero"] == 3
    assert r["n_pos"] == 3 and r["n_neg"] == 0


def test_sign_test_symmetric_pvalue():
    # 22 pos / 8 neg out of 30: matches the real artifact's sign-test shape.
    diffs = [0.5] * 22 + [-0.5] * 8
    r = rgs.sign_test(diffs)
    assert r["n_pos"] == 22 and r["n_neg"] == 8
    assert r["p_value"] < 0.05


# --------------------------------------------------------------------------- #
# Bootstrap CI
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_excludes_zero_for_clear_lift():
    base = [0.0] * 100
    multi = [1.0] * 100  # every probe improved by +1.0
    r = rgs.bootstrap_mrr_delta_ci(base, multi, n_boot=2000, seed=1)
    assert r["delta_mrr_point"] == pytest.approx(1.0)
    assert r["excludes_zero"] is True


def test_bootstrap_ci_includes_zero_for_null():
    # Symmetric noise around 0 -> CI should straddle 0.
    base = [0.5] * 50
    multi = [0.5] * 50
    r = rgs.bootstrap_mrr_delta_ci(base, multi, n_boot=2000, seed=1)
    assert r["delta_mrr_point"] == pytest.approx(0.0)
    assert r["excludes_zero"] is False


# --------------------------------------------------------------------------- #
# analyze end-to-end on a tiny artifact
# --------------------------------------------------------------------------- #
def test_analyze_reproduces_hand_counts():
    base = [
        {"query": "q1", "rank": 1, "rr": 1.0},
        {"query": "q2", "rank": None, "rr": 0.0},
        {"query": "q3", "rank": 6, "rr": 1 / 6},
    ]
    multi = [
        {"query": "q1", "rank": 1, "rr": 1.0},     # unchanged
        {"query": "q2", "rank": 2, "rr": 0.5},     # gained into top-5
        {"query": "q3", "rank": 3, "rr": 1 / 3},   # improved within top-10
    ]
    res = rgs.analyze(_artifact(base, multi), n_boot=500, seed=3)
    assert res["n_probes"] == 3
    # R@5: baseline 1/3 hits (q1); multi 3/3 (q1 held, q2 6->2 wait no q2 was
    # a miss that became rank 2, q3 6->3 both now in top-5).
    assert res["recall_at_5_pct"]["baseline"] == pytest.approx(100 / 3)
    assert res["recall_at_5_pct"]["multi"] == pytest.approx(100.0)
    # q2 and q3 both improved their RR; q1 unchanged.
    assert res["n_improved"] == 2 and res["n_worsened"] == 0
