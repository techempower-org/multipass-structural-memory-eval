"""Unit tests for scripts/k_curve_router_analysis.py — issue #85."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "k_curve_router_analysis",
    Path(__file__).resolve().parents[1] / "scripts" / "k_curve_router_analysis.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


# ---- K-curve primitives ------------------------------------------------
def test_r_at_k_counts_ranks_within_k():
    probes = [{"rank": 1}, {"rank": 3}, {"rank": None}, {"rank": 11}]
    assert mod.r_at_k(probes, 1) == pytest.approx(25.0)
    assert mod.r_at_k(probes, 3) == pytest.approx(50.0)
    assert mod.r_at_k(probes, 10) == pytest.approx(50.0)


def test_r_at_k_empty():
    assert mod.r_at_k([], 5) == 0.0


def test_mrr_at_k_truncates():
    probes = [{"rank": 1}, {"rank": 2}, {"rank": 20}]
    # rank-1 -> 1.0, rank-2 -> 0.5, rank-20 dropped at k=10
    assert mod.mrr_at_k(probes, 10) == pytest.approx(1.5 / 3)
    # at k=1 only the rank-1 probe contributes
    assert mod.mrr_at_k(probes, 1) == pytest.approx(1.0 / 3)


def test_k_curve_for_source_deltas():
    src = {
        "strategies": {
            "A": {"probes": [{"rank": 5}, {"rank": None}]},   # R@1=0
            "B": {"probes": [{"rank": 1}, {"rank": None}]},   # R@1=50
        }
    }
    out = mod.k_curve_for_source(src, ["A", "B"])
    assert out["A"]["R@K"]["1"] == 0.0
    assert out["B"]["R@K"]["1"] == 50.0
    assert out["_deltas_R@K"]["B_minus_A"]["1"] == 50.0


def test_build_k_curve_handles_single_encoder():
    src = {"strategies": {"A": {"probes": [{"rank": 1}]}}}
    block = mod.build_k_curve(src, None)
    assert block["strategies"] == ["A"]
    assert "baseline" in block["by_encoder"]
    assert "FT-300" not in block["by_encoder"]


def test_build_k_curve_none_when_no_sources():
    assert mod.build_k_curve(None, None) is None


# ---- per-question normalisation ---------------------------------------
def test_norm_record_regex_idf_style():
    rec = {
        "question_id": "q1", "question_type": "single-session-user",
        "top1": "s_gold", "top1_score": 2.5,
        "gold": ["s_gold"], "hit_at_1": 1,
    }
    n = mod._norm_record(rec)
    assert n["qid"] == "q1"
    assert n["qtype"] == "single-session-user"
    assert n["top1"] == "s_gold"
    assert n["top1_score"] == 2.5
    assert n["gold"] == {"s_gold"}
    assert n["hit1"] == 1


def test_norm_record_encoder_swap_style_recomputes_hit():
    rec = {
        "question_id": "q2", "question_type": "multi-session",
        "retrieved_rank_1": "s_x", "expected_sources": ["s_gold"],
    }
    n = mod._norm_record(rec)
    assert n["qid"] == "q2"
    assert n["top1"] == "s_x"
    assert n["gold"] == {"s_gold"}
    # no hit flag in source -> recomputed from top1 vs gold (miss)
    assert n["hit1"] == 0


def test_norm_record_at_sign_hit_key():
    rec = {"qid": "q3", "qtype": "t", "top1": "g", "gold": ["g"], "hit@1": 1}
    n = mod._norm_record(rec)
    assert n["hit1"] == 1


def test_load_per_q_reads_per_q_block(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"summary": {}, "per_q": [
        {"qid": "a", "top1": "g", "gold": ["g"], "hit@1": 1},
        {"qid": "b", "top1": "x", "gold": ["g"], "hit@1": 0},
    ]}))
    m = mod.load_per_q(f)
    assert set(m) == {"a", "b"}
    assert m["a"]["hit1"] == 1


# ---- router cross-table ------------------------------------------------
def _mk(qid, top1, gold, score=None, qtype="t"):
    return mod._norm_record({
        "qid": qid, "qtype": qtype, "top1": top1,
        "gold": list(gold), "top1_score": score,
    })


def test_router_cross_table_max_lift():
    # q1: both hit same pick; q2: only graph; q3: only encoder; q4: neither
    graph = {
        "q1": _mk("q1", "g", {"g"}),
        "q2": _mk("q2", "g", {"g"}),
        "q3": _mk("q3", "x", {"g"}),
        "q4": _mk("q4", "x", {"g"}),
    }
    encoder = {
        "q1": _mk("q1", "g", {"g"}),
        "q2": _mk("q2", "x", {"g"}),
        "q3": _mk("q3", "g", {"g"}),
        "q4": _mk("q4", "x", {"g"}),
    }
    ct = mod.router_cross_table(graph, encoder)
    assert ct["n_shared"] == 4
    assert ct["both_hit"] == 1
    assert ct["both_same_pick"] == 1
    assert ct["only_graph"] == 1
    assert ct["only_encoder"] == 1
    assert ct["neither"] == 1
    # max router gets q1, q2, q3 = 3/4
    assert ct["R@1_max_router"] == pytest.approx(0.75)
    assert ct["R@1_graph_alone"] == pytest.approx(0.5)   # q1, q2
    assert ct["R@1_encoder_alone"] == pytest.approx(0.5)  # q1, q3


def test_router_cross_table_different_pick():
    # multi-gold question where both hit but pick different gold sessions
    graph = {"q1": _mk("q1", "g1", {"g1", "g2"})}
    encoder = {"q1": _mk("q1", "g2", {"g1", "g2"})}
    ct = mod.router_cross_table(graph, encoder)
    assert ct["both_hit"] == 1
    assert ct["both_different_pick"] == 1
    assert ct["both_same_pick"] == 0


def test_confidence_routing_signal_separates():
    # graph wins (hit, enc miss) carry high scores; losses carry low scores
    graph = {
        "w1": _mk("w1", "g", {"g"}, score=5.0),
        "w2": _mk("w2", "g", {"g"}, score=4.0),
        "l1": _mk("l1", "x", {"g"}, score=0.5),
        "l2": _mk("l2", "x", {"g"}, score=0.2),
    }
    encoder = {
        "w1": _mk("w1", "x", {"g"}),   # enc misses -> graph unique win
        "w2": _mk("w2", "x", {"g"}),
        "l1": _mk("l1", "g", {"g"}),
        "l2": _mk("l2", "g", {"g"}),
    }
    sig = mod.confidence_routing_signal(graph, encoder)
    assert sig["n_unique_wins"] == 2
    assert sig["n_losses"] == 2
    assert sig["win_median_score"] == pytest.approx(4.5)
    assert sig["lose_median_score"] == pytest.approx(0.35)
    assert sig["separates"] is True


def test_confidence_routing_signal_no_scores():
    graph = {"q1": _mk("q1", "g", {"g"}, score=None)}
    encoder = {"q1": _mk("q1", "x", {"g"})}
    sig = mod.confidence_routing_signal(graph, encoder)
    assert sig["n_unique_wins"] == 0
    assert sig["separates"] is False


def test_per_category_router_gain():
    graph = {
        "q1": _mk("q1", "g", {"g"}, qtype="temporal"),
        "q2": _mk("q2", "x", {"g"}, qtype="temporal"),
    }
    encoder = {
        "q1": _mk("q1", "x", {"g"}, qtype="temporal"),  # only graph
        "q2": _mk("q2", "x", {"g"}, qtype="temporal"),  # neither
    }
    out = mod.per_category_router_gain(graph, encoder)
    assert out["temporal"]["n"] == 2
    assert out["temporal"]["R@1_encoder_alone"] == 0.0
    assert out["temporal"]["R@1_max_router"] == pytest.approx(0.5)
    assert out["temporal"]["lift"] == pytest.approx(0.5)


# ---- end-to-end main ---------------------------------------------------
def test_main_router_end_to_end(tmp_path):
    graph_f = tmp_path / "graph.json"
    enc_f = tmp_path / "enc.json"
    graph_f.write_text(json.dumps({"per_q": [
        {"qid": "q1", "qtype": "t", "top1": "g", "gold": ["g"],
         "top1_score": 3.0, "hit@1": 1},
        {"qid": "q2", "qtype": "t", "top1": "x", "gold": ["g"],
         "top1_score": 0.1, "hit@1": 0},
    ]}))
    enc_f.write_text(json.dumps({"per_question": [
        {"question_id": "q1", "question_type": "t",
         "retrieved_rank_1": "x", "expected_sources": ["g"], "hit_at_1": 0},
        {"question_id": "q2", "question_type": "t",
         "retrieved_rank_1": "g", "expected_sources": ["g"], "hit_at_1": 1},
    ]}))
    out = tmp_path / "out.json"
    rc = mod.main([
        "--graph-results", str(graph_f),
        "--encoder-results", str(enc_f),
        "--score-system", "graph",
        "--out", str(out),
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert "router" in report
    ct = report["router"]["cross_table"]
    # q1: only graph hits; q2: only encoder hits -> max router 2/2
    assert ct["R@1_max_router"] == pytest.approx(1.0)
    assert ct["only_graph"] == 1
    assert ct["only_encoder"] == 1


def test_main_kcurve_end_to_end(tmp_path):
    base_f = tmp_path / "base.json"
    base_f.write_text(json.dumps({"strategies": {
        "A": {"probes": [{"rank": 5}, {"rank": None}]},
        "B": {"probes": [{"rank": 1}, {"rank": 2}]},
    }}))
    out = tmp_path / "out.json"
    rc = mod.main(["--kcurve-baseline", str(base_f), "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert "k_curve" in report
    assert report["k_curve"]["strategies"] == ["A", "B"]
