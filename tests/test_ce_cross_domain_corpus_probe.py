"""Unit tests for the corpus-side CE cross-domain probe (#104).

Covers relevance labeling, dual-ordering reconstruction (rrf_score vs
rerank_score), per-query scoring, aggregation, and the degenerate-measurement
guard that refuses to dress a non-distinguishing run up as an H3 verdict. No
network.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC_PATH = Path(__file__).resolve().parent.parent / "scripts" / "ce_cross_domain_corpus_probe.py"
_spec = importlib.util.spec_from_file_location("ce_cross_domain_corpus_probe", _SPEC_PATH)
ccp = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(ccp)


# --------------------------------------------------------------------------- #
# Relevance
# --------------------------------------------------------------------------- #
def test_hit_relevant_substring():
    hit = {"text": "uses plainto_tsquery with an ILIKE fallback"}
    assert ccp.hit_relevant(hit, {"content_any": ["plainto_tsquery"]})
    assert not ccp.hit_relevant(hit, {"content_any": ["trigram"]})


def test_hit_relevant_case_insensitive():
    hit = {"text": "The HyDE Bridge Was A Dead End"}
    assert ccp.hit_relevant(hit, {"content_any": ["hyde"]})


def test_hit_relevant_empty_content_never_matches():
    hit = {"text": "anything at all"}
    assert not ccp.hit_relevant(hit, {"content_any": []})
    assert not ccp.hit_relevant(hit, {})


def test_hit_relevant_source_glob_gate():
    hit = {"text": "trigram index", "source_file": "/x/postgres.py"}
    assert ccp.hit_relevant(hit, {"content_any": ["trigram"], "source_glob": "postgres.py"})
    assert not ccp.hit_relevant(hit, {"content_any": ["trigram"], "source_glob": "searcher.py"})


# --------------------------------------------------------------------------- #
# Ordering reconstruction
# --------------------------------------------------------------------------- #
def test_order_by_descending_missing_sinks():
    hits = [{"rrf_score": 0.1}, {"rrf_score": 0.3}, {"rrf_score": None}]
    ordered = ccp.order_by(hits, "rrf_score")
    assert [h["rrf_score"] for h in ordered] == [0.3, 0.1, None]


def test_first_relevant_rank():
    hits = [{"text": "no"}, {"text": "yes trigram"}, {"text": "no"}]
    assert ccp.first_relevant_rank(hits, {"content_any": ["trigram"]}) == 2
    assert ccp.first_relevant_rank(hits, {"content_any": ["absent"]}) is None


def test_rank_to_metrics():
    assert ccp.rank_to_metrics(1, 5) == (1, 1.0)
    assert ccp.rank_to_metrics(5, 5) == (1, 0.2)
    assert ccp.rank_to_metrics(6, 5) == (0, pytest.approx(1 / 6))
    assert ccp.rank_to_metrics(None, 5) == (0, 0.0)


def test_score_query_rerank_moves_relevant_up():
    # Relevant doc is rank 3 by rrf (off) but rank 1 by rerank (on).
    hits = [
        {"text": "irrelevant", "rrf_score": 0.9, "rerank_score": 0.1},
        {"text": "irrelevant", "rrf_score": 0.8, "rerank_score": 0.2},
        {"text": "the trigram answer", "rrf_score": 0.5, "rerank_score": 0.99},
    ]
    s = ccp.score_query(hits, {"content_any": ["trigram"]})
    assert s["rank_off"] == 3
    assert s["rank_on"] == 1
    assert s["hit_on"] == 1 and s["rr_on"] == 1.0
    assert s["rr_off"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# Aggregation + degenerate guard
# --------------------------------------------------------------------------- #
def _row(domain, off_rank, on_rank, in_pool=1):
    h_off, rr_off = ccp.rank_to_metrics(off_rank)
    h_on, rr_on = ccp.rank_to_metrics(on_rank)
    return {
        "domain": domain,
        "score": {
            "n_candidates": 10, "n_relevant_in_pool": in_pool,
            "rank_off": off_rank, "rank_on": on_rank,
            "hit_off": h_off, "hit_on": h_on, "rr_off": rr_off, "rr_on": rr_on,
        },
    }


def test_aggregate_excludes_zero_pool_queries():
    rows = [
        _row("code", 3, 1, in_pool=1),
        _row("code", None, None, in_pool=0),   # excluded (no relevant in pool)
    ]
    agg = ccp.aggregate(rows)
    assert agg["by_domain"]["code"]["n"] == 2
    assert agg["by_domain"]["code"]["n_with_relevant_in_pool"] == 1


def test_aggregate_asymmetry_conversational_gains_more():
    # code: rerank does nothing (1->1); conversational: rerank lifts 3->1.
    rows = [
        _row("code", 1, 1), _row("code", 1, 1), _row("code", 1, 1),
        _row("conversational", 3, 1), _row("conversational", 3, 1),
        _row("conversational", 3, 1),
    ]
    agg = ccp.aggregate(rows)
    a = agg["asymmetry"]
    assert a["code_mrr_delta"] == pytest.approx(0.0)
    assert a["conversational_mrr_delta"] > 0
    assert a["conversational_minus_code"] > 0
    verdict = ccp.interpret(agg)["verdict"]
    assert "asymmetry present" in verdict


def test_interpret_flags_degenerate_ceiling():
    # Both domains: every query already rank-1 under both orderings -> degenerate.
    rows = [_row("code", 1, 1) for _ in range(4)] + \
           [_row("conversational", 1, 1) for _ in range(4)]
    interp = ccp.interpret(ccp.aggregate(rows))
    assert interp["verdict"].startswith("INCONCLUSIVE")
    assert "rank 1 under BOTH" in interp["why"]


def test_interpret_flags_too_few_in_pool():
    rows = [_row("code", 1, 1, in_pool=1)] + \
           [_row("conversational", 3, 1) for _ in range(4)]
    interp = ccp.interpret(ccp.aggregate(rows))
    assert interp["verdict"].startswith("INCONCLUSIVE")
    assert "too few" in interp["why"]


def test_interpret_detects_code_harmed():
    # code: rerank pushes relevant DOWN (1->4); conversational: neutral.
    rows = [_row("code", 1, 4) for _ in range(4)] + \
           [_row("conversational", 1, 1) for _ in range(4)]
    interp = ccp.interpret(ccp.aggregate(rows))
    assert "HURTS the code domain" in interp["verdict"]
    assert "domain-aware gate" in interp["recommendation"]
