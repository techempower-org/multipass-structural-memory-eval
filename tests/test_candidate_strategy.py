"""Tests for sme.eval.candidate_strategy — pure-logic helpers, no daemon.

The end-to-end MCP path is exercised by scripts/candidate_strategy_eval.py
and the sme-eval candidate-strategy subcommand against a live daemon;
this file covers the relevance + aggregation primitives in isolation.
"""
from __future__ import annotations

from sme.eval.candidate_strategy import (
    aggregate,
    is_relevant,
    rank_of_first_relevant,
)


# --- is_relevant ----------------------------------------------------------


def test_is_relevant_matches_on_content_any():
    hit = {"source_file": "notes.md", "text": "the kill cascade incident"}
    assert is_relevant(hit, {"content_any": ["kill cascade"]}) is True


def test_is_relevant_misses_when_no_substring_match():
    hit = {"source_file": "notes.md", "text": "something unrelated"}
    assert is_relevant(hit, {"content_any": ["kill cascade"]}) is False


def test_is_relevant_with_source_glob_filters_first():
    """source_glob is an AND constraint — content_any only checked if glob matches."""
    hit = {"source_file": "diary.md", "text": "the kill cascade"}
    # glob doesn't match → not relevant even though content_any matches
    assert is_relevant(hit, {
        "source_glob": "notes/*",
        "content_any": ["kill cascade"],
    }) is False


def test_is_relevant_source_glob_only_matches_when_content_empty():
    """If content_any is empty, source_glob alone is sufficient."""
    hit = {"source_file": "notes/kill-cascade.md", "text": "anything"}
    assert is_relevant(hit, {"source_glob": "notes/*"}) is True


# --- rank_of_first_relevant ----------------------------------------------


def test_rank_first_relevant_returns_1_based_rank():
    results = [
        {"text": "miss"},
        {"text": "miss"},
        {"text": "the kill cascade hit"},
        {"text": "miss"},
    ]
    assert rank_of_first_relevant(results, {"content_any": ["kill cascade"]}) == 3


def test_rank_first_relevant_returns_none_when_no_match():
    results = [{"text": "miss"}, {"text": "also miss"}]
    assert rank_of_first_relevant(results, {"content_any": ["kill cascade"]}) is None


# --- aggregate -----------------------------------------------------------


def test_aggregate_computes_R_at_K_and_MRR():
    """Two queries, one strategy. Both gold sessions land in top-5."""
    per_query = {
        "q1": {"vector": {"rank": 1, "r5": 1, "r10": 1, "rr": 1.0,
                          "n_hits": 5, "latency_ms": 100}},
        "q2": {"vector": {"rank": 4, "r5": 1, "r10": 1, "rr": 0.25,
                          "n_hits": 5, "latency_ms": 200}},
    }
    summary = aggregate(per_query, ["vector"], n=2)
    vec = summary["per_strategy"]["vector"]
    assert vec["R@5"] == 1.0
    assert vec["R@10"] == 1.0
    assert vec["MRR"] == 0.625  # (1.0 + 0.25) / 2
    assert vec["p50_ms"] == 150  # median of [100, 200]


def test_aggregate_strategy_flips_capture_movements():
    """Strategy-flip diagnostic — the #57 actionable signal.

    Three flip categories:
      - moved_up: rank improved from A→B
      - moved_down: rank worsened from A→B
      - new_hits: A had no relevant; B did
      - lost_hits: A had relevant; B didn't
    """
    per_query = {
        "q_up": {  # rank 3 → 1 (moved up)
            "a": {"rank": 3, "r5": 1, "r10": 1, "rr": 0.33,
                  "n_hits": 5, "latency_ms": 50},
            "b": {"rank": 1, "r5": 1, "r10": 1, "rr": 1.0,
                  "n_hits": 5, "latency_ms": 60},
        },
        "q_down": {  # rank 1 → 3
            "a": {"rank": 1, "r5": 1, "r10": 1, "rr": 1.0,
                  "n_hits": 5, "latency_ms": 50},
            "b": {"rank": 3, "r5": 1, "r10": 1, "rr": 0.33,
                  "n_hits": 5, "latency_ms": 60},
        },
        "q_new": {  # no relevant in A; rank 2 in B
            "a": {"rank": None, "r5": 0, "r10": 0, "rr": 0.0,
                  "n_hits": 5, "latency_ms": 50},
            "b": {"rank": 2, "r5": 1, "r10": 1, "rr": 0.5,
                  "n_hits": 5, "latency_ms": 60},
        },
        "q_lost": {  # rank 1 in A; no relevant in B
            "a": {"rank": 1, "r5": 1, "r10": 1, "rr": 1.0,
                  "n_hits": 5, "latency_ms": 50},
            "b": {"rank": None, "r5": 0, "r10": 0, "rr": 0.0,
                  "n_hits": 5, "latency_ms": 60},
        },
        "q_same": {  # rank 1 in both
            "a": {"rank": 1, "r5": 1, "r10": 1, "rr": 1.0,
                  "n_hits": 5, "latency_ms": 50},
            "b": {"rank": 1, "r5": 1, "r10": 1, "rr": 1.0,
                  "n_hits": 5, "latency_ms": 60},
        },
    }
    summary = aggregate(per_query, ["a", "b"], n=5)
    flip = summary["strategy_flips"]["a_to_b"]
    moved_up_qids = [m["qid"] for m in flip["moved_up"]]
    moved_down_qids = [m["qid"] for m in flip["moved_down"]]
    assert moved_up_qids == ["q_up"]
    assert moved_down_qids == ["q_down"]
    assert flip["new_hits"] == ["q_new"]
    assert flip["lost_hits"] == ["q_lost"]
    assert flip["unchanged_count"] == 1  # only q_same


def test_aggregate_excludes_error_rows_from_latency_stats():
    """#65 Gemini HIGH — failed queries have latency_ms=0.0 in the error
    fallback; including them in p50/p95 skews the stats downward. Now
    they're excluded from latency aggregation and counted via n_errors."""
    per_query = {
        "q_ok": {"v": {"rank": 1, "r5": 1, "r10": 1, "rr": 1.0,
                       "n_hits": 5, "latency_ms": 200}},
        "q_err": {"v": {"rank": None, "r5": 0, "r10": 0, "rr": 0.0,
                        "n_hits": 0, "latency_ms": 0.0, "error": "boom"}},
    }
    summary = aggregate(per_query, ["v"], n=2)
    v = summary["per_strategy"]["v"]
    assert v["n_ok"] == 1
    assert v["n_errors"] == 1
    assert v["p50_ms"] == 200  # NOT median([0.0, 200]) = 100
    assert v["p95_ms"] == 200


def test_aggregate_all_errors_returns_None_latency():
    """When every query errors, latency stats are None (not IndexError)."""
    per_query = {
        "q1": {"v": {"rank": None, "r5": 0, "r10": 0, "rr": 0.0,
                     "n_hits": 0, "latency_ms": 0.0, "error": "boom"}},
        "q2": {"v": {"rank": None, "r5": 0, "r10": 0, "rr": 0.0,
                     "n_hits": 0, "latency_ms": 0.0, "error": "boom2"}},
    }
    summary = aggregate(per_query, ["v"], n=2)
    v = summary["per_strategy"]["v"]
    assert v["n_errors"] == 2
    assert v["p50_ms"] is None
    assert v["p95_ms"] is None


def test_is_relevant_tolerates_metadata_nested_source_file():
    """#65 Gemini HIGH — /search GET nests source_file under metadata;
    the predicate must find it whether it's at top-level or in metadata."""
    hit = {"metadata": {"source_file": "notes/abc.md"}, "text": "match"}
    assert is_relevant(hit, {
        "source_glob": "notes/*",
        "content_any": ["match"],
    }) is True


def test_aggregate_headline_picks_best_R5_strategy():
    per_query = {
        "q1": {
            "vector": {"rank": 5, "r5": 1, "r10": 1, "rr": 0.2,
                       "n_hits": 5, "latency_ms": 50},
            "hybrid": {"rank": 1, "r5": 1, "r10": 1, "rr": 1.0,
                       "n_hits": 5, "latency_ms": 500},
        },
    }
    summary = aggregate(per_query, ["vector", "hybrid"], n=1)
    # Both strategies have R@5 = 1.0 here, so headline picks one (either OK)
    assert "best_R@5" in summary["headline"]
    assert "(1.000)" in summary["headline"]["best_R@5"]


# --- probe_rerank (#113) -------------------------------------------------


def test_probe_rerank_reads_block_from_search_response(monkeypatch):
    """The probe reports the daemon's rerank state from the /search response's
    `rerank` block, so a baseline JSON is self-describing."""
    from sme.eval import candidate_strategy as cs

    def _fake_mcp(api_url, api_key, *, query, strategy, limit, timeout=60.0):
        return {
            "results": [{"text": "x"}],
            "rerank": {
                "enabled": True,
                "enabled_source": "env",
                "model": "ms-marco-MiniLM-L-12-v2",
                "status": "ok",
            },
        }, 12.0

    monkeypatch.setattr(cs, "mcp_search", _fake_mcp)
    meta = cs.probe_rerank("http://fake", "k")
    assert meta["rerank_enabled"] is True
    assert meta["rerank_model"] == "ms-marco-MiniLM-L-12-v2"
    assert meta["rerank_enabled_source"] == "env"
    assert meta["rerank_status"] == "ok"
    assert "rerank_probe_note" not in meta


def test_probe_rerank_no_block_records_none(monkeypatch):
    """Older daemon (no rerank block) → None + explanatory note, no raise."""
    from sme.eval import candidate_strategy as cs

    def _fake_mcp(api_url, api_key, *, query, strategy, limit, timeout=60.0):
        return {"results": [{"text": "x"}]}, 10.0  # no 'rerank' key

    monkeypatch.setattr(cs, "mcp_search", _fake_mcp)
    meta = cs.probe_rerank("http://fake", "k")
    assert meta["rerank_enabled"] is None
    assert meta["rerank_model"] is None
    assert "no rerank block" in meta["rerank_probe_note"]


def test_probe_rerank_swallows_daemon_error(monkeypatch):
    """A down daemon must not crash the run — probe records None + the error."""
    from sme.eval import candidate_strategy as cs

    def _boom(api_url, api_key, *, query, strategy, limit, timeout=60.0):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(cs, "mcp_search", _boom)
    meta = cs.probe_rerank("http://fake", "k")
    assert meta["rerank_enabled"] is None
    assert meta["rerank_model"] is None
    assert "connection refused" in meta["rerank_probe_note"]


def test_probe_rerank_disabled_block_reports_false(monkeypatch):
    """rerank present but disabled → enabled=False, model still recorded."""
    from sme.eval import candidate_strategy as cs

    def _fake_mcp(api_url, api_key, *, query, strategy, limit, timeout=60.0):
        return {
            "results": [{"text": "x"}],
            "rerank": {"enabled": False, "enabled_source": "env",
                       "model": "ms-marco-MiniLM-L-12-v2", "status": "noop"},
        }, 8.0

    monkeypatch.setattr(cs, "mcp_search", _fake_mcp)
    meta = cs.probe_rerank("http://fake", "k")
    assert meta["rerank_enabled"] is False
    assert meta["rerank_model"] == "ms-marco-MiniLM-L-12-v2"
    assert meta["rerank_status"] == "noop"


# --- run_eval_multi_limit ------------------------------------------------


def test_run_eval_multi_limit_dispatches_each_limit(monkeypatch):
    """Multi-limit sweep — each (strategy, limit) calls mcp_search once."""
    from sme.eval import candidate_strategy as cs

    calls: list[tuple] = []

    def _fake_mcp(api_url, api_key, *, query, strategy, limit, timeout=60.0):
        calls.append((query, strategy, limit))
        results = [{"text": "kill-cascade match" if limit >= 5 else "miss"}]
        return {"results": results}, 100.0

    monkeypatch.setattr(cs, "mcp_search", _fake_mcp)

    queries = [{
        "id": "q1", "query": "kill-cascade",
        "relevant": {"content_any": ["kill-cascade"]},
    }]
    report = cs.run_eval_multi_limit(
        api_url="http://fake", api_key="k",
        queries=queries, strategies=["vector", "hybrid"],
        limits=[5, 10, 20],
    )

    # 1 query × 2 strategies × 3 limits = 6 probes
    assert len(calls) == 6
    assert set(calls) == {
        ("kill-cascade", s, lim)
        for s in ["vector", "hybrid"] for lim in [5, 10, 20]
    }

    by_limit = report["summary_by_limit"]
    assert set(by_limit.keys()) == {5, 10, 20}
    for summary in by_limit.values():
        assert set(summary["per_strategy"].keys()) == {"vector", "hybrid"}


def test_run_eval_multi_limit_records_per_call_errors(monkeypatch):
    """When mcp_search raises for one cell, the rest still run and the
    failure is recorded as ``error`` on that record."""
    from sme.eval import candidate_strategy as cs

    def _fake_mcp(api_url, api_key, *, query, strategy, limit, timeout=60.0):
        if strategy == "hybrid" and limit == 20:
            raise RuntimeError("synthetic mcp failure")
        return {"results": [{"text": "match"}]}, 50.0

    monkeypatch.setattr(cs, "mcp_search", _fake_mcp)

    queries = [{"id": "q1", "query": "x", "relevant": {"content_any": ["match"]}}]
    report = cs.run_eval_multi_limit(
        api_url="http://fake", api_key="k",
        queries=queries, strategies=["vector", "hybrid"],
        limits=[10, 20],
    )

    failed = report["per_query_by_limit"][20]["q1"]["hybrid"]
    assert "error" in failed
    assert "synthetic mcp failure" in failed["error"]

    # Other cells unaffected
    assert "error" not in report["per_query_by_limit"][20]["q1"]["vector"]
    assert "error" not in report["per_query_by_limit"][10]["q1"]["vector"]
    assert "error" not in report["per_query_by_limit"][10]["q1"]["hybrid"]
