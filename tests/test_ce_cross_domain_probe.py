"""Unit tests for scripts/ce_cross_domain_probe.py — issue #85."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ce_cross_domain_probe",
    Path(__file__).resolve().parents[1] / "scripts" / "ce_cross_domain_probe.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _items(*ids):
    return [{"corpus_id": i, "text": f"text for {i}"} for i in ids]


# ---- trust-gated rerank ------------------------------------------------
def test_no_scorer_keeps_bi_top1():
    items = _items("a", "b", "c")
    top1, override = mod.trust_gated_rerank("q", items, None, margin=1.0)
    assert top1 == "a"
    assert override is False


def test_single_item_keeps_bi_top1():
    items = _items("a")
    scorer = lambda q, texts: [9.0]  # noqa: E731
    top1, override = mod.trust_gated_rerank("q", items, scorer, margin=1.0)
    assert top1 == "a"
    assert override is False


def test_override_when_margin_exceeded():
    items = _items("a", "b", "c")
    # CE strongly prefers index 2 ("c") over index 0
    scorer = lambda q, texts: [0.1, 0.2, 5.0]  # noqa: E731
    top1, override = mod.trust_gated_rerank("q", items, scorer, margin=1.0)
    assert top1 == "c"
    assert override is True


def test_no_override_when_below_margin():
    items = _items("a", "b", "c")
    # CE prefers index 1 but only by 0.3 < margin 1.0
    scorer = lambda q, texts: [1.0, 1.3, 0.5]  # noqa: E731
    top1, override = mod.trust_gated_rerank("q", items, scorer, margin=1.0)
    assert top1 == "a"
    assert override is False


def test_no_override_when_ce_agrees_with_bi():
    items = _items("a", "b", "c")
    scorer = lambda q, texts: [9.0, 0.1, 0.1]  # noqa: E731  (idx 0 wins)
    top1, override = mod.trust_gated_rerank("q", items, scorer, margin=1.0)
    assert top1 == "a"
    assert override is False


# ---- evaluate aggregation ----------------------------------------------
def _run(qid, qtype, *ids, question="q"):
    return {
        "question_id": qid, "question_type": qtype, "question": question,
        "retrieval_results": {"ranked_items": _items(*ids)},
    }


def test_evaluate_baseline_only():
    runs = [
        _run("q1", "t", "g", "x"),   # baseline hit
        _run("q2", "t", "x", "g"),   # baseline miss
    ]
    gold = {"q1": {"g"}, "q2": {"g"}}
    res = mod.evaluate(runs, gold, None, margin=1.0, top_k=20)
    assert res["n"] == 2
    assert res["R@1_baseline"] == pytest.approx(0.5)
    assert res["R@1_final"] == pytest.approx(0.5)  # no scorer, no override
    assert res["total_overrides"] == 0


def test_evaluate_override_helps():
    runs = [_run("q1", "t", "x", "g", "z")]  # bi top1 "x" misses; "g" at idx1
    gold = {"q1": {"g"}}
    # CE strongly prefers idx 1 ("g")
    scorer = lambda q, texts: [0.0, 5.0, 0.0]  # noqa: E731
    res = mod.evaluate(runs, gold, scorer, margin=1.0, top_k=20)
    assert res["R@1_baseline"] == pytest.approx(0.0)
    assert res["R@1_final"] == pytest.approx(1.0)
    assert res["total_overrides"] == 1
    assert res["total_helped"] == 1
    assert res["total_hurt"] == 0


def test_evaluate_override_hurts():
    runs = [_run("q1", "t", "g", "x", "z")]  # bi top1 "g" hits
    gold = {"q1": {"g"}}
    scorer = lambda q, texts: [0.0, 5.0, 0.0]  # noqa: E731  (drags to "x")
    res = mod.evaluate(runs, gold, scorer, margin=1.0, top_k=20)
    assert res["R@1_baseline"] == pytest.approx(1.0)
    assert res["R@1_final"] == pytest.approx(0.0)
    assert res["total_overrides"] == 1
    assert res["total_hurt"] == 1


def test_evaluate_by_type_breakdown():
    runs = [
        _run("q1", "temporal", "g", "x"),
        _run("q2", "preference", "x", "g"),
    ]
    gold = {"q1": {"g"}, "q2": {"g"}}
    res = mod.evaluate(runs, gold, None, margin=1.0, top_k=20)
    assert res["by_type"]["temporal"]["R@1_baseline"] == pytest.approx(1.0)
    assert res["by_type"]["preference"]["R@1_baseline"] == pytest.approx(0.0)


def test_evaluate_empty():
    res = mod.evaluate([], {}, None, margin=1.0, top_k=20)
    assert res["n"] == 0
    assert res["R@1_baseline"] == 0.0


def test_top_k_clip_limits_rerank_window():
    # gold sits at index 2 but top_k=2 means CE never sees it
    runs = [_run("q1", "t", "x", "y", "g")]
    gold = {"q1": {"g"}}
    scorer = lambda q, texts: [0.0] * len(texts)  # noqa: E731
    res = mod.evaluate(runs, gold, scorer, margin=1.0, top_k=2)
    assert res["R@1_final"] == pytest.approx(0.0)


# ---- loaders + scorer-builder ------------------------------------------
def test_load_gold(tmp_path):
    f = tmp_path / "gold.json"
    f.write_text(json.dumps([
        {"question_id": "q1", "answer_session_ids": ["a", "b"]},
        {"question_id": "q2", "answer_session_ids": None},
    ]))
    gold = mod.load_gold(f)
    assert gold["q1"] == {"a", "b"}
    assert gold["q2"] == set()


def test_load_runs_jsonl(tmp_path):
    f = tmp_path / "run.jsonl"
    f.write_text(
        json.dumps(_run("q1", "t", "a")) + "\n"
        + "\n"  # blank line tolerated
        + json.dumps(_run("q2", "t", "b")) + "\n"
    )
    runs = mod.load_runs(f)
    assert [r["question_id"] for r in runs] == ["q1", "q2"]


def test_build_scorer_missing_checkpoint_returns_none(tmp_path):
    # nonexistent path -> None (graceful skip), regardless of ST install
    scorer = mod.build_cross_encoder_scorer(str(tmp_path / "nope"))
    assert scorer is None


def test_main_baseline_only_end_to_end(tmp_path):
    run_f = tmp_path / "run.jsonl"
    gold_f = tmp_path / "gold.json"
    run_f.write_text(
        json.dumps(_run("q1", "t", "g", "x")) + "\n"
        + json.dumps(_run("q2", "t", "x", "g")) + "\n"
    )
    gold_f.write_text(json.dumps([
        {"question_id": "q1", "answer_session_ids": ["g"]},
        {"question_id": "q2", "answer_session_ids": ["g"]},
    ]))
    out = tmp_path / "out.json"
    rc = mod.main(["--run", str(run_f), "--gold", str(gold_f),
                   "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["ce_active"] is False
    assert report["R@1_baseline"] == pytest.approx(0.5)
    assert report["R@1_final"] == pytest.approx(0.5)
