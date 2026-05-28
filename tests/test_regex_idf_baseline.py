"""Unit tests for scripts/regex_idf_baseline.py — issue #82."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "regex_idf_baseline",
    Path(__file__).resolve().parents[1] / "scripts" / "regex_idf_baseline.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_extract_entities_proper_nouns():
    ents = mod.extract_entities("Susan visited Apache Kafka in New York.")
    assert "susan" in ents
    assert "apache kafka" in ents
    assert "new york" in ents


def test_extract_entities_dates_and_numbers():
    # Regex caveat carried over from upstream: the unit list is singular
    # only (hour|min|year|...). "10 hours" with plural-s does NOT match;
    # "5km" / "10kg" do. Picking inputs that exercise actually-supported
    # units, leaving the plural gap as a known limitation worth a
    # follow-up if false-negatives matter (cf. techempower-org/adaptmem's
    # benchmarks/structural_memory_eval/entity_graph_baseline.py).
    ents = mod.extract_entities(
        "We met on 2026-05-15 and ran 5km that day; weight was 10kg."
    )
    assert "2026-05-15" in ents
    assert "5km" in ents
    assert "10kg" in ents


def test_extract_entities_dollar_and_url():
    ents = mod.extract_entities("Paid $50,000 to https://example.com/invoice")
    assert "$50,000" in ents
    assert "https://example.com/invoice" in ents


def test_extract_entities_stopwords_filtered():
    ents = mod.extract_entities("The User said Hello to the Assistant.")
    assert "user" not in ents
    assert "assistant" not in ents
    assert "hello" not in ents


def test_extract_entities_tech_tokens():
    ents = mod.extract_entities("Use the api_key with snake_case and CamelCase.")
    assert "api_key" in ents
    assert "snake_case" in ents
    assert "camelcase" in ents


def test_score_one_question_gold_at_rank_1():
    q = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What did Susan say about Apache Kafka?",
        "answer_session_ids": ["sess_gold"],
        "haystack_session_ids": ["sess_distractor", "sess_gold"],
        "haystack_sessions": [
            [{"role": "user", "content": "I like ice cream."}],
            [{"role": "user", "content": "Susan said Apache Kafka rules."}],
        ],
    }
    r = mod.score_one_question(q)
    assert r["question_id"] == "q1"
    assert r["ranked_top10"][0] == "sess_gold"
    assert r["hit_at_1"] == 1
    assert r["hit_at_5"] == 1
    assert r["hit_at_10"] == 1


def test_score_one_question_no_entity_overlap():
    q = {
        "question_id": "q2",
        "question_type": "abstention",
        "question": "What is the weather like on Mars?",
        "answer_session_ids": ["sess_gold"],
        "haystack_session_ids": ["sess_distractor", "sess_gold"],
        "haystack_sessions": [
            [{"role": "user", "content": "I like coffee in the morning."}],
            [{"role": "user", "content": "I had eggs for breakfast."}],
        ],
    }
    r = mod.score_one_question(q)
    assert r["top1_score"] == 0.0


def test_aggregate_computes_per_type():
    per_q = [
        {"question_id": "a", "question_type": "single-session", "hit_at_1": 1, "hit_at_5": 1, "hit_at_10": 1},
        {"question_id": "b", "question_type": "single-session", "hit_at_1": 0, "hit_at_5": 1, "hit_at_10": 1},
        {"question_id": "c", "question_type": "multi-session", "hit_at_1": 0, "hit_at_5": 0, "hit_at_10": 1},
    ]
    agg = mod.aggregate(per_q)
    assert agg["n"] == 3
    assert agg["R@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert agg["R@5"] == pytest.approx(2 / 3, abs=1e-4)
    assert agg["R@10"] == pytest.approx(3 / 3, abs=1e-4)
    by = agg["by_type"]
    assert by["single-session"]["n"] == 2
    assert by["single-session"]["R@5"] == 1.0
    assert by["multi-session"]["n"] == 1
    assert by["multi-session"]["R@1"] == 0.0
    assert by["multi-session"]["R@10"] == 1.0


def test_aggregate_empty_input():
    assert mod.aggregate([]) == {"n": 0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "by_type": {}}


def test_main_end_to_end(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps([
        {
            "question_id": "qa",
            "question_type": "single-session",
            "question": "What did Susan say about Apache Kafka?",
            "answer_session_ids": ["sess_gold"],
            "haystack_session_ids": ["sess_distractor", "sess_gold"],
            "haystack_sessions": [
                [{"role": "user", "content": "I like ice cream."}],
                [{"role": "user", "content": "Susan said Apache Kafka rules."}],
            ],
        },
        {
            "question_id": "qb",
            "question_type": "multi-session",
            "question": "When did Alice meet Bob?",
            "answer_session_ids": ["sess_bob"],
            "haystack_session_ids": ["sess_charlie", "sess_bob"],
            "haystack_sessions": [
                [{"role": "user", "content": "Charlie had lunch yesterday."}],
                [{"role": "user", "content": "Alice and Bob met on 2026-04-01."}],
            ],
        },
    ]))
    out = tmp_path / "out.json"
    rc = mod.main(["--questions", str(fixture), "--out", str(out), "--progress-every", "1"])
    assert rc == 0
    report = json.loads(out.read_text())
    assert "summary" in report and "per_q" in report
    assert report["summary"]["n"] == 2
    assert report["summary"]["R@5"] == 1.0
    assert {q["question_id"] for q in report["per_q"]} == {"qa", "qb"}
