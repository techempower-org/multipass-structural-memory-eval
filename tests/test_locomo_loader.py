"""Tests for sme.corpora.locomo.loader.

Uses a small inline fixture (2 samples) rather than the upstream
~2.8 MB locomo10.json, so the tests run without any external download.
The fixture's schema mirrors the released
``snap-research/locomo/data/locomo10.json`` format documented in the
loader module docstring.

The pinned-subset assertions (SUBSET_QA_COUNT == 1986, adversarial
included) guard the comparability contract from
docs/research/2026-05-29-comparison-readiness.md §1.3 — they pin the
constants, not the fixture, so a future edit that silently changes the
pinned subset fails loudly.
"""
from __future__ import annotations

import json

import pytest

from sme.corpora.locomo import (
    ADVERSARIAL_INCLUDED,
    LOCOMO_CATEGORY_NAMES,
    LOCOMO_CATEGORY_TO_SME,
    SUBSET,
    SUBSET_QA_COUNT,
    SUBSET_SAMPLE_COUNT,
    LoCoMoQuestion,
    LoCoMoSession,
    LoCoMoTurn,
    load_questions,
    materialize_sme_corpus,
)


# Two samples mirroring locomo10.json's shape:
#  - conv-aa: 2 sessions, a multimodal turn, a multi-hop question whose
#    evidence spans both sessions, a single-hop question (int answer),
#    and an adversarial (category-5) question.
#  - conv-bb: 1 session, one temporal question with a non-conforming
#    evidence entry to exercise the skip path.
FIXTURE = [
    {
        "sample_id": "conv-aa",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "10:00 am on 1 Jan, 2023",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
                {
                    "speaker": "Bob",
                    "dia_id": "D1:2",
                    "text": "Cute! Here's a pic from the shelter.",
                    "img_url": ["https://example.com/rex.jpg"],
                    "blip_caption": "a brown dog sitting on a porch",
                    "query": "shelter dog",
                },
            ],
            "session_2_date_time": "3:00 pm on 14 Feb, 2023",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "Rex turned two years old today."},
                {"speaker": "Bob", "dia_id": "D2:2", "text": "Happy birthday Rex!"},
            ],
        },
        "qa": [
            {
                "question": "How old is Alice's dog and what is its name?",
                "answer": "Rex, two years old",
                "evidence": ["D1:1", "D2:1"],
                "category": 1,  # multi-hop (scorer numbering)
            },
            {
                "question": "How many dogs did Alice adopt?",
                "answer": 1,  # int answer -> coerced to str
                "evidence": ["D1:1"],
                "category": 2,  # single-hop
            },
            {
                "question": "What breed is Alice's cat?",
                "adversarial_answer": "tabby",
                "evidence": ["D1:1"],
                "category": 5,  # adversarial -> abstain
            },
        ],
        "event_summary": {},
        "observation": {},
        "session_summary": {},
    },
    {
        "sample_id": "conv-bb",
        "conversation": {
            "speaker_a": "Carol",
            "speaker_b": "Dave",
            "session_1_date_time": "9:00 am on 3 Mar, 2023",
            "session_1": [
                {"speaker": "Carol", "dia_id": "D1:1", "text": "I start my new job next Monday."},
            ],
        },
        "qa": [
            {
                "question": "When does Carol start her new job?",
                "answer": "next Monday",
                # one valid ref + one non-conforming ref (should be skipped)
                "evidence": ["D1:1", "bad-ref"],
                "category": 3,  # temporal
            },
        ],
    },
]


@pytest.fixture
def fixture_path(tmp_path):
    p = tmp_path / "locomo_fixture.json"
    p.write_text(json.dumps(FIXTURE))
    return p


# --- pinned subset contract (guards the comparability claim) ---------


def test_pinned_subset_constants():
    """The pinned subset must stay locomo10 / 1986 QA / adversarial
    included — these three facts are the comparability contract."""
    assert SUBSET == "locomo10"
    assert SUBSET_QA_COUNT == 1986
    assert SUBSET_SAMPLE_COUNT == 10
    assert ADVERSARIAL_INCLUDED is True


def test_category_names_and_sme_map_cover_all_five():
    assert set(LOCOMO_CATEGORY_NAMES) == {1, 2, 3, 4, 5}
    assert set(LOCOMO_CATEGORY_TO_SME) == {1, 2, 3, 4, 5}
    # scorer numbering: category 1 is multi-hop, not single-hop
    assert LOCOMO_CATEGORY_NAMES[1] == "multi-hop"
    assert LOCOMO_CATEGORY_NAMES[2] == "single-hop"
    assert LOCOMO_CATEGORY_NAMES[5] == "adversarial"
    assert LOCOMO_CATEGORY_TO_SME[2] == "cat_1"
    assert LOCOMO_CATEGORY_TO_SME[1] == "cat_2c"
    assert LOCOMO_CATEGORY_TO_SME[5] == "cat_1_negative"


# --- record schema ----------------------------------------------------


def test_load_questions_yields_dataclasses(fixture_path):
    questions = list(load_questions(fixture_path))
    assert len(questions) == 4  # 3 from conv-aa + 1 from conv-bb
    assert all(isinstance(q, LoCoMoQuestion) for q in questions)


def test_record_has_full_schema(fixture_path):
    q = next(load_questions(fixture_path))
    # every harness-expected field is present and typed
    assert q.question_id == "conv-aa::q0"
    assert q.sample_id == "conv-aa"
    assert q.category == 1
    assert q.question_type == "multi-hop"
    assert q.sme_category == "cat_2c"
    assert q.question.startswith("How old is Alice's dog")
    assert q.gold_answer == "Rex, two years old"
    assert q.is_adversarial is False
    assert q.adversarial_answer == ""
    assert q.evidence == ["D1:1", "D2:1"]
    assert q.speaker_a == "Alice" and q.speaker_b == "Bob"
    # shared haystack attached
    assert len(q.sessions) == 2
    assert all(isinstance(s, LoCoMoSession) for s in q.sessions)
    assert all(isinstance(t, LoCoMoTurn) for s in q.sessions for t in s.turns)


def test_multi_hop_evidence_spans_sessions(fixture_path):
    q = next(load_questions(fixture_path))
    # multi-session evidence -> two distinct session ids, order-preserving
    assert q.expected_sources_session_level() == ["D1", "D2"]
    turns = q.expected_sources_turn_level()
    assert any("adopted a dog named Rex" in t for t in turns)
    assert any("turned two years old" in t for t in turns)


def test_int_answer_coerced_to_str(fixture_path):
    q = list(load_questions(fixture_path))[1]
    assert q.question_type == "single-hop"
    assert q.sme_category == "cat_1"
    assert q.answer == "1"  # int 1 -> "1"


# --- adversarial flag -------------------------------------------------


def test_adversarial_item_carries_flag(fixture_path):
    q = list(load_questions(fixture_path))[2]
    assert q.is_adversarial is True
    assert q.category == 5
    assert q.question_type == "adversarial"
    assert q.sme_category == "cat_1_negative"
    # adversarial_answer preserved; gold answer is empty (abstain)
    assert q.adversarial_answer == "tabby"
    assert q.gold_answer == ""


def test_to_sme_question_preserves_adversarial(fixture_path):
    adv = list(load_questions(fixture_path))[2]
    sme_q = adv.to_sme_question()
    assert sme_q["id"] == "conv-aa::q2"
    assert sme_q["sme_category"] == "cat_1_negative"
    assert sme_q["locomo"]["is_adversarial"] is True
    assert sme_q["locomo"]["adversarial_answer"] == "tabby"
    assert sme_q["locomo"]["category"] == 5
    assert sme_q["locomo"]["question_type"] == "adversarial"


# --- evidence resolution edge cases -----------------------------------


def test_nonconforming_evidence_is_skipped(fixture_path):
    q = list(load_questions(fixture_path))[3]
    assert q.question_type == "temporal"
    assert q.sme_category == "cat_6"
    # "bad-ref" does not match D<n>:<m> -> dropped, only D1 survives
    assert q.expected_sources_session_level() == ["D1"]


def test_to_sme_question_shape(fixture_path):
    q = next(load_questions(fixture_path))
    sme_q = q.to_sme_question()
    assert sme_q["id"] == "conv-aa::q0"
    assert sme_q["sme_category"] == "cat_2c"
    assert sme_q["expected_sources"] == ["D1", "D2"]
    assert sme_q["gold_answer"] == "Rex, two years old"
    assert sme_q["locomo"]["sample_id"] == "conv-aa"
    assert sme_q["locomo"]["evidence"] == ["D1:1", "D2:1"]


def test_load_questions_rejects_non_list_top_level(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"samples": []}))
    with pytest.raises(ValueError, match="expected top-level JSON array"):
        list(load_questions(bad))


# --- materialization (per-sample vaults) ------------------------------


def test_materialize_writes_per_sample_vault(fixture_path, tmp_path):
    out = tmp_path / "locomo_corpus"
    summary = materialize_sme_corpus(load_questions(fixture_path), out)
    # 4 questions, but only 2 sample vaults (shared conversation)
    assert summary["questions_count"] == 4
    assert summary["samples_count"] == 2
    assert summary["sample_ids"] == ["conv-aa", "conv-bb"]
    vault = out / "vault"
    assert (vault / "conv-aa").is_dir()
    assert (vault / "conv-bb").is_dir()
    # conv-aa has 2 sessions -> 2 md files
    sessions = sorted((vault / "conv-aa").glob("*.md"))
    assert [p.stem for p in sessions] == ["D1", "D2"]
    sample = (vault / "conv-aa" / "D1.md").read_text()
    assert "sample_id: conv-aa" in sample
    assert "session_id: D1" in sample
    assert "source: locomo" in sample
    # multimodal turn folds the BLIP caption into the body
    assert "_[shared image: a brown dog sitting on a porch]_" in sample
    assert "<!-- img_url: https://example.com/rex.jpg -->" in sample


def test_materialize_respects_max_questions(fixture_path, tmp_path):
    out = tmp_path / "locomo_corpus"
    # cap at 3 questions -> only conv-aa's vault is written (its 3 qa)
    summary = materialize_sme_corpus(load_questions(fixture_path), out, max_questions=3)
    assert summary["questions_count"] == 3
    assert summary["samples_count"] == 1
    assert (out / "vault" / "conv-aa").is_dir()
    assert not (out / "vault" / "conv-bb").exists()


def test_materialize_writes_questions_yaml_with_subset_metadata(fixture_path, tmp_path):
    import yaml

    out = tmp_path / "locomo_corpus"
    materialize_sme_corpus(load_questions(fixture_path), out)
    qy = yaml.safe_load((out / "questions.yaml").read_text())
    # subset metadata is the comparability contract — must be recorded
    assert qy["subset"] == "locomo10"
    assert qy["subset_qa_count"] == 1986
    assert qy["adversarial_included"] is True
    assert "github.com/snap-research/locomo" in qy["source"]
    assert len(qy["questions"]) == 4
    assert qy["questions"][0]["id"] == "conv-aa::q0"
