"""Tests for sme.corpora.beam.loader.

Two fixtures:
  - an inline 1-conversation fixture (no download, no committed file) for
    fast schema/edge-case assertions — mirrors the LoCoMo loader test;
  - the committed pinned sample (sample/beam_100K_sample.json), a 12-turn
    / 10-question slice of the real 100K split, to prove the loader
    parses the actual released schema (string-encoded probing_questions,
    2D chat, rubric nuggets) without any network access.

The pinned-contract assertions (VALID_BUCKETS, QUESTIONS_PER_CONVERSATION,
ABILITY_TYPES) guard the comparability claim — a BEAM reading is
meaningless without its bucket, so the loader must always carry it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sme.corpora.beam import (
    ABILITY_TYPES,
    BEAM_ABILITY_TO_SME,
    QUESTIONS_PER_CONVERSATION,
    VALID_BUCKETS,
    BEAMQuestion,
    BEAMSession,
    BEAMTurn,
    load_questions,
    materialize_sme_corpus,
)

# The committed pinned sample (no download needed).
SAMPLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "sme" / "corpora" / "beam" / "sample" / "beam_100K_sample.json"
)


# An inline conversation mirroring the released BEAM shape:
#  - conversation_id "c1", conversation_seed with a category,
#  - chat as a 2D list (2 sessions x a couple of turns each), turns
#    carrying id / role / content / time_anchor,
#  - probing_questions stored as a STRING (as the real release does),
#    keyed by ability type: an information_extraction item with evidence
#    spanning one session, a multi_session_reasoning item spanning both,
#    and an abstention item (no source_chat_ids, ideal_response gold).
_INLINE_PQ = {
    "information_extraction": [
        {
            "question": "What database did I pick?",
            "answer": "You picked SQLite.",
            "difficulty": "easy",
            "question_type": "context_fact",
            "rubric": ["LLM response should state: SQLite"],
            "source_chat_ids": [1],
        }
    ],
    "multi_session_reasoning": [
        {
            "question": "How did the schema evolve?",
            "answer": "From SQLite to Postgres.",
            "difficulty": "medium",
            "question_type": "synthesis",
            "rubric": ["SQLite first", "then Postgres"],
            "source_chat_ids": [1, 10],
        }
    ],
    "abstention": [
        {
            "question": "What is my dog's name?",
            "ideal_response": "There is no information about a dog in the chat.",
            "difficulty": "medium",
            "abstention_type": "missing_detail",
            "rubric": ["no information about a dog"],
        }
    ],
}

FIXTURE = [
    {
        "conversation_id": "c1",
        "conversation_seed": {
            "category": "Coding",
            "id": 1,
            "subtopics": ["db"],
            "theme": "build an app",
            "title": "Budget tracker",
        },
        "user_profile": {"user_info": "dev", "user_relationships": ""},
        "chat": [
            [
                {"content": "I'm starting a budget app.", "id": 0, "index": "1,1",
                 "question_type": "main_question", "role": "user",
                 "time_anchor": "March-15-2024"},
                {"content": "I'll use SQLite for storage.", "id": 1, "index": "1,2",
                 "question_type": "follow_up", "role": "user",
                 "time_anchor": "March-15-2024"},
            ],
            [
                {"content": "Actually I migrated to Postgres.", "id": 10, "index": "2,1",
                 "question_type": "follow_up", "role": "user",
                 "time_anchor": "April-20-2024"},
                {"content": "Great, that scales better.", "id": 11, "index": "2,2",
                 "question_type": "assistant_reply", "role": "assistant",
                 "time_anchor": "April-20-2024"},
            ],
        ],
        # probing_questions is a STRING in the real release — exercise the parse path.
        "probing_questions": json.dumps(_INLINE_PQ),
    }
]


@pytest.fixture
def fixture_path(tmp_path):
    p = tmp_path / "beam_fixture.json"
    p.write_text(json.dumps(FIXTURE))
    return p


# --- pinned contract (guards the comparability claim) -----------------


def test_pinned_contract_constants():
    """Bucket grading + 20 questions/conversation are the comparability
    contract — pin them so a silent edit fails loudly."""
    assert VALID_BUCKETS == ("100K", "500K", "1M", "10M")
    assert QUESTIONS_PER_CONVERSATION == 20


def test_ability_types_and_sme_map_cover_all_ten():
    assert len(ABILITY_TYPES) == 10
    # every ability has an SME mapping (mapped or explicitly "unmapped")
    assert set(BEAM_ABILITY_TO_SME) == set(ABILITY_TYPES)
    assert BEAM_ABILITY_TO_SME["information_extraction"] == "cat_1"
    assert BEAM_ABILITY_TO_SME["multi_session_reasoning"] == "cat_2c"
    assert BEAM_ABILITY_TO_SME["abstention"] == "cat_1_negative"
    assert BEAM_ABILITY_TO_SME["temporal_reasoning"] == "cat_6"
    # generation-behavior abilities are deliberately unmapped
    assert BEAM_ABILITY_TO_SME["summarization"] == "unmapped"
    assert BEAM_ABILITY_TO_SME["instruction_following"] == "unmapped"


def test_load_requires_valid_bucket(fixture_path):
    with pytest.raises(ValueError, match="bucket must be one of"):
        list(load_questions(fixture_path, bucket="42K"))


# --- record schema (inline fixture) -----------------------------------


def test_load_questions_yields_dataclasses(fixture_path):
    qs = list(load_questions(fixture_path, bucket="100K"))
    assert len(qs) == 3  # IE + MSR + abstention
    assert all(isinstance(q, BEAMQuestion) for q in qs)


def test_record_has_full_schema(fixture_path):
    qs = list(load_questions(fixture_path, bucket="100K"))
    # iteration is in ABILITY_TYPES order -> abstention comes first
    ie = next(q for q in qs if q.ability_type == "information_extraction")
    assert ie.question_id == "100K::c1::q1"  # abstention is q0
    assert ie.conversation_id == "c1"
    assert ie.bucket == "100K"
    assert ie.sme_category == "cat_1"
    assert ie.question_subtype == "context_fact"
    assert ie.question == "What database did I pick?"
    assert ie.gold_answer == "You picked SQLite."
    assert ie.is_abstention is False
    assert ie.rubric == ["LLM response should state: SQLite"]
    assert ie.ground_truth_nuggets == "LLM response should state: SQLite"
    assert ie.source_chat_ids == [1]
    assert ie.category == "Coding"
    # shared haystack attached, two sessions
    assert len(ie.sessions) == 2
    assert all(isinstance(s, BEAMSession) for s in ie.sessions)
    assert all(isinstance(t, BEAMTurn) for s in ie.sessions for t in s.turns)


def test_question_id_is_bucket_scoped(fixture_path):
    """Bucket is part of the id so 100K and 1M readings of the same
    conversation never collide."""
    qs100 = list(load_questions(fixture_path, bucket="100K"))
    qs1m = list(load_questions(fixture_path, bucket="1M"))
    assert qs100[0].question_id.startswith("100K::")
    assert qs1m[0].question_id.startswith("1M::")
    assert qs100[0].bucket == "100K" and qs1m[0].bucket == "1M"


def test_evidence_resolution_single_and_multi_session(fixture_path):
    qs = list(load_questions(fixture_path, bucket="100K"))
    ie = next(q for q in qs if q.ability_type == "information_extraction")
    # turn id 1 lives in session 0 -> S0 only
    assert ie.expected_sources_session_level() == ["S0"]
    assert any("SQLite" in t for t in ie.expected_sources_turn_level())

    msr = next(q for q in qs if q.ability_type == "multi_session_reasoning")
    # ids 1 (S0) + 10 (S1) -> spans both sessions, order-preserving
    assert msr.expected_sources_session_level() == ["S0", "S1"]
    turns = msr.expected_sources_turn_level()
    assert any("SQLite" in t for t in turns)
    assert any("Postgres" in t for t in turns)


# --- abstention -------------------------------------------------------


def test_abstention_item_carries_flag(fixture_path):
    qs = list(load_questions(fixture_path, bucket="100K"))
    abs_q = next(q for q in qs if q.ability_type == "abstention")
    assert abs_q.is_abstention is True
    assert abs_q.sme_category == "cat_1_negative"
    # gold is the ideal refusal; the raw `answer` field is empty
    assert abs_q.answer == ""
    assert "no information about a dog" in abs_q.gold_answer
    # no evidence by construction -> empty expected sources
    assert abs_q.expected_sources_session_level() == []


def test_to_sme_question_shape(fixture_path):
    qs = list(load_questions(fixture_path, bucket="100K"))
    msr = next(q for q in qs if q.ability_type == "multi_session_reasoning")
    sme_q = msr.to_sme_question()
    assert sme_q["id"] == msr.question_id
    assert sme_q["sme_category"] == "cat_2c"
    assert sme_q["expected_sources"] == ["S0", "S1"]
    assert sme_q["gold_answer"] == "From SQLite to Postgres."
    assert sme_q["beam"]["bucket"] == "100K"
    assert sme_q["beam"]["ability_type"] == "multi_session_reasoning"
    assert sme_q["beam"]["rubric"] == ["SQLite first", "then Postgres"]
    assert sme_q["beam"]["ground_truth_nuggets"] == "SQLite first | then Postgres"
    assert sme_q["beam"]["source_chat_ids"] == [1, 10]


def test_load_questions_rejects_non_list_top_level(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"conversations": []}))
    with pytest.raises(ValueError, match="expected top-level JSON array"):
        list(load_questions(bad, bucket="100K"))


# --- materialization (per-conversation vaults) ------------------------


def test_materialize_writes_per_conversation_vault(fixture_path, tmp_path):
    out = tmp_path / "beam_corpus"
    summary = materialize_sme_corpus(load_questions(fixture_path, bucket="100K"), out)
    # 3 questions, 1 conversation vault (shared chat)
    assert summary["questions_count"] == 3
    assert summary["conversations_count"] == 1
    assert summary["conversation_ids"] == ["c1"]
    assert summary["buckets"] == ["100K"]
    vault = out / "vault"
    assert (vault / "c1").is_dir()
    # 2 sessions -> 2 md files
    sessions = sorted((vault / "c1").glob("*.md"))
    assert [p.stem for p in sessions] == ["S0", "S1"]
    note = (vault / "c1" / "S0.md").read_text()
    assert "conversation_id: c1" in note
    assert "session_id: S0" in note
    assert "bucket: 100K" in note
    assert "source: beam" in note
    # turn id preserved for the source_chat_ids join; time_anchor folded in
    assert "<!-- turn_id: 1 -->" in note
    assert "_Time anchor: March-15-2024_" in note


def test_materialize_writes_questions_yaml_with_contract_metadata(fixture_path, tmp_path):
    import yaml

    out = tmp_path / "beam_corpus"
    materialize_sme_corpus(load_questions(fixture_path, bucket="100K"), out)
    qy = yaml.safe_load((out / "questions.yaml").read_text())
    assert qy["buckets"] == ["100K"]
    assert qy["questions_per_conversation"] == 20
    assert "huggingface.co/datasets/Mohammadta/BEAM" in qy["dataset"]
    assert "github.com/mem0ai/memory-benchmarks" in qy["source"]
    assert len(qy["questions"]) == 3


def test_materialize_respects_max_questions(fixture_path, tmp_path):
    out = tmp_path / "beam_corpus"
    summary = materialize_sme_corpus(
        load_questions(fixture_path, bucket="100K"), out, max_questions=1
    )
    assert summary["questions_count"] == 1
    assert summary["conversations_count"] == 1


# --- the committed pinned sample (real released schema) ---------------


def test_pinned_sample_parses_real_schema():
    """The committed sample is a verbatim slice of the real 100K split
    (string-encoded probing_questions, 2D chat). Parsing it proves the
    loader handles the actual release, not just the inline fixture."""
    assert SAMPLE_PATH.exists(), f"missing pinned sample at {SAMPLE_PATH}"
    qs = list(load_questions(SAMPLE_PATH, bucket="100K"))
    # one question per ability type was kept -> 10 questions
    assert len(qs) == 10
    assert {q.ability_type for q in qs} == set(ABILITY_TYPES)
    # exactly one abstention item, flagged and mapped to the negative class
    abst = [q for q in qs if q.is_abstention]
    assert len(abst) == 1
    assert abst[0].sme_category == "cat_1_negative"
    # every non-abstention item carries rubric nuggets (the judge target)
    for q in qs:
        if not q.is_abstention:
            assert q.rubric, f"{q.ability_type} has no rubric nuggets"
    # the conversation seed category survives
    assert all(q.category for q in qs)
    # at least one question's evidence spans both sessions (the BEAM point)
    assert any(
        len(q.expected_sources_session_level()) >= 2 for q in qs
    )
