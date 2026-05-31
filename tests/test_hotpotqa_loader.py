"""Tests for sme.corpora.hotpotqa.loader.

Uses a small inline fixture (3 questions) rather than the upstream ~44 MB
hotpot_dev_distractor_v1.json, so the tests run without any external
download. The fixture's schema mirrors the released
``hotpotqa/hotpot/data`` format documented in the loader module docstring:
each record carries ``_id`` / ``question`` / ``answer`` / ``type`` /
``level`` / ``supporting_facts`` ([title, sent_id]) / ``context``
([title, [sentence, …]]).

The pinned-subset assertions guard the comparability contract (dev
distractor split, 7405 questions, every question 2-hop) — they pin the
constants, not the fixture, so a future edit that silently changes the
pinned subset fails loudly.
"""
from __future__ import annotations

import json

import pytest

from sme.corpora.hotpotqa import (
    HOTPOT_MIN_HOPS,
    HOTPOT_SME_CATEGORY,
    HOTPOT_TYPE_NAMES,
    SETTING,
    SUBSET,
    SUBSET_QUESTION_COUNT,
    HotpotParagraph,
    HotpotQuestion,
    load_questions,
    materialize_sme_corpus,
)


# Three records mirroring hotpot_dev_distractor_v1.json's shape:
#  - q-bridge: a bridge question whose two supporting facts span two
#    distinct gold paragraphs, plus one distractor paragraph.
#  - q-compare: a comparison question (yes/no answer), two gold paragraphs.
#  - q-edge: a record with an out-of-range supporting-fact sent_id and a
#    malformed supporting_facts entry, to exercise the tolerant skip paths.
FIXTURE = [
    {
        "_id": "q-bridge",
        "question": "What city is the company that employs Jane Doe headquartered in?",
        "answer": "Springfield",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": [
            ["Jane Doe", 0],
            ["Acme Corp", 1],
        ],
        "context": [
            ["Jane Doe", ["Jane Doe is an engineer.", "She works at Acme Corp."]],
            [
                "Acme Corp",
                ["Acme Corp makes widgets.", "Acme Corp is headquartered in Springfield."],
            ],
            ["Unrelated Topic", ["This paragraph is a distractor.", "It is not gold."]],
        ],
    },
    {
        "_id": "q-compare",
        "question": "Were the Eiffel Tower and the Statue of Liberty designed by the same person?",
        "answer": "no",
        "type": "comparison",
        "level": "medium",
        "supporting_facts": [
            ["Eiffel Tower", 0],
            ["Statue of Liberty", 0],
        ],
        "context": [
            ["Eiffel Tower", ["The Eiffel Tower was designed by Gustave Eiffel."]],
            ["Statue of Liberty", ["The Statue of Liberty was designed by Bartholdi."]],
        ],
    },
    {
        "_id": "q-edge",
        "question": "Edge-case question.",
        "answer": "n/a",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": [
            ["Only Para", 0],
            ["Only Para", 99],   # out-of-range sent_id -> skipped at sentence level
            ["Only Para"],        # malformed (length 1) -> dropped entirely
        ],
        "context": [
            ["Only Para", ["The single supporting sentence."]],
        ],
    },
]


@pytest.fixture
def fixture_path(tmp_path):
    p = tmp_path / "hotpot_fixture.json"
    p.write_text(json.dumps(FIXTURE))
    return p


# --- pinned subset contract (guards the comparability claim) ---------


def test_pinned_subset_constants():
    """The pinned subset must stay dev_distractor / distractor / 7405 /
    2-hop — these facts are the comparability contract."""
    assert SUBSET == "dev_distractor"
    assert SETTING == "distractor"
    assert SUBSET_QUESTION_COUNT == 7405
    assert HOTPOT_MIN_HOPS == 2


def test_type_names_and_sme_category():
    assert set(HOTPOT_TYPE_NAMES) == {"comparison", "bridge"}
    assert "2-hop" in HOTPOT_TYPE_NAMES["bridge"]
    assert "2-hop" in HOTPOT_TYPE_NAMES["comparison"]
    # all HotpotQA questions are the multi-hop SME category
    assert HOTPOT_SME_CATEGORY == "cat_2c"


# --- record schema ----------------------------------------------------


def test_load_questions_yields_dataclasses(fixture_path):
    questions = list(load_questions(fixture_path))
    assert len(questions) == 3
    assert all(isinstance(q, HotpotQuestion) for q in questions)


def test_record_has_full_schema(fixture_path):
    q = next(load_questions(fixture_path))
    assert q.question_id == "q-bridge"
    assert q.question.startswith("What city")
    assert q.gold_answer == "Springfield"
    assert q.qtype == "bridge"
    assert q.type_name == HOTPOT_TYPE_NAMES["bridge"]
    assert q.level == "hard"
    # every question is 2-hop, mapped to cat_2c
    assert q.min_hops == 2
    assert q.sme_category == "cat_2c"
    # supporting facts coerced to (str, int) tuples
    assert q.supporting_facts == [("Jane Doe", 0), ("Acme Corp", 1)]
    # haystack attached as typed paragraphs
    assert len(q.paragraphs) == 3
    assert all(isinstance(p, HotpotParagraph) for p in q.paragraphs)


def test_gold_paragraphs_flagged(fixture_path):
    """Paragraphs carrying a supporting fact are gold; the distractor is not."""
    q = next(load_questions(fixture_path))
    gold = {p.title for p in q.paragraphs if p.is_gold}
    not_gold = {p.title for p in q.paragraphs if not p.is_gold}
    assert gold == {"Jane Doe", "Acme Corp"}
    assert not_gold == {"Unrelated Topic"}


def test_paragraph_text_joins_sentences(fixture_path):
    q = next(load_questions(fixture_path))
    acme = next(p for p in q.paragraphs if p.title == "Acme Corp")
    assert acme.text == "Acme Corp makes widgets. Acme Corp is headquartered in Springfield."


# --- multi-hop evidence (the Cat 2c calibration target) --------------


def test_bridge_evidence_spans_two_gold_paragraphs(fixture_path):
    """A bridge question's expected sources are exactly its two gold
    paragraph titles — the multi-hop recall target."""
    q = next(load_questions(fixture_path))
    assert q.expected_sources_paragraph_level() == ["Jane Doe", "Acme Corp"]
    # multi-hop => at least two distinct gold paragraphs
    assert len(q.expected_sources_paragraph_level()) == 2


def test_sentence_level_evidence_texts(fixture_path):
    q = next(load_questions(fixture_path))
    sents = q.expected_sources_sentence_level()
    assert "Jane Doe is an engineer." in sents
    assert "Acme Corp is headquartered in Springfield." in sents


def test_comparison_question_yes_no_answer(fixture_path):
    q = list(load_questions(fixture_path))[1]
    assert q.qtype == "comparison"
    assert q.gold_answer == "no"
    assert q.expected_sources_paragraph_level() == ["Eiffel Tower", "Statue of Liberty"]


# --- tolerant parsing edge cases -------------------------------------


def test_out_of_range_and_malformed_supporting_facts(fixture_path):
    """An out-of-range sent_id is skipped at the sentence level; a
    length-1 (malformed) supporting_facts entry is dropped entirely."""
    q = list(load_questions(fixture_path))[2]
    # malformed ["Only Para"] dropped; the two well-formed (title, int) kept
    assert q.supporting_facts == [("Only Para", 0), ("Only Para", 99)]
    # only the in-range sentence text survives (sent_id 99 is out of range)
    assert q.expected_sources_sentence_level() == ["The single supporting sentence."]
    # gold_titles dedups the repeated title
    assert q.gold_titles == ["Only Para"]


def test_load_questions_rejects_non_list_top_level(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"questions": []}))
    with pytest.raises(ValueError, match="expected top-level JSON array"):
        list(load_questions(bad))


# --- to_sme_question shape -------------------------------------------


def test_to_sme_question_shape(fixture_path):
    q = next(load_questions(fixture_path))
    sme_q = q.to_sme_question()
    assert sme_q["id"] == "q-bridge"
    assert sme_q["sme_category"] == "cat_2c"
    assert sme_q["min_hops"] == 2
    assert sme_q["expected_sources"] == ["Jane Doe", "Acme Corp"]
    assert sme_q["gold_answer"] == "Springfield"
    assert sme_q["hotpotqa"]["type"] == "bridge"
    assert sme_q["hotpotqa"]["level"] == "hard"
    assert sme_q["hotpotqa"]["supporting_facts"] == [["Jane Doe", 0], ["Acme Corp", 1]]
    assert sme_q["hotpotqa"]["gold_titles"] == ["Jane Doe", "Acme Corp"]


# --- materialization (per-question vaults) ---------------------------


def test_materialize_writes_per_question_vault(fixture_path, tmp_path):
    out = tmp_path / "hotpot_corpus"
    summary = materialize_sme_corpus(load_questions(fixture_path), out)
    assert summary["questions_count"] == 3
    # 3 + 2 + 1 paragraphs across the three questions = 6
    assert summary["paragraphs_written"] == 6
    assert summary["gold_only"] is False
    vault = out / "vault"
    # each question gets its own vault dir keyed on _id
    assert (vault / "q-bridge").is_dir()
    assert (vault / "q-compare").is_dir()
    # bridge question wrote all 3 of its context paragraphs (incl. distractor)
    bridge_files = sorted(p.stem for p in (vault / "q-bridge").glob("*.md"))
    assert bridge_files == ["Acme_Corp", "Jane_Doe", "Unrelated_Topic"]
    acme = (vault / "q-bridge" / "Acme_Corp.md").read_text()
    assert "question_id: q-bridge" in acme
    assert "is_gold: true" in acme
    assert "source: hotpotqa" in acme
    assert "Acme Corp is headquartered in Springfield." in acme


def test_materialize_gold_only_drops_distractors(fixture_path, tmp_path):
    out = tmp_path / "hotpot_gold"
    summary = materialize_sme_corpus(
        load_questions(fixture_path), out, gold_only=True
    )
    assert summary["gold_only"] is True
    # gold-only: 2 + 2 + 1 = 5 paragraphs (the distractor is dropped)
    assert summary["paragraphs_written"] == 5
    vault = out / "vault"
    # the distractor paragraph is absent from the bridge vault
    bridge_files = sorted(p.stem for p in (vault / "q-bridge").glob("*.md"))
    assert bridge_files == ["Acme_Corp", "Jane_Doe"]
    assert not (vault / "q-bridge" / "Unrelated_Topic.md").exists()


def test_materialize_respects_max_questions(fixture_path, tmp_path):
    out = tmp_path / "hotpot_corpus"
    summary = materialize_sme_corpus(
        load_questions(fixture_path), out, max_questions=1
    )
    assert summary["questions_count"] == 1
    assert (out / "vault" / "q-bridge").is_dir()
    assert not (out / "vault" / "q-compare").exists()


def test_materialize_writes_questions_yaml_with_subset_metadata(fixture_path, tmp_path):
    import yaml

    out = tmp_path / "hotpot_corpus"
    materialize_sme_corpus(load_questions(fixture_path), out)
    qy = yaml.safe_load((out / "questions.yaml").read_text())
    # subset metadata is the comparability contract — must be recorded
    assert qy["subset"] == "dev_distractor"
    assert qy["setting"] == "distractor"
    assert qy["license"] == "CC BY-SA 4.0"
    assert "github.com/hotpotqa/hotpot" in qy["source"]
    assert len(qy["questions"]) == 3
    assert qy["questions"][0]["id"] == "q-bridge"
    assert qy["questions"][0]["min_hops"] == 2
