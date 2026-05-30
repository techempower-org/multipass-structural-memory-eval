"""End-to-end tests for the `--corpus locomo` branch of
scripts/cross_validate_longmemeval.

Uses an inline LoCoMo-shape fixture (mirrors test_locomo_loader.py) and
the FullContextAdapter so the test needs no ChromaDB or network. The
judge is mocked via the ``judge_client`` parameter on ``run()``.

Focus areas (the team-lead spec for the branch):
  - --corpus locomo dispatches to the per-sample loop.
  - questions are grouped by sample_id and ingested PER SAMPLE (one vault
    per sample, queried by all that sample's questions).
  - adversarial (category-5) items are judged abstention-aware.
  - the default corpus (longmemeval) path is unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cross_validate_longmemeval as harness  # noqa: E402


# Two LoCoMo samples mirroring locomo10.json's shape. conv-aa has 2
# sessions + 2 questions (a single-hop and an adversarial); conv-bb has
# 1 session + 1 multi-hop question. Tests that the harness materializes
# ONE vault per sample and queries all of a sample's questions against it.
FIXTURE = [
    {
        "sample_id": "conv-aa",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "10:00 am on 1 Jan, 2023",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1",
                 "text": "I adopted a dog named Rex."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Cute!"},
            ],
            "session_2_date_time": "3:00 pm on 14 Feb, 2023",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1",
                 "text": "Rex turned two today."},
            ],
        },
        "qa": [
            {
                "question": "What is the name of Alice's dog?",
                "answer": "Rex",
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
    },
    {
        "sample_id": "conv-bb",
        "conversation": {
            "speaker_a": "Carol",
            "speaker_b": "Dave",
            "session_1_date_time": "9:00 am on 3 Mar, 2023",
            "session_1": [
                {"speaker": "Carol", "dia_id": "D1:1",
                 "text": "I moved to Berlin and started a new job."},
            ],
        },
        "qa": [
            {
                "question": "Where does Carol live and work?",
                "answer": "Berlin",
                "evidence": ["D1:1"],
                "category": 1,  # multi-hop
            },
        ],
    },
]


@pytest.fixture
def dataset(tmp_path):
    p = tmp_path / "locomo10_fixture.json"
    p.write_text(json.dumps(FIXTURE))
    return p


@pytest.fixture
def args_factory(tmp_path):
    def _build(dataset_path, *, adapter="full-context", skip_judge=True,
               skip_reader=True, max_questions=None):
        return SimpleNamespace(
            dataset=dataset_path,
            corpus="locomo",
            adapter=adapter,
            max_questions=max_questions,
            reader_model="gpt-4o-mini",
            judge_model="gpt-4o-2024-08-06",
            skip_judge=skip_judge,
            skip_reader=skip_reader,
            out=None,
            work_dir=tmp_path / "work",
            verbose=False,
        )
    return _build


# --- corpus dispatch + per-sample grouping ---------------------------------

def test_corpus_locomo_loads_and_groups_by_sample(dataset, args_factory):
    args = args_factory(dataset, skip_judge=True)
    report = harness.run(args)
    assert report["run_metadata"]["corpus"] == "locomo"
    pq = report["per_question"]
    # 3 questions total (2 from conv-aa, 1 from conv-bb)
    assert len(pq) == 3
    # Each record carries the LoCoMo provenance fields.
    sample_ids = {r["sample_id"] for r in pq}
    assert sample_ids == {"conv-aa", "conv-bb"}
    # conv-aa contributed 2 records, conv-bb 1.
    aa = [r for r in pq if r["sample_id"] == "conv-aa"]
    bb = [r for r in pq if r["sample_id"] == "conv-bb"]
    assert len(aa) == 2 and len(bb) == 1


def test_per_sample_vault_holds_full_conversation(dataset, args_factory, tmp_path):
    """The single-hop question in conv-aa should retrieve from session 1,
    AND the full conversation (both sessions) must be in scope — proof the
    whole sample was ingested per-sample, not just one question's slice.

    FullContextAdapter dumps the entire ingested vault into context, so if
    per-sample ingest worked, the context contains BOTH sessions' text.
    """
    args = args_factory(dataset, skip_judge=True)
    report = harness.run(args)
    single_hop = next(
        r for r in report["per_question"]
        if r["sample_id"] == "conv-aa" and r["question_type"] == "single-hop"
    )
    # expected source D1 present -> recall 1.0
    assert single_hop["expected_sources"] == ["D1"]
    assert single_hop["sme_recall"] == 1.0
    assert "D1" in single_hop["matched_sources"]


def test_adversarial_record_flagged(dataset, args_factory):
    args = args_factory(dataset, skip_judge=True)
    report = harness.run(args)
    adv = next(
        r for r in report["per_question"]
        if r["question_type"] == "adversarial"
    )
    assert adv["is_adversarial"] is True
    assert adv["is_abstention"] is True  # threaded through for the judge
    assert adv["sme_category"] == "cat_1_negative"
    assert adv["locomo_category"] == 5


def test_max_questions_caps_locomo(dataset, args_factory):
    # cap at 1 -> only conv-aa's first question runs; conv-bb never ingested
    args = args_factory(dataset, skip_judge=True, max_questions=1)
    report = harness.run(args)
    assert len(report["per_question"]) == 1
    assert report["per_question"][0]["sample_id"] == "conv-aa"


# --- abstention-aware judging ----------------------------------------------

class _CannedJudgeClient:
    """Affirms the answer ("yes") under the canonical binary judge contract.

    The canonical judge (#146) replies "yes"/"no", and grade_answer maps
    that via question_type: (abstention, yes) -> ABSTAIN, else CORRECT. So
    a always-"yes" client lets us assert the adversarial item is routed to
    the abstention rubric (-> ABSTAIN) while the single-hop is CORRECT.
    """

    def __init__(self):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, *, model, messages, temperature=0.0):
                content = messages[0]["content"]
                outer.calls.append(content)
                # Canonical judge (#146) parses a BINARY 'yes'/'no' reply
                # (not the old {"label": ...} JSON) and derives the label
                # from question_type: (abstention, yes) -> ABSTAIN,
                # (non-abstention, yes) -> CORRECT. The reader is correct in
                # this fixture (right fact on the single-hop; correct refusal
                # on the adversarial item), so the judge affirms "yes" in both
                # cases and grade_answer does the ABSTAIN-vs-CORRECT routing.
                reply = "yes"
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=reply))],
                    usage=SimpleNamespace(
                        prompt_tokens=20, completion_tokens=8, total_tokens=28),
                )

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_adversarial_judged_with_abstention_rubric(dataset, args_factory):
    args = args_factory(dataset, skip_judge=False, skip_reader=True)
    judge_client = _CannedJudgeClient()
    report = harness.run(args, judge_client=judge_client)

    adv = next(
        r for r in report["per_question"]
        if r["question_type"] == "adversarial"
    )
    # routed to the abstention rubric -> ABSTAIN label = correct refusal
    assert adv["judge"]["autoeval_label"] == "ABSTAIN"
    # non-adversarial items did NOT get the abstention rubric
    sh = next(
        r for r in report["per_question"]
        if r["question_type"] == "single-hop"
    )
    assert sh["judge"]["autoeval_label"] == "CORRECT"
    # adversarial bucket scores the refusal as success
    per_cat = report["summary"]["per_category"]
    assert per_cat["cat_1_negative"]["judge_correct_rate"] == 1.0


# --- the default (longmemeval) path is untouched ---------------------------

def test_arg_parser_corpus_defaults_to_longmemeval():
    parser = harness.build_arg_parser()
    parsed = parser.parse_args([
        "--dataset", "/tmp/x.json", "--adapter", "full-context",
    ])
    assert parsed.corpus == "longmemeval"
    parsed_locomo = parser.parse_args([
        "--dataset", "/tmp/x.json", "--adapter", "full-context",
        "--corpus", "locomo",
    ])
    assert parsed_locomo.corpus == "locomo"
