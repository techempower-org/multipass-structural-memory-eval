"""End-to-end tests for scripts/cross_validate_longmemeval.

Uses an inline 2-record LongMemEval-shape fixture (mirrors
test_longmemeval_loader.py) and the FullContextAdapter so the test
doesn't need ChromaDB or a network. The judge is mocked via the
``judge_client`` parameter on ``run()``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# scripts/ isn't a package — pull the harness in directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cross_validate_longmemeval as harness  # noqa: E402


FIXTURE = [
    {
        "question_id": "test_001_temporal",
        "question_type": "temporal-reasoning",
        "question": "What was the first issue I had with my new car?",
        "answer": "GPS system not functioning correctly",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_session_ids": ["sess_001_a", "sess_001_b"],
        "haystack_dates": [
            "2023/03/01 (Wed) 10:00",
            "2023/03/15 (Wed) 14:30",
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I just bought a new car."},
                {"role": "assistant", "content": "Congrats!"},
            ],
            [
                {"role": "user",
                 "content": "GPS not working since firmware update.",
                 "has_answer": True},
                {"role": "assistant", "content": "Annoying."},
            ],
        ],
        "answer_session_ids": ["sess_001_b"],
    },
    {
        "question_id": "test_002_abstain_abs",
        "question_type": "single-session-user",
        "question": "What did I say about my submarine?",
        "answer": "abstain",
        "question_date": "2023/05/01 (Mon) 09:00",
        "haystack_session_ids": ["sess_002_a"],
        "haystack_dates": ["2023/04/20 (Thu) 12:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I bought a kayak last week."},
                {"role": "assistant", "content": "Sounds fun!"},
            ],
        ],
        "answer_session_ids": [],
    },
]


@pytest.fixture
def dataset(tmp_path):
    p = tmp_path / "lme_oracle.json"
    p.write_text(json.dumps(FIXTURE))
    return p


@pytest.fixture
def args_factory(tmp_path):
    """Build an argparse.Namespace mirroring the CLI defaults."""

    def _build(dataset_path, *, adapter="full-context", skip_judge=True,
               skip_reader=True, max_questions=None):
        return SimpleNamespace(
            dataset=dataset_path,
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


# --- skip-judge path (SME-only) --------------------------------------------

def test_skip_judge_runs_sme_only(dataset, args_factory):
    args = args_factory(dataset, skip_judge=True)
    report = harness.run(args)
    assert report["run_metadata"]["adapter"] == "full-context"
    assert report["run_metadata"]["skip_judge"] is True
    assert report["run_metadata"]["judge_model"] is None
    pq = report["per_question"]
    assert len(pq) == 2
    # FullContextAdapter dumps the entire vault into context, so the
    # session id ("sess_001_b") should appear and SME recall should be 1.0
    rec0 = next(r for r in pq if r["question_id"] == "test_001_temporal")
    assert rec0["sme_recall"] == 1.0
    assert "sess_001_b" in rec0["matched_sources"]
    assert rec0["judge"] is None
    # Abstention question: no expected sources, recall stays at 0.0.
    rec1 = next(r for r in pq if r["question_id"] == "test_002_abstain_abs")
    assert rec1["expected_sources"] == []
    assert rec1["sme_recall"] == 0.0

    # Per-category aggregation populated.
    per_cat = report["summary"]["per_category"]
    assert "cat_6" in per_cat
    assert per_cat["cat_6"]["n"] == 1
    assert per_cat["cat_6"]["judge_correct_rate"] is None  # judge skipped
    assert per_cat["cat_6"]["sme_recall_mean"] == 1.0


def test_max_questions_caps_iteration(dataset, args_factory):
    args = args_factory(dataset, skip_judge=True, max_questions=1)
    report = harness.run(args)
    assert len(report["per_question"]) == 1


# --- mocked judge path -----------------------------------------------------

class _CannedJudgeClient:
    """Always votes 'yes'. The canonical judge maps yes -> ABSTAIN on the
    abstention (unanswerable) template and yes -> CORRECT otherwise, so the
    temporal q grades CORRECT and the abstention q grades ABSTAIN."""

    def __init__(self):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, *, model, messages, temperature=0.0,
                       max_tokens=None):
                outer.calls.append(messages[0]["content"])
                # Canonical binary verdict: the parser reads "yes" in the reply.
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="yes"))],
                    usage=SimpleNamespace(
                        prompt_tokens=20, completion_tokens=8,
                        total_tokens=28,
                    ),
                )

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_with_mocked_judge_records_per_question_labels(dataset, args_factory):
    args = args_factory(dataset, skip_judge=False, skip_reader=True)
    judge_client = _CannedJudgeClient()
    report = harness.run(args, judge_client=judge_client)

    pq = report["per_question"]
    assert len(pq) == 2
    rec0 = next(r for r in pq if r["question_id"] == "test_001_temporal")
    assert rec0["judge"]["autoeval_label"] == "CORRECT"
    assert rec0["judge"]["usage"]["total_tokens"] == 28
    rec1 = next(r for r in pq if r["question_id"] == "test_002_abstain_abs")
    # Abstention question gets routed to the abstention rubric.
    assert rec1["judge"]["autoeval_label"] == "ABSTAIN"
    # Two judge calls happened.
    assert len(judge_client.calls) == 2


def test_aggregation_reports_judge_correct_rate(dataset, args_factory):
    args = args_factory(dataset, skip_judge=False, skip_reader=True)
    judge_client = _CannedJudgeClient()
    report = harness.run(args, judge_client=judge_client)

    per_cat = report["summary"]["per_category"]
    assert per_cat["cat_6"]["judge_correct_rate"] == 1.0
    assert per_cat["cat_1_negative"]["judge_correct_rate"] == 1.0
    # KU caveat is preserved in the report.
    assert "Cat 3" in report["summary"]["ku_caveat"]
    # Token usage is summed.
    assert report["summary"]["judge_total_usage"]["total_tokens"] == 56


def test_aggregation_emits_r1_misses_with_qtype_histogram(dataset, args_factory):
    """#53 sub-task 2 — r1_misses + per-type histogram in the report.

    Verifies the aggregate output includes a top-level r1_misses list and
    r1_miss_by_type histogram. The flat-baseline adapter (used by the
    test fixture's args_factory) returns no ranked retrieval, so every
    fixture question is a rank-1 miss — each row should land in the list
    bucketed by question_type.
    """
    args = args_factory(dataset, skip_judge=True, skip_reader=True)
    report = harness.run(args)
    summary = report["summary"]
    assert "r1_misses" in summary
    assert "r1_miss_by_type" in summary
    assert isinstance(summary["r1_misses"], list)
    assert isinstance(summary["r1_miss_by_type"], dict)
    # Histogram rows match the question_type values seen in the fixture.
    total_misses = sum(summary["r1_miss_by_type"].values())
    assert total_misses == len(summary["r1_misses"])


def test_per_question_record_has_rank_diagnostics(dataset, args_factory):
    """#53 sub-task 2 — every per-question record carries rank-aware fields."""
    args = args_factory(dataset, skip_judge=True, skip_reader=True)
    report = harness.run(args)
    for rec in report["per_question"]:
        for key in ("retrieved_rank_1", "hit_at_1", "hit_at_5", "hit_at_10"):
            assert key in rec, f"record missing {key!r}: {rec.keys()}"


def test_aggregation_records_disagreements(tmp_path, args_factory):
    """If SME recall is high but judge says INCORRECT, harness flags it."""
    # Construct a fixture where the abstention case has SME recall=0
    # but a misbehaving judge marks it INCORRECT — that's a
    # disagreement (sme says wrong, judge says wrong → AGREE) so let's
    # instead force the temporal record to disagree.
    dataset_path = tmp_path / "lme_oracle.json"
    dataset_path.write_text(json.dumps(FIXTURE))
    args = args_factory(dataset_path, skip_judge=False, skip_reader=True)

    class _AlwaysIncorrect:
        def __init__(self):
            class _C:
                def create(self, *, model, messages, temperature=0.0,
                           max_tokens=None):
                    # Canonical binary verdict: "no" -> INCORRECT.
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(
                            content="no"
                        ))],
                        usage=SimpleNamespace(
                            prompt_tokens=1, completion_tokens=1,
                            total_tokens=2),
                    )

            class _Chat:
                completions = _C()

            self.chat = _Chat()

    report = harness.run(args, judge_client=_AlwaysIncorrect())
    disagreements = report["summary"]["disagreements"]
    # The temporal question has SME recall 1.0 (full-context dumps the
    # whole vault, sess_001_b appears) but judge says INCORRECT → disagree.
    qids = {d["question_id"] for d in disagreements}
    assert "test_001_temporal" in qids


# --- helpers ---------------------------------------------------------------

def test_sme_substring_recall_basic():
    score, matched = harness.sme_substring_recall(
        retrieved="alpha and bravo are present",
        expected=["alpha", "bravo", "charlie"],
    )
    assert matched == ["alpha", "bravo"]
    assert score == pytest.approx(2 / 3)


def test_sme_substring_recall_empty_expected():
    score, matched = harness.sme_substring_recall(
        retrieved="anything", expected=[],
    )
    assert score == 0.0
    assert matched == []


def test_judge_label_to_correct():
    assert harness.judge_label_to_correct("CORRECT") is True
    assert harness.judge_label_to_correct("ABSTAIN") is True
    assert harness.judge_label_to_correct("INCORRECT") is False
    assert harness.judge_label_to_correct("PARTIAL") is False
    assert harness.judge_label_to_correct("ERROR") is None


# --- arg parser smoke ------------------------------------------------------

def test_arg_parser_requires_dataset_and_adapter():
    parser = harness.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args([
        "--dataset", "/tmp/x.json",
        "--adapter", "full-context",
    ])
    assert parsed.adapter == "full-context"
    assert parsed.skip_judge is False
    assert parsed.reader_model == "gpt-4o-mini"


# --- BEAM bucket retrieval depth + hit-metric id normalization -------------

def test_arg_parser_beam_n_results_default_and_override():
    """``--beam-n-results`` defaults to 5 (the 100K full-context regime)
    and is overridable for the 500K/1M retrieval regime."""
    parser = harness.build_arg_parser()
    default = parser.parse_args(["--dataset", "/tmp/x.json", "--adapter", "flat"])
    assert default.beam_n_results == 5
    override = parser.parse_args(
        ["--dataset", "/tmp/x.json", "--adapter", "flat",
         "--corpus", "beam", "--bucket", "1M", "--beam-n-results", "2"]
    )
    assert override.beam_n_results == 2


def test_run_beam_questions_threads_n_results_to_adapter():
    """The per-query retrieval depth reaches ``adapter.query`` so the
    500K/1M buckets can bound the retrieved context to the reader window."""
    seen: dict[str, int] = {}

    class _RecordingAdapter:
        def query(self, question, *, n_results=None):
            seen["n_results"] = n_results
            return harness.QueryResult(answer="", context_string="ctx",
                                       retrieved_entities=[])

        def close(self):
            pass

    # A minimal BEAMQuestion-shaped stand-in (only the attrs the runner
    # touches). One conversation, one question, no abstention.
    class _Q:
        question_id = "1M::c1::q0"
        conversation_id = "c1"
        bucket = "1M"
        ability_type = "information_extraction"
        sme_category = "cat_1"
        is_abstention = False
        question = "what?"
        gold_answer = "x"
        ground_truth_nuggets = "x"
        sessions: list = []

        @staticmethod
        def expected_sources_session_level():
            return []

        def to_sme_question(self):
            return {"id": self.question_id, "bucket": self.bucket}

    import tempfile

    with tempfile.TemporaryDirectory() as wd:
        # adapter_factory ignores the vault path and returns our recorder;
        # the runner still materializes the (empty-session) vault once.
        recs = harness.run_beam_questions(
            [_Q()],
            adapter_factory=lambda _vault: _RecordingAdapter(),
            work_dir=Path(wd),
            skip_judge=True,
            skip_reader=True,
            reader_model="x",
            judge_model="x",
            n_results=2,
        )
    assert seen["n_results"] == 2
    assert recs[0]["bucket"] == "1M"


def test_chunk_prefixed_ids_normalize_for_hit_metric():
    """The flat baseline labels entities ``chunk:S0``; expected sources
    are bare ``S0``. Hit-at-K must normalize the prefix so per-session
    retrieval recall is real, while the raw ids are preserved."""
    from sme.adapters.base import Entity

    result = harness.QueryResult(
        answer="", context_string="S0 body",
        retrieved_entities=[
            Entity(id="chunk:S0", name="S0.md", entity_type="chunk"),
            Entity(id="chunk:S3", name="S3.md", entity_type="chunk"),
        ],
    )
    rec = harness._score_and_judge(
        question="q", question_id="100K::c1::q0",
        question_type="information_extraction", sme_category="cat_1",
        is_abstention=False, gold_answer="x", expected=["S0"],
        result=result, skip_judge=True, skip_reader=True,
        reader_model="x", judge_model="x",
    )
    assert rec["hit_at_1"] is True          # chunk:S0 -> S0 matches expected
    assert rec["hit_at_5"] is True
    # raw ids preserved verbatim for provenance
    assert rec["retrieved_entity_ids"] == ["chunk:S0", "chunk:S3"]
