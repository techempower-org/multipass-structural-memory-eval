"""End-to-end tests for scripts/run_longmemeval_mempalace.

Exercises the issue #19 run script with mocked adapters, mocked daemon
ingest, and mocked judge so no live daemon or OpenAI key is needed.
Mirrors the fixture pattern from test_cross_validate_longmemeval to keep
the two harnesses' tests visibly comparable.
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

import run_longmemeval_mempalace as runner  # noqa: E402
from sme.adapters.base import QueryResult, SMEAdapter  # noqa: E402
from sme.corpora.longmemeval import load_questions  # noqa: E402


# --- Inline 2-record fixture (mirrors test_cross_validate_longmemeval) -----

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
    """Build an argparse.Namespace mirroring the runner CLI defaults."""

    def _build(
        dataset_path,
        *,
        adapter="mempalace-daemon",
        api_url="http://fake-daemon:8085",
        api_key="test-key",
        dry_run=False,
        skip_judge=True,
        skip_reader=True,
        max_questions=None,
        familiar_url="http://familiar:8080",
        familiar_inference=False,
    ):
        return SimpleNamespace(
            adapter=adapter,
            questions=dataset_path,
            api_url=api_url,
            api_key=api_key,
            kind=None,
            familiar_url=familiar_url,
            familiar_inference=familiar_inference,
            answer_model="o4-mini",
            judge="gpt-5.3-chat",
            max_questions=max_questions,
            skip_judge=skip_judge,
            skip_reader=skip_reader,
            dry_run=dry_run,
            json=None,
            work_dir=tmp_path / "work",
            verbose=False,
        )
    return _build


# --- Mock adapter + ingest client -------------------------------------------

class FakeAdapter(SMEAdapter):
    """Returns the question text verbatim as the retrieved context.

    For Cat 1 substring scoring this means the only expected_source that
    matches is one that appears literally in the question — which means
    test_001_temporal (expected_source: sess_001_b) gets recall 0.0 and
    test_002 (no expected sources) also gets recall 0.0. We don't need
    high recall to test the run-script wiring; we need the adapter to
    not raise.
    """

    def __init__(self, wing=None):
        self.wing = wing
        self.queries: list[str] = []

    def ingest_corpus(self, corpus):
        return {"entities_created": 0, "edges_created": 0,
                "errors": [], "warnings": []}

    def query(self, question, n_results=5):
        self.queries.append(question)
        return QueryResult(
            answer="",
            context_string=f"[fake] {question}",
            retrieved_entities=[],
        )

    def get_graph_snapshot(self):
        return [], []


class FakeIngestClient:
    """Records every post_memory call so the test can assert the
    per-question wing + session count."""

    def __init__(self):
        self.posted: list[dict] = []

    def post_memory(self, *, content, wing, room="longmemeval"):
        self.posted.append({"content": content, "wing": wing, "room": room})
        return 200, {}

    def post_flush(self):
        return 200, {}


class _CannedJudgeClient:
    """Returns CORRECT for the temporal q, ABSTAIN for the abstention q."""

    def __init__(self):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, *, model, messages, temperature=0.0):
                outer.calls.append(messages[0]["content"])
                content = messages[0]["content"]
                if "Abstention" in content:
                    label, rationale = "ABSTAIN", "system refused"
                else:
                    label, rationale = "CORRECT", "matches gold"
                payload = (
                    '{"label": "' + label + '", '
                    '"rationale": "' + rationale + '"}'
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=payload))],
                    usage=SimpleNamespace(
                        prompt_tokens=20, completion_tokens=8,
                        total_tokens=28,
                    ),
                )

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


# --- Dry-run path -----------------------------------------------------------

def test_dry_run_estimates_cost_without_calling_apis(dataset, args_factory):
    args = args_factory(dataset, dry_run=True)
    report = runner.run(args)
    assert report["run_metadata"]["mode"] == "dry-run"
    cost = report["cost_estimate"]
    assert cost["n_questions"] == 2
    assert cost["reader_model"] == "o4-mini"
    assert cost["judge_model"] == "gpt-5.3-chat"
    # Cost should be small but non-zero for both models.
    assert cost["reader_usd"] > 0
    assert cost["judge_usd"] > 0
    assert cost["total_usd"] == round(cost["reader_usd"] + cost["judge_usd"], 4)


def test_dry_run_respects_max_questions(dataset, args_factory):
    args = args_factory(dataset, dry_run=True, max_questions=1)
    report = runner.run(args)
    assert report["cost_estimate"]["n_questions"] == 1


# --- Live (mocked) path -----------------------------------------------------

def test_run_with_mocked_daemon_ingests_per_question(dataset, args_factory):
    args = args_factory(dataset, skip_judge=True)
    ingest = FakeIngestClient()

    def _factory(q, _vault):
        return FakeAdapter(wing=f"lme_{q.question_id}")

    report = runner.run(
        args,
        ingest_client=ingest,
        factory_fn=_factory,
    )

    # Each question's sessions were posted under its own wing
    wings_posted = {p["wing"] for p in ingest.posted}
    assert wings_posted == {"lme_test_001_temporal", "lme_test_002_abstain_abs"}

    # Question 1 has 2 sessions, question 2 has 1
    by_wing = {}
    for p in ingest.posted:
        by_wing.setdefault(p["wing"], 0)
        by_wing[p["wing"]] += 1
    assert by_wing["lme_test_001_temporal"] == 2
    assert by_wing["lme_test_002_abstain_abs"] == 1

    # Every posted drawer landed in the references room
    assert all(p["room"] == "references" for p in ingest.posted)

    # Per-question records were emitted by the harness
    assert report["run_metadata"]["adapter"] == "mempalace-daemon"
    assert report["run_metadata"]["ingested_per_question"] is True
    assert len(report["per_question"]) == 2


def test_run_with_mocked_judge_records_qa_accuracy(dataset, args_factory):
    args = args_factory(dataset, skip_judge=False, skip_reader=True)
    ingest = FakeIngestClient()

    def _factory(q, _vault):
        return FakeAdapter(wing=f"lme_{q.question_id}")

    judge = _CannedJudgeClient()
    report = runner.run(
        args,
        ingest_client=ingest,
        factory_fn=_factory,
        judge_client=judge,
    )

    pq = report["per_question"]
    rec_temporal = next(r for r in pq if r["question_id"] == "test_001_temporal")
    assert rec_temporal["judge"]["autoeval_label"] == "CORRECT"
    rec_abs = next(r for r in pq if r["question_id"] == "test_002_abstain_abs")
    assert rec_abs["judge"]["autoeval_label"] == "ABSTAIN"

    # The dual-metric aggregator emits per-category and overall numbers.
    dual = report["summary"]["dual_metric"]
    assert "cat_6" in dual["per_category"]
    assert dual["per_category"]["cat_6"]["qa_accuracy"] == 1.0
    # Overall QA accuracy is 2/2 = 1.0 (one CORRECT + one ABSTAIN; both true)
    assert dual["overall"]["qa_accuracy"] == 1.0


# --- Ingest helper unit tests ---------------------------------------------

def test_ingest_question_haystack_posts_one_drawer_per_session(dataset):
    questions = list(load_questions(dataset))
    q = questions[0]  # 2 sessions

    ingest = FakeIngestClient()
    report = runner.ingest_question_haystack(q, ingest)

    assert report["wing"] == "lme_test_001_temporal"
    assert report["posted"] == 2
    assert report["errors"] == []
    assert len(ingest.posted) == 2
    # Evidence marker is preserved in the rendered drawer text
    assert any(
        "<!-- evidence -->" in p["content"] for p in ingest.posted
    )


def test_ingest_question_haystack_records_per_session_failures(dataset):
    """When the daemon returns 500 on one session, the helper records the
    error but continues with the rest."""
    questions = list(load_questions(dataset))
    q = questions[0]

    class _FlakyIngest:
        def __init__(self):
            self.calls = 0
            self.posted = []

        def post_memory(self, *, content, wing, room="longmemeval"):
            self.calls += 1
            if self.calls == 1:
                return 500, {"_raw": "boom"}
            self.posted.append({"content": content, "wing": wing, "room": room})
            return 200, {}

    flaky = _FlakyIngest()
    report = runner.ingest_question_haystack(q, flaky)

    assert report["posted"] == 1  # one session got in
    assert len(report["errors"]) == 1  # one session failed
    assert "HTTP 500" in report["errors"][0]


# --- Adapter selection guardrails ------------------------------------------

def test_mempalace_daemon_requires_api_url(dataset, args_factory):
    args = args_factory(dataset, adapter="mempalace-daemon", api_url=None)
    with pytest.raises(SystemExit, match="api-url"):
        runner._build_factory(args)


def test_mempalace_daemon_requires_api_key(dataset, args_factory, monkeypatch):
    monkeypatch.delenv("PALACE_API_KEY", raising=False)
    args = args_factory(dataset, adapter="mempalace-daemon", api_key=None)
    with pytest.raises(SystemExit, match="API key"):
        runner._build_factory(args)


def test_mempalace_daemon_picks_up_env_api_key(dataset, args_factory,
                                                monkeypatch):
    monkeypatch.setenv("PALACE_API_KEY", "env-key")
    args = args_factory(dataset, adapter="mempalace-daemon", api_key=None)
    # _build_factory should not raise
    factory = runner._build_factory(args)
    assert callable(factory)


# --- Cost estimator unit tests ---------------------------------------------

def test_estimate_run_cost_known_models():
    questions = list(load_questions(_REPO_ROOT / "tests" / "_lme_fixture.json")) \
        if (_REPO_ROOT / "tests" / "_lme_fixture.json").exists() else []
    # Use the inline fixture instead of relying on a file
    from sme.corpora.longmemeval.loader import _parse_record
    parsed = [_parse_record(r) for r in FIXTURE]
    est = runner.estimate_run_cost(
        parsed,
        reader_model="gpt-4.1-mini",
        judge_model="gpt-4o-2024-08-06",
    )
    assert est["n_questions"] == 2
    assert est["reader_usd"] > 0
    assert est["judge_usd"] > est["reader_usd"]  # judge is more expensive


def test_estimate_run_cost_unknown_model_zero_priced():
    """Unknown model → cost falls through to zero rather than crashing."""
    from sme.corpora.longmemeval.loader import _parse_record
    parsed = [_parse_record(FIXTURE[0])]
    est = runner.estimate_run_cost(
        parsed,
        reader_model="gpt-99-fictional",
        judge_model="gpt-4o-2024-08-06",
    )
    assert est["reader_usd"] == 0  # unknown model in price table
    assert est["judge_usd"] > 0


# --- Arg parser smoke ------------------------------------------------------

def test_arg_parser_requires_adapter_and_questions():
    parser = runner.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args([
        "--adapter", "mempalace-daemon",
        "--questions", "/tmp/x.json",
        "--api-url", "http://localhost:8085",
    ])
    assert parsed.adapter == "mempalace-daemon"
    assert parsed.answer_model == "o4-mini"
    assert parsed.judge == "gpt-5.3-chat"
    assert parsed.dry_run is False


def test_arg_parser_familiar_adapter_choice():
    parser = runner.build_arg_parser()
    parsed = parser.parse_args([
        "--adapter", "familiar",
        "--questions", "/tmp/x.json",
        "--familiar-url", "http://familiar.realm.watch:8080",
    ])
    assert parsed.adapter == "familiar"
    assert parsed.familiar_url == "http://familiar.realm.watch:8080"
