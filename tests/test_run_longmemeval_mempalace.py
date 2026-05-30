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
        # Deterministic synthetic drawer_id per call so #58 rank-aware
        # scoring has something stable to map session→drawer against.
        drawer_id = f"drawer_{wing}_{room}_{len(self.posted):04d}"
        self.posted.append({
            "content": content, "wing": wing, "room": room,
            "drawer_id": drawer_id,
        })
        return 200, {"drawer_id": drawer_id, "success": True}

    def post_flush(self):
        return 200, {}


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


def test_per_question_record_has_drawer_rank_fields_and_ingest(dataset, args_factory):
    """#58 + #59 — every per-question record carries drawer_hit_at_{1,5,10}
    (from #58) AND an ``ingest`` diagnostics block (from #59).

    With the FakeAdapter returning no retrieved entities, the hit_at_K
    fields all evaluate False — but the keys are present, which is what
    downstream aggregators rely on. The ingest block lets operators tell
    whether a low recall is from incomplete haystack ingest vs from
    retrieval missing the gold session.
    """
    args = args_factory(dataset, skip_judge=True)
    ingest = FakeIngestClient()

    def _factory(q, _vault):
        return FakeAdapter(wing=f"lme_{q.question_id}")

    report = runner.run(args, ingest_client=ingest, factory_fn=_factory)

    for rec in report["per_question"]:
        for key in (
            "expected_drawer_ids", "retrieved_drawer_ids",
            "drawer_hit_at_1", "drawer_hit_at_5", "drawer_hit_at_10",
        ):
            assert key in rec, f"missing {key}: {sorted(rec.keys())}"
        # FakeAdapter returns no retrieved_entity_ids, so all hits are False.
        assert rec["drawer_hit_at_1"] is False
        assert rec["drawer_hit_at_5"] is False
        # #59 — ingest diagnostics
        assert "ingest" in rec, f"missing ingest diagnostics: {rec.keys()}"
        block = rec["ingest"]
        assert isinstance(block.get("posted"), int)
        assert isinstance(block.get("errors"), list)
        # Successful FakeIngestClient runs leave errors empty
        assert block["errors"] == []

    # Per-question posted counts: Q1 has 2 sessions, Q2 has 1
    by_qid = {r["question_id"]: r["ingest"]["posted"] for r in report["per_question"]}
    assert by_qid["test_001_temporal"] == 2
    assert by_qid["test_002_abstain_abs"] == 1


def test_drawer_rank_scoring_with_session_map(dataset, args_factory):
    """#58 — when the daemon's drawer_id is in the top-K of retrieved
    entities, drawer_hit_at_K flips to True.

    Builds a custom adapter that returns the FakeIngestClient's stored
    drawer_id at rank-1, simulating a perfect retrieval. The session→drawer
    map from ingest then resolves the gold session into the expected drawer
    id, and the hit_at_1 check passes.
    """
    args = args_factory(dataset, skip_judge=True)
    ingest = FakeIngestClient()

    class _RankAdapter(SMEAdapter):
        """Always returns the most-recently-posted drawer at rank 1."""

        def __init__(self, wing):
            self.wing = wing

        def ingest_corpus(self, corpus):
            return {"entities_created": 0, "edges_created": 0,
                    "errors": [], "warnings": []}

        def query(self, question, n_results=5):
            from sme.adapters.base import Entity
            wing_posts = [p for p in ingest.posted if p["wing"] == self.wing]
            if not wing_posts:
                return QueryResult(answer="", context_string="", retrieved_entities=[])
            # Put the LAST-posted drawer at rank 1 — the test fixture's
            # answer_session_id is sess_001_b, which is the 2nd (and
            # last) session for the temporal question. So this rigs
            # rank-1 to match the gold.
            last_drawer = wing_posts[-1]["drawer_id"]
            return QueryResult(
                answer="",
                context_string="x",  # non-empty so harness doesn't short-circuit
                retrieved_entities=[Entity(id=last_drawer, name="x", entity_type="drawer")],
            )

        def get_graph_snapshot(self):
            return [], []

    def _factory(q, _vault):
        return _RankAdapter(wing=f"lme_{q.question_id}")

    report = runner.run(args, ingest_client=ingest, factory_fn=_factory)

    rec_temporal = next(
        r for r in report["per_question"] if r["question_id"] == "test_001_temporal"
    )
    # Gold = sess_001_b = 2nd session = last posted to the wing.
    # The _RankAdapter returned that drawer at rank 1.
    assert rec_temporal["drawer_hit_at_1"] is True
    assert rec_temporal["drawer_hit_at_5"] is True
    assert rec_temporal["drawer_hit_at_10"] is True


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
    # Use the inline fixture instead of relying on a file.
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


# --- Familiar adapter ingest gating (#46) ----------------------------------


def test_familiar_adapter_with_api_url_ingests_haystack(dataset, args_factory):
    """#46 — familiar wraps palace-daemon and reads its drawer store, so
    the per-question ingest needs to fire for the familiar adapter too,
    not just for mempalace-daemon. With --api-url + --api-key supplied,
    `run()` builds an ingest client for familiar."""
    args = args_factory(
        dataset, adapter="familiar", api_url="http://fake-daemon:8085",
        api_key="test-key", skip_judge=True,
    )
    ingest = FakeIngestClient()

    def _factory(q, _vault):
        return FakeAdapter(wing=f"lme_{q.question_id}")

    report = runner.run(args, ingest_client=ingest, factory_fn=_factory)

    # Both questions' haystacks were posted to the daemon
    assert {p["wing"] for p in ingest.posted} == {
        "lme_test_001_temporal", "lme_test_002_abstain_abs"
    }
    # Run metadata reports ingested_per_question=True for familiar too
    assert report["run_metadata"]["adapter"] == "familiar"
    assert report["run_metadata"]["ingested_per_question"] is True


def test_familiar_adapter_auto_builds_ingest_client_when_kwarg_omitted(
    dataset, args_factory, monkeypatch,
):
    """#46 — when ingest_client is not injected by the test, run() should
    auto-build one for familiar (mirroring the daemon-direct path) so the
    haystack still lands. We don't need a real HTTP daemon here — the
    auto-build path uses args.api_url + api_key resolution; we inject a
    factory_fn so the per-question adapter calls never actually fire."""
    args = args_factory(
        dataset, adapter="familiar", api_url="http://fake-daemon:8085",
        api_key="env-key", skip_judge=True,
    )

    # Stub DaemonIngestClient so the auto-build doesn't open a socket; we
    # just want to confirm the gate now lets familiar through.
    built = {}

    class _StubIngestClient:
        def __init__(self, *, api_url, api_key):
            built["api_url"] = api_url
            built["api_key"] = api_key
            self.posted = []

        def post_memory(self, *, content, wing, room="longmemeval"):
            drawer_id = f"drawer_{wing}_{len(self.posted):04d}"
            self.posted.append({
                "content": content, "wing": wing, "room": room,
                "drawer_id": drawer_id,
            })
            return 200, {"drawer_id": drawer_id, "success": True}

    monkeypatch.setattr(runner, "DaemonIngestClient", _StubIngestClient)

    def _factory(q, _vault):
        return FakeAdapter(wing=f"lme_{q.question_id}")

    report = runner.run(args, factory_fn=_factory)

    # Ingest client was auto-built with the supplied api_url
    assert built["api_url"] == "http://fake-daemon:8085"
    assert built["api_key"] == "env-key"
    # And the run metadata reflects per-question ingest
    assert report["run_metadata"]["ingested_per_question"] is True


def test_familiar_adapter_without_api_url_skips_ingest(dataset, args_factory):
    """When --api-url isn't supplied for familiar, the ingest gate skips
    the auto-build — familiar can still run against whatever's already
    in the palace, but no per-question loading happens."""
    args = args_factory(
        dataset, adapter="familiar", api_url=None,
        api_key=None, skip_judge=True,
    )

    def _factory(q, _vault):
        return FakeAdapter(wing=f"lme_{q.question_id}")

    report = runner.run(args, factory_fn=_factory)
    assert report["run_metadata"]["ingested_per_question"] is False


# --- --search-endpoint flag plumbing (#45) ---------------------------------


def test_arg_parser_search_endpoint_defaults_to_search():
    """--search-endpoint defaults to /search so existing runs are unaffected."""
    parser = runner.build_arg_parser()
    parsed = parser.parse_args([
        "--adapter", "mempalace-daemon",
        "--questions", "/tmp/x.json",
        "--api-url", "http://localhost:8085",
    ])
    assert parsed.search_endpoint == "/search"


def test_arg_parser_accepts_age_fused_search_endpoint():
    """--search-endpoint /search/age-fused parses cleanly and threads
    through to args.search_endpoint for the factory to consume."""
    parser = runner.build_arg_parser()
    parsed = parser.parse_args([
        "--adapter", "mempalace-daemon",
        "--questions", "/tmp/x.json",
        "--api-url", "http://localhost:8085",
        "--search-endpoint", "/search/age-fused",
    ])
    assert parsed.search_endpoint == "/search/age-fused"


def test_run_metadata_surfaces_search_endpoint(dataset, args_factory):
    """#83 — run_metadata.search_endpoint should reflect the actual
    args.search_endpoint value so JSON consumers can tell which endpoint
    produced a given reading. Previously hardcoded to 'default' even
    when /search/age-fused was the actual endpoint queried."""
    args = args_factory(dataset, skip_judge=True)
    args.search_endpoint = "/search/age-fused"
    ingest = FakeIngestClient()

    def _factory(q, _vault):
        return FakeAdapter(wing=f"lme_{q.question_id}")

    report = runner.run(args, ingest_client=ingest, factory_fn=_factory)
    assert report["run_metadata"]["search_endpoint"] == "/search/age-fused"


def test_run_metadata_default_search_endpoint_logged_as_search(
    dataset, args_factory,
):
    """When --search-endpoint isn't passed, run_metadata.search_endpoint
    falls through to the argparse default '/search' (the daemon's default
    vector + BM25 endpoint)."""
    args = args_factory(dataset, skip_judge=True)
    args.search_endpoint = "/search"
    ingest = FakeIngestClient()

    def _factory(q, _vault):
        return FakeAdapter(wing=f"lme_{q.question_id}")

    report = runner.run(args, ingest_client=ingest, factory_fn=_factory)
    assert report["run_metadata"]["search_endpoint"] == "/search"


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
