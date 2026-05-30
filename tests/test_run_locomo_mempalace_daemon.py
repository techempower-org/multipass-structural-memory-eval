"""Tests for scripts/run_locomo_mempalace_daemon.py (issue #176).

No live daemon, no network, no Azure: a fake ingest client records POSTs and
hands back synthetic drawer_ids, a fake wing-scoped adapter returns a canned
retrieval, and ``_score_and_judge`` runs with ``skip_judge``/``skip_reader``
collapsed by injecting a fake judge client. Focus:

  - isolation_guard refuses non-localhost URLs and non-empty palaces;
  - sessions are ingested PER SAMPLE under wing ``locomo_<sample_id>``;
  - the adapter is built once per sample and scoped to that wing;
  - drawer-based R@5 uses the session->drawer map (chunk-suffix stripped).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_locomo_mempalace_daemon as runner  # noqa: E402
from sme.adapters.base import Entity, QueryResult, SMEAdapter  # noqa: E402
from sme.corpora.locomo.loader import load_questions  # noqa: E402

# Reuse the LoCoMo fixture shape from test_cross_validate_locomo.
FIXTURE = [
    {
        "sample_id": "conv-aa",
        "conversation": {
            "speaker_a": "Alice", "speaker_b": "Bob",
            "session_1_date_time": "10:00 am on 1 Jan, 2023",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Cute!"},
            ],
            "session_2_date_time": "3:00 pm on 14 Feb, 2023",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "Rex turned two today."},
            ],
        },
        "qa": [
            {"question": "What is the name of Alice's dog?", "answer": "Rex",
             "evidence": ["D1:1"], "category": 2},
        ],
    },
    {
        "sample_id": "conv-bb",
        "conversation": {
            "speaker_a": "Carol", "speaker_b": "Dave",
            "session_1_date_time": "9:00 am on 3 Mar, 2023",
            "session_1": [
                {"speaker": "Carol", "dia_id": "D1:1",
                 "text": "I moved to Berlin and started a new job."},
            ],
        },
        "qa": [
            {"question": "Where does Carol live?", "answer": "Berlin",
             "evidence": ["D1:1"], "category": 1},
        ],
    },
]


@pytest.fixture
def questions(tmp_path):
    p = tmp_path / "locomo_fixture.json"
    p.write_text(json.dumps(FIXTURE))
    return list(load_questions(p))


class FakeIngest:
    """Records POSTs; returns a deterministic drawer_id per (wing, session)."""

    def __init__(self):
        self.posts = []

    def post_memory(self, *, content, wing, room):
        idx = len(self.posts)
        self.posts.append({"wing": wing, "room": room, "content": content})
        return 200, {"drawer_id": f"drawer_{wing}_{idx}"}

    def post_flush(self):
        return 200, {}


class FakeJudge:
    """OpenAI-shaped fake: always grades 'yes' (CORRECT)."""

    class _Resp:
        class _Choice:
            class _Msg:
                content = "yes"
            message = _Msg()
        choices = [_Choice()]

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **_kw):
        return self._Resp()


def _make_adapter_for(wing, hit_drawer_id):
    """A wing-scoped adapter whose retrieval returns one hit with the given
    drawer id (chunked, to exercise the parent-id strip)."""

    class _A(SMEAdapter):
        def query(self, question, *, n_results=5, **_kw):
            return QueryResult(
                answer="ctx", context_string="Rex Berlin",
                retrieved_entities=[Entity(
                    id=f"{hit_drawer_id}_chunk_000000", name="hit",
                    entity_type="drawer:sessions", properties={"wing": wing},
                )],
            )

        def get_graph_snapshot(self):
            return [], []

        def ingest_corpus(self, corpus):
            return {}

        def close(self):
            pass

    return _A()


def test_isolation_guard_rejects_remote_url():
    with pytest.raises(SystemExit, match="not a localhost"):
        runner.isolation_guard("http://familiar:8085", "k", count_fn=lambda *_: 0)


def test_isolation_guard_rejects_nonempty_palace():
    with pytest.raises(SystemExit, match="drawers=42"):
        runner.isolation_guard("http://localhost:8086", "k", count_fn=lambda *_: 42)


def test_isolation_guard_rejects_unverifiable_count():
    with pytest.raises(SystemExit, match="drawers=-1"):
        runner.isolation_guard("http://localhost:8086", "k", count_fn=lambda *_: -1)


def test_isolation_guard_passes_localhost_empty():
    runner.isolation_guard("http://127.0.0.1:8086", "k", count_fn=lambda *_: 0)


def test_sample_wing_normalises_hyphens():
    assert runner.sample_wing("conv-26") == "locomo_conv_26"


def _run_all(questions, ingest, factory_fn):
    """Ingest then query, mirroring main()'s ingest→query order. Backfill is a
    live-daemon concern, skipped in unit tests."""
    by_sample, order = runner._group_by_sample(questions, None)
    sample_s2d, ingest_total = runner.ingest_all_samples(
        by_sample=by_sample, order=order, ingest_client=ingest,
    )
    records = runner.query_all_samples(
        by_sample=by_sample, order=order, sample_s2d=sample_s2d,
        factory_fn=factory_fn, reader_model="m", judge_model="m",
        judge_client=FakeJudge(), reader_client=FakeJudge(),
    )
    return records, ingest_total


def test_ingest_is_per_sample_under_wing(questions):
    ingest = FakeIngest()
    # conv-aa has 2 sessions, conv-bb has 1.
    _run_all(questions, ingest, lambda wing: _make_adapter_for(wing, "x"))
    wings = {p["wing"] for p in ingest.posts}
    assert wings == {"locomo_conv_aa", "locomo_conv_bb"}
    aa = [p for p in ingest.posts if p["wing"] == "locomo_conv_aa"]
    bb = [p for p in ingest.posts if p["wing"] == "locomo_conv_bb"]
    assert len(aa) == 2  # two sessions
    assert len(bb) == 1
    assert all(p["room"] == "sessions" for p in ingest.posts)


def test_drawer_recall_uses_session_map(questions):
    """conv-aa's question evidence is D1 → its first ingested drawer. The fake
    adapter returns that drawer (chunked); drawer_hit_at_5 must be True after
    the parent-id strip."""
    ingest = FakeIngest()
    # conv-aa session D1 is the 0th post → drawer_locomo_conv_aa_0
    records, ingest_total = _run_all(
        questions, ingest, lambda wing: _make_adapter_for(wing, f"drawer_{wing}_0")
    )
    assert ingest_total["posted"] == 3
    aa_rec = next(r for r in records if r["sample_id"] == "conv-aa")
    assert aa_rec["expected_drawer_ids"] == ["drawer_locomo_conv_aa_0"]
    assert aa_rec["drawer_hit_at_5"] is True
    assert aa_rec["drawer_hit_at_1"] is True


def test_build_report_aggregates_per_type_and_overall(questions):
    ingest = FakeIngest()
    records, ingest_total = _run_all(
        questions, ingest, lambda wing: _make_adapter_for(wing, f"drawer_{wing}_0")
    )
    report = runner.build_report(records, ingest_total, meta={"corpus": "locomo"})
    assert report["qa_overall"]["n"] == 2
    # FakeJudge always says 'yes' → both CORRECT.
    assert report["qa_overall"]["qa_accuracy"] == 1.0
    assert set(report["qa_by_locomo_type"]) == {"single-hop", "multi-hop"}
    assert report["run_metadata"]["ingest_posted"] == 3
