"""Tests for scripts/run_longmemeval_omega — the OMEGA competitor runner.

Exercises the session-level R@K scoring and the report shape with a mocked
OmegaAdapter, so no omega-memory install or Azure key is needed. The crux
this guards: the runner computes ``omega_hit_at_K`` by matching each retrieved
Entity's ``properties["session_id"]`` against the question's expected session
ids — the OMEGA analogue of the daemon's session→drawer_id map. The generic
``cross_validate --adapter omega`` substring path can't see this (session ids
never appear in upstream-exact user-turn text), so this runner + this test are
what make the OMEGA R@K number real rather than a metric artifact.
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

import run_longmemeval_omega as runner  # noqa: E402
from sme.adapters.base import Entity, QueryResult  # noqa: E402


# --- 2-record fixture: one question whose evidence session OMEGA surfaces,
# one whose evidence session it misses. ------------------------------------

FIXTURE = [
    {
        "question_id": "q_hit",
        "question_type": "single-session-user",
        "question": "What pet does Maria have?",
        "answer": "A golden retriever named Biscuit",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_session_ids": ["sess_hit", "sess_noise"],
        "haystack_dates": ["2023/03/01 (Wed) 10:00", "2023/03/15 (Wed) 14:30"],
        "haystack_sessions": [
            [{"role": "user", "content": "Maria adopted a golden retriever "
              "named Biscuit.", "has_answer": True}],
            [{"role": "user", "content": "The standup moved to 9:30am."}],
        ],
        "answer_session_ids": ["sess_hit"],
    },
    {
        "question_id": "q_miss",
        "question_type": "multi-session",
        "question": "What did I say about the submarine?",
        "answer": "A blue submarine",
        "question_date": "2023/05/01 (Mon) 09:00",
        "haystack_session_ids": ["sess_target", "sess_other"],
        "haystack_dates": ["2023/04/20 (Thu) 12:00", "2023/04/21 (Fri) 12:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "I bought a blue submarine.",
              "has_answer": True}],
            [{"role": "user", "content": "I like kayaks."}],
        ],
        "answer_session_ids": ["sess_target"],
    },
]


@pytest.fixture
def dataset(tmp_path):
    p = tmp_path / "lme_s.json"
    p.write_text(json.dumps(FIXTURE))
    return p


def _args(dataset_path, **over):
    base = dict(
        questions=dataset_path,
        max_questions=None,
        stratify_by=None,
        shuffle=None,
        content_rules="upstream-exact",
        answer_model="o4-mini",
        judge="gpt-5.3-chat",
        n_results=5,
        skip_judge=True,
        skip_reader=False,
        json=None,
        verbose=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _FakeOmega:
    """Stand-in for OmegaAdapter.

    For ``q_hit`` it returns the evidence session ``sess_hit`` at rank 1; for
    ``q_miss`` it returns only non-evidence sessions (the retrieval miss). The
    per-question store id is irrelevant — what matters is the ``session_id``
    carried on each retrieved Entity, exactly as the real adapter surfaces it.
    """

    # Map the first user-turn substring → the session_id the runner tagged it
    # with, so the fake can echo the right session ids back on query.
    def __init__(self, *, omega_home=None, n_results=5):
        self.n_results = n_results
        self._by_session: dict[str, str] = {}

    def ingest_corpus(self, corpus):
        for row in corpus:
            self._by_session[row["session_id"]] = row["content"]
        return {"entities_created": len(corpus), "edges_created": 0,
                "errors": [], "warnings": []}

    def query(self, question, n_results=5):
        # Evidence sessions are tagged sess_hit / sess_target; surface
        # sess_hit (a hit) for the pet question, and only sess_other (a miss)
        # for the submarine question.
        if "pet" in question or "Maria" in question:
            ranked = ["sess_hit", "sess_noise"]
        else:
            ranked = ["sess_other"]  # misses sess_target on purpose
        ents = [
            Entity(
                id=f"omega:mem-{sid}",
                name=sid,
                entity_type="memory:summary",
                properties={"session_id": sid, "score": 0.9 - i * 0.1},
            )
            for i, sid in enumerate(ranked)
        ]
        ctx = "\n\n".join(self._by_session.get(s, "") for s in ranked)
        return QueryResult(answer=ctx, context_string=ctx,
                           retrieved_entities=ents)

    def close(self):
        pass


@pytest.fixture
def patch_adapter(monkeypatch):
    monkeypatch.setattr(
        "sme.adapters.omega.OmegaAdapter", _FakeOmega, raising=True
    )


def test_session_level_hit_at_k(dataset, patch_adapter):
    """omega_hit_at_5 is True for the question whose evidence session is
    retrieved, False for the one it's missed — and sme_recall reflects the
    session-level recall, not substring 0."""
    report = runner.run(_args(dataset))
    recs = {r["question_id"]: r for r in report["per_question"]}

    hit = recs["q_hit"]
    assert hit["omega_hit_at_1"] is True
    assert hit["omega_hit_at_5"] is True
    assert hit["retrieved_session_ids"][0] == "sess_hit"
    assert hit["sme_recall"] == 1.0  # session-level recall, not substring

    miss = recs["q_miss"]
    assert miss["omega_hit_at_1"] is False
    assert miss["omega_hit_at_5"] is False
    assert miss["sme_recall"] == 0.0


def test_summary_retrieval_session_level(dataset, patch_adapter):
    """The aggregate carries overall + per-category session-level R@K."""
    report = runner.run(_args(dataset))
    rsl = report["summary"]["retrieval_session_level"]
    assert rsl["overall"]["n"] == 2
    # one hit of two questions
    assert rsl["overall"]["r_at_5"] == 0.5
    assert rsl["overall"]["r_at_1"] == 0.5
    assert set(rsl["per_category"]) == {"cat_1", "cat_2c"}


def test_metadata_discloses_run_conditions(dataset, patch_adapter):
    """Run metadata must disclose the conditions a published number needs:
    semantic embedding (not FTS5), local isolation, content rules, models."""
    report = runner.run(_args(dataset))
    meta = report["run_metadata"]
    assert meta["adapter"] == "omega"
    assert meta["content_rules"] == "upstream-exact"
    assert "bge-small" in meta["embedding_model"]
    assert "session-level" in meta["retrieval_metric"]
    # skip_judge True → models suppressed
    assert meta["answer_model"] is None
    assert meta["judge_model"] is None


def test_stratify_matches_mempalace_subset_logic(dataset, patch_adapter):
    """--stratify-by uses the same _stratified_cap as the mempalace runner,
    so a capped competitor run lands on the same subset shape."""
    report = runner.run(_args(dataset, max_questions=2, stratify_by="question_type"))
    assert report["run_metadata"]["n_questions"] == 2
    assert report["run_metadata"]["stratify_by"] == "question_type"
