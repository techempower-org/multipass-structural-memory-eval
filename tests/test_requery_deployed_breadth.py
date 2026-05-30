"""Tests for scripts/requery_deployed_breadth.py (#117).

The script re-queries already-ingested lme_* wings at a chosen retrieval limit
and writes a pinned-context JSON for the offline reader sweep. These tests mock
the wing-scoped adapter so no daemon or network is touched — they verify the
pinned-doc assembly, the source-pinned question passthrough, and that the limit
flows through to ``n_results``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "requery_deployed_breadth.py"


def _load_module():
    if str(_REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(_REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("requery_deployed_breadth", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeQR:
    def __init__(self, n_hits: int):
        self.context_string = "\n\n".join(f"[{i+1}] hit text {i}" for i in range(n_hits))
        self.retrieved_entities = [object()] * n_hits


@pytest.fixture
def source_pinned(tmp_path: Path) -> Path:
    doc = {
        "run_metadata": {"subset": "stratified-25-per-type"},
        "pinned_context": [
            {
                "question_id": "q1", "question": "What was my 5K time?",
                "gold_answer": "25:50", "question_type": "single-session-user",
                "sme_category": "cat_1", "is_abstention": False,
                "context_string": "OLD limit=5 capture — must NOT be reused",
                "context_chars": 31, "hit_at_5": True,
            },
            {
                "question_id": "q2", "question": "Where did Rachel move?",
                "gold_answer": "Berlin", "question_type": "multi-session",
                "sme_category": "cat_2", "is_abstention": False,
                "context_string": "OLD", "context_chars": 3, "hit_at_5": False,
            },
        ],
    }
    p = tmp_path / "source.json"
    p.write_text(json.dumps(doc))
    return p


def test_requery_passes_limit_and_reassembles_context(
    monkeypatch, source_pinned, tmp_path
):
    mod = _load_module()
    monkeypatch.setattr(mod, "_api_key", lambda: "FAKE")

    captured = {"limits": []}

    def fake_adapter(*, api_url, api_key, wing, kind, search_endpoint):
        class _A:
            def query(self, question, *, n_results, wing=None):
                captured["limits"].append(n_results)
                # fresh context grows with the limit — proves we re-retrieve
                return _FakeQR(n_results)
        return _A()

    monkeypatch.setattr(mod, "_make_wing_scoped_daemon_adapter", fake_adapter)

    out = tmp_path / "pinned_limit20.json"
    rc = mod.main([
        "--limit", "20", "--out", str(out),
        "--source-pinned", str(source_pinned),
    ])
    assert rc == 0

    # limit flowed through to n_results for every question
    assert captured["limits"] == [20, 20]

    doc = json.loads(out.read_text())
    assert doc["run_metadata"]["retrieval_limit"] == 20
    assert doc["run_metadata"]["n_questions"] == 2
    assert doc["run_metadata"]["n_empty_context"] == 0
    pc = doc["pinned_context"]
    assert [p["question_id"] for p in pc] == ["q1", "q2"]
    # gold/question/type carried through from the source subset
    assert pc[0]["gold_answer"] == "25:50"
    assert pc[1]["question_type"] == "multi-session"
    # context is FRESHLY re-retrieved (20 hits), not the old limit=5 capture
    assert pc[0]["n_hits"] == 20
    assert "OLD" not in pc[0]["context_string"]
    assert pc[0]["context_chars"] == len(pc[0]["context_string"]) > 0


def test_empty_context_is_counted(monkeypatch, source_pinned, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "_api_key", lambda: "FAKE")

    def fake_adapter(*, api_url, api_key, wing, kind, search_endpoint):
        class _A:
            def query(self, question, *, n_results, wing=None):
                return _FakeQR(0)  # empty wing
        return _A()

    monkeypatch.setattr(mod, "_make_wing_scoped_daemon_adapter", fake_adapter)
    out = tmp_path / "pinned.json"
    mod.main(["--limit", "5", "--out", str(out), "--source-pinned", str(source_pinned)])
    doc = json.loads(out.read_text())
    assert doc["run_metadata"]["n_empty_context"] == 2
