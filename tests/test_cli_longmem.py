"""CLI smoke test for ``sme-eval longmemeval``.

Uses the FullContextAdapter so the test needs no ChromaDB and no network.
Judge is mocked via the LongMemEval-shape JSON fixture from
``test_cross_validate_longmemeval.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sme import cli

# Re-use the same 2-record fixture pattern as the harness tests.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


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
def dataset_path(tmp_path):
    p = tmp_path / "lme_oracle.json"
    p.write_text(json.dumps(FIXTURE))
    return p


def test_longmem_subcommand_help_lists_dual_metric_options(capsys):
    """`sme-eval longmemeval --help` should expose --answer-model + --judge."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["longmemeval", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--answer-model" in out
    assert "--judge" in out
    assert "--skip-judge" in out
    assert "--adapter" in out


def test_longmem_subcommand_runs_full_context_skip_judge(
        dataset_path, tmp_path, capsys):
    """Skip-judge path: no API key needed, R@5 reported, JSON written."""
    out_json = tmp_path / "report.json"
    rc = cli.main([
        "longmemeval",
        "--adapter", "full-context",
        "--questions", str(dataset_path),
        "--skip-judge",
        "--json", str(out_json),
        "--work-dir", str(tmp_path / "work"),
    ])
    assert rc == 0
    assert out_json.exists()
    report = json.loads(out_json.read_text())
    # Structure sanity.
    assert report["run_metadata"]["adapter"] == "full-context"
    assert report["run_metadata"]["skip_judge"] is True
    assert "dual_metric" in report["summary"]
    overall = report["summary"]["dual_metric"]["overall"]
    assert overall["n"] == 2
    # Judge skipped → QA accuracy is None.
    assert overall["qa_accuracy"] is None
    assert overall["retrieval_qa_gap"] is None

    captured = capsys.readouterr().out
    assert "R@5" in captured
    assert "QA-acc" in captured
    assert "overall" in captured


def test_longmem_subcommand_requires_api_url_for_daemon(dataset_path, tmp_path):
    """The mempalace-daemon adapter needs --api-url; verify clear error."""
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "longmemeval",
            "--adapter", "mempalace-daemon",
            "--questions", str(dataset_path),
            "--skip-judge",
            "--work-dir", str(tmp_path / "work"),
        ])
    msg = str(exc.value)
    assert "api-url" in msg.lower() or "mempalace-daemon" in msg


def test_longmem_subcommand_defaults_to_gpt_4_1_mini(dataset_path, tmp_path):
    """Issue #17 default reader model is gpt-4.1-mini, not gpt-4o-mini."""
    # Build the parser without running, by hijacking parse_args.
    captured = {}

    def _fake_run(args, **kwargs):
        captured["reader_model"] = args.reader_model
        # Return a minimal valid report.
        return {
            "run_metadata": {"adapter": args.adapter,
                             "skip_judge": True,
                             "judge_model": None,
                             "reader_model": None,
                             "timestamp_utc": "x"},
            "summary": {
                "per_category": {},
                "total_questions": 0,
                "judge_total_usage": {"prompt_tokens": 0,
                                      "completion_tokens": 0,
                                      "total_tokens": 0},
                "disagreements": [],
                "dual_metric": {"per_category": {},
                                "overall": {"n": 0,
                                            "n_judged": 0,
                                            "sme_recall_mean": 0.0,
                                            "qa_accuracy": None,
                                            "retrieval_qa_gap": None,
                                            "judge_label_counts": {}}},
                "ku_caveat": "",
            },
            "per_question": [],
        }

    import cross_validate_longmemeval as harness
    orig_run = harness.run
    harness.run = _fake_run
    try:
        rc = cli.main([
            "longmemeval",
            "--adapter", "full-context",
            "--questions", str(dataset_path),
            "--skip-judge",
            "--json", str(tmp_path / "x.json"),
        ])
    finally:
        harness.run = orig_run
    assert rc == 0
    assert captured["reader_model"] == "gpt-4.1-mini"
