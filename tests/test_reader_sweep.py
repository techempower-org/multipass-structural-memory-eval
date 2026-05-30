"""Tests for the #116 reader-sweep core (offline, no daemon, no real LLM).

Fakes the reader + judge clients so the matrix expansion, per-config replay,
QA-acc aggregation, and pinned-context loading are all covered in isolation.
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from sme.eval.reader_sweep import (
    PROMPT_VARIANTS,
    ReaderConfig,
    SweepMatrix,
    aggregate_labels,
    estimate_sweep_calls,
    load_pinned_context,
    run_one_config,
    run_sweep,
)


# --- fakes ----------------------------------------------------------------


class _FakeReader:
    """OpenAI-shaped client whose answer encodes the model+prompt it saw,
    so tests can confirm the sweep actually varies the reader config."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kw):
        prompt = messages[0]["content"]
        self.calls.append((model, prompt))
        # Echo a deterministic "answer" keyed on model so judge fakes can map it.
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"ans::{model}"))]
        )


class _FakeJudge:
    """Judge that marks an answer CORRECT iff it came from a 'good' model."""

    def __init__(self, good_models):
        self.good = set(good_models)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kw):
        body = messages[-1]["content"]
        label = "INCORRECT"
        for m in self.good:
            if f"ans::{m}" in body:
                label = "CORRECT"
        # The judge parser reads the "label" key (see _parse_judge_reply).
        payload = json.dumps({"label": label, "rationale": "x"})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
        )


class _QuestionEchoReader:
    """Reader whose answer embeds the question text it saw, so a parity test
    can detect any answer<->question misordering under concurrency. Thread-safe
    call log (the point is to run this from multiple threads)."""

    def __init__(self, delay: float = 0.0):
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._delay = delay
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kw):
        prompt = messages[0]["content"]
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self.calls.append((model, prompt))
        # Echo the question line back so the judge can verify the answer is
        # paired with the right question. The prompt embeds "Question: <q>".
        q = ""
        for line in prompt.splitlines():
            if line.startswith("Question:"):
                q = line[len("Question:"):].strip()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"ans-for::{q}"))]
        )


class _QuestionMatchJudge:
    """Judge that marks CORRECT iff the answer it sees was generated for the
    SAME question it is grading — i.e. the reader hypothesis and the question
    line up. A misordering would pair the wrong answer with the question and
    flip the label, so the parity test would fail loudly."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kw):
        body = messages[-1]["content"]
        # The judge prompt carries "Question: <q>" and "System answer:
        # ans-for::<the question the reader actually saw>". CORRECT iff those
        # two questions match — a misordering pairs the wrong answer with the
        # question and flips this to INCORRECT.
        asked_q = ""
        answered_q = None
        for line in body.splitlines():
            if line.startswith("Question:"):
                asked_q = line[len("Question:"):].strip()
            if "ans-for::" in line:
                answered_q = line.split("ans-for::", 1)[1].strip()
        label = "CORRECT" if (answered_q is not None and answered_q == asked_q) \
            else "INCORRECT"
        payload = json.dumps({"label": label, "rationale": "x"})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
        )


def _records(n=3):
    return [
        {
            "question_id": f"q{i}",
            "question": f"question {i}?",
            "gold_answer": f"gold {i}",
            "question_type": "single-session-user",
            "sme_category": "cat_1",
            "is_abstention": False,
            "context_string": f"context for {i}",
        }
        for i in range(n)
    ]


# --- SweepMatrix / ReaderConfig -------------------------------------------


def test_matrix_is_cartesian_product():
    m = SweepMatrix(
        reader_models=["a", "b"], prompts=["baseline", "cot"],
        context_widths=[None, 2000],
    )
    cfgs = m.configs()
    assert len(cfgs) == 2 * 2 * 2
    labels = {c.label for c in cfgs}
    assert "a|baseline|ctx=full" in labels
    assert "b|cot|ctx=2000" in labels


def test_reader_config_label_full_vs_capped():
    assert ReaderConfig("m").label == "m|baseline|ctx=full"
    assert ReaderConfig("m", "cot", 500).label == "m|cot|ctx=500"


# --- aggregate_labels -----------------------------------------------------


def test_aggregate_qa_acc_counts_correct():
    per_q = [
        {"question_type": "t", "autoeval_label": "CORRECT"},
        {"question_type": "t", "autoeval_label": "INCORRECT"},
        {"question_type": "t", "autoeval_label": "CORRECT"},
    ]
    agg = aggregate_labels(per_q)
    assert agg["overall"]["qa_acc"] == round(2 / 3, 4)
    assert agg["overall"]["labels"]["CORRECT"] == 2


def test_aggregate_abstain_correct_only_on_abstention_questions():
    per_q = [
        {"question_type": "abstention", "autoeval_label": "ABSTAIN"},   # right
        {"question_type": "single-session-user", "autoeval_label": "ABSTAIN"},  # wrong
    ]
    agg = aggregate_labels(per_q)
    assert agg["overall"]["qa_acc"] == 0.5


# --- run_one_config / run_sweep -------------------------------------------


def test_run_one_config_replays_pinned_context_without_retrieval():
    reader = _FakeReader()
    judge = _FakeJudge(good_models={"good-model"})
    cfg = ReaderConfig("good-model", "baseline", None)
    res = run_one_config(
        records=_records(3), config=cfg, judge_model="judge",
        reader_client=reader, judge_client=judge,
    )
    # 3 questions → 3 reader calls, all using the configured model.
    assert len(reader.calls) == 3
    assert all(model == "good-model" for model, _ in reader.calls)
    assert res["summary"]["overall"]["qa_acc"] == 1.0


def test_run_sweep_picks_best_config():
    reader = _FakeReader()
    judge = _FakeJudge(good_models={"good-model"})
    matrix = SweepMatrix(reader_models=["good-model", "bad-model"])
    res = run_sweep(
        records=_records(4), matrix=matrix, judge_model="judge",
        reader_client=reader, judge_client=judge,
    )
    assert res["n_configs"] == 2
    assert res["best"]["config"] == "good-model|baseline|ctx=full"
    assert res["best"]["qa_acc"] == 1.0


def test_run_one_config_unknown_prompt_raises():
    with pytest.raises(KeyError):
        run_one_config(records=_records(1),
                       config=ReaderConfig("m", "nonexistent-prompt"),
                       judge_model="j")


def test_context_width_truncates_what_reader_sees():
    reader = _FakeReader()
    judge = _FakeJudge(good_models=set())
    recs = [{
        "question_id": "q0", "question": "q?", "gold_answer": "g",
        "question_type": "t", "is_abstention": False,
        "context_string": "X" * 5000,
    }]
    run_one_config(records=recs, config=ReaderConfig("m", "baseline", 100),
                   judge_model="j", reader_client=reader, judge_client=judge)
    _, prompt = reader.calls[0]
    # The baseline template wraps the context; the context slice itself must
    # be capped at 100 chars, so total X-run in the prompt is exactly 100.
    assert prompt.count("X") == 100


# --- estimate_sweep_calls -------------------------------------------------


def test_estimate_sweep_calls_counts_reader_and_judge():
    m = SweepMatrix(reader_models=["a", "b"], prompts=["baseline", "cot"],
                    context_widths=[None])
    est = estimate_sweep_calls(50, m)
    assert est["n_configs"] == 4
    assert est["reader_calls"] == 200
    assert est["judge_calls"] == 200
    assert est["total_llm_calls"] == 400


# --- load_pinned_context --------------------------------------------------


def test_load_pinned_context_ok(tmp_path):
    doc = {"run_metadata": {"snippet_width": "/search"},
           "pinned_context": _records(2)}
    p = tmp_path / "pinned.json"
    p.write_text(json.dumps(doc))
    meta, recs = load_pinned_context(p)
    assert meta["snippet_width"] == "/search"
    assert len(recs) == 2


def test_load_pinned_context_rejects_missing_context_string(tmp_path):
    bad = [{"question_id": "q0", "question": "q", "gold_answer": "g",
            "question_type": "t"}]  # no context_string
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"pinned_context": bad}))
    with pytest.raises(ValueError, match="context_string"):
        load_pinned_context(p)


def test_prompt_variants_all_have_required_fields():
    """Every prompt variant must expose {context} and {question} so the
    reader can format it — a malformed variant would crash mid-sweep."""
    for name, tmpl in PROMPT_VARIANTS.items():
        formatted = tmpl.format(context="C", question="Q")
        assert "C" in formatted and "Q" in formatted, name


# --- concurrency ----------------------------------------------------------


def test_concurrent_run_one_config_matches_serial():
    """Identical per-question rows + aggregate whether concurrency=1 or >1.

    The reader answer encodes the question it saw and the judge marks CORRECT
    only when the answer is paired with the right question, so a concurrency
    bug that mis-pairs answers and questions would flip labels and break this
    assertion. Out-of-order completion is forced via a per-question delay."""
    cfg = ReaderConfig("m", "baseline", None)
    recs = _records(12)

    serial = run_one_config(
        records=recs, config=cfg, judge_model="judge",
        reader_client=_QuestionEchoReader(), judge_client=_QuestionMatchJudge(),
        concurrency=1,
    )
    concurrent = run_one_config(
        records=recs, config=cfg, judge_model="judge",
        reader_client=_QuestionEchoReader(delay=0.01),
        judge_client=_QuestionMatchJudge(), concurrency=6,
    )

    # Per-question order preserved and every field identical.
    assert [r["question_id"] for r in concurrent["per_question"]] == \
        [f"q{i}" for i in range(12)]
    assert concurrent["per_question"] == serial["per_question"]
    assert concurrent["summary"] == serial["summary"]
    # Every answer was paired with its own question → all CORRECT.
    assert concurrent["summary"]["overall"]["qa_acc"] == 1.0


def test_concurrent_actually_runs_in_parallel():
    """Wall-clock check: N delayed calls under concurrency>1 finish in roughly
    one delay, not N — proving the pool runs them concurrently, not serially."""
    cfg = ReaderConfig("m", "baseline", None)
    recs = _records(8)
    reader = _QuestionEchoReader(delay=0.05)
    t0 = time.monotonic()
    run_one_config(
        records=recs, config=cfg, judge_model="judge",
        reader_client=reader, judge_client=_QuestionMatchJudge(),
        concurrency=8,
    )
    elapsed = time.monotonic() - t0
    # Serial would be >= 8 * 0.05 = 0.4s; concurrent should be well under half.
    assert elapsed < 0.2, f"expected concurrent run, took {elapsed:.3f}s"


def test_run_sweep_threads_concurrency_and_matches_serial():
    """run_sweep forwards concurrency to each config and the full report is
    identical to the serial sweep."""
    matrix = SweepMatrix(reader_models=["m1", "m2"], prompts=["baseline", "cot"])
    recs = _records(6)

    serial = run_sweep(
        records=recs, matrix=matrix, judge_model="judge",
        reader_client=_QuestionEchoReader(), judge_client=_QuestionMatchJudge(),
        concurrency=1,
    )
    concurrent = run_sweep(
        records=recs, matrix=matrix, judge_model="judge",
        reader_client=_QuestionEchoReader(delay=0.005),
        judge_client=_QuestionMatchJudge(), concurrency=4,
    )
    assert concurrent["n_configs"] == serial["n_configs"] == 4
    assert concurrent["configs"] == serial["configs"]
    assert concurrent["best"] == serial["best"]


def test_concurrency_zero_or_negative_falls_back_to_serial():
    """Defensive: a non-positive K must not raise (ThreadPoolExecutor rejects
    max_workers<=0) — it clamps to serial."""
    cfg = ReaderConfig("m", "baseline", None)
    recs = _records(3)
    res = run_one_config(
        records=recs, config=cfg, judge_model="judge",
        reader_client=_QuestionEchoReader(), judge_client=_QuestionMatchJudge(),
        concurrency=0,
    )
    assert len(res["per_question"]) == 3
    assert res["summary"]["overall"]["qa_acc"] == 1.0
