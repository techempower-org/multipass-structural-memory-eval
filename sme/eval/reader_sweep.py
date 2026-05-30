"""Reader-sweep core for the #116 orchestrator experiment.

The unified finding (PR #100/#102): on LongMemEval oracle, retrieval is solved
(~97% R@5) but end-to-end QA-accuracy is ~38pp lower — the *reader* (the model
that turns retrieved context into an answer) is the bottleneck, not retrieval.
The encoder swap (#84) was a null. #116 isolates the reader: hold retrieval +
encoder fixed and sweep reader configs (model × extraction-prompt ×
context-width), measuring QA-acc against the fixed retrieval ceiling.

Design — two phases, so the daemon is touched minimally:

  Phase 1 (pin-context, daemon, run by the orchestrator serially):
    Retrieve once per question over the oracle haystack and persist the full
    ``context_string`` per question to a pinned-context JSON. One file per
    daemon snippet-width setting (palace-daemon#150: /search default vs
    /search/age-fused return ~5.5x different snippet widths). This is the
    ONLY daemon-touching step; everything downstream is offline.

  Phase 2 (reader-sweep, offline, no daemon — this module):
    Load the pinned-context JSON(s) and replay each question's fixed context
    through the reader matrix. Retrieval never re-runs, so R@5 is constant by
    construction and any QA-acc delta is attributable to the reader config.

This module is the Phase-2 core: pure functions over pinned-context records +
injected reader/judge clients, so it's fully unit-testable with fakes and never
imports a daemon client.
"""

from __future__ import annotations

import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from sme.eval.answer_generator import (
    READER_PROMPT_TEMPLATE,
    generate_answer,
)
from sme.eval.longmemeval_judge import grade_answer

# Default fan-out for the offline reader/judge sweep. Reader + judge calls are
# network-I/O-bound (1-3s each, mostly latency), so thread-based concurrency is
# the right lever — the OpenAI/Azure and AnthropicBedrock sync clients are
# thread-safe for concurrent .create() calls. 8 is conservative; raising it
# risks provider 429s (rate limits), so document that before cranking it up.
DEFAULT_CONCURRENCY = 8

# Extraction-prompt variants for the prompt axis. The "terse" default is the
# answer_generator baseline; the others probe whether prompt scaffolding moves
# the reader's QA-acc at a fixed retrieval ceiling.
PROMPT_VARIANTS: dict[str, str] = {
    "baseline": READER_PROMPT_TEMPLATE,
    "cot": (
        "You are answering a question from a user's own conversation history.\n"
        "Read the history, think step by step about which turns are relevant, "
        "then give a single final answer. If the answer is not present, say "
        "'I don't know.'\n\n"
        "Conversation history:\n{context}\n\n"
        "Question: {question}\n\n"
        "Reason briefly, then end with 'Answer: <answer>'."
    ),
    "extractive": (
        "Using ONLY the conversation history below, extract the exact answer to "
        "the question. Quote the relevant detail verbatim where possible. If the "
        "history does not contain the answer, respond exactly 'I don't know.'\n\n"
        "Conversation history:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    # --- #116 diagnosed-failure fixes -------------------------------------
    # Pass B showed Opus scoring WORST under the baseline prompt, driven by two
    # failure modes the baseline prompt actively induces:
    #   (1) knowledge-update PARTIALs — the reader hedges by presenting BOTH the
    #       old and the updated value instead of committing to the latest one;
    #   (2) single-session-preference 0.00 — the reader REFUSES to make a
    #       recommendation ("the answer is not present"), because the baseline's
    #       "say I don't know" instruction over-fires on inference questions.
    # "committed" drops the over-eager abstention and tells the reader to commit
    # to the single most-recent value on update questions.
    "committed": (
        "Answer the user's question using the conversation history below. Give a "
        "single, direct, committed answer — do not hedge or present multiple "
        "candidate answers. If the history shows a value that was later changed "
        "or updated, answer with the MOST RECENT value only; do not also restate "
        "the old value. Only say 'I don't know.' if the history is genuinely "
        "silent on the question.\n\n"
        "Conversation history:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    # "preference" additionally targets recommendation/preference questions: the
    # reader must INFER the user's preferences from the history and tailor a
    # recommendation, rather than refusing because no answer is stated verbatim.
    "preference": (
        "Answer the user's question using the conversation history below. Give a "
        "single, direct, committed answer — do not hedge. If the history shows a "
        "value that was later updated, answer with the MOST RECENT value only.\n"
        "If the question asks for a recommendation, suggestion, or what the user "
        "would prefer, INFER the user's preferences from what they have said and "
        "done in the history and tailor your answer to them. Do NOT refuse just "
        "because no explicit answer is stated — a well-reasoned inference from "
        "their stated tastes is the expected answer. Only say 'I don't know.' if "
        "the history contains nothing relevant to base an answer on.\n\n"
        "Conversation history:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
}


@dataclass(frozen=True)
class ReaderConfig:
    """One cell of the reader sweep."""

    reader_model: str
    prompt: str = "baseline"            # key into PROMPT_VARIANTS
    max_context_chars: Optional[int] = None

    @property
    def label(self) -> str:
        w = self.max_context_chars if self.max_context_chars is not None else "full"
        return f"{self.reader_model}|{self.prompt}|ctx={w}"


@dataclass
class SweepMatrix:
    """The cartesian sweep space. Each axis is a list; configs = product."""

    reader_models: list[str]
    prompts: list[str] = field(default_factory=lambda: ["baseline"])
    context_widths: list[Optional[int]] = field(default_factory=lambda: [None])

    def configs(self) -> list[ReaderConfig]:
        out: list[ReaderConfig] = []
        for model, prompt, width in itertools.product(
            self.reader_models, self.prompts, self.context_widths
        ):
            out.append(ReaderConfig(reader_model=model, prompt=prompt,
                                    max_context_chars=width))
        return out


def load_pinned_context(path: Path) -> tuple[dict, list[dict]]:
    """Load a Phase-1 pinned-context JSON.

    Returns ``(meta, records)`` where each record carries at least
    ``question_id, question, gold_answer, question_type, is_abstention,
    context_string``. Raises if the file lacks the pinned context (so we
    never silently sweep over empty contexts).
    """
    doc = json.loads(Path(path).read_text())
    records = doc.get("pinned_context") or doc.get("per_question") or []
    if not records:
        raise ValueError(f"{path}: no pinned_context records")
    missing = [r.get("question_id") for r in records if "context_string" not in r]
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} record(s) lack 'context_string' — this file "
            f"was not produced by the Phase-1 pin-context step (e.g. {missing[:3]})"
        )
    return doc.get("run_metadata", {}), records


def _grade_one_record(
    r: dict,
    *,
    config: ReaderConfig,
    prompt_template: str,
    judge_model: str,
    reader_client: Optional[Any],
    judge_client: Optional[Any],
) -> dict:
    """Reader + judge for a single pinned record. Pure over its inputs (no
    shared mutable state), so it's safe to run concurrently across questions —
    the unit of work submitted to the ThreadPoolExecutor in ``run_one_config``.
    """
    hypothesis = generate_answer(
        r["question"], r["context_string"],
        reader_model=config.reader_model,
        client=reader_client,
        max_context_chars=config.max_context_chars,
        prompt_template=prompt_template,
    )
    qtype = "abstention" if r.get("is_abstention") else r["question_type"]
    judge = grade_answer(
        question_type=qtype,
        question=r["question"],
        gold_answer=r["gold_answer"],
        hypothesis=hypothesis,
        judge_model=judge_model,
        client=judge_client,
    )
    return {
        "question_id": r["question_id"],
        "question_type": r["question_type"],
        "sme_category": r.get("sme_category"),
        "hypothesis": hypothesis,
        "autoeval_label": judge.get("autoeval_label"),
    }


def run_one_config(
    *,
    records: list[dict],
    config: ReaderConfig,
    judge_model: str,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    concurrency: int = 1,
) -> dict:
    """Replay every pinned record through one reader config, then judge.

    Retrieval is NOT re-run — ``context_string`` comes straight from the
    pinned file, so R@5 is constant. Returns a per-config result block with
    per-question judge labels + the aggregate QA-acc.

    ``concurrency`` fans the per-question reader+judge calls out across a
    ``ThreadPoolExecutor`` (the calls are network-I/O-bound). Results are
    identical to serial regardless of K: per-question output is reassembled in
    the original record order, so answers, judge labels, and the aggregate are
    independent of execution order. ``concurrency=1`` runs strictly serially.
    """
    prompt_template = PROMPT_VARIANTS.get(config.prompt)
    if prompt_template is None:
        raise KeyError(f"unknown prompt variant {config.prompt!r}; "
                       f"known: {sorted(PROMPT_VARIANTS)}")

    n = len(records)
    per_q: list[Optional[dict]] = [None] * n

    def _work(i: int, r: dict) -> tuple[int, dict]:
        return i, _grade_one_record(
            r, config=config, prompt_template=prompt_template,
            judge_model=judge_model, reader_client=reader_client,
            judge_client=judge_client,
        )

    k = max(1, int(concurrency))
    if k == 1 or n <= 1:
        # Serial path — identical behaviour to the pre-concurrency loop.
        for i, r in enumerate(records):
            if progress:
                progress(i + 1, n, r["question_id"])
            per_q[i] = _grade_one_record(
                r, config=config, prompt_template=prompt_template,
                judge_model=judge_model, reader_client=reader_client,
                judge_client=judge_client,
            )
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=min(k, n)) as pool:
            futures = [pool.submit(_work, i, r) for i, r in enumerate(records)]
            for fut in futures:
                i, row = fut.result()
                per_q[i] = row
                done += 1
                if progress:
                    progress(done, n, row["question_id"])

    rows = [row for row in per_q if row is not None]
    return {"config": config.label, "summary": aggregate_labels(rows),
            "per_question": rows}


def aggregate_labels(per_q: list[dict]) -> dict:
    """QA-acc + label histogram, overall and per question_type.

    QA-acc counts CORRECT (and ABSTAIN on abstention questions) as right,
    matching the LongMemEval judge convention used elsewhere in the harness.
    """
    def _acc(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        hist: dict[str, int] = {}
        correct = 0
        for r in rows:
            lab = r.get("autoeval_label") or "ERROR"
            hist[lab] = hist.get(lab, 0) + 1
            if lab == "CORRECT" or (
                lab == "ABSTAIN" and r["question_type"] == "abstention"
            ):
                correct += 1
        return {"n": n, "qa_acc": round(correct / n, 4), "labels": hist}

    by_type: dict[str, list[dict]] = {}
    for r in per_q:
        by_type.setdefault(r["question_type"], []).append(r)
    return {
        "overall": _acc(per_q),
        "by_question_type": {qt: _acc(rs) for qt, rs in sorted(by_type.items())},
    }


def run_sweep(
    *,
    records: list[dict],
    matrix: SweepMatrix,
    judge_model: str,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    concurrency: int = 1,
) -> dict:
    """Run the full reader sweep over a pinned-context record set.

    Returns ``{configs: [...], best, ceiling_note}``. ``best`` is the config
    with the highest overall QA-acc — the headline #116 signal.

    ``concurrency`` is forwarded to each config's per-question fan-out (see
    ``run_one_config``). Configs themselves run sequentially so the per-config
    pool size stays bounded at K; the speed-up is within each config's question
    set, which is where the call volume lives.
    """
    results = []
    for cfg in matrix.configs():
        results.append(run_one_config(
            records=records, config=cfg, judge_model=judge_model,
            reader_client=reader_client, judge_client=judge_client,
            progress=progress, concurrency=concurrency,
        ))
    best = max(
        results, key=lambda r: r["summary"]["overall"].get("qa_acc", 0.0),
        default=None,
    )
    return {
        "n_questions": len(records),
        "n_configs": len(results),
        "configs": results,
        "best": {"config": best["config"],
                 "qa_acc": best["summary"]["overall"]["qa_acc"]} if best else None,
    }


def estimate_sweep_calls(n_questions: int, matrix: SweepMatrix) -> dict:
    """Dry-run sizing: how many reader + judge LLM calls the sweep will make."""
    n_cfg = len(matrix.configs())
    calls = n_questions * n_cfg
    return {
        "n_questions": n_questions,
        "n_configs": n_cfg,
        "reader_calls": calls,
        "judge_calls": calls,
        "total_llm_calls": 2 * calls,
        "configs": [c.label for c in matrix.configs()],
    }


def merge_pinned_sources(paths: Iterable[Path]) -> dict[str, list[dict]]:
    """Group pinned-context files by their snippet-width axis label.

    Returns ``{snippet_width_label: records}`` so the sweep can report
    QA-acc per (reader-config × snippet-width) — the palace-daemon#150 axis.
    """
    grouped: dict[str, list[dict]] = {}
    for p in paths:
        meta, records = load_pinned_context(Path(p))
        label = meta.get("snippet_width") or meta.get("search_endpoint") or Path(p).stem
        grouped[label] = records
    return grouped
