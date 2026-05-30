"""Category 9: Harness Integration — The Handshake.

Tests whether the memory system is actually reachable through its
declared invocation surfaces (MCP servers, Claude Code hooks, tool
calls, slash commands, custom actions). Every other SME category
measures offline retrieval — this category measures the layer between
retrieval and a running model.

Current scope:

  9a  Invocation rate
      Given a real model orchestrating the memory system through a
      declared tool surface, the fraction of questions on which the
      model actually *issues* at least one tool call before answering.
      This is the willingness-to-invoke layer — it is the dominant
      lever on effective memory once retrieval is healthy, and it
      tracks the orchestrator model's tool-agent ability (Tau2) rather
      than its parameter count. The scorer here (``run_cat9a``) is
      model-agnostic: a driver runs each question through a real model
      and hands back a ``Cat9aQueryOutcome`` per question; the scorer
      tallies invocation rate (and, when expected sources are present,
      a comparable substring recall). Drivers for specific runtimes
      (Bedrock / ollama tool-use loops) live in
      ``sme.eval.cat9a_orchestrators``; they are optional and import
      lazily so the core stays dependency-light.

  9b  Call-through success
      For each ``HarnessDescriptor`` returned by the adapter, invoke
      ``probe_fn`` once and report whether the call completed. A low
      9b means the integration is broken (bad schema, timeout, wrong
      parameters, tool not registered, MCP server unreachable). A high
      9b means the surface is live — it says nothing about whether the
      model will actually invoke it, which is what 9a measures.

Planned (not implemented here; see spec v8 § Category 9):

  9c  Result usage          — needs real model API + Cat 1 matcher
  9d  Negative-control rate — needs real model API + held-out set
  9e  Per-model sensitivity — needs multi-model API access
  9f  Per-harness portability — needs per-harness runners
  9g  Hook-driven access    — needs per-harness shims (Claude Code,
                               Cursor, LangGraph, etc.)

9b was implemented first because — per the spec — it is the one
sub-test that "can be measured against a mock model that always invokes
the tool" (no API keys, no cost). 9a was added next (issue #194): it
needs a real model API, but the scorer/IO split keeps the cost and the
runtime dependency confined to the driver, not the category logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from sme.adapters.base import HarnessDescriptor, ProbeResult, SMEAdapter

log = logging.getLogger(__name__)


# --- Bands for the Reading section ------------------------------------

_CALL_THROUGH_HEALTHY = 1.00   # all surfaces callable
_CALL_THROUGH_WARN = 0.80      # 80-99% — partial integration


def _band(value: float, healthy: float, warn: float) -> str:
    if value >= healthy:
        return "healthy"
    if value >= warn:
        return "warn"
    return "concerning"


# --- Result types -----------------------------------------------------


@dataclass
class ProbeReading:
    """One descriptor's probe outcome with SME-side wrapping."""

    descriptor: HarnessDescriptor
    result: ProbeResult


@dataclass
class Cat9bResult:
    """Category 9b — call-through success — scorecard."""

    total_probes: int
    successful_probes: int
    failed_probes: int
    # Probes grouped by HarnessDescriptor.kind for per-surface breakdown.
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)
    readings: list[ProbeReading] = field(default_factory=list)
    # Present when the adapter declares no manifest — distinct from zero
    # probes succeeding (which is a real failure).
    empty_manifest: bool = False

    @property
    def call_through_rate(self) -> Optional[float]:
        """Fraction of declared surfaces that answered.

        Returns ``None`` when the adapter declares no harness manifest —
        a "not measured" signal distinct from "every probe failed"
        (which would be ``0.0``). Consumers reading the JSON should
        treat ``null`` as "Cat 9b does not apply to this system" and
        ``0.0`` as a measured floor.
        """
        if self.empty_manifest:
            return None
        if self.total_probes == 0:
            return 0.0
        return self.successful_probes / self.total_probes

    @property
    def band(self) -> str:
        if self.empty_manifest:
            return "n/a"
        rate = self.call_through_rate
        if rate is None:
            return "n/a"
        return _band(rate, _CALL_THROUGH_HEALTHY, _CALL_THROUGH_WARN)


# --- Sub-test: 9b call-through success --------------------------------


def run_cat9b(adapter: SMEAdapter, *, timeout_per_probe_s: float = 10.0) -> Cat9bResult:
    """Execute Category 9b against an adapter's declared harness manifest.

    Invokes each ``HarnessDescriptor.probe_fn`` exactly once and tallies
    successes. A probe that raises an exception counts as a failure with
    the exception message captured in ``ProbeResult.error``. A probe
    that exceeds ``timeout_per_probe_s`` is not automatically aborted —
    the probe_fn owns its own timeout semantics — but the elapsed
    wall-clock is recorded in ``ProbeResult.latency_ms``.

    Consumers of this function (the CLI, a CI harness, a test) should
    treat an empty manifest as "the adapter doesn't declare a harness
    surface" — a reporting outcome, not a pass/fail.
    """
    manifest = adapter.get_harness_manifest()

    if not manifest:
        return Cat9bResult(
            total_probes=0,
            successful_probes=0,
            failed_probes=0,
            empty_manifest=True,
        )

    readings: list[ProbeReading] = []
    by_kind: dict[str, dict[str, int]] = {}
    successful = 0
    failed = 0

    for descriptor in manifest:
        start = time.perf_counter()
        try:
            result = descriptor.probe_fn()
            if not isinstance(result, ProbeResult):
                # Tolerate the common mistake of returning a bool.
                result = ProbeResult(
                    success=bool(result),
                    latency_ms=(time.perf_counter() - start) * 1000,
                    error=None if result else "probe_fn returned non-ProbeResult falsy value",
                )
            # Fill latency if the probe forgot to.
            if result.latency_ms == 0.0:
                result.latency_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:  # noqa: BLE001 — intentional; probe_fn is user code
            latency = (time.perf_counter() - start) * 1000
            log.debug("probe %r raised %s", descriptor.name, exc)
            result = ProbeResult(
                success=False,
                latency_ms=latency,
                error=f"{type(exc).__name__}: {exc}",
            )

        readings.append(ProbeReading(descriptor=descriptor, result=result))

        kind_bucket = by_kind.setdefault(descriptor.kind, {"success": 0, "fail": 0})
        if result.success:
            successful += 1
            kind_bucket["success"] += 1
        else:
            failed += 1
            kind_bucket["fail"] += 1

    return Cat9bResult(
        total_probes=len(manifest),
        successful_probes=successful,
        failed_probes=failed,
        by_kind=by_kind,
        readings=readings,
        empty_manifest=False,
    )


# --- Formatting helpers (used by the CLI) -----------------------------


def format_cat9b_report(result: Cat9bResult, *, source_label: str = "") -> str:
    """Return a human-readable scorecard for Category 9b.

    Follows the same banded-reading shape as ``cat4`` / ``cat5`` so the
    CLI output stays consistent. Probe-level detail is included so a
    failing call-through rate is actionable.
    """
    lines: list[str] = []

    header = "Category 9b — Harness Integration (Call-Through Success)"
    if source_label:
        header = f"{header} — {source_label}"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")

    if result.empty_manifest:
        lines.append(
            "  Adapter declared no harness manifest — pure library usage only. "
            "Cat 9b does not apply."
        )
        lines.append("")
        lines.append(
            "  If this memory system is reachable via MCP, hooks, tool calls, or "
            "slash commands, implement ``get_harness_manifest()`` so SME can probe "
            "each surface."
        )
        return "\n".join(lines) + "\n"

    rate = result.call_through_rate
    rate_pct = (rate * 100) if rate is not None else 0.0
    lines.append(
        f"  Probes: {result.total_probes} total — "
        f"{result.successful_probes} succeeded, "
        f"{result.failed_probes} failed "
        f"({rate_pct:.1f}% call-through, band: {result.band})"
    )
    lines.append("")

    if result.by_kind:
        lines.append("  By surface kind:")
        for kind, counts in sorted(result.by_kind.items()):
            total = counts["success"] + counts["fail"]
            success = counts["success"]
            pct = (success / total * 100) if total else 0.0
            lines.append(f"    {kind:22} {success}/{total} ({pct:.0f}%)")
        lines.append("")

    if result.failed_probes:
        lines.append("  Failed probes:")
        for reading in result.readings:
            if reading.result.success:
                continue
            error = reading.result.error or "(no error captured)"
            lines.append(
                f"    - {reading.descriptor.kind}/{reading.descriptor.name}: {error}"
            )
        lines.append("")

    lines.append("  Reading:")
    if result.band == "healthy":
        lines.append(
            "    All declared surfaces answered. The memory system is live at every "
            "harness point it claims to support."
        )
    elif result.band == "warn":
        lines.append(
            "    Some surfaces are unreachable. Likely integration regressions rather "
            "than retrieval problems — inspect the failed probes above."
        )
    else:
        lines.append(
            "    Most surfaces failed to respond. The memory system is effectively "
            "unavailable to callers using the declared harness contract. Cat 9a "
            "(invocation rate) readings downstream of this will be artificially low."
        )
    lines.append("")

    lines.append(
        "  Note: 9b measures whether a mock-invoker can reach each surface. It does "
        "NOT measure whether a real model would actually invoke the tool (9a), use "
        "the result (9c), or skip when unnecessary (9d). Those sub-tests require a "
        "real model runtime and are tracked separately."
    )
    return "\n".join(lines) + "\n"


# --- Sub-test: 9a invocation rate -------------------------------------
#
# 9a needs a real model. To keep the category logic free of any runtime
# dependency (boto3, the ollama server, an API key), the model side is a
# *driver*: a callable that takes a question string and returns a
# ``Cat9aQueryOutcome`` describing how many tool calls the model issued,
# the context those calls returned, and the model's final answer. The
# scorer below is pure: given a list of outcomes (plus the questions'
# expected sources) it tallies the invocation rate and a comparable
# substring recall. This is the same split 9b uses (probe_fn is the
# driver, run_cat9b is the scorer) and it keeps 9a unit-testable with a
# fake driver — no model required.

_INVOCATION_HEALTHY = 0.90   # ≥90% of questions trigger a tool call
_INVOCATION_WARN = 0.60      # 60-89% — orchestrator under-invokes


@dataclass
class Cat9aQueryOutcome:
    """One question's result from a real-model orchestration run.

    ``tool_calls`` is the number of times the model invoked the memory
    tool (the load-bearing 9a signal — distinct from how many drawers
    came back, which inflated the original 2026-04-30 baselines; see
    ``docs/ideas.md`` § "Caveat on the fine-grained call-count
    histogram"). ``context`` is the concatenation of what those calls
    returned, used by the substring scorer so 9a recall is comparable
    to the Cat-1 / ``retrieve`` numbers.
    """

    question_id: str
    tool_calls: int
    context: str = ""
    answer: str = ""
    error: Optional[str] = None

    @property
    def invoked(self) -> bool:
        """Did the model issue at least one tool call? The binary 9a
        signal — robust even on runs where the per-call count is noisy."""
        return self.tool_calls > 0


@dataclass
class Cat9aQuestionReading:
    """Per-question 9a row with scoring attached."""

    question_id: str
    text: str
    expected_sources: list[str]
    outcome: Cat9aQueryOutcome
    matched_sources: list[str] = field(default_factory=list)
    recall: float = 0.0

    @property
    def hit(self) -> bool:
        return self.recall > 0.0


@dataclass
class Cat9aResult:
    """Category 9a — invocation rate — scorecard for one orchestrator.

    ``tau2`` is the orchestrator's published tool-agent (Tau2) score, if
    known. Per the spec and ``reference_tau2_predicts_cat9a``, every 9a
    reading should carry it so cross-model comparisons are apples-to-
    apples — invocation rate tracks Tau2, not parameter count.
    """

    orchestrator: str
    total_questions: int
    invoked_questions: int
    errored_questions: int
    readings: list[Cat9aQuestionReading] = field(default_factory=list)
    tau2: Optional[float] = None
    tau2_note: str = ""

    @property
    def invocation_rate(self) -> Optional[float]:
        """Fraction of questions on which the model issued ≥1 tool call.

        Errored questions stay in the denominator — a model that times
        out or 400s never reached the tool, so it didn't invoke. Returns
        ``None`` only when there were no questions at all.
        """
        if self.total_questions == 0:
            return None
        return self.invoked_questions / self.total_questions

    @property
    def mean_recall(self) -> float:
        if not self.readings:
            return 0.0
        return sum(r.recall for r in self.readings) / len(self.readings)

    @property
    def hit_rate(self) -> float:
        if not self.readings:
            return 0.0
        return sum(1 for r in self.readings if r.hit) / len(self.readings)

    @property
    def band(self) -> str:
        rate = self.invocation_rate
        if rate is None:
            return "n/a"
        return _band(rate, _INVOCATION_HEALTHY, _INVOCATION_WARN)


def _score_recall(context: str, expected_sources: list[str]) -> tuple[list[str], float]:
    """Substring recall — identical contract to ``cmd_retrieve``: a source
    is matched if its token appears anywhere in the orchestration context.
    Returns (matched, recall)."""
    if not expected_sources:
        return [], 0.0
    matched = [src for src in expected_sources if src in context]
    return matched, len(matched) / len(expected_sources)


def run_cat9a(
    questions: list[dict],
    driver,
    *,
    orchestrator: str,
    tau2: Optional[float] = None,
    tau2_note: str = "",
    on_question=None,
) -> Cat9aResult:
    """Execute Category 9a against a real-model orchestration ``driver``.

    Args:
        questions: list of question dicts (``id``, ``text``,
            ``expected_sources``) — the jp-realm-v0.1 corpus shape.
        driver: callable ``(question_text) -> Cat9aQueryOutcome``. The
            driver owns the model + the tool loop + the memory backend;
            this function never touches a model directly.
        orchestrator: human-readable model id for the reading.
        tau2: the orchestrator's published Tau2 score (0-100), if known.
        tau2_note: provenance for the Tau2 number (source / domain).
        on_question: optional ``(reading) -> None`` progress callback.

    The driver is called once per question. A driver exception is
    captured as an errored outcome (zero tool calls) rather than
    aborting the run — one model timing out shouldn't lose the rest.
    """
    readings: list[Cat9aQuestionReading] = []
    invoked = 0
    errored = 0

    for q in questions:
        qid = q.get("id", "?")
        text = q.get("text", "")
        expected = q.get("expected_sources", []) or []
        try:
            outcome = driver(text)
            if not isinstance(outcome, Cat9aQueryOutcome):
                raise TypeError(
                    f"driver returned {type(outcome).__name__}, expected "
                    "Cat9aQueryOutcome"
                )
            outcome.question_id = qid
        except Exception as exc:  # noqa: BLE001 — isolate one bad question
            log.warning("9a driver raised on %s: %s", qid, exc)
            outcome = Cat9aQueryOutcome(
                question_id=qid, tool_calls=0, error=f"{type(exc).__name__}: {exc}"
            )

        matched, recall = _score_recall(outcome.context, expected)
        reading = Cat9aQuestionReading(
            question_id=qid,
            text=text,
            expected_sources=expected,
            outcome=outcome,
            matched_sources=matched,
            recall=recall,
        )
        readings.append(reading)
        if outcome.invoked:
            invoked += 1
        if outcome.error:
            errored += 1
        if on_question is not None:
            on_question(reading)

    return Cat9aResult(
        orchestrator=orchestrator,
        total_questions=len(questions),
        invoked_questions=invoked,
        errored_questions=errored,
        readings=readings,
        tau2=tau2,
        tau2_note=tau2_note,
    )


def format_cat9a_report(result: Cat9aResult, *, source_label: str = "") -> str:
    """Human-readable scorecard for Category 9a, matching the 9b shape."""
    lines: list[str] = []

    header = "Category 9a — Harness Integration (Invocation Rate)"
    if source_label:
        header = f"{header} — {source_label}"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")

    rate = result.invocation_rate
    rate_pct = (rate * 100) if rate is not None else 0.0
    tau2_str = f"{result.tau2:.1f}" if result.tau2 is not None else "—"
    lines.append(f"  Orchestrator: {result.orchestrator}  (Tau2: {tau2_str})")
    if result.tau2_note:
        lines.append(f"    Tau2 source: {result.tau2_note}")
    lines.append(
        f"  Invocation rate: {result.invoked_questions}/{result.total_questions} "
        f"({rate_pct:.1f}%, band: {result.band})"
    )
    lines.append(
        f"  Recall (substring): {result.mean_recall:.1%} mean, "
        f"{result.hit_rate:.1%} hit-rate"
    )
    if result.errored_questions:
        lines.append(f"  Errored questions: {result.errored_questions} (counted as no-invoke)")
    lines.append("")

    lines.append("  Reading:")
    if result.band == "healthy":
        lines.append(
            "    The orchestrator reliably invokes the memory tool. Effective memory "
            "is gated by retrieval quality, not by willingness to invoke."
        )
    elif result.band == "warn":
        lines.append(
            "    The orchestrator under-invokes — it answers from its own weights on "
            "a meaningful fraction of questions instead of reaching the memory. This "
            "is the dominant lever on effective memory: a higher-Tau2 orchestrator "
            "typically lifts this number more than a better retriever does."
        )
    else:
        lines.append(
            "    The orchestrator rarely invokes the memory tool — most answers come "
            "from training data, not retrieval. Cat 1-8 retrieval quality is largely "
            "moot at this invocation rate. Swap to a higher-Tau2 orchestrator before "
            "tuning the substrate."
        )
    lines.append("")
    lines.append(
        "  Note: per reference_tau2_predicts_cat9a, invocation rate tracks the "
        "orchestrator's Tau2 (tool-agent) score, not its parameter count. Record "
        "Tau2 alongside every 9a reading so cross-model comparisons stay honest."
    )
    return "\n".join(lines) + "\n"
