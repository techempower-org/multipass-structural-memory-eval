"""Tests for Category 9: Harness Integration — The Handshake.

Covers 9b (call-through success) and 9a (invocation rate). The 9a tests
use a fake driver — no model API, no daemon — exercising the
model-agnostic scorer in isolation, the same way the 9b tests use a
stub adapter. The runtime drivers in ``sme.eval.cat9a_orchestrators``
need a live model and are exercised by the runner script, not here.
Other sub-tests (9c–9g) are spec'd but not implemented.
"""

from __future__ import annotations

from sme.adapters.base import HarnessDescriptor, ProbeResult, SMEAdapter
from sme.categories.harness_integration import (
    Cat9aQueryOutcome,
    format_cat9a_report,
    format_cat9b_report,
    run_cat9a,
    run_cat9b,
)


class _StubAdapter(SMEAdapter):
    """Minimal adapter that lets a test pre-declare a harness manifest.

    Only overrides the three abstract methods + get_harness_manifest().
    """

    def __init__(self, manifest: list[HarnessDescriptor]) -> None:
        self._manifest = manifest

    def ingest_corpus(self, corpus):
        return {"entities_created": 0, "edges_created": 0, "errors": [], "warnings": []}

    def query(self, question):
        from sme.adapters.base import QueryResult

        return QueryResult(answer="")

    def get_graph_snapshot(self):
        return [], []

    def get_harness_manifest(self):
        return self._manifest


# --- 9b: empty manifest ------------------------------------------------


def test_cat9b_empty_manifest_reports_not_applicable():
    adapter = _StubAdapter(manifest=[])
    result = run_cat9b(adapter)
    assert result.empty_manifest is True
    assert result.total_probes == 0
    assert result.band == "n/a"
    # call_through_rate must be None on empty manifest — distinct from
    # "every probe failed" which is the measured 0.0 floor. JSON
    # consumers reading the rate field without checking empty_manifest
    # would otherwise see "0%" when the truthful answer is "not measured."
    assert result.call_through_rate is None
    report = format_cat9b_report(result)
    assert "declared no harness manifest" in report


# --- 9b: all probes succeed --------------------------------------------


def test_cat9b_all_probes_succeed_is_healthy():
    descriptors = [
        HarnessDescriptor(
            name=f"probe_{i}",
            kind="mcp_resource",
            probe_fn=lambda: ProbeResult(success=True, latency_ms=1.0),
        )
        for i in range(3)
    ]
    adapter = _StubAdapter(manifest=descriptors)
    result = run_cat9b(adapter)
    assert result.total_probes == 3
    assert result.successful_probes == 3
    assert result.failed_probes == 0
    assert result.call_through_rate == 1.0
    assert result.band == "healthy"


# --- 9b: probe raises an exception -------------------------------------


def test_cat9b_raising_probe_counts_as_failure():
    def boom() -> ProbeResult:
        raise RuntimeError("integration broken")

    adapter = _StubAdapter(
        manifest=[
            HarnessDescriptor(name="boom", kind="tool_call", probe_fn=boom),
        ]
    )
    result = run_cat9b(adapter)
    assert result.failed_probes == 1
    assert result.successful_probes == 0
    assert result.readings[0].result.success is False
    assert "RuntimeError" in (result.readings[0].result.error or "")


# --- 9b: mixed outcomes give correct banding ---------------------------


def test_cat9b_mixed_outcomes_produce_warn_band():
    # 4 probes total, 3 succeed, 1 fails → 75% → concerning (< 80% warn floor)
    descriptors = [
        HarnessDescriptor(
            name=f"ok_{i}",
            kind="mcp_resource",
            probe_fn=lambda: ProbeResult(success=True, latency_ms=1.0),
        )
        for i in range(3)
    ]
    descriptors.append(
        HarnessDescriptor(
            name="bad",
            kind="mcp_resource",
            probe_fn=lambda: ProbeResult(success=False, error="timeout"),
        )
    )
    adapter = _StubAdapter(manifest=descriptors)
    result = run_cat9b(adapter)
    assert result.total_probes == 4
    assert result.successful_probes == 3
    assert result.failed_probes == 1
    assert abs(result.call_through_rate - 0.75) < 1e-9
    assert result.band == "concerning"  # 75% < 80% warn threshold


def test_cat9b_warn_band_at_exactly_80_percent():
    # 5 probes, 4 succeed → 80% → warn band
    descriptors = [
        HarnessDescriptor(
            name=f"ok_{i}",
            kind="mcp_resource",
            probe_fn=lambda: ProbeResult(success=True, latency_ms=1.0),
        )
        for i in range(4)
    ]
    descriptors.append(
        HarnessDescriptor(
            name="bad",
            kind="mcp_resource",
            probe_fn=lambda: ProbeResult(success=False, error="timeout"),
        )
    )
    adapter = _StubAdapter(manifest=descriptors)
    result = run_cat9b(adapter)
    assert result.band == "warn"


# --- 9b: by-kind breakdown ---------------------------------------------


def test_cat9b_by_kind_counts_are_accurate():
    descriptors = [
        HarnessDescriptor(
            name="mcp_ok",
            kind="mcp_resource",
            probe_fn=lambda: ProbeResult(success=True),
        ),
        HarnessDescriptor(
            name="mcp_bad",
            kind="mcp_resource",
            probe_fn=lambda: ProbeResult(success=False, error="x"),
        ),
        HarnessDescriptor(
            name="hook_ok",
            kind="claude_code_hook",
            probe_fn=lambda: ProbeResult(success=True),
        ),
    ]
    adapter = _StubAdapter(manifest=descriptors)
    result = run_cat9b(adapter)
    assert result.by_kind["mcp_resource"] == {"success": 1, "fail": 1}
    assert result.by_kind["claude_code_hook"] == {"success": 1, "fail": 0}


# --- 9b: bool-returning probe is tolerated -----------------------------


def test_cat9b_probe_returning_bool_is_coerced():
    """Defensive: users may reasonably write `return True` instead of
    `return ProbeResult(success=True)`. The runner coerces and continues.
    """
    adapter = _StubAdapter(
        manifest=[
            HarnessDescriptor(
                name="naive_true",
                kind="tool_call",
                probe_fn=lambda: True,  # type: ignore[return-value]
            ),
            HarnessDescriptor(
                name="naive_false",
                kind="tool_call",
                probe_fn=lambda: False,  # type: ignore[return-value]
            ),
        ]
    )
    result = run_cat9b(adapter)
    assert result.successful_probes == 1
    assert result.failed_probes == 1


# --- 9a: invocation-rate scorer (fake driver, no model) ----------------

# A few jp-realm-shaped questions. The fake driver's behavior is keyed on
# the question text so each test controls invoke/no-invoke + recall.
_Q = [
    {"id": "q1", "text": "What is familiar?", "expected_sources": ["familiar", "palace"]},
    {"id": "q2", "text": "What is sigil?", "expected_sources": ["sigil", "version"]},
    {"id": "q3", "text": "What is status?", "expected_sources": ["status"]},
]


def _driver_always_invokes(text):
    # Issues a tool call and returns context containing every expected token.
    ctx = "drawer: familiar palace sigil version status realm"
    return Cat9aQueryOutcome(question_id="", tool_calls=1, context=ctx, answer="ans")


def _driver_never_invokes(text):
    # Answers from weights — zero tool calls, empty retrieval context.
    return Cat9aQueryOutcome(question_id="", tool_calls=0, context="", answer="from memory")


def test_cat9a_full_invocation_is_healthy():
    result = run_cat9a(_Q, _driver_always_invokes, orchestrator="fake-high-tau2", tau2=99.0)
    assert result.total_questions == 3
    assert result.invoked_questions == 3
    assert result.invocation_rate == 1.0
    assert result.band == "healthy"
    assert result.errored_questions == 0
    # All expected tokens were in the context → full recall.
    assert abs(result.mean_recall - 1.0) < 1e-9
    assert result.hit_rate == 1.0
    assert result.tau2 == 99.0


def test_cat9a_zero_invocation_is_concerning():
    result = run_cat9a(_Q, _driver_never_invokes, orchestrator="fake-low-tau2", tau2=20.0)
    assert result.invoked_questions == 0
    assert result.invocation_rate == 0.0
    assert result.band == "concerning"
    # No retrieval context → zero recall, the substrate is moot.
    assert result.mean_recall == 0.0
    assert result.hit_rate == 0.0


def test_cat9a_partial_invocation_bands_warn():
    # Invoke on 2 of 3 → 66.7% → warn (≥60%, <90%).
    def driver(text):
        if "sigil" in text:
            return Cat9aQueryOutcome(question_id="", tool_calls=0, context="")
        return Cat9aQueryOutcome(
            question_id="", tool_calls=2, context="familiar palace status"
        )

    result = run_cat9a(_Q, driver, orchestrator="fake-mid", tau2=70.0)
    assert result.invoked_questions == 2
    assert abs(result.invocation_rate - 2 / 3) < 1e-9
    assert result.band == "warn"


def test_cat9a_driver_exception_counts_as_no_invoke():
    # A driver that raises on one question must not abort the run; that
    # question is recorded as errored + no-invoke (it never reached the tool).
    def driver(text):
        if "status" in text:
            raise TimeoutError("model timed out")
        return Cat9aQueryOutcome(question_id="", tool_calls=1, context="familiar palace sigil version")

    result = run_cat9a(_Q, driver, orchestrator="fake-flaky", tau2=50.0)
    assert result.total_questions == 3
    assert result.invoked_questions == 2
    assert result.errored_questions == 1
    errored = [r for r in result.readings if r.outcome.error]
    assert len(errored) == 1
    assert "TimeoutError" in errored[0].outcome.error
    assert errored[0].outcome.invoked is False


def test_cat9a_recall_uses_substring_match_against_context():
    # Context carries only one of two expected tokens → recall 0.5, hit True.
    def driver(text):
        return Cat9aQueryOutcome(question_id="", tool_calls=1, context="mentions familiar only")

    q = [{"id": "q", "text": "?", "expected_sources": ["familiar", "palace"]}]
    result = run_cat9a(q, driver, orchestrator="fake")
    assert abs(result.readings[0].recall - 0.5) < 1e-9
    assert result.readings[0].hit is True
    assert result.readings[0].matched_sources == ["familiar"]


def test_cat9a_report_carries_tau2_and_band():
    result = run_cat9a(_Q, _driver_always_invokes, orchestrator="claude-opus-4-8", tau2=99.3,
                       tau2_note="tau2-bench telecom")
    report = format_cat9a_report(result, source_label="jp-realm-v0.1")
    assert "Invocation Rate" in report
    assert "claude-opus-4-8" in report
    assert "99.3" in report
    assert "tau2-bench telecom" in report
    assert "100.0%" in report  # invocation rate
