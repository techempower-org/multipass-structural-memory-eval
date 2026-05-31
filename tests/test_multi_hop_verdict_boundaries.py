"""Boundary conditions for the Cat 2c multi-hop verdict.

``test_multi_hop.py`` already covers the clear-cut verdict cases (a solid
win, a solid loss, a clean neutral tax). What it does *not* pin are the
threshold boundaries — and the verdict uses strict inequalities at three
of them, so a value sitting exactly on the line is silently classified
the *other* way from what a casual reader expects:

  * "beats A" requires recall_delta_pp **> 5**, so a +5.0pp delta does
    NOT count as a win.
  * "loses to A" requires recall_delta_pp **< -5**, so a -5.0pp delta
    does NOT count as a loss.
  * "ratio grows" requires last **> first * 1.2**, so a ratio that lands
    exactly on first*1.2 is "uniform scale", not "earns complexity".

These off-by-epsilon boundaries are precisely where a verdict would flip
under a tiny measurement perturbation, so they are worth nailing down.
The expected verdicts were verified against the live ``_verdict`` before
being pinned here.
"""
from __future__ import annotations

from sme.categories.multi_hop import Cat2cReport, _verdict


def _report(deltas_a, ratios, deltas_c=None) -> Cat2cReport:
    r = Cat2cReport()
    r.delta_B_minus_A = deltas_a
    r.ratio_B_over_A = ratios
    if deltas_c is not None:
        r.delta_B_minus_C = deltas_c
    return r


# ── ±5pp win/loss boundaries (strict inequality) ──────────────────


def test_exactly_plus_5pp_is_not_a_win():
    """+5.0pp at every depth is on the boundary; the strict ``> 5`` test
    means it doesn't count as beating flat → neutral tax, not a win."""
    r = _report(
        {1: {"recall_delta_pp": 5.0, "tokens_delta": 0},
         2: {"recall_delta_pp": 5.0, "tokens_delta": 0}},
        {1: 1.05, 2: 1.05},
    )
    assert _verdict(r)[0] == "structure is a neutral tax"


def test_exactly_minus_5pp_is_not_a_loss():
    """-5.0pp is on the boundary; the strict ``< -5`` test means it
    doesn't count as losing → neutral tax, not harmful."""
    r = _report(
        {1: {"recall_delta_pp": -5.0, "tokens_delta": 0}},
        {1: 0.95},
    )
    assert _verdict(r)[0] == "structure is a neutral tax"


def test_just_over_5pp_is_a_win():
    """5.01pp clears the boundary → counts as beating flat."""
    r = _report(
        {1: {"recall_delta_pp": 5.01, "tokens_delta": 0},
         2: {"recall_delta_pp": 5.01, "tokens_delta": 0}},
        {1: 1.05, 2: 1.05},  # flat ratio → uniform scale, but it IS a win
    )
    assert _verdict(r)[0] == "structure adds value at uniform scale"


def test_just_under_minus_5pp_is_a_loss():
    """-5.01pp clears the boundary the other way → counts as a loss."""
    r = _report(
        {1: {"recall_delta_pp": -5.01, "tokens_delta": 0}},
        {1: 0.9},
    )
    assert _verdict(r)[0] == "structure harmful at multi-hop"


# ── ratio-grows boundary (last > first * 1.2) ─────────────────────


def test_ratio_exactly_at_1_2x_does_not_grow():
    """last == first * 1.2 is on the boundary; the strict ``>`` means the
    ratio is NOT considered to grow → uniform scale, not earns-complexity."""
    r = _report(
        {1: {"recall_delta_pp": 10.0, "tokens_delta": 0},
         2: {"recall_delta_pp": 10.0, "tokens_delta": 0}},
        {1: 1.0, 2: 1.2},  # 1.2 == 1.0 * 1.2 exactly
    )
    assert _verdict(r)[0] == "structure adds value at uniform scale"


def test_ratio_just_over_1_2x_earns_complexity():
    """last just past first * 1.2 → ratio grows → earns complexity."""
    r = _report(
        {1: {"recall_delta_pp": 10.0, "tokens_delta": 0},
         2: {"recall_delta_pp": 40.0, "tokens_delta": 0}},
        {1: 1.0, 2: 1.21},
    )
    assert _verdict(r)[0] == "structure earns complexity (scales with depth)"


# ── single-ratio guard (ratio_grows needs >= 2 points) ────────────


def test_single_hop_ratio_cannot_grow():
    """With only one hop bucket the ratio has nothing to grow against, so
    a win is reported as uniform scale, never earns-complexity."""
    r = _report(
        {1: {"recall_delta_pp": 50.0, "tokens_delta": 0}},
        {1: 5.0},  # huge but single-point → ratio_grows stays False
    )
    assert _verdict(r)[0] == "structure adds value at uniform scale"


# ── infinite-ratio entries are skipped in the grows check ─────────


def test_infinite_ratio_skipped_in_grows_check():
    """An inf ratio (A had zero recall at a depth) is excluded from the
    grows comparison; the remaining finite points decide it. Here only
    one finite ratio survives, so it can't grow → uniform scale."""
    r = _report(
        {1: {"recall_delta_pp": 60.0, "tokens_delta": 0},
         2: {"recall_delta_pp": 70.0, "tokens_delta": 0}},
        {1: float("inf"), 2: 3.0},
    )
    assert _verdict(r)[0] == "structure adds value at uniform scale"


# ── C-absent narration when A is present ──────────────────────────


def test_no_c_overlap_still_notes_missing_isolation():
    """A populated but no B-C overlap → the verdict still narrates that
    the structural contribution could not be isolated, so the reader
    isn't misled into thinking C was tested."""
    r = _report(
        {1: {"recall_delta_pp": 10.0, "tokens_delta": 0}},
        {1: 1.3},
    )
    verdict, details = _verdict(r)
    assert verdict == "structure adds value at uniform scale"
    assert any("no Condition C" in d for d in details)
