"""Claim-scoring dispatch edges for Cat 8 ontology coherence.

The existing ``test_ontology_coherence.py`` covers the common claim
branches (temporal, provenance, cat7 pass/fail, cat3, cat2b, the inline
override happy path). This module fills the dispatch edges that change
a claim's *status* — the field that feeds the Cat 8 pass rate — and so
silently move the published number if wrong:

  * ``_score_override`` for ``cat7_delta_recall`` when the Cat 7 results
    are present but missing the recall fields → ``skipped`` (distinct
    from "no Cat 7 data passed"), and an override with no registered
    handler → ``untestable``.
  * The ``five_standard_halls`` vocabulary-claim threshold: partial
    population below 0.5 fails, exactly 0.5 passes (``>=`` boundary).
  * ``_score_hall_usage`` in-vocabulary counting, including the
    ``hall_`` prefix-stripping match and the entity_type-suffix
    fallback when the ``hall`` property is empty.

All assertions were verified against the live functions before being
pinned.
"""
from __future__ import annotations

import pytest

from sme.adapters.base import Entity
from sme.categories.ontology_coherence import (
    ImpliedOntology,
    _score_hall_usage,
    _score_override,
    score_cat8,
)

_EMPTY_LIBRARY = {"claims": [], "untestable_patterns": []}


# ── _score_override: cat7_delta_recall skip branches ──────────────


def test_override_cat7_skipped_when_recall_fields_missing():
    """Cat 7 results present but lacking graph/flat recall → skipped with
    a note distinct from the no-data case (so the diagnosis is honest
    about *why* the claim couldn't be scored)."""
    override = {
        "metric": "cat7_delta_recall",
        "description": "structure is not a moat",
        "pass_condition": "abs(delta) < 0.1",
    }
    res = _score_override(
        "c1",
        "structure is not a moat",
        override,
        {},
        cat7_results={"unrelated_key": 1},
        cat3_results=None,
        cat2b_results=None,
    )
    assert res.status == "skipped"
    assert "missing recall" in res.notes


def test_override_cat7_skipped_when_no_results():
    override = {"metric": "cat7_delta_recall", "description": "d"}
    res = _score_override(
        "c1",
        "structure is not a moat",
        override,
        {},
        cat7_results=None,
        cat3_results=None,
        cat2b_results=None,
    )
    assert res.status == "skipped"
    assert "no Cat 7 data passed" in res.notes


def test_override_unknown_metric_is_untestable():
    override = {"metric": "no_such_handler", "description": "d"}
    res = _score_override(
        "c1",
        "x",
        override,
        {},
        cat7_results=None,
        cat3_results=None,
        cat2b_results=None,
    )
    assert res.status == "untestable"
    assert "no override handler" in res.notes


def test_override_cat7_pass_within_band():
    """Sanity floor for the pass path: |delta| < 0.1 → 'not a moat' passes."""
    override = {"metric": "cat7_delta_recall", "description": "not a moat"}
    res = _score_override(
        "c1",
        "structure is not a moat",
        override,
        {},
        cat7_results={"graph_mean_recall": 0.62, "flat_mean_recall": 0.60},
        cat3_results=None,
        cat2b_results=None,
    )
    assert res.status == "pass"
    assert res.metrics["delta_recall"] == pytest.approx(0.02)


def test_override_cat7_fail_outside_band():
    override = {"metric": "cat7_delta_recall", "description": "not a moat"}
    res = _score_override(
        "c1",
        "structure is not a moat",
        override,
        {},
        cat7_results={"graph_mean_recall": 0.80, "flat_mean_recall": 0.55},
        cat3_results=None,
        cat2b_results=None,
    )
    assert res.status == "fail"  # 0.25 delta is outside ±0.1


# ── five_standard_halls vocabulary-claim threshold ────────────────


def _drawer(idx: str, hall: str) -> Entity:
    return Entity(idx, idx, "drawer", properties={"hall": hall})


def _score_halls(drawers):
    implied = ImpliedOntology(
        version="t",
        source="declared",
        hall_vocabulary=["facts", "decisions"],
        vocabulary_claims=[{"id": "five_standard_halls", "text": "five standard halls"}],
    )
    rep = score_cat8(implied, drawers, [], {}, claim_library=_EMPTY_LIBRARY)
    return next(c for c in rep.claims if c.claim_id == "five_standard_halls")


def test_five_standard_halls_fails_below_half():
    # 1 of 4 populated → 0.25 < 0.5 → fail
    drawers = [
        _drawer("d1", "facts"),
        _drawer("d2", ""),
        _drawer("d3", ""),
        _drawer("d4", ""),
    ]
    res = _score_halls(drawers)
    assert res.status == "fail"
    assert res.metrics["fraction_populated"] == pytest.approx(0.25)


def test_five_standard_halls_passes_at_exactly_half():
    # 1 of 2 populated → 0.5, and the threshold is >= 0.5 → pass
    drawers = [_drawer("d1", "facts"), _drawer("d2", "")]
    res = _score_halls(drawers)
    assert res.status == "pass"
    assert res.metrics["fraction_populated"] == pytest.approx(0.5)


# ── _score_hall_usage in-vocabulary accounting ────────────────────


def test_hall_usage_in_vocab_matches_prefixed_and_bare():
    """A declared hall ``facts`` matches both the bare ``facts`` value and
    the ``hall_facts`` prefixed form (via ``hall_`` stripping). A populated
    hall outside the vocabulary is counted as populated but not in-vocab."""
    drawers = [
        _drawer("d1", "hall_facts"),  # prefixed → in vocab
        _drawer("d2", "facts"),       # bare → in vocab
        _drawer("d3", "weird_hall"),  # populated, out of vocab
        _drawer("d4", ""),            # unpopulated
    ]
    hu = _score_hall_usage(drawers, ["facts", "decisions"])
    assert hu["total_drawers"] == 4
    assert hu["populated_count"] == 3
    assert hu["fraction_populated"] == pytest.approx(0.75)
    assert hu["in_vocabulary_count"] == 2
    assert hu["in_vocabulary_fraction"] == pytest.approx(2 / 4)


def test_hall_usage_falls_back_to_entity_type_suffix():
    """When the ``hall`` property is empty/untyped, the hall is recovered
    from the entity_type suffix (``drawer:facts``)."""
    drawers = [
        Entity("d1", "d1", "drawer:facts", properties={}),
        Entity("d2", "d2", "drawer:untyped", properties={"hall": "untyped"}),
    ]
    hu = _score_hall_usage(drawers, ["facts"])
    # d1 recovers "facts" from the suffix → populated + in vocab.
    # d2 is "untyped" both ways → not populated.
    assert hu["populated_count"] == 1
    assert hu["in_vocabulary_count"] == 1
    assert hu["distribution"] == {"facts": 1}


def test_hall_usage_ignores_non_drawer_entities():
    """Only ``drawer``-typed entities count toward hall population."""
    ents = [
        _drawer("d1", "facts"),
        Entity("r1", "r1", "room", properties={"hall": "facts"}),
        Entity("w1", "w1", "wing", properties={}),
    ]
    hu = _score_hall_usage(ents, ["facts"])
    assert hu["total_drawers"] == 1  # the room/wing are not drawers
    assert hu["populated_count"] == 1
