"""Tests for _stratified_cap — techempower-org/...#122.

The oracle/S LongMemEval corpora are sorted by question_type, so a bare
``questions[:n]`` cap is a single-category slice. ``_stratified_cap`` draws
round-robin across the field's values to keep a cap representative.
"""
from dataclasses import dataclass

from scripts.run_longmemeval_mempalace import _stratified_cap


@dataclass
class Q:
    question_id: str
    question_type: str


def _sorted_corpus():
    """A category-sorted corpus, like the real S/oracle files: all of type A,
    then all of type B, then C."""
    qs = []
    for t, count in (("A", 50), ("B", 50), ("C", 50)):
        qs.extend(Q(f"{t}{i}", t) for i in range(count))
    return qs


def test_bare_cap_would_be_single_category():
    """Guards the bug: the first 30 of a sorted corpus are one category."""
    qs = _sorted_corpus()
    naive = qs[:30]
    assert {q.question_type for q in naive} == {"A"}  # the bug #122 fixes


def test_stratified_cap_is_even_across_categories():
    qs = _sorted_corpus()
    out = _stratified_cap(qs, 30, "question_type")
    assert len(out) == 30
    from collections import Counter
    dist = Counter(q.question_type for q in out)
    assert dist == {"A": 10, "B": 10, "C": 10}


def test_stratified_cap_preserves_within_group_order():
    qs = _sorted_corpus()
    out = _stratified_cap(qs, 6, "question_type")
    a_ids = [q.question_id for q in out if q.question_type == "A"]
    assert a_ids == ["A0", "A1"]  # first-in-group, not shuffled


def test_stratified_cap_handles_uneven_groups():
    """A small group is exhausted; remaining slots fill from larger groups."""
    qs = [Q("A0", "A"), Q("A1", "A"), Q("A2", "A"), Q("A3", "A"), Q("B0", "B")]
    out = _stratified_cap(qs, 4, "question_type")
    assert len(out) == 4
    from collections import Counter
    dist = Counter(q.question_type for q in out)
    assert dist["B"] == 1  # only one B existed
    assert dist["A"] == 3  # remainder filled from A


def test_stratified_cap_n_exceeds_corpus_returns_all():
    qs = _sorted_corpus()
    out = _stratified_cap(qs, 1000, "question_type")
    assert len(out) == len(qs)


def test_stratified_cap_works_on_dict_records():
    """Falls back to dict.get when records aren't attribute objects."""
    qs = [{"question_type": "A"}, {"question_type": "B"}, {"question_type": "A"}]
    out = _stratified_cap(qs, 2, "question_type")
    assert len(out) == 2
    assert {r["question_type"] for r in out} == {"A", "B"}
