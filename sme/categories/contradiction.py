"""Category 3: Contradiction Detection — The Dissonance.

Scores whether a memory system explicitly surfaces conflicting facts,
not merely whether it retrieves the most recent value (spec v8 §3).

The structured channel is ``ContradictionPair[]``: a system that models
contradictions returns one pair per seeded conflict via the adapter's
``get_contradiction_pairs()`` (or on ``QueryResult.contradictions`` for a
per-query reading). A flat retriever surfaces zero by construction.

Two readings:

  * **structural** — pairs the system surfaces from the graph's typed
    ``contradicts`` edges. Detection rate = surfaced seeded pairs /
    seeded pairs; precision = surfaced seeded / total surfaced.
  * **flat** — the substring-recall floor (no structured pairs). Always
    0 detection for the structured metric; carried so the headline is
    ``(structural − flat)``, per the diagnostic-not-benchmark posture.

The corpus seeds its contradiction pairs as ``contradicts`` edges in the
vault frontmatter; the adapter surfaces them as ContradictionPair[]. A
*seeded* pair is matched when the system surfaces a pair whose
unordered endpoint set equals the seeded edge's ``{from, to}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sme.adapters.base import (
    ContradictionPair,
    Edge,
    Entity,
    is_contradicts_edge,
)


@dataclass
class Cat3Report:
    """Cat 3 scorecard. ``flat_detection_rate`` is 0 by construction
    (the flat baseline surfaces no structured pairs); it's carried so a
    consumer reads the ``(structural − flat)`` delta, not the absolute."""

    seeded_pairs: int
    surfaced_pairs: int
    surfaced_seeded: int  # surfaced pairs that match a seeded pair
    detection_rate: float  # surfaced_seeded / seeded_pairs
    precision: float  # surfaced_seeded / surfaced_pairs
    flat_detection_rate: float = 0.0
    flat_precision: float = 0.0
    pairs: list[ContradictionPair] = field(default_factory=list)
    seeded_pair_keys: list[list[str]] = field(default_factory=list)
    matched_pair_keys: list[list[str]] = field(default_factory=list)
    detection_level: str = "structured"

    @property
    def detection_delta(self) -> float:
        """(structural − flat) detection rate — the headline metric."""
        return self.detection_rate - self.flat_detection_rate

    def to_dict(self) -> dict:
        return {
            "category": "cat_3_contradiction",
            "seeded_pairs": self.seeded_pairs,
            "surfaced_pairs": self.surfaced_pairs,
            "surfaced_seeded": self.surfaced_seeded,
            "structural_detection_rate": self.detection_rate,
            "structural_precision": self.precision,
            "flat_detection_rate": self.flat_detection_rate,
            "flat_precision": self.flat_precision,
            "detection_delta": self.detection_delta,
            "detection_level": self.detection_level,
            # Mirror the field name ontology_coherence.py reads for the
            # "conflict detection" claim cross-reference.
            "contradiction_pairs": self.surfaced_pairs,
            "seeded_pair_keys": self.seeded_pair_keys,
            "matched_pair_keys": self.matched_pair_keys,
            "pairs": [
                {
                    "claim_a": p.claim_a,
                    "claim_b": p.claim_b,
                    "source_a": p.source_a,
                    "source_b": p.source_b,
                }
                for p in self.pairs
            ],
        }


def _seeded_keys_from_edges(edges: list[Edge]) -> set[frozenset[str]]:
    """The ground-truth contradiction pairs are the corpus's typed
    ``contradicts`` edges, keyed on the unordered endpoint set."""
    return {
        frozenset({e.source_id, e.target_id})
        for e in edges
        if is_contradicts_edge(e.edge_type)
    }


def score_cat3(
    entities: list[Entity],
    edges: list[Edge],
    surfaced_pairs: list[ContradictionPair],
    *,
    flat_detection_rate: float = 0.0,
    flat_precision: float = 0.0,
) -> Cat3Report:
    """Produce a Cat 3 scorecard.

    Args:
        entities, edges: the structural graph snapshot. The seeded
            ground-truth pairs are derived from the ``contradicts`` edges
            in this snapshot.
        surfaced_pairs: what the system surfaced (from
            ``adapter.get_contradiction_pairs()``).
        flat_detection_rate / flat_precision: the flat-baseline floor
            (0 for a pure retriever; carried for the delta).
    """
    seeded_keys = _seeded_keys_from_edges(edges)
    surfaced_keys = [
        frozenset({p.source_a, p.source_b}) for p in surfaced_pairs
    ]
    matched = [k for k in surfaced_keys if k in seeded_keys]

    n_seeded = len(seeded_keys)
    n_surfaced = len(surfaced_pairs)
    n_matched = len(set(matched))

    detection_rate = (n_matched / n_seeded) if n_seeded else 0.0
    precision = (n_matched / n_surfaced) if n_surfaced else 0.0

    return Cat3Report(
        seeded_pairs=n_seeded,
        surfaced_pairs=n_surfaced,
        surfaced_seeded=n_matched,
        detection_rate=detection_rate,
        precision=precision,
        flat_detection_rate=flat_detection_rate,
        flat_precision=flat_precision,
        pairs=list(surfaced_pairs),
        seeded_pair_keys=[sorted(k) for k in seeded_keys],
        matched_pair_keys=[sorted(k) for k in set(matched)],
    )


def format_report(report: Cat3Report) -> str:
    lines = [
        "Cat 3 — The Dissonance (Contradiction Detection)",
        "═" * 56,
        "",
        f"  Seeded contradiction pairs:   {report.seeded_pairs}",
        f"  Surfaced pairs:               {report.surfaced_pairs}",
        f"  Surfaced ∩ seeded:            {report.surfaced_seeded}",
        "",
        f"  Structural detection rate:    {report.detection_rate:.2f}",
        f"  Structural precision:         {report.precision:.2f}",
        f"  Flat-baseline detection rate: {report.flat_detection_rate:.2f}",
        f"  (structural − flat) delta:    {report.detection_delta:+.2f}",
        "",
        "Reading",
        "─" * 56,
    ]
    if report.seeded_pairs == 0:
        lines.append(
            "  No seeded contradiction edges in this snapshot — Cat 3 "
            "is not exercised by this corpus/graph."
        )
        return "\n".join(lines)
    if report.detection_rate >= 0.99:
        lines.append(
            "  ● Full structured detection: the system surfaced every "
            "seeded contradiction as a ContradictionPair. The flat "
            "baseline surfaces 0 by construction, so the entire reading "
            "is the structural channel's contribution."
        )
    elif report.detection_rate > 0:
        lines.append(
            "  ● Partial structured detection: the system surfaced some "
            "but not all seeded contradictions. Investigate which "
            "contradicts edges did not project into pairs."
        )
    else:
        lines.append(
            "  ● No structured detection: the system retrieves but does "
            "not model contradictions. This is the honest 0 for a flat "
            "retriever — the (structural − flat) delta is also 0."
        )
    return "\n".join(lines)
