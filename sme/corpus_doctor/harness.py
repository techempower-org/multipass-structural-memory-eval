"""Verification harness — the inject → detect → assert loop.

Ties the injector to the detectors and scores detection against the
injection manifest. This is what makes corpus-doctor a *verification*
tool rather than just a corruptor: for each defect type it asserts the
relevant SME signal recalls the KNOWN injected defects.

Recall here is over injected defects (how many of the known defects the
detector flagged). Precision is over flagged ids (how many flagged ids
were actually injected) — but precision must account for defects that
pre-exist in the clean corpus (a real graph has its own isolates and
collisions). The harness therefore computes precision against the
*delta*: ids the detector flags on the corrupted graph that it did NOT
flag on the clean graph are attributable to the injection. This avoids
penalizing the detector for correctly flagging native defects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sme.adapters.base import Edge, Entity
from sme.corpus_doctor.detectors import DETECTORS
from sme.corpus_doctor.injector import (
    DEFECT_TYPES,
    CorpusDoctor,
    InjectionResult,
)


@dataclass
class DetectionResult:
    """Outcome of one inject → detect → score cycle."""

    defect_type: str
    injected: int  # number of defects injected
    recalled: int  # injected defects the detector flagged
    recall: float  # recalled / injected
    # Ids the detector newly flags vs the clean baseline (injection-
    # attributable), and how many of those were actually injected.
    new_flagged: int
    new_true_positives: int
    delta_precision: float  # new_true_positives / new_flagged
    missed_ids: list[str]   # injected defect ids the detector missed
    detected_all: bool      # recall == 1.0

    def summary(self) -> str:
        return (
            f"{self.defect_type}: recall {self.recalled}/{self.injected} "
            f"({self.recall:.2f}), Δprecision {self.delta_precision:.2f} "
            f"({self.new_true_positives}/{self.new_flagged} new flags injected)"
        )


def _expected_ids(result: InjectionResult, defect_type: str) -> set[str]:
    if defect_type == "duplicate_entity":
        return result.expected_duplicate_ids()
    if defect_type == "orphan_node":
        return result.expected_orphan_ids()
    if defect_type == "broken_ref":
        return result.expected_dangling_target_ids()
    raise ValueError(f"unknown defect_type {defect_type!r}")


def verify_defect(
    defect_type: str,
    clean_entities: list[Entity],
    clean_edges: list[Edge],
    *,
    count: int = 3,
    seed: int = 0,
) -> DetectionResult:
    """Inject ``count`` defects of ``defect_type`` and score detection.

    Runs the detector on both the clean and corrupted graphs so
    precision can be measured against the injection delta (native
    defects in the clean corpus are not counted against the detector).
    """
    if defect_type not in DEFECT_TYPES:
        raise ValueError(
            f"unknown defect_type {defect_type!r}; expected one of {DEFECT_TYPES}"
        )
    detector = DETECTORS[defect_type]

    # Baseline: what the detector flags on the clean graph.
    baseline_flagged = detector(clean_entities, clean_edges)

    # Inject, then detect on the corrupted graph.
    doctor = CorpusDoctor(seed=seed)
    result = doctor.inject(defect_type, clean_entities, clean_edges, count=count)
    flagged = detector(result.entities, result.edges)

    expected = _expected_ids(result, defect_type)
    recalled_ids = expected & flagged
    missed = sorted(expected - flagged)

    new_flagged = flagged - baseline_flagged
    new_true_positives = new_flagged & expected
    delta_precision = (
        len(new_true_positives) / len(new_flagged) if new_flagged else 1.0
    )

    injected = len(expected)
    recall = len(recalled_ids) / injected if injected else 1.0
    return DetectionResult(
        defect_type=defect_type,
        injected=injected,
        recalled=len(recalled_ids),
        recall=recall,
        new_flagged=len(new_flagged),
        new_true_positives=len(new_true_positives),
        delta_precision=delta_precision,
        missed_ids=missed,
        detected_all=recall == 1.0,
    )


def run_all_defects(
    clean_entities: list[Entity],
    clean_edges: list[Edge],
    *,
    count: int = 3,
    seed: int = 0,
    defect_types: Optional[tuple[str, ...]] = None,
) -> dict[str, DetectionResult]:
    """Run the inject → detect → assert loop for every defect type.

    Returns a ``{defect_type: DetectionResult}`` map. A clean bill of
    health is ``all(r.detected_all for r in results.values())`` — every
    injected defect was caught by its category.
    """
    types = defect_types or DEFECT_TYPES
    return {
        dt: verify_defect(dt, clean_entities, clean_edges, count=count, seed=seed)
        for dt in types
    }
