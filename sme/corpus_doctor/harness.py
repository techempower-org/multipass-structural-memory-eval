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

from dataclasses import dataclass, field
from typing import Optional

from sme.adapters.base import Edge, Entity
from sme.corpus_doctor.detectors import (
    DETECTORS,
    ID_RECALL_DEFECTS,
    detect_phantom_edges,
    monoculture_signal,
)
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
    # Scalar-signal defects (edge_type_monoculture) carry a before/after
    # reading instead of id recall. Empty for id-recall / edge-key defects.
    signal_clean: dict[str, float] = field(default_factory=dict)
    signal_dirty: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        if self.signal_clean:
            parts = ", ".join(
                f"{k} {self.signal_clean.get(k, 0):.3f}->{self.signal_dirty.get(k, 0):.3f}"
                for k in self.signal_dirty
            )
            verdict = "moved" if self.detected_all else "NO MOVE"
            return f"{self.defect_type}: signal {verdict} ({parts})"
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


def _verify_id_recall(
    defect_type: str,
    clean_entities: list[Entity],
    clean_edges: list[Edge],
    *,
    count: int,
    seed: int,
) -> DetectionResult:
    """id-recall path: duplicate_entity / orphan_node / broken_ref.

    Runs the detector on both the clean and corrupted graphs so precision
    is measured against the injection delta (native defects in the clean
    corpus are not counted against the detector)."""
    detector = DETECTORS[defect_type]
    baseline_flagged = detector(clean_entities, clean_edges)

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


def _verify_phantom_edge(
    clean_entities: list[Entity],
    clean_edges: list[Edge],
    *,
    count: int,
    seed: int,
    source_bodies: Optional[dict[str, str]],
) -> DetectionResult:
    """phantom_edge path: recall over injected ``(src, dst, type)`` edge
    keys, scored against the phantom-edge detector.

    The injected edges are grounded against an EMPTY body (so they are
    unambiguously phantom). The corpus's real ``source_bodies`` are merged
    in so native edges are still checkable for the delta-precision
    baseline — an injected phantom must be a NEW flag, not one the corpus
    already had."""
    bodies = dict(source_bodies or {})

    # Baseline: phantom edges the detector flags on the clean graph (with
    # only the real corpus bodies). Edge keys, not ids.
    baseline_flagged = detect_phantom_edges(clean_entities, clean_edges, bodies)

    doctor = CorpusDoctor(seed=seed)
    result = doctor.inject(
        "phantom_edge", clean_entities, clean_edges, count=count
    )
    # The injected edges carry a synthetic source_note with an empty body;
    # merge it in so they are checkable (absent notes are skipped).
    dirty_bodies = {**bodies, **result.phantom_source_bodies()}
    flagged = detect_phantom_edges(result.entities, result.edges, dirty_bodies)

    expected = result.expected_phantom_edge_keys()
    recalled = expected & flagged
    missed = sorted(f"{s}-[{t}]->{d}" for (s, d, t) in (expected - flagged))

    new_flagged = flagged - baseline_flagged
    new_true_positives = new_flagged & expected
    delta_precision = (
        len(new_true_positives) / len(new_flagged) if new_flagged else 1.0
    )

    injected = len(expected)
    recall = len(recalled) / injected if injected else 1.0
    return DetectionResult(
        defect_type="phantom_edge",
        injected=injected,
        recalled=len(recalled),
        recall=recall,
        new_flagged=len(new_flagged),
        new_true_positives=len(new_true_positives),
        delta_precision=delta_precision,
        missed_ids=missed,
        detected_all=recall == 1.0,
    )


def _verify_monoculture(
    clean_entities: list[Entity],
    clean_edges: list[Edge],
    *,
    count: int,
    seed: int,
) -> DetectionResult:
    """edge_type_monoculture path: scalar-signal detection.

    There is no per-id defect set; the test is whether Cat 4c's
    monoculture reading moved in the expected direction — the dominant
    edge-type fraction UP and the normalized edge-type entropy DOWN —
    relative to the clean baseline. ``detected_all`` is True iff both move
    correctly."""
    clean_signal = monoculture_signal(clean_entities, clean_edges)

    doctor = CorpusDoctor(seed=seed)
    result = doctor.inject(
        "edge_type_monoculture", clean_entities, clean_edges, count=count
    )
    dirty_signal = monoculture_signal(result.entities, result.edges)

    frac_up = (
        dirty_signal["dominant_edge_type_fraction"]
        > clean_signal["dominant_edge_type_fraction"]
    )
    # Entropy should not INCREASE (amplifying the dominant type can only
    # hold entropy flat in the degenerate single-type case, or lower it).
    entropy_down = (
        dirty_signal["edge_type_entropy_normalized"]
        <= clean_signal["edge_type_entropy_normalized"]
    )
    moved = frac_up and entropy_down

    return DetectionResult(
        defect_type="edge_type_monoculture",
        injected=count,
        recalled=count if moved else 0,
        recall=1.0 if moved else 0.0,
        new_flagged=count,
        new_true_positives=count if moved else 0,
        delta_precision=1.0 if moved else 0.0,
        missed_ids=[] if moved else ["dominant-fraction-did-not-rise"],
        detected_all=moved,
        signal_clean=clean_signal,
        signal_dirty=dirty_signal,
    )


def verify_defect(
    defect_type: str,
    clean_entities: list[Entity],
    clean_edges: list[Edge],
    *,
    count: int = 3,
    seed: int = 0,
    source_bodies: Optional[dict[str, str]] = None,
) -> DetectionResult:
    """Inject ``count`` defects of ``defect_type`` and score detection.

    Dispatches by defect family:
      - id-recall (duplicate_entity / orphan_node / broken_ref): recall
        over flagged entity/target ids, delta-precision vs the clean
        baseline.
      - phantom_edge: recall over flagged ``(src, dst, type)`` edge keys.
        Requires ``source_bodies`` (the prose the edges ground against);
        if omitted, only the injected synthetic note is used so the
        injected edges are still checkable and recall is still meaningful.
      - edge_type_monoculture: scalar-signal — the Cat 4c monoculture
        reading must move in the expected direction.
    """
    if defect_type not in DEFECT_TYPES:
        raise ValueError(
            f"unknown defect_type {defect_type!r}; expected one of {DEFECT_TYPES}"
        )
    if defect_type in ID_RECALL_DEFECTS:
        return _verify_id_recall(
            defect_type, clean_entities, clean_edges, count=count, seed=seed
        )
    if defect_type == "phantom_edge":
        return _verify_phantom_edge(
            clean_entities,
            clean_edges,
            count=count,
            seed=seed,
            source_bodies=source_bodies,
        )
    if defect_type == "edge_type_monoculture":
        return _verify_monoculture(
            clean_entities, clean_edges, count=count, seed=seed
        )
    raise ValueError(f"unhandled defect_type {defect_type!r}")  # pragma: no cover


def run_all_defects(
    clean_entities: list[Entity],
    clean_edges: list[Edge],
    *,
    count: int = 3,
    seed: int = 0,
    defect_types: Optional[tuple[str, ...]] = None,
    source_bodies: Optional[dict[str, str]] = None,
) -> dict[str, DetectionResult]:
    """Run the inject → detect → assert loop for every defect type.

    Returns a ``{defect_type: DetectionResult}`` map. A clean bill of
    health is ``all(r.detected_all for r in results.values())`` — every
    injected defect was caught by its category.

    ``source_bodies`` is forwarded to the phantom_edge path (the prose the
    edges ground against). For the good-dog corpus, get it from
    :func:`sme.corpora.good_dog_graph.load_source_bodies`.
    """
    types = defect_types or DEFECT_TYPES
    return {
        dt: verify_defect(
            dt,
            clean_entities,
            clean_edges,
            count=count,
            seed=seed,
            source_bodies=source_bodies,
        )
        for dt in types
    }
