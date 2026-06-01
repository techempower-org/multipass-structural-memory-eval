"""corpus-doctor verification harness.

Closes the calibration loop for issue #27: inject a KNOWN defect into a
clean snapshot, run the relevant category against the dirtied snapshot,
and grade the reading against the PROV-O ``defects`` manifest. The
headline number is the **recovery rate** — the fraction of injected
defects the category actually moved its reading by, relative to its
clean baseline.

This is what turns Cat 4 / Cat 5's "reads clean on clean corpora" from
an unfalsifiable claim into a measured sensitivity:

  * For count-delta defects (``duplicate_evidence`` → +1
    ``canonical_collisions`` each; ``orphan_inflation`` → +1
    ``isolated_nodes`` each) the harness compares the dirty reading minus
    the clean baseline against the manifest's summed expected delta, and
    recovery = ``min(observed_delta, expected_delta) / expected_delta``.
  * For direction-only defects (``monoculture_edge_type`` →
    ``dominant_edge_type_fraction`` should *increase*) recovery is a
    boolean: did the reading move in the expected direction by a
    non-trivial margin.

The harness imports the real Cat 4 / Cat 5 scorers — no reimplementation
of the detection logic. It is lightweight (numpy/networkx only, via the
scorers) and runs entirely in-process on the core dataclasses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from sme.adapters.base import Edge, Entity
from sme.categories.gap_detection import score_gap_detection
from sme.categories.ingestion_integrity import score_ingestion_integrity
from sme.corpus_doctor import Defect, DoctorResult, inject

# Direction-only defects need the reading to move by at least this much
# (absolute) to count as recovered — guards against floating-point noise.
_DIRECTION_EPSILON = 1e-6


@dataclass
class VerificationResult:
    """Grade for one pathology's injection + detection round-trip."""

    pathology: str
    severity: float
    seed: int
    n_defects: int
    field: str
    category: str
    clean_reading: float
    dirty_reading: float
    observed_delta: float
    expected_delta: Optional[float]  # None for direction-only defects
    recovery_rate: float  # 0..1
    detected: bool  # recovery_rate >= detection threshold
    notes: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        exp = (
            f"expected +{self.expected_delta:g}"
            if self.expected_delta is not None
            else "expected ↑ (direction)"
        )
        return (
            f"{self.pathology:<22} {self.field:<28} "
            f"clean={self.clean_reading:g} dirty={self.dirty_reading:g} "
            f"obs={self.observed_delta:+g} {exp} "
            f"recovery={self.recovery_rate:.0%} "
            f"{'DETECTED' if self.detected else 'MISSED'}"
        )


def _read_field(category: str, field_name: str, entities, edges) -> float:
    """Run the right scorer and pull the named field as a float."""
    if category == "cat4":
        report = score_ingestion_integrity(entities, edges)
        return float(getattr(report, field_name))
    if category == "cat5":
        # Homology is irrelevant to isolate counting and pulls in the
        # optional [topology] extra; skip it so the harness runs core-only.
        report = score_gap_detection(entities, edges, run_homology=False)
        return float(getattr(report, field_name))
    raise ValueError(f"unknown category {category!r} for field {field_name!r}")


def _expectation(defects: list[Defect]) -> tuple[str, str, Optional[float]]:
    """Collapse a homogeneous defect list to (category, field, expected_delta).

    All defects from a single pathology share one (category, field) and
    one ``expect`` mode. Count-delta pathologies sum their per-defect
    ``delta``; direction-only pathologies return ``None`` for the delta.
    """
    cats = {d.expect["category"] for d in defects}
    fields = {d.expect["field"] for d in defects}
    if len(cats) != 1 or len(fields) != 1:
        raise ValueError(
            "verify_pathology expects a homogeneous defect list "
            f"(got categories={cats}, fields={fields})"
        )
    category = next(iter(cats))
    field_name = next(iter(fields))
    if all("delta" in d.expect for d in defects):
        expected = float(sum(d.expect["delta"] for d in defects))
        return category, field_name, expected
    # direction-only
    return category, field_name, None


def verify_pathology(
    clean_entities: list[Entity],
    clean_edges: list[Edge],
    pathology: str,
    *,
    severity: float = 0.3,
    seed: int = 0,
    detection_threshold: float = 0.99,
    **inject_kwargs,
) -> VerificationResult:
    """Inject one pathology and grade whether the category recovers it.

    Args:
        clean_entities, clean_edges: the clean baseline snapshot.
        pathology: a name from ``sme.corpus_doctor.PATHOLOGIES``.
        severity, seed: forwarded to the injector (deterministic).
        detection_threshold: recovery rate at/above which the pathology
            is marked ``detected``. Defaults to 0.99 — count-delta
            defects should recover essentially fully; the harness is a
            calibration gate, not a soft score.
        inject_kwargs: extra per-pathology knobs (e.g. ``dominant_type``).

    Returns a ``VerificationResult`` with the clean/dirty readings, the
    observed vs expected delta, and the recovery rate.
    """
    result: DoctorResult = inject(
        clean_entities,
        clean_edges,
        pathology,
        severity=severity,
        seed=seed,
        **inject_kwargs,
    )
    defects = result.defects
    notes: list[str] = []

    if not defects:
        notes.append(
            "injector produced 0 defects (corpus too small or severity 0) — "
            "nothing to detect; recovery is vacuously 0."
        )
        category, field_name, expected = "n/a", "n/a", None
        # Best-effort: still surface the pathology's nominal field.
        return VerificationResult(
            pathology=pathology,
            severity=severity,
            seed=seed,
            n_defects=0,
            field=field_name,
            category=category,
            clean_reading=0.0,
            dirty_reading=0.0,
            observed_delta=0.0,
            expected_delta=expected,
            recovery_rate=0.0,
            detected=False,
            notes=notes,
        )

    category, field_name, expected_delta = _expectation(defects)

    clean_reading = _read_field(
        category, field_name, clean_entities, clean_edges
    )
    dirty_reading = _read_field(
        category, field_name, result.entities, result.edges
    )
    observed_delta = dirty_reading - clean_reading

    if expected_delta is not None:
        # Count-delta: recovery is the fraction of the expected rise that
        # the reading actually shows, clamped to [0, 1]. Over-shoot (the
        # detector counted MORE than we injected — e.g. an injected
        # duplicate happened to also collide with a pre-existing near-dup)
        # caps at 1.0 rather than rewarding it.
        if expected_delta == 0:
            recovery = 1.0 if abs(observed_delta) <= _DIRECTION_EPSILON else 0.0
        else:
            recovery = max(0.0, min(observed_delta, expected_delta) / expected_delta)
        if observed_delta > expected_delta + _DIRECTION_EPSILON:
            if pathology == "orphan_inflation":
                cause = (
                    "stripping a node's edges stranded a degree-1 neighbour "
                    "that only connected through it (collateral isolate)"
                )
            elif pathology == "duplicate_evidence":
                cause = (
                    "an injected clone collided with a pre-existing near-dup, "
                    "tipping a latent collision group over the threshold"
                )
            else:
                cause = "the corpus had a latent condition the injection tipped over"
            notes.append(
                f"observed delta {observed_delta:g} exceeds injected "
                f"{expected_delta:g} — {cause}. Recovery is capped at 100% "
                f"(the detector is correct about the extra, not penalised)."
            )
    else:
        # Direction-only: recovered iff it moved up by a real margin.
        recovery = 1.0 if observed_delta > _DIRECTION_EPSILON else 0.0
        if recovery < 1.0:
            notes.append(
                f"reading did not increase ({observed_delta:+g}); the "
                f"monoculture signal may be saturated on this corpus "
                f"(already low-entropy) or severity too low."
            )

    return VerificationResult(
        pathology=pathology,
        severity=severity,
        seed=seed,
        n_defects=len(defects),
        field=field_name,
        category=category,
        clean_reading=clean_reading,
        dirty_reading=dirty_reading,
        observed_delta=observed_delta,
        expected_delta=expected_delta,
        recovery_rate=recovery,
        detected=recovery >= detection_threshold,
        notes=notes,
    )


def verify_all(
    clean_entities: list[Entity],
    clean_edges: list[Edge],
    *,
    severity: float = 0.3,
    seed: int = 0,
    pathologies: Optional[list[str]] = None,
) -> list[VerificationResult]:
    """Run the verification round-trip for several pathologies.

    Defaults to the full implemented set. Each pathology is injected
    INDEPENDENTLY against the same clean baseline (not composed), so a
    miss in one cannot mask another.
    """
    from sme.corpus_doctor import PATHOLOGIES

    names = pathologies if pathologies is not None else sorted(PATHOLOGIES)
    return [
        verify_pathology(
            clean_entities, clean_edges, name, severity=severity, seed=seed
        )
        for name in names
    ]


def format_verification(results: list[VerificationResult]) -> str:
    """Render a human-readable calibration report."""
    if not results:
        return "corpus-doctor verification: no pathologies run."
    lines = [
        "corpus-doctor verification — inject → detect recovery",
        "=" * 70,
    ]
    detected = sum(1 for r in results if r.detected)
    by_cat: Counter[str] = Counter(r.category for r in results)
    for r in results:
        lines.append("  " + r.summary_line())
        for note in r.notes:
            lines.append(f"      ↳ {note}")
    lines.append("-" * 70)
    lines.append(
        f"  {detected}/{len(results)} pathologies recovered "
        f"(categories exercised: {', '.join(sorted(by_cat))})"
    )
    return "\n".join(lines)
