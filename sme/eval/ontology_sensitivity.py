"""Cat 4 / Cat 5 ontology-sensitivity sweep (upstream #45).

Cat 4 (ingestion integrity) and Cat 5 (structural gap detection) both
read on top of an ontology that is *exogenous* to SME — the entity types
and edge types come from the adapter's underlying graph, not from SME.
That makes ontology design an **unmeasured confound** for any cross-system
or cross-corpus comparison: "system X scores 0.4 on Cat 4 vs system Y at
0.8" is partly a statement about ontology granularity, not adapter quality.

This module quantifies the confound. It takes one corpus graph
``(entities, edges)`` and re-reads Cat 4 / Cat 5 under N deliberately
different ontology granularities — typically *flat* (one type),
*moderate* (the corpus as-authored), and *fine-grained* (split types) —
then reports how much the headline structural numbers move across them.

Two possible outcomes, both publishable (the issue's framing):

- **Readings move a lot** → methodological caveat: Cat 4 / Cat 5 must be
  reported alongside the ontology choice; cross-system comparison is only
  valid when ontologies are matched.
- **Readings are stable** → robustness claim: Cat 4 / Cat 5 are robust to
  ontology design within the observed range; comparison is valid.

The sweep is deliberately *adapter-free*: it operates on an already-loaded
graph snapshot and remaps the ``entity_type`` / ``edge_type`` fields with
pure functions. Same corpus, same adapter, same questions — only the
ontology projection changes. That isolation is the whole point.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

from sme.adapters.base import Edge, Entity
from sme.categories.gap_detection import score_gap_detection
from sme.categories.ingestion_integrity import score_ingestion_integrity

# A remapper takes the original type string (and the full entity/edge, in
# case the projection depends on name or properties) and returns the type
# under the target ontology. Returning the input unchanged is the identity
# (moderate) projection.
EntityTypeMapper = Callable[[Entity], str]
EdgeTypeMapper = Callable[[Edge], str]


@dataclass(frozen=True)
class OntologyCondition:
    """One ontology projection to read the graph under.

    ``entity_mapper`` / ``edge_mapper`` rewrite the type fields. The
    default mappers are identity — i.e. the corpus as-authored, which is
    the ``moderate`` baseline.
    """

    name: str
    description: str
    entity_mapper: EntityTypeMapper = lambda e: e.entity_type
    edge_mapper: EdgeTypeMapper = lambda e: e.edge_type


def remap_graph(
    entities: list[Entity],
    edges: list[Edge],
    condition: OntologyCondition,
) -> tuple[list[Entity], list[Edge]]:
    """Return a copy of the graph with types projected under ``condition``.

    Identity, name, and properties of every node/edge are preserved — only
    ``entity_type`` / ``edge_type`` change. The graph *topology* (which
    nodes, which edges) is untouched: a flat projection has the exact same
    connectivity as the moderate one, so any Cat 5 component/Betti movement
    is attributable to the type-driven signals (isolate-by-type), not to a
    different graph.
    """
    new_entities = [
        Entity(
            id=e.id,
            name=e.name,
            entity_type=condition.entity_mapper(e),
            properties=dict(e.properties),
            embedding=e.embedding,
        )
        for e in entities
    ]
    new_edges = [
        Edge(
            source_id=x.source_id,
            target_id=x.target_id,
            edge_type=condition.edge_mapper(x),
            properties=dict(x.properties),
        )
        for x in edges
    ]
    return new_entities, new_edges


@dataclass
class ConditionReading:
    """Cat 4 + Cat 5 headline metrics under one ontology condition."""

    condition: str
    description: str
    n_entity_types: int
    n_edge_types: int

    # Cat 4 (ingestion integrity) headlines
    canonical_collisions: int
    edge_type_entropy_normalized: float
    dominant_edge_type_fraction: float

    # Cat 5 (gap detection) headlines
    components: int
    largest_component_size: int
    isolated_nodes: int
    betti_0_largest: int
    betti_1_largest: int

    def as_metric_dict(self) -> dict[str, float]:
        """The numeric headlines, keyed, for movement computation."""
        return {
            "canonical_collisions": float(self.canonical_collisions),
            "edge_type_entropy_normalized": self.edge_type_entropy_normalized,
            "dominant_edge_type_fraction": self.dominant_edge_type_fraction,
            "components": float(self.components),
            "largest_component_size": float(self.largest_component_size),
            "isolated_nodes": float(self.isolated_nodes),
            "betti_0_largest": float(self.betti_0_largest),
            "betti_1_largest": float(self.betti_1_largest),
        }


@dataclass
class MetricMovement:
    """How far one headline metric moved across the ontology conditions."""

    metric: str
    values: dict[str, float]  # condition name -> value
    min_value: float
    max_value: float
    spread: float  # max - min (absolute movement)
    relative_spread: Optional[float]  # spread / |mean|, None if mean == 0

    @property
    def stable(self) -> bool:
        """A metric is 'stable' if it moves < 10% relative across conditions.

        Heuristic band only — the human reads the numbers. Metrics whose
        mean is ~0 (no relative reference) fall back to an absolute < 1.0
        movement check.
        """
        if self.relative_spread is None:
            return self.spread < 1.0
        return self.relative_spread < 0.10


@dataclass
class SensitivitySweepResult:
    corpus: str
    readings: list[ConditionReading] = field(default_factory=list)
    movements: list[MetricMovement] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """'robust' if every headline metric is stable, else 'sensitive'."""
        if not self.movements:
            return "undetermined"
        return "robust" if all(m.stable for m in self.movements) else "sensitive"

    def to_dict(self) -> dict:
        return {
            "corpus": self.corpus,
            "verdict": self.verdict,
            "conditions": [
                {
                    "name": r.condition,
                    "description": r.description,
                    "n_entity_types": r.n_entity_types,
                    "n_edge_types": r.n_edge_types,
                    "cat4": {
                        "canonical_collisions": r.canonical_collisions,
                        "edge_type_entropy_normalized": round(r.edge_type_entropy_normalized, 4),
                        "dominant_edge_type_fraction": round(r.dominant_edge_type_fraction, 4),
                    },
                    "cat5": {
                        "components": r.components,
                        "largest_component_size": r.largest_component_size,
                        "isolated_nodes": r.isolated_nodes,
                        "betti_0_largest": r.betti_0_largest,
                        "betti_1_largest": r.betti_1_largest,
                    },
                }
                for r in self.readings
            ],
            "movements": [
                {
                    "metric": m.metric,
                    "values": {k: round(v, 4) for k, v in m.values.items()},
                    "min": round(m.min_value, 4),
                    "max": round(m.max_value, 4),
                    "spread": round(m.spread, 4),
                    "relative_spread": (
                        round(m.relative_spread, 4) if m.relative_spread is not None else None
                    ),
                    "stable": m.stable,
                }
                for m in self.movements
            ],
        }


def _read_condition(
    entities: list[Entity],
    edges: list[Edge],
    condition: OntologyCondition,
) -> ConditionReading:
    remapped_ents, remapped_edges = remap_graph(entities, edges, condition)

    cat4 = score_ingestion_integrity(remapped_ents, remapped_edges)
    cat5 = score_gap_detection(remapped_ents, remapped_edges)

    return ConditionReading(
        condition=condition.name,
        description=condition.description,
        n_entity_types=len({e.entity_type for e in remapped_ents}),
        n_edge_types=len({x.edge_type for x in remapped_edges}),
        canonical_collisions=cat4.canonical_collisions,
        edge_type_entropy_normalized=cat4.edge_type_entropy_normalized,
        dominant_edge_type_fraction=cat4.dominant_edge_type_fraction,
        components=cat5.components,
        largest_component_size=cat5.largest_component_size,
        isolated_nodes=cat5.isolated_nodes,
        betti_0_largest=cat5.betti_0_largest,
        betti_1_largest=cat5.betti_1_largest,
    )


def _compute_movements(readings: list[ConditionReading]) -> list[MetricMovement]:
    if not readings:
        return []
    metric_keys = list(readings[0].as_metric_dict().keys())
    movements: list[MetricMovement] = []
    for key in metric_keys:
        values = {r.condition: r.as_metric_dict()[key] for r in readings}
        nums = list(values.values())
        lo, hi = min(nums), max(nums)
        spread = hi - lo
        mean = statistics.fmean(nums)
        relative = (spread / abs(mean)) if mean != 0 else None
        movements.append(
            MetricMovement(
                metric=key,
                values=values,
                min_value=lo,
                max_value=hi,
                spread=spread,
                relative_spread=relative,
            )
        )
    return movements


def run_sensitivity_sweep(
    entities: list[Entity],
    edges: list[Edge],
    conditions: list[OntologyCondition],
    *,
    corpus: str = "unknown",
) -> SensitivitySweepResult:
    """Read Cat 4 / Cat 5 under each ontology condition and report movement.

    The graph topology is identical across conditions; only the type
    projection changes. The returned ``movements`` quantify how far each
    headline metric travels, and ``verdict`` summarizes robust vs sensitive.
    """
    readings = [_read_condition(entities, edges, c) for c in conditions]
    movements = _compute_movements(readings)
    return SensitivitySweepResult(corpus=corpus, readings=readings, movements=movements)


def format_sweep(result: SensitivitySweepResult) -> str:
    """Human-readable table of the sweep."""
    lines: list[str] = []
    lines.append(f"Ontology-sensitivity sweep — corpus: {result.corpus}")
    lines.append(f"Verdict: {result.verdict.upper()}")
    lines.append("")

    # Conditions header
    cond_names = [r.condition for r in result.readings]
    lines.append("Conditions:")
    for r in result.readings:
        lines.append(
            f"  {r.condition:<14} {r.n_entity_types:>2} entity types, "
            f"{r.n_edge_types:>2} edge types — {r.description}"
        )
    lines.append("")

    # Per-metric movement table
    lines.append(
        f"{'metric':<32} " + " ".join(f"{c:>12}" for c in cond_names) + "   spread  rel    stable"
    )
    lines.append("-" * (32 + 13 * len(cond_names) + 26))
    for m in result.movements:
        row = f"{m.metric:<32} "
        row += " ".join(f"{m.values[c]:>12.3f}" for c in cond_names)
        rel = f"{m.relative_spread:.2%}" if m.relative_spread is not None else "  n/a"
        flag = "yes" if m.stable else "NO"
        row += f"   {m.spread:>6.3f}  {rel:>6}  {flag}"
        lines.append(row)
    return "\n".join(lines)
