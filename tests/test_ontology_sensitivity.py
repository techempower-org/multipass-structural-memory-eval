"""Tests for the Cat 4 / Cat 5 ontology-sensitivity sweep (upstream #45).

The sweep's load-bearing invariant is **topology preservation**: remapping
entity_type / edge_type must NOT change the node set, the edge set, or any
node/edge identity — only the type labels. If that invariant holds, then
Cat 5's topology-driven metrics (components, isolates, Betti) are
necessarily stable across ontology conditions, and any Cat 4 movement is
attributable to ontology granularity alone. These tests pin both halves.
"""

from __future__ import annotations

from sme.adapters.base import Edge, Entity
from sme.corpora.good_dog_graph import load_graph
from sme.corpora.good_dog_ontologies import (
    GOOD_DOG_CONDITIONS,
    GOOD_DOG_FINE_GRAINED,
    GOOD_DOG_FLAT,
    GOOD_DOG_MODERATE,
)
from sme.eval.ontology_sensitivity import (
    OntologyCondition,
    remap_graph,
    run_sensitivity_sweep,
)


def _toy_graph() -> tuple[list[Entity], list[Edge]]:
    ents = [
        Entity(id="a", name="Alpha", entity_type="person"),
        Entity(id="b", name="Beta", entity_type="organization"),
        Entity(id="c", name="Gamma", entity_type="publication"),
    ]
    edges = [
        Edge(source_id="a", target_id="b", edge_type="affiliated_with"),
        Edge(source_id="c", target_id="a", edge_type="authored_by"),
    ]
    return ents, edges


# --- remap_graph: topology preservation -------------------------------


def test_remap_preserves_node_and_edge_sets():
    ents, edges = _toy_graph()
    flat = OntologyCondition(
        name="flat",
        description="one type",
        entity_mapper=lambda e: "node",
        edge_mapper=lambda x: "related",
    )
    new_ents, new_edges = remap_graph(ents, edges, flat)

    # Same nodes by id + name, same count.
    assert [e.id for e in new_ents] == [e.id for e in ents]
    assert [e.name for e in new_ents] == [e.name for e in ents]
    # Same edges by endpoints, same count.
    assert [(x.source_id, x.target_id) for x in new_edges] == [
        (x.source_id, x.target_id) for x in edges
    ]


def test_remap_only_changes_types():
    ents, edges = _toy_graph()
    flat = OntologyCondition(
        name="flat",
        description="one type",
        entity_mapper=lambda e: "node",
        edge_mapper=lambda x: "related",
    )
    new_ents, new_edges = remap_graph(ents, edges, flat)

    assert {e.entity_type for e in new_ents} == {"node"}
    assert {x.edge_type for x in new_edges} == {"related"}


def test_remap_does_not_mutate_input():
    ents, edges = _toy_graph()
    flat = OntologyCondition(
        name="flat",
        description="one type",
        entity_mapper=lambda e: "node",
        edge_mapper=lambda x: "related",
    )
    remap_graph(ents, edges, flat)
    # Originals untouched.
    assert ents[0].entity_type == "person"
    assert edges[0].edge_type == "affiliated_with"


def test_identity_condition_is_a_noop():
    """The default (moderate) mappers must leave types unchanged."""
    ents, edges = _toy_graph()
    identity = OntologyCondition(name="moderate", description="as-authored")
    new_ents, new_edges = remap_graph(ents, edges, identity)
    assert [e.entity_type for e in new_ents] == [e.entity_type for e in ents]
    assert [x.edge_type for x in new_edges] == [x.edge_type for x in edges]


# --- good-dog conditions: the three granularities ----------------------


def test_good_dog_conditions_have_distinct_granularities():
    ents, edges = load_graph()

    flat_e, flat_x = remap_graph(ents, edges, GOOD_DOG_FLAT)
    mod_e, mod_x = remap_graph(ents, edges, GOOD_DOG_MODERATE)
    fine_e, fine_x = remap_graph(ents, edges, GOOD_DOG_FINE_GRAINED)

    n_flat = len({e.entity_type for e in flat_e})
    n_mod = len({e.entity_type for e in mod_e})
    n_fine = len({e.entity_type for e in fine_e})

    # flat collapses to one; fine-grained strictly out-splits moderate.
    assert n_flat == 1
    assert n_mod >= 5  # the issue's "moderate = 5-8 types" band
    assert n_fine >= 15  # the issue's "fine-grained = 15+ types" band
    assert n_flat < n_mod < n_fine

    # edge types likewise: flat=1, fine >= moderate
    assert len({x.edge_type for x in flat_x}) == 1
    assert len({x.edge_type for x in fine_x}) >= len({x.edge_type for x in mod_x})


def test_good_dog_fine_grained_splits_are_subtypes_of_moderate():
    """Fine-grained must only SPLIT moderate types, never merge across them.

    A node's fine type and its moderate type must be consistent — i.e.
    every fine type maps back to exactly one moderate type. This guards
    against an accidental cross-type collapse that would corrupt the
    'same graph, finer labels' contract.
    """
    ents, _ = load_graph()
    fine_to_moderate: dict[str, set[str]] = {}
    for e in ents:
        fine = GOOD_DOG_FINE_GRAINED.entity_mapper(e)
        fine_to_moderate.setdefault(fine, set()).add(e.entity_type)
    for fine_type, moderate_types in fine_to_moderate.items():
        assert len(moderate_types) == 1, (
            f"fine type {fine_type!r} spans moderate types {moderate_types} "
            f"— fine-grained must split, not merge"
        )


# --- the sweep + its central finding -----------------------------------


def test_sweep_is_deterministic():
    ents, edges = load_graph()
    r1 = run_sensitivity_sweep(ents, edges, GOOD_DOG_CONDITIONS, corpus="good-dog")
    r2 = run_sensitivity_sweep(ents, edges, GOOD_DOG_CONDITIONS, corpus="good-dog")
    assert r1.to_dict() == r2.to_dict()


def test_cat5_topology_metrics_are_ontology_invariant():
    """THE FINDING (half 1): Cat 5 topology metrics do not move.

    Components / largest-component / isolates / Betti are functions of the
    graph topology, which the remap preserves — so they must read
    identically under flat, moderate, and fine-grained.
    """
    ents, edges = load_graph()
    result = run_sensitivity_sweep(ents, edges, GOOD_DOG_CONDITIONS, corpus="good-dog")
    topology_metrics = {
        "components",
        "largest_component_size",
        "isolated_nodes",
        "betti_0_largest",
        "betti_1_largest",
    }
    for m in result.movements:
        if m.metric in topology_metrics:
            assert m.spread == 0.0, (
                f"{m.metric} moved ({m.values}) — topology must be ontology-invariant"
            )
            assert m.stable


def test_cat4_type_metrics_move_with_ontology():
    """THE FINDING (half 2): Cat 4 monoculture/entropy track ontology.

    edge_type_entropy and dominant-edge-fraction are definitionally
    driven by the type count, so they MUST move between flat (1 type)
    and moderate/fine. This is the methodological caveat the sweep lands.
    """
    ents, edges = load_graph()
    result = run_sensitivity_sweep(ents, edges, GOOD_DOG_CONDITIONS, corpus="good-dog")
    by_metric = {m.metric: m for m in result.movements}

    # Flat has a single edge type → zero entropy, full concentration.
    entropy = by_metric["edge_type_entropy_normalized"]
    assert entropy.values["flat"] == 0.0
    assert entropy.values["moderate"] > 0.5
    assert not entropy.stable  # it moved

    dom = by_metric["dominant_edge_type_fraction"]
    assert dom.values["flat"] == 1.0  # one type = 100% dominant
    assert dom.values["moderate"] < 0.5
    assert not dom.stable


def test_overall_verdict_is_sensitive():
    """With Cat 4 moving and Cat 5 stable, the all-metrics verdict is
    'sensitive' — the honest summary is the nuanced split, not a single
    word, but the aggregate flag should not falsely claim robustness."""
    ents, edges = load_graph()
    result = run_sensitivity_sweep(ents, edges, GOOD_DOG_CONDITIONS, corpus="good-dog")
    assert result.verdict == "sensitive"
