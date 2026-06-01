"""Tests for corpus-doctor — synthetic defect injection (upstream #27).

Two layers:
  - injector tests: each defect type produces the right corruption, the
    manifest matches what was injected, inputs are never mutated, and a
    fixed seed is deterministic.
  - harness tests: the inject → detect → assert loop recalls every
    injected defect via its category, and delta-precision is not
    penalized by defects native to the clean corpus (the harness's whole
    point — proving the category catches the KNOWN injected defects).
"""
from __future__ import annotations

import pytest

from sme.adapters.base import Edge, Entity
from sme.categories.ingestion_integrity import default_canonical_key
from sme.corpus_doctor import (
    DEFECT_TYPES,
    ID_RECALL_DEFECTS,
    CorpusDoctor,
    run_all_defects,
    verify_defect,
)
from sme.corpus_doctor.detectors import (
    detect_broken_refs,
    detect_duplicate_entities,
    detect_orphan_nodes,
)


@pytest.fixture
def clean_graph():
    """A small clean graph with NO native defects: 4 entities, all
    connected in a cycle, distinct canonical names.

    Edge types are deliberately DIVERSE (not all one type) so the
    edge_type_monoculture defect has headroom to raise the dominant
    fraction — an already-monoculture clean graph is a degenerate case
    where the signal can't move. Still defect-free for the other
    categories: no canonical collisions, no isolates, no dangling refs."""
    ents = [
        Entity(id="a", name="Alpha", entity_type="Thing"),
        Entity(id="b", name="Beta", entity_type="Thing"),
        Entity(id="c", name="Gamma", entity_type="Thing"),
        Entity(id="d", name="Delta", entity_type="Thing"),
    ]
    edges = [
        Edge(source_id="a", target_id="b", edge_type="rel_ab"),
        Edge(source_id="b", target_id="c", edge_type="rel_bc"),
        Edge(source_id="c", target_id="d", edge_type="rel_cd"),
        Edge(source_id="d", target_id="a", edge_type="rel_da"),
    ]
    return ents, edges


# --- injector: duplicate_entity ---------------------------------------


def test_inject_duplicate_adds_canonical_collision(clean_graph):
    ents, edges = clean_graph
    doc = CorpusDoctor(seed=1)
    res = doc.inject_duplicate_entity(ents, edges, count=2)
    # two new duplicate entities added, edges untouched
    assert len(res.entities) == len(ents) + 2
    assert len(res.edges) == len(edges)
    assert len(res.defects) == 2
    for d in res.defects:
        assert d.defect_type == "duplicate_entity"
        dup = next(e for e in res.entities if e.id == d.duplicate_id)
        orig = next(e for e in res.entities if e.id == d.original_id)
        # surface name differs, canonical key is identical -> a collision
        assert dup.name != orig.name
        assert default_canonical_key(dup.name, dup.entity_type) == default_canonical_key(
            orig.name, orig.entity_type
        )


# --- injector: orphan_node --------------------------------------------


def test_inject_orphan_has_no_incident_edges(clean_graph):
    ents, edges = clean_graph
    doc = CorpusDoctor(seed=1)
    res = doc.inject_orphan_node(ents, edges, count=3)
    assert len(res.entities) == len(ents) + 3
    assert len(res.edges) == len(edges)  # no edges added
    incident = {e.source_id for e in res.edges} | {e.target_id for e in res.edges}
    for oid in res.expected_orphan_ids():
        assert oid not in incident


# --- injector: broken_ref ---------------------------------------------


def test_inject_broken_ref_targets_missing_entity(clean_graph):
    ents, edges = clean_graph
    doc = CorpusDoctor(seed=1)
    res = doc.inject_broken_ref(ents, edges, count=2)
    assert len(res.entities) == len(ents)  # no entities added
    assert len(res.edges) == len(edges) + 2
    entity_ids = {e.id for e in res.entities}
    for d in res.defects:
        # the dangling target is absent from the entity set...
        assert d.dangling_target_id not in entity_ids
        # ...but the source endpoint is a real entity (isolates the defect)
        assert d.edge_source_id in entity_ids


# --- injector: hygiene ------------------------------------------------


def test_injection_does_not_mutate_inputs(clean_graph):
    ents, edges = clean_graph
    n_ents, n_edges = len(ents), len(edges)
    doc = CorpusDoctor(seed=1)
    doc.inject_duplicate_entity(ents, edges, count=2)
    doc.inject_orphan_node(ents, edges, count=2)
    doc.inject_broken_ref(ents, edges, count=2)
    # caller's lists are untouched
    assert len(ents) == n_ents
    assert len(edges) == n_edges


def test_injection_is_deterministic_under_seed(clean_graph):
    ents, edges = clean_graph
    r1 = CorpusDoctor(seed=99).inject_duplicate_entity(ents, edges, count=2)
    r2 = CorpusDoctor(seed=99).inject_duplicate_entity(ents, edges, count=2)
    assert r1.expected_duplicate_ids() == r2.expected_duplicate_ids()


def test_inject_dispatch_rejects_unknown_type(clean_graph):
    ents, edges = clean_graph
    with pytest.raises(ValueError, match="unknown defect_type"):
        CorpusDoctor().inject("not_a_defect", ents, edges)


def test_defect_types_constant_matches_dispatch(clean_graph):
    ents, edges = clean_graph
    doc = CorpusDoctor(seed=3)
    # every declared defect type dispatches without error
    for dt in DEFECT_TYPES:
        res = doc.inject(dt, ents, edges, count=1)
        assert res.defect_type == dt
        assert len(res.defects) == 1


# --- detectors: direct ------------------------------------------------


def test_detector_duplicate_flags_extra_only(clean_graph):
    ents, edges = clean_graph
    res = CorpusDoctor(seed=5).inject_duplicate_entity(ents, edges, count=2)
    flagged = detect_duplicate_entities(res.entities, res.edges)
    # exactly the injected duplicate ids (one kept representative per group)
    assert flagged == res.expected_duplicate_ids()


def test_detector_orphan_flags_isolates(clean_graph):
    ents, edges = clean_graph
    res = CorpusDoctor(seed=5).inject_orphan_node(ents, edges, count=2)
    flagged = detect_orphan_nodes(res.entities, res.edges)
    assert res.expected_orphan_ids() <= flagged
    # clean cycle has no native isolates -> only the injected ones
    assert flagged == res.expected_orphan_ids()


def test_detector_broken_ref_flags_dangling(clean_graph):
    ents, edges = clean_graph
    res = CorpusDoctor(seed=5).inject_broken_ref(ents, edges, count=2)
    flagged = detect_broken_refs(res.entities, res.edges)
    assert flagged == res.expected_dangling_target_ids()


# --- harness: the inject → detect → assert loop -----------------------


@pytest.mark.parametrize("defect_type", DEFECT_TYPES)
def test_verify_recalls_every_injected_defect(clean_graph, defect_type):
    ents, edges = clean_graph
    result = verify_defect(defect_type, ents, edges, count=3, seed=11)
    assert result.injected == 3
    assert result.detected_all, f"{defect_type} missed {result.missed_ids}"
    assert result.recall == 1.0
    # on a clean corpus, every new flag is injection-attributable
    assert result.delta_precision == 1.0


def test_run_all_defects_clean_bill(clean_graph):
    ents, edges = clean_graph
    results = run_all_defects(ents, edges, count=4, seed=2)
    assert set(results) == set(DEFECT_TYPES)
    assert all(r.detected_all for r in results.values())


def test_delta_precision_ignores_native_defects():
    """The harness must not penalize a detector for native corpus
    defects — only injection-attributable flags count. Here the clean
    graph already has an isolate; injecting more must still score
    delta-precision 1.0 (the native isolate is in the baseline)."""
    ents = [
        Entity(id="a", name="Alpha", entity_type="Thing"),
        Entity(id="b", name="Beta", entity_type="Thing"),
        Entity(id="native_orphan", name="Lonely", entity_type="Thing"),  # native isolate
    ]
    edges = [Edge(source_id="a", target_id="b", edge_type="rel")]
    # sanity: the native orphan is present in the clean baseline
    assert "native_orphan" in detect_orphan_nodes(ents, edges)
    result = verify_defect("orphan_node", ents, edges, count=2, seed=4)
    assert result.detected_all
    # native_orphan is in the baseline, so it is NOT counted as a new flag
    assert result.delta_precision == 1.0
    assert result.new_flagged == 2  # only the 2 injected orphans are new


def test_verify_rejects_unknown_defect(clean_graph):
    ents, edges = clean_graph
    with pytest.raises(ValueError, match="unknown defect_type"):
        verify_defect("not_a_real_defect", ents, edges)


# --- integration: real good-dog corpus --------------------------------


def test_harness_on_good_dog_corpus():
    """Smoke the full loop on the real good-dog graph (has native
    defects) — proves the harness works beyond a hand-built fixture.

    Passes the real source bodies so the phantom_edge path grounds against
    actual prose; without them the phantom edges still inject+recall, but
    this exercises the real grounding map too."""
    from sme.corpora import good_dog_graph

    ents, edges = good_dog_graph.load_graph()
    bodies = good_dog_graph.load_source_bodies()
    assert len(ents) > 0 and len(edges) > 0
    results = run_all_defects(ents, edges, count=5, seed=7, source_bodies=bodies)
    assert set(results) == set(DEFECT_TYPES)  # all five
    for dt, r in results.items():
        assert r.detected_all, f"{dt} missed {r.missed_ids} on good-dog"
        if dt in ID_RECALL_DEFECTS or dt == "phantom_edge":
            assert r.delta_precision == 1.0


# --- injector: phantom_edge -------------------------------------------


def test_inject_phantom_edge_between_real_entities(clean_graph):
    ents, edges = clean_graph
    res = CorpusDoctor(seed=1).inject_phantom_edge(ents, edges, count=2)
    assert len(res.entities) == len(ents)  # no entities added
    assert len(res.edges) == len(edges) + 2
    entity_ids = {e.id for e in res.entities}
    for s, t, _typ in res.expected_phantom_edge_keys():
        # both endpoints are REAL — distinguishes from broken_ref
        assert s in entity_ids
        assert t in entity_ids
        assert s != t  # no self-loops


def test_phantom_edge_source_bodies_are_empty(clean_graph):
    ents, edges = clean_graph
    res = CorpusDoctor(seed=1).inject_phantom_edge(ents, edges, count=2)
    bodies = res.phantom_source_bodies()
    assert bodies  # at least one note
    assert all(body == "" for body in bodies.values())


def test_phantom_edge_needs_two_entities():
    ents = [Entity(id="solo", name="Solo", entity_type="Thing")]
    with pytest.raises(ValueError, match="at least 2 entities"):
        CorpusDoctor().inject_phantom_edge(ents, [], count=1)


def test_verify_phantom_edge_recalls_injected(clean_graph):
    ents, edges = clean_graph
    # clean fixture has no source bodies → real edges are skipped, the
    # injected phantom edges ground against their empty synthetic note.
    result = verify_defect("phantom_edge", ents, edges, count=3, seed=2)
    assert result.injected == 3
    assert result.detected_all, f"missed {result.missed_ids}"
    assert result.recall == 1.0
    assert result.delta_precision == 1.0


# --- injector: edge_type_monoculture ----------------------------------


def test_inject_monoculture_adds_edges_of_one_type(clean_graph):
    ents, edges = clean_graph
    res = CorpusDoctor(seed=1).inject_edge_type_monoculture(ents, edges, count=10)
    assert len(res.entities) == len(ents)  # no entities added
    assert len(res.edges) == len(edges) + 10
    mono_type = res.monoculture_type()
    assert mono_type is not None
    injected = [e for e in res.edges if e.properties.get("_injected_monoculture")]
    assert len(injected) == 10
    assert all(e.edge_type == mono_type for e in injected)


def test_monoculture_default_amplifies_existing_dominant():
    # An all-one-type graph: the default collapse target must be that
    # existing type (amplify the skew, don't invent a new type).
    ents = [
        Entity(id="a", name="Alpha", entity_type="Thing"),
        Entity(id="b", name="Beta", entity_type="Thing"),
    ]
    edges = [
        Edge(source_id="a", target_id="b", edge_type="rel"),
        Edge(source_id="b", target_id="a", edge_type="rel"),
    ]
    res = CorpusDoctor(seed=1).inject_edge_type_monoculture(ents, edges, count=5)
    assert res.monoculture_type() == "rel"


def test_verify_monoculture_signal_moves(clean_graph):
    # Give the fixture a diverse edge mix so the dominant fraction has
    # room to rise when monoculture edges pile onto the top type.
    ents = [
        Entity(id="a", name="Alpha", entity_type="Thing"),
        Entity(id="b", name="Beta", entity_type="Thing"),
        Entity(id="c", name="Gamma", entity_type="Thing"),
    ]
    edges = [
        Edge(source_id="a", target_id="b", edge_type="rel_x"),
        Edge(source_id="b", target_id="c", edge_type="rel_y"),
        Edge(source_id="c", target_id="a", edge_type="rel_z"),
    ]
    result = verify_defect("edge_type_monoculture", ents, edges, count=12, seed=3)
    assert result.detected_all
    # dominant fraction rose, normalized entropy fell
    assert (
        result.signal_dirty["dominant_edge_type_fraction"]
        > result.signal_clean["dominant_edge_type_fraction"]
    )
    assert (
        result.signal_dirty["edge_type_entropy_normalized"]
        < result.signal_clean["edge_type_entropy_normalized"]
    )


def test_all_five_defect_types_dispatch(clean_graph):
    ents, edges = clean_graph
    assert len(DEFECT_TYPES) == 5
    doc = CorpusDoctor(seed=3)
    for dt in DEFECT_TYPES:
        res = doc.inject(dt, ents, edges, count=1)
        assert res.defects and res.defects[0].defect_type == dt
