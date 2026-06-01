"""Tests for corpus-doctor (issue #27, first slice).

The core claim under test is the calibration loop: inject a KNOWN
defect → the relevant category RECOVERS it. Every detection assertion
runs the REAL Cat 4 / Cat 5 scorer (no reimplemented detection logic),
so a passing test proves the shipped detector is sensitive to the
defect, not that a stub agrees with itself.

The clean baseline is the hand-built ``synthetic_gap_graph`` /
``synthetic_duplicates_graph`` fixtures plus the in-tree good-dog
corpus graph — exercising both the small deterministic case (exact
counts) and a realistic corpus (recovery rate).
"""

from __future__ import annotations

import json

import pytest

from sme.adapters.base import Edge, Entity
from sme.categories.corpus_doctor_harness import (
    format_verification,
    verify_all,
    verify_pathology,
)
from sme.categories.gap_detection import score_gap_detection
from sme.categories.ingestion_integrity import score_ingestion_integrity
from sme.corpus_doctor import (
    PATHOLOGIES,
    PATHOLOGY_BACKLOG,
    Defect,
    inject,
    inject_duplicate_evidence,
    inject_many,
    inject_monoculture_edge_type,
    inject_orphan_inflation,
    load_manifest,
    write_manifest,
)
from sme.topology.fixtures import (
    synthetic_duplicates_graph,
    synthetic_gap_graph,
)


# --- a small, fully-connected clean baseline -------------------------
# Every entity touches at least one edge and every canonical key is
# unique → clean Cat 4 (0 collisions) and clean Cat 5 (0 isolates).


def _clean_connected_graph() -> tuple[list[Entity], list[Edge]]:
    entities = [
        Entity(id="n1", name="Alpha", entity_type="topic"),
        Entity(id="n2", name="Bravo", entity_type="topic"),
        Entity(id="n3", name="Charlie", entity_type="topic"),
        Entity(id="n4", name="Delta", entity_type="note"),
        Entity(id="n5", name="Echo", entity_type="note"),
        Entity(id="n6", name="Foxtrot", entity_type="person"),
    ]
    edges = [
        Edge("n1", "n2", "RELATED"),
        Edge("n2", "n3", "MENTIONS"),
        Edge("n3", "n4", "CITES"),
        Edge("n4", "n5", "AUTHORED_BY"),
        Edge("n5", "n6", "LOCATED_IN"),
        Edge("n6", "n1", "MEMBER_OF"),
    ]
    return entities, edges


# --- the clean baseline really is clean ------------------------------


def test_baseline_is_clean_cat4():
    entities, edges = _clean_connected_graph()
    report = score_ingestion_integrity(entities, edges)
    assert report.canonical_collisions == 0


def test_baseline_is_clean_cat5():
    entities, edges = _clean_connected_graph()
    report = score_gap_detection(entities, edges, run_homology=False)
    assert report.isolated_nodes == 0


# --- duplicate_evidence → Cat 4a -------------------------------------


def test_duplicate_evidence_raises_collisions_by_exact_count():
    entities, edges = _clean_connected_graph()
    clean = score_ingestion_integrity(entities, edges).canonical_collisions
    result = inject_duplicate_evidence(entities, edges, severity=0.5, seed=7)
    assert result.defects, "severity 0.5 must inject at least one dupe"
    dirty = score_ingestion_integrity(
        result.entities, result.edges
    ).canonical_collisions
    # Each clone is one EXTRA duplicate ID on an existing canonical key.
    assert dirty - clean == len(result.defects)


def test_duplicate_evidence_does_not_mutate_input():
    entities, edges = _clean_connected_graph()
    n_before = len(entities)
    inject_duplicate_evidence(entities, edges, severity=1.0, seed=1)
    assert len(entities) == n_before  # clean snapshot untouched


def test_duplicate_clone_shares_name_and_type():
    entities, edges = _clean_connected_graph()
    result = inject_duplicate_evidence(entities, edges, severity=0.2, seed=3)
    by_id = {e.id: e for e in result.entities}
    for d in result.defects:
        clone = by_id[d.target["ids"][0]]
        src = by_id[d.target["source_id"]]
        assert clone.name == src.name
        assert clone.entity_type == src.entity_type
        assert clone.properties.get("_dupe_of") == src.id


def test_duplicate_evidence_detected_on_fixture():
    # The hand-built duplicates fixture already carries 3 collisions;
    # injecting more must be recovered on top of that nonzero baseline.
    entities, edges, _ = synthetic_duplicates_graph()
    res = verify_pathology(entities, edges, "duplicate_evidence", severity=0.5, seed=2)
    assert res.detected
    assert res.recovery_rate == pytest.approx(1.0)
    assert res.observed_delta == pytest.approx(res.expected_delta)


# --- orphan_inflation → Cat 5 ----------------------------------------


def test_orphan_inflation_raises_isolates_by_exact_count():
    entities, edges = _clean_connected_graph()
    clean = score_gap_detection(
        entities, edges, run_homology=False
    ).isolated_nodes
    result = inject_orphan_inflation(entities, edges, severity=0.34, seed=5)
    assert result.defects
    dirty = score_gap_detection(
        result.entities, result.edges, run_homology=False
    ).isolated_nodes
    assert dirty - clean == len(result.defects)


def test_orphan_inflation_removes_all_incident_edges():
    entities, edges = _clean_connected_graph()
    result = inject_orphan_inflation(entities, edges, severity=0.34, seed=5)
    orphaned = {d.target["ids"][0] for d in result.defects}
    # No surviving edge may touch an orphaned node.
    for e in result.edges:
        assert e.source_id not in orphaned
        assert e.target_id not in orphaned


def test_orphan_inflation_manifest_records_removed_edges():
    entities, edges = _clean_connected_graph()
    result = inject_orphan_inflation(entities, edges, severity=0.5, seed=11)
    for d in result.defects:
        assert d.detail["removed_edges"], "an orphaned node had incident edges"


def test_orphan_inflation_detected_on_gap_fixture():
    entities, edges, _ = synthetic_gap_graph()
    res = verify_pathology(entities, edges, "orphan_inflation", severity=0.3, seed=4)
    assert res.detected
    assert res.recovery_rate == pytest.approx(1.0)


# --- monoculture_edge_type → Cat 4c ----------------------------------


def test_monoculture_raises_dominant_fraction():
    entities, edges = _clean_connected_graph()
    clean = score_ingestion_integrity(entities, edges).dominant_edge_type_fraction
    result = inject_monoculture_edge_type(
        entities, edges, severity=1.0, seed=0, dominant_type="RELATED"
    )
    dirty = score_ingestion_integrity(
        result.entities, result.edges
    ).dominant_edge_type_fraction
    assert dirty > clean


def test_monoculture_full_severity_collapses_to_one_type():
    entities, edges = _clean_connected_graph()
    result = inject_monoculture_edge_type(
        entities, edges, severity=1.0, seed=0, dominant_type="RELATED"
    )
    types = {e.edge_type for e in result.edges}
    assert types == {"RELATED"}


def test_monoculture_records_original_edge_type():
    entities, edges = _clean_connected_graph()
    result = inject_monoculture_edge_type(
        entities, edges, severity=1.0, seed=0, dominant_type="RELATED"
    )
    for d in result.defects:
        assert d.detail["original_edge_type"] != "RELATED"
        assert d.detail["rewritten_to"] == "RELATED"


def test_monoculture_default_target_is_existing_dominant():
    # On the good-dog corpus the dominant type is `mentions`; the default
    # collapse target must be that existing dominant, not a fixed string.
    from sme.corpora.good_dog_graph import load_graph

    entities, edges = load_graph()
    result = inject_monoculture_edge_type(entities, edges, severity=0.5, seed=0)
    assert result.defects
    for d in result.defects:
        assert d.detail["rewritten_to"] == "mentions"


def test_monoculture_detected_direction_only():
    entities, edges = _clean_connected_graph()
    res = verify_pathology(entities, edges, "monoculture_edge_type", severity=0.6, seed=1)
    assert res.expected_delta is None  # direction-only
    assert res.detected
    assert res.observed_delta > 0


# --- determinism -----------------------------------------------------


@pytest.mark.parametrize("pathology", sorted(PATHOLOGIES))
def test_injection_is_deterministic(pathology):
    entities, edges = _clean_connected_graph()
    a = inject(entities, edges, pathology, severity=0.5, seed=42)
    b = inject(entities, edges, pathology, severity=0.5, seed=42)
    assert [d.defect_id for d in a.defects] == [d.defect_id for d in b.defects]
    assert {e.id for e in a.entities} == {e.id for e in b.entities}


def test_different_seeds_select_differently():
    # With a 6-node graph and partial severity, two seeds should
    # generally pick different targets. Use a severity < 1 so there's
    # actually a choice to make.
    entities, edges = _clean_connected_graph()
    a = inject_orphan_inflation(entities, edges, severity=0.34, seed=1)
    b = inject_orphan_inflation(entities, edges, severity=0.34, seed=99)
    ids_a = {d.target["ids"][0] for d in a.defects}
    ids_b = {d.target["ids"][0] for d in b.defects}
    # Not strictly guaranteed, but for these seeds the selections differ.
    assert ids_a != ids_b


# --- severity scaling ------------------------------------------------


def test_severity_zero_injects_nothing():
    entities, edges = _clean_connected_graph()
    result = inject_duplicate_evidence(entities, edges, severity=0.0, seed=0)
    assert result.defects == []


def test_higher_severity_injects_more():
    entities, edges, _ = synthetic_gap_graph()
    low = inject_orphan_inflation(entities, edges, severity=0.1, seed=0)
    high = inject_orphan_inflation(entities, edges, severity=0.8, seed=0)
    assert len(high.defects) >= len(low.defects)


# --- dispatcher + composition ----------------------------------------


def test_inject_unknown_pathology_raises_with_backlog_hint():
    entities, edges = _clean_connected_graph()
    with pytest.raises(KeyError) as exc:
        inject(entities, edges, "stale_facts", severity=0.5)
    # The error names the deferred backlog so the gap is discoverable.
    assert "stale_facts" in str(exc.value)


def test_inject_many_accumulates_defects():
    entities, edges = _clean_connected_graph()
    names = ["duplicate_evidence", "orphan_inflation"]
    result = inject_many(entities, edges, names, severity=0.5, seed=0)
    assert set(result.pathologies) == set(names)
    assert result.defects


# --- manifest round-trip (PROV-O shape) ------------------------------


def test_manifest_has_prov_o_keys():
    entities, edges = _clean_connected_graph()
    result = inject_duplicate_evidence(entities, edges, severity=0.5, seed=0)
    rec = result.defects[0].to_manifest_record()
    assert rec["prov:activity"] == "inject_defect"
    assert rec["prov:wasAttributedTo"].startswith("corpus-doctor/")
    assert "expect" in rec and "target" in rec


def test_manifest_write_read_roundtrip(tmp_path):
    entities, edges = _clean_connected_graph()
    result = inject_many(
        entities,
        edges,
        ["duplicate_evidence", "orphan_inflation", "monoculture_edge_type"],
        severity=0.5,
        seed=3,
    )
    path = write_manifest(result.defects, tmp_path / "defects.jsonl")
    loaded = load_manifest(path)
    assert len(loaded) == len(result.defects)
    assert {d.defect_id for d in loaded} == {d.defect_id for d in result.defects}
    # Every line is valid JSON carrying the PROV activity key.
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        assert rec["prov:activity"] == "inject_defect"


def test_load_manifest_tolerates_blank_lines(tmp_path):
    p = tmp_path / "defects.jsonl"
    d = Defect(
        defect_id="x::1",
        pathology="duplicate_evidence",
        severity=0.3,
        seed=0,
        target={"kind": "entity", "ids": ["a"]},
        expect={"category": "cat4", "field": "canonical_collisions", "delta": 1},
    )
    p.write_text(json.dumps(d.to_manifest_record()) + "\n\n\n")
    loaded = load_manifest(p)
    assert len(loaded) == 1


# --- end-to-end on the real in-tree corpus ---------------------------


def test_verify_all_on_good_dog_corpus():
    from sme.corpora.good_dog_graph import load_graph

    entities, edges = load_graph()
    results = verify_all(entities, edges, severity=0.3, seed=0)
    # All three implemented pathologies must be recovered on a real corpus.
    assert len(results) == 3
    for r in results:
        assert r.detected, f"{r.pathology} not detected: {r.summary_line()}"
    # The formatter renders without error and names every pathology.
    rendered = format_verification(results)
    for name in PATHOLOGIES:
        assert name in rendered


def test_backlog_documents_deferred_pathologies():
    # The deferred pathologies named in issue #27 are discoverable in code.
    for name in ("zipfian_degree", "hotspot_entity", "stale_facts", "phantom_edge"):
        assert name in PATHOLOGY_BACKLOG
