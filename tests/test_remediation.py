"""Tests for diagnostic actionability — the Remediation field (#44).

Each category's scorer attaches a ``Remediation`` ("fix this and re-run")
to every non-healthy finding, and nothing on a clean reading. These tests
construct controlled graphs that trip a specific band, then assert the
remediation is present, carries the right band, and renders.

The shared shape lives in ``sme.categories._remediation``; the per-category
``_build_remediations`` helpers mirror the bands the Reading sections use.
"""

from __future__ import annotations

from sme.adapters.base import Edge, Entity
from sme.categories._remediation import Remediation, render_remediations
from sme.categories.gap_detection import format_report as format_gap_report
from sme.categories.gap_detection import score_gap_detection
from sme.categories.ingestion_integrity import (
    format_report as format_integrity_report,
)
from sme.categories.ingestion_integrity import score_ingestion_integrity
from sme.categories.ontology_coherence import (
    ImpliedOntology,
    score_cat8,
)


# --- Shared Remediation shape ----------------------------------------


def test_remediation_to_dict_roundtrips():
    rem = Remediation(
        finding="248 collisions",
        fix="normalize the key",
        reverify="re-run ingest",
        band="warning",
    )
    d = rem.to_dict()
    assert d == {
        "finding": "248 collisions",
        "fix": "normalize the key",
        "reverify": "re-run ingest",
        "band": "warning",
    }


def test_render_remediations_empty_is_no_lines():
    """A clean reading appends nothing — callers extend unconditionally."""
    assert render_remediations([]) == []


def test_render_remediations_titles_and_numbers():
    rems = [
        Remediation("f1", "fix1", "rerun1", "warning"),
        Remediation("f2", "fix2", "rerun2", "concerning"),
    ]
    rendered = "\n".join(render_remediations(rems))
    assert "Remediation — fix this and re-run" in rendered
    assert "1. [warning] f1" in rendered
    assert "2. [concerning] f2" in rendered
    assert "Fix:" in rendered
    assert "Re-run:" in rendered


# --- Cat 4 (ingestion integrity) -------------------------------------


def _monoculture_graph():
    """A graph where one edge type holds the overwhelming majority —
    trips the edge-type-entropy band (normalized entropy well below the
    0.80 healthy floor)."""
    entities = [Entity(id=f"e{i}", name=f"E{i}", entity_type="thing") for i in range(9)]
    # 'related' on 7 edges, one 'other_rel' — ~0.55 normalized entropy.
    edges = [Edge(source_id="e0", target_id=f"e{i}", edge_type="related") for i in range(1, 8)]
    edges.append(Edge(source_id="e1", target_id="e8", edge_type="other_rel"))
    return entities, edges


def test_cat4_monoculture_emits_remediation():
    entities, edges = _monoculture_graph()
    report = score_ingestion_integrity(entities, edges)
    # Dominant 'related' is 75% — not healthy.
    monoculture = [r for r in report.remediations if "monoculture" in r.finding.lower()]
    assert len(monoculture) == 1
    rem = monoculture[0]
    assert rem.band in ("warning", "concerning")
    assert "re-run" in rem.reverify.lower()


def test_cat4_collisions_emit_remediation():
    """Two distinct IDs canonicalizing to the same key → collision band."""
    entities = [
        Entity(id="e1", name="Docker", entity_type="tool"),
        Entity(id="e2", name="docker", entity_type="tool"),
    ]
    edges = [Edge(source_id="e1", target_id="e2", edge_type="related")]
    report = score_ingestion_integrity(entities, edges)
    coll = [r for r in report.remediations if "collision" in r.finding.lower()]
    assert len(coll) == 1
    assert "highest-degree" in coll[0].fix.lower() or "merge" in coll[0].fix.lower()


def test_cat4_field_gap_emits_remediation():
    """An entity with no name is a required-field gap → coverage band."""
    entities = [
        Entity(id="e1", name="Docker", entity_type="tool"),
        Entity(id="e2", name="", entity_type=""),
    ]
    edges = [Edge(source_id="e1", target_id="e2", edge_type="related")]
    report = score_ingestion_integrity(entities, edges)
    gaps = [r for r in report.remediations if "coverage" in r.finding.lower()]
    assert len(gaps) == 1


def test_cat4_clean_graph_no_remediation():
    """Balanced vocab, no collisions, full coverage → empty list."""
    entities = [Entity(id=f"e{i}", name=f"Distinct{i}", entity_type="thing") for i in range(6)]
    edges = [
        Edge(source_id="e0", target_id="e1", edge_type="a"),
        Edge(source_id="e1", target_id="e2", edge_type="b"),
        Edge(source_id="e2", target_id="e3", edge_type="c"),
        Edge(source_id="e3", target_id="e4", edge_type="d"),
        Edge(source_id="e4", target_id="e5", edge_type="e"),
    ]
    report = score_ingestion_integrity(entities, edges)
    assert report.remediations == []


def test_cat4_remediation_renders_in_format_report():
    entities, edges = _monoculture_graph()
    report = score_ingestion_integrity(entities, edges)
    rendered = format_integrity_report(report)
    assert "Remediation — fix this and re-run" in rendered


def test_cat4_empty_graph_no_remediation():
    report = score_ingestion_integrity([], [])
    assert report.remediations == []


# --- Cat 5 (gap detection) -------------------------------------------


def test_cat5_isolates_and_fragmentation_emit_remediation():
    """A graph that's mostly disconnected with many orphans → multiple
    non-healthy bands (connectivity + isolates)."""
    # One tiny connected pair + four isolated nodes.
    entities = [
        Entity(id="a", name="A", entity_type="t"),
        Entity(id="b", name="B", entity_type="t"),
        Entity(id="c", name="C", entity_type="t"),
        Entity(id="d", name="D", entity_type="t"),
        Entity(id="e", name="E", entity_type="t"),
        Entity(id="f", name="F", entity_type="t"),
    ]
    edges = [Edge(source_id="a", target_id="b", edge_type="x")]
    report = score_gap_detection(entities, edges, run_homology=False)
    findings = " ".join(r.finding.lower() for r in report.remediations)
    assert "isolates" in findings
    assert "connectivity" in findings
    # Isolate remediation should name the most-affected type.
    iso = [r for r in report.remediations if "isolates" in r.finding.lower()][0]
    assert iso.band in ("warning", "concerning")


def test_cat5_clean_connected_graph_no_remediation():
    """A dense, fully-connected, redundant graph → all healthy."""
    entities = [Entity(id=f"n{i}", name=f"N{i}", entity_type="t") for i in range(6)]
    # Ring + chords → connected, low bridge ratio, no isolates.
    edges = []
    for i in range(6):
        edges.append(Edge(source_id=f"n{i}", target_id=f"n{(i + 1) % 6}", edge_type="r"))
    edges.append(Edge(source_id="n0", target_id="n3", edge_type="r"))
    edges.append(Edge(source_id="n1", target_id="n4", edge_type="r"))
    report = score_gap_detection(entities, edges, run_homology=False)
    assert report.remediations == []


def test_cat5_remediation_renders_in_format_report():
    entities = [
        Entity(id="a", name="A", entity_type="t"),
        Entity(id="b", name="B", entity_type="t"),
        Entity(id="c", name="C", entity_type="t"),
    ]
    edges = [Edge(source_id="a", target_id="b", edge_type="x")]
    report = score_gap_detection(entities, edges, run_homology=False)
    rendered = format_gap_report(report)
    assert "Remediation — fix this and re-run" in rendered


def test_cat5_empty_graph_no_remediation():
    report = score_gap_detection([], [], run_homology=False)
    assert report.remediations == []


# --- Cat 8 (ontology coherence) --------------------------------------


def test_cat8_missing_types_and_edges_emit_remediation():
    """Declared vocabulary the graph never produces → drift + coverage
    remediations, including the 'dead extraction rule' edge-vocab item."""
    implied = ImpliedOntology(
        version="t",
        source="declared",
        entity_types=["person", "place", "thing"],
        edge_types=["knows", "located_in", "owns"],
    )
    # Graph only ever produces 'person' entities and 'knows' edges.
    entities = [
        Entity(id="p1", name="Alice", entity_type="person"),
        Entity(id="p2", name="Bob", entity_type="person"),
    ]
    edges = [Edge(source_id="p1", target_id="p2", edge_type="knows")]
    report = score_cat8(implied, entities, edges, structural_health={})

    findings = " ".join(r.finding.lower() for r in report.remediations)
    assert "type coverage" in findings
    assert "edge-vocabulary coverage" in findings
    assert "drift" in findings

    edge_rem = [r for r in report.remediations if "edge-vocabulary" in r.finding.lower()][0]
    assert (
        "dead extraction rule" in edge_rem.fix.lower() or "ingestion rule" in edge_rem.fix.lower()
    )


def test_cat8_coherent_ontology_no_remediation():
    """Declared == effective AND no single-type over-concentration → no
    remediation. Two declared types, both present and roughly balanced,
    both declared edges present."""
    implied = ImpliedOntology(
        version="t",
        source="declared",
        entity_types=["person", "place"],
        edge_types=["knows", "located_in"],
    )
    entities = [
        Entity(id="p1", name="Alice", entity_type="person"),
        Entity(id="p2", name="Bob", entity_type="person"),
        Entity(id="l1", name="Paris", entity_type="place"),
        Entity(id="l2", name="London", entity_type="place"),
    ]
    edges = [
        Edge(source_id="p1", target_id="p2", edge_type="knows"),
        Edge(source_id="p1", target_id="l1", edge_type="located_in"),
    ]
    report = score_cat8(implied, entities, edges, structural_health={})
    assert report.remediations == []


def test_cat8_remediations_in_to_dict():
    implied = ImpliedOntology(
        version="t",
        source="declared",
        entity_types=["person", "ghost"],
        edge_types=["knows"],
    )
    entities = [Entity(id="p1", name="Alice", entity_type="person")]
    edges = [Edge(source_id="p1", target_id="p1", edge_type="knows")]
    report = score_cat8(implied, entities, edges, structural_health={})
    d = report.to_dict()
    assert "remediations" in d
    assert isinstance(d["remediations"], list)
    assert any("ghost" in r["finding"] for r in d["remediations"])
