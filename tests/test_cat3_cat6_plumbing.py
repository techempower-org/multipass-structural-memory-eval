"""Tests for Cat 3 / Cat 6 structured-field plumbing (issue #200).

Pins the contract that:

  * ``annotate_superseded_edges`` derives the reserved ``_superseded_by``
    property from ``supersedes``-predicate edges (Cat 6).
  * ``contradiction_pairs_from_edges`` extracts ContradictionPair[] from
    ``contradicts``-predicate edges (Cat 3).
  * Both palace paths surface these: the daemon ``project_graph`` and the
    direct ``MemPalaceAdapter`` snapshot.
  * The good-dog vault → graph loader carries the seeded edges.
  * The Cat 3 / Cat 6 scorers compute the (structural − flat) delta.

These are the exact fields the matrix marked "uncomputable" before #200.
"""

from __future__ import annotations

from sme.adapters.base import (
    ContradictionPair,
    Edge,
    Entity,
    annotate_superseded_edges,
    contradiction_pairs_from_edges,
    is_contradicts_edge,
    is_supersedes_edge,
    normalize_predicate,
)


# ── Predicate normalization ────────────────────────────────────────


def test_normalize_predicate_variants():
    assert normalize_predicate("CONTRADICTS") == "contradicts"
    assert normalize_predicate("conflicts-with") == "conflicts_with"
    assert normalize_predicate("  Supersedes  ") == "supersedes"
    assert normalize_predicate(None) == ""


def test_is_supersedes_and_contradicts():
    assert is_supersedes_edge("supersedes")
    assert is_supersedes_edge("replaces")
    assert is_supersedes_edge("REPLACED_BY")
    assert not is_supersedes_edge("mentions")
    assert is_contradicts_edge("contradicts")
    assert is_contradicts_edge("conflicts_with")
    assert not is_contradicts_edge("cites")


# ── annotate_superseded_edges (Cat 6) ──────────────────────────────


def test_annotate_stamps_superseded_target_on_supersedes_edge():
    edges = [Edge(source_id="new", target_id="old", edge_type="supersedes")]
    annotate_superseded_edges(edges)
    # The supersedes edge records its direction so a scorer holding only
    # the typed edge can resolve the linkage.
    assert edges[0].properties["_supersedes_target"] == "old"
    assert edges[0].properties["_superseded_target_by"] == "new"


def test_annotate_stamps_superseded_by_on_outgoing_claims():
    """An edge originating from a superseded entity gets _superseded_by."""
    edges = [
        Edge(source_id="new", target_id="old", edge_type="supersedes"),
        Edge(source_id="old", target_id="claim", edge_type="mentions"),
    ]
    annotate_superseded_edges(edges)
    claim_edge = next(e for e in edges if e.edge_type == "mentions")
    assert claim_edge.properties["_superseded_by"] == "new"


def test_annotate_is_idempotent():
    edges = [
        Edge(source_id="new", target_id="old", edge_type="supersedes"),
        Edge(source_id="old", target_id="claim", edge_type="mentions"),
    ]
    first = annotate_superseded_edges(edges)
    second = annotate_superseded_edges(edges)
    assert first >= 1
    assert second == 0  # nothing new to stamp on the second pass


def test_annotate_ignores_non_supersedes():
    edges = [Edge(source_id="a", target_id="b", edge_type="mentions")]
    annotate_superseded_edges(edges)
    assert "_superseded_by" not in edges[0].properties


# ── contradiction_pairs_from_edges (Cat 3) ─────────────────────────


def test_contradiction_pairs_extracted():
    edges = [
        Edge(source_id="a", target_id="b", edge_type="contradicts",
             properties={"evidence": "a says X, b says not-X"}),
        Edge(source_id="c", target_id="d", edge_type="mentions"),
    ]
    pairs = contradiction_pairs_from_edges(edges)
    assert len(pairs) == 1
    assert pairs[0].source_a == "a"
    assert pairs[0].source_b == "b"
    assert pairs[0].claim_a == "a says X, b says not-X"


def test_contradiction_pairs_dedup_unordered():
    """A corpus declaring the same contradiction both ways yields one pair."""
    edges = [
        Edge(source_id="a", target_id="b", edge_type="contradicts"),
        Edge(source_id="b", target_id="a", edge_type="contradicts"),
    ]
    pairs = contradiction_pairs_from_edges(edges)
    assert len(pairs) == 1


def test_contradiction_pairs_use_node_names():
    edges = [Edge(source_id="a", target_id="b", edge_type="contradicts")]
    pairs = contradiction_pairs_from_edges(
        edges, node_names={"a": "Framing A", "b": "Framing B"}
    )
    assert pairs[0].claim_a == "Framing A"
    assert pairs[0].claim_b == "Framing B"


def test_no_contradicts_edges_returns_empty():
    edges = [Edge(source_id="a", target_id="b", edge_type="mentions")]
    assert contradiction_pairs_from_edges(edges) == []


# ── Daemon /graph projection surfaces both fields ──────────────────


def test_project_graph_surfaces_superseded_and_contradicts():
    from sme.adapters._graph_mapping import project_graph

    body = {
        "kg_entities": [
            {"id": "new", "name": "New", "type": "publication"},
            {"id": "old", "name": "Old", "type": "publication"},
            {"id": "x", "name": "X framing", "type": "publication"},
            {"id": "y", "name": "Y framing", "type": "publication"},
        ],
        "kg_triples": [
            {"subject": "new", "predicate": "supersedes", "object": "old"},
            {"subject": "x", "predicate": "contradicts", "object": "y"},
        ],
    }
    entities, edges = project_graph(body)
    sup = next(e for e in edges if e.edge_type == "supersedes")
    assert sup.properties["_superseded_target_by"] == "kg:new"
    pairs = contradiction_pairs_from_edges(
        edges, node_names={e.id: e.name for e in entities}
    )
    assert len(pairs) == 1
    assert pairs[0].source_a == "kg:x"


# ── good-dog vault → graph loader ──────────────────────────────────


def test_good_dog_loader_carries_seeded_edges():
    from sme.corpora.good_dog_graph import load_graph

    entities, edges = load_graph()
    assert len(entities) > 50
    contradicts = [e for e in edges if e.edge_type == "contradicts"]
    supersedes = [e for e in edges if e.edge_type == "supersedes"]
    assert len(contradicts) >= 2  # at least the 2 canonical seeded pairs
    assert len(supersedes) >= 2
    # _superseded_by derivation fired on the supersession chains.
    assert any(e.properties.get("_superseded_by") for e in edges)


def test_good_dog_adapter_surfaces_pairs_and_supersession():
    from sme.adapters.good_dog_graph import GoodDogGraphAdapter

    adapter = GoodDogGraphAdapter()
    pairs = adapter.get_contradiction_pairs()
    assert len(pairs) >= 2
    assert all(isinstance(p, ContradictionPair) for p in pairs)

    entities, edges = adapter.get_graph_snapshot()
    sup = [e for e in edges if e.edge_type == "supersedes"]
    assert sup
    assert all(e.properties.get("_superseded_target_by") for e in sup)

    # query() populates the per-query contradictions channel.
    result = adapter.query("grain-free DCM FDA")
    assert isinstance(result.contradictions, list)
    adapter.close()


# ── Cat 3 / Cat 6 scorers ──────────────────────────────────────────


def test_score_cat3_full_detection_delta():
    from sme.categories.contradiction import score_cat3

    entities = [
        Entity(id="a", name="A", entity_type="publication"),
        Entity(id="b", name="B", entity_type="publication"),
    ]
    edges = [Edge(source_id="a", target_id="b", edge_type="contradicts")]
    surfaced = contradiction_pairs_from_edges(edges)
    report = score_cat3(entities, edges, surfaced)
    assert report.seeded_pairs == 1
    assert report.detection_rate == 1.0
    assert report.precision == 1.0
    assert report.detection_delta == 1.0  # flat is 0 by construction


def test_score_cat3_no_surfaced_is_zero():
    from sme.categories.contradiction import score_cat3

    edges = [Edge(source_id="a", target_id="b", edge_type="contradicts")]
    report = score_cat3([], edges, [])  # system surfaced nothing
    assert report.detection_rate == 0.0
    assert report.detection_delta == 0.0


def test_score_cat6_completeness_and_chains():
    from sme.categories.supersession import score_cat6

    edges = [
        Edge(source_id="mid", target_id="old", edge_type="supersedes"),
        Edge(source_id="new", target_id="mid", edge_type="supersedes"),
    ]
    annotate_superseded_edges(edges)
    report = score_cat6([], edges)
    assert report.seeded_supersedes_edges == 2
    assert report.resolved_supersedes_edges == 2
    assert report.completeness == 1.0
    assert report.completeness_delta == 1.0
    # Chain reconstructed oldest-first, current state last.
    assert report.chains == [["old", "mid", "new"]]


def test_score_cat6_no_supersedes_is_not_exercised():
    from sme.categories.supersession import score_cat6

    edges = [Edge(source_id="a", target_id="b", edge_type="mentions")]
    report = score_cat6([], edges)
    assert report.seeded_supersedes_edges == 0
    assert report.completeness == 0.0


def test_mempalace_adapter_surfaces_kg_contradictions(tmp_path):
    """The direct ChromaDB MemPalaceAdapter must surface contradiction
    pairs and _superseded_by from its SQLite KG — no chromadb needed for
    this path since get_contradiction_pairs reads only the KG.

    This is the #200 ask for the direct adapter: a palace whose KG holds
    `contradicts` / `supersedes` triples gets correct Cat 3 / Cat 6
    structured fields without any backend change.
    """
    import sqlite3

    from sme.adapters.mempalace import MemPalaceAdapter

    kg_path = tmp_path / "kg.sqlite3"
    conn = sqlite3.connect(kg_path)
    conn.execute(
        "CREATE TABLE entities (id TEXT, name TEXT, type TEXT, "
        "properties TEXT, created_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE triples (subject TEXT, predicate TEXT, object TEXT, "
        "valid_from TEXT, valid_to TEXT, confidence REAL, "
        "source_closet TEXT, source_file TEXT)"
    )
    conn.executemany(
        "INSERT INTO entities VALUES (?,?,?,?,?)",
        [
            ("fda_2022", "FDA 2022 framing", "publication", "{}", None),
            ("fda_2018", "FDA 2018 framing", "publication", "{}", None),
            ("hills_nov", "Hills warning letter", "publication", "{}", None),
            ("hills_jan", "Hills announcement", "publication", "{}", None),
        ],
    )
    conn.executemany(
        "INSERT INTO triples VALUES (?,?,?,?,?,?,?,?)",
        [
            ("fda_2022", "contradicts", "fda_2018", None, None, None, None, None),
            ("hills_nov", "supersedes", "hills_jan", "2019-11", None, None, None, None),
        ],
    )
    conn.commit()
    conn.close()

    # Build the adapter without chromadb __init__; only the KG path runs.
    a = MemPalaceAdapter.__new__(MemPalaceAdapter)
    a.include_kg = True
    a.kg_path = str(kg_path)
    a._kg_conn = None

    pairs = a.get_contradiction_pairs()
    assert len(pairs) == 1
    assert pairs[0].source_a == "kg:fda_2022"
    assert pairs[0].source_b == "kg:fda_2018"

    # _read_kg edges carry the supersession linkage after annotation.
    _, edges = a._read_kg()
    annotate_superseded_edges(edges)
    sup = next(e for e in edges if e.edge_type == "supersedes")
    assert sup.properties["_superseded_target_by"] == "kg:hills_nov"
    a.close()


def test_good_dog_cat3_cat6_end_to_end():
    """The headline #200 numbers: structural detection on the real corpus."""
    from sme.adapters.good_dog_graph import GoodDogGraphAdapter
    from sme.categories.contradiction import score_cat3
    from sme.categories.supersession import score_cat6

    adapter = GoodDogGraphAdapter()
    entities, edges = adapter.get_graph_snapshot()
    r3 = score_cat3(entities, edges, adapter.get_contradiction_pairs())
    r6 = score_cat6(entities, edges)
    # Every seeded contradiction surfaces; every supersedes edge resolves.
    assert r3.detection_rate == 1.0
    assert r6.completeness == 1.0
    # The delta is the whole reading — flat surfaces neither.
    assert r3.detection_delta == 1.0
    assert r6.completeness_delta == 1.0
    adapter.close()
