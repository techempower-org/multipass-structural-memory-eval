"""Correctness tests for the proposed Phantom-Edge category (upstream #4).

Two halves:

* **Synthetic** — a tiny hand-built graph with one deliberately phantom
  edge (endpoints named nowhere in the source body) and grounded edges
  alongside it. Asserts the phantom is caught and the grounded edges are
  NOT flagged. This is the seed-a-known-defect test the task asks for.

* **good-dog corpus** — the real fixture. Asserts the detector tracks the
  corpus's own ``needs_grounding`` flag (flagged edges phantom at a
  strictly higher rate than the rest), which is the validity proof: the
  detector is finding what the maintainer already knows is weakly
  grounded, not noise.
"""

from __future__ import annotations

from sme.adapters.base import Edge, Entity
from sme.categories.phantom_edge import (
    PhantomEdgeReport,
    format_report,
    score_phantom_edges,
)


# --- Synthetic seed-a-known-phantom fixture ---------------------------


def _synthetic_graph() -> tuple[list[Entity], list[Edge], dict[str, str]]:
    """A 1-note graph with three edges:

    * ``apbt -authored_by-> ukc``  GROUNDED (both named in body)
    * ``ukc -regulates-> apbt``    GROUNDED (both named in body)
    * ``apbt -contradicts-> tibet`` PHANTOM — the Tibetan Mastiff is
      asserted as an endpoint but is named NOWHERE in the body, and shares
      no distinctive tokens with anything that is. No source supports this
      edge. (Models the #4 'auto-detection fired on coincidental/absent
      basis' failure mode.)
    """
    entities = [
        Entity(
            id="apbt",
            name="American Pit Bull Terrier",
            entity_type="breed",
            properties={"aliases": ["APBT", "pit bull"]},
        ),
        Entity(
            id="ukc",
            name="United Kennel Club",
            entity_type="organization",
            properties={"aliases": ["UKC"]},
        ),
        Entity(
            id="tibet",
            name="Tibetan Mastiff",
            entity_type="breed",
            properties={"aliases": ["Do-Khyi"]},
        ),
    ]
    body = (
        "The United Kennel Club (UKC) recognizes the American Pit Bull "
        "Terrier as a distinct breed. The UKC standard describes the APBT "
        "temperament and conformation in detail. The UKC regulates the "
        "breed standard for the American Pit Bull Terrier."
    )
    note = "breed_standards/ukc-apbt.md"
    edges = [
        Edge(
            source_id="apbt",
            target_id="ukc",
            edge_type="authored_by",
            properties={"source_note": note, "evidence": "UKC standard"},
        ),
        Edge(
            source_id="ukc",
            target_id="apbt",
            edge_type="regulates",
            properties={"source_note": note, "evidence": "UKC regulates breed"},
        ),
        # The phantom: the Tibetan Mastiff appears nowhere in the body.
        Edge(
            source_id="apbt",
            target_id="tibet",
            edge_type="contradicts",
            properties={"source_note": note, "evidence": ""},
        ),
    ]
    return entities, edges, {note: body}


def test_seeded_phantom_edge_is_caught():
    """The apbt -contradicts-> tibet edge must be flagged phantom — the
    Tibetan Mastiff is asserted as an endpoint but never appears in the
    source body."""
    ents, edges, bodies = _synthetic_graph()
    report = score_phantom_edges(ents, edges, bodies)
    phantom_pairs = {(pe.source_id, pe.target_id) for pe in report.phantom_edges}
    assert ("apbt", "tibet") in phantom_pairs
    assert report.phantom == 1
    assert report.phantom_rate == 1 / 3


def test_grounded_edges_are_not_flagged():
    """Both edges whose endpoints ARE named in the body must ground."""
    ents, edges, bodies = _synthetic_graph()
    report = score_phantom_edges(ents, edges, bodies)
    phantom_pairs = {(pe.source_id, pe.target_id) for pe in report.phantom_edges}
    assert ("apbt", "ukc") not in phantom_pairs
    assert ("ukc", "apbt") not in phantom_pairs
    assert report.grounded == 2


def test_phantom_edge_records_missing_endpoint():
    """The phantom edge should record which endpoint failed — here the
    target (tibet), since the source (apbt) IS in the body."""
    ents, edges, bodies = _synthetic_graph()
    report = score_phantom_edges(ents, edges, bodies)
    pe = next(p for p in report.phantom_edges if p.target_id == "tibet")
    assert pe.missing == "target"
    assert pe.edge_type == "contradicts"
    assert pe.source_note == "breed_standards/ukc-apbt.md"


def test_alias_grounds_endpoint():
    """An endpoint named only by its alias in the body still grounds.

    Replace the body so APBT appears only as 'APBT' (alias), never as the
    full canonical name. With alias-aware matching the authored_by edge
    still grounds.
    """
    ents, edges, bodies = _synthetic_graph()
    note = "breed_standards/ukc-apbt.md"
    bodies[note] = (
        "The United Kennel Club recognizes the APBT. "
        "The UKC standard covers the APBT in detail."
    )
    report = score_phantom_edges(ents, edges, bodies)
    phantom_pairs = {(pe.source_id, pe.target_id) for pe in report.phantom_edges}
    # authored_by(apbt, ukc): both 'APBT' alias and 'United Kennel Club' present
    assert ("apbt", "ukc") not in phantom_pairs


def test_edge_without_source_body_is_excluded_not_assumed_phantom():
    """An edge whose source_note isn't in the bodies map can't be
    grounded — it must be EXCLUDED from the rate, not counted phantom."""
    ents, edges, bodies = _synthetic_graph()
    edges.append(
        Edge(
            source_id="apbt",
            target_id="ukc",
            edge_type="mentions",
            properties={"source_note": "missing/nowhere.md"},
        )
    )
    report = score_phantom_edges(ents, edges, bodies)
    assert report.edges_total == 4
    assert report.edges_checked == 3  # the missing-source edge excluded
    assert report.edges_missing_source == 1


def test_uncheckable_stopword_only_name():
    """An endpoint whose name is all stop-words can't be lexically
    checked — the edge is a phantom CANDIDATE but flagged uncheckable."""
    entities = [
        Entity(id="x", name="The Dog", entity_type="concept", properties={}),
        Entity(id="y", name="Border Collie", entity_type="breed", properties={}),
    ]
    body = "Border Collie herding behaviour is well documented."
    note = "n.md"
    edges = [
        Edge(
            source_id="x",
            target_id="y",
            edge_type="mentions",
            properties={"source_note": note},
        )
    ]
    report = score_phantom_edges(entities, edges, {note: body})
    assert report.uncheckable == 1
    assert report.phantom == 1
    pe = report.phantom_edges[0]
    assert pe.uncheckable is True


def test_empty_graph_is_all_zeros():
    report = score_phantom_edges([], [], {})
    assert report.edges_total == 0
    assert report.edges_checked == 0
    assert report.phantom == 0
    assert report.phantom_rate == 0.0
    assert report.phantom_edges == []


def test_format_report_renders_synthetic():
    ents, edges, bodies = _synthetic_graph()
    report = score_phantom_edges(ents, edges, bodies)
    rendered = format_report(report)
    assert "Phantom Edges" in rendered
    assert "proposed category" in rendered
    assert "Phantom rate" in rendered
    # The phantom example should surface.
    assert "tibet" in rendered


def test_min_overlap_threshold_is_echoed():
    ents, edges, bodies = _synthetic_graph()
    report = score_phantom_edges(ents, edges, bodies, min_overlap=0.75)
    assert report.min_overlap == 0.75


# --- good-dog corpus: calibration against needs_grounding -------------


def _good_dog_report(min_overlap: float = 0.5) -> PhantomEdgeReport:
    from sme.corpora.good_dog_graph import load_graph, load_source_bodies

    ents, edges = load_graph()
    bodies = load_source_bodies()
    return score_phantom_edges(ents, edges, bodies, min_overlap=min_overlap)


def test_good_dog_loads_and_checks_every_edge():
    """All good-dog edges carry a source_note that resolves to a body —
    none should fall into edges_missing_source.

    Counts track the good-dog corpus expansion (upstream 54→62 sources /
    v0.3): 469 edges across 62 vault notes, every one grounded to a source.
    """
    report = _good_dog_report()
    assert report.edges_total == 469
    assert report.edges_checked == 469
    assert report.edges_missing_source == 0


def test_good_dog_needs_grounding_flag_propagates():
    """The loader must propagate needs_grounding onto edges so the
    detector can calibrate — there are 39 such edges in the expanded
    corpus (good-dog 54→62 / v0.3)."""
    report = _good_dog_report()
    assert report.flagged_total == 39


def test_good_dog_calibration_delta():
    """VALIDITY PROOF: in the usable threshold band (0.5–0.75), the
    detector flags ``needs_grounding`` edges as phantom at a strictly
    higher rate than the rest — it tracks the maintainer's own weak-
    grounding judgement, not noise.

    The band matters: at the strict ``1.0`` reading the signal inverts
    (alias-rich LEGITIMATE edges in the unflagged set fail to ground
    because every token of a long canonical title rarely appears
    verbatim), and at the permissive ``0.34`` reading every edge grounds
    so there's no separation. Both extremes are documented limitations in
    the module proposal; the default ``0.5`` sits in the usable band.
    """
    for ov in (0.5, 0.6, 0.75):
        report = _good_dog_report(min_overlap=ov)
        assert report.flagged_total > 0
        assert report.unflagged_total > 0
        assert report.flagged_phantom_rate > report.unflagged_phantom_rate, (
            f"calibration inverted at min_overlap={ov}: "
            f"flagged {report.flagged_phantom_rate:.3f} vs "
            f"unflagged {report.unflagged_phantom_rate:.3f}"
        )


def test_good_dog_format_report_surfaces_calibration():
    report = _good_dog_report()
    rendered = format_report(report)
    assert "Calibration" in rendered
    assert "needs_grounding" in rendered


# --- load_source_bodies contract --------------------------------------


def test_load_source_bodies_strips_frontmatter():
    """The body map must contain prose, NOT the YAML frontmatter — the
    grounding check would be circular if it could see the edge
    declarations it's auditing.

    Asserted via ``note_id:`` / ``source_url:`` — frontmatter-only keys
    that never appear in the markdown prose body. (``needs_grounding:``
    can't be used: the corpus notes discuss their own grounding flags in
    prose, so the literal string legitimately survives in the body.)
    """
    from sme.corpora.good_dog_graph import load_source_bodies

    bodies = load_source_bodies()
    assert bodies, "expected at least one source body"
    for note, body in bodies.items():
        assert "note_id:" not in body, note
        assert "source_url:" not in body, note
        assert not body.lstrip().startswith("---"), note


def test_load_source_bodies_keys_match_edge_source_notes():
    """Body-map keys must align with the source_note stamped on edges,
    so a consumer can look up the text behind any edge."""
    from sme.corpora.good_dog_graph import load_graph, load_source_bodies

    _, edges = load_graph()
    bodies = load_source_bodies()
    edge_notes = {e.properties.get("source_note") for e in edges}
    edge_notes.discard(None)
    missing = edge_notes - set(bodies)
    assert not missing, f"edges reference notes with no body: {missing}"
