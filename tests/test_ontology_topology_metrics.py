"""Topology-metric correctness for Cat 8 ontology coherence.

The three networkx-backed graph metrics — modularity, type homogeneity,
inter-community edge density — are the *measurement surface* for Cat 8's
hierarchical / partitioning / cross-domain claims. They are exercised
indirectly through ``_score_claim`` elsewhere, but their numeric
correctness and boundary behavior were never asserted directly. A wrong
value here silently mis-scores the structural claims it feeds, which is
exactly the failure mode a diagnostic tool must not have.

These tests pin the metrics on graphs with a known community structure:

  * Two disjoint pure-type triangles → Louvain finds two communities,
    modularity 0.5, homogeneity 1.0, inter-community density 0.0.
  * One cross-community edge → density becomes 1 / total_edges.
  * Mixed-type community → homogeneity = dominant_count / size.

Plus the boundary conditions every metric shares: no entities → 0.0,
entities-but-no-edges → 0.0, and edges whose endpoints are absent from
the entity list are dropped (so a dangling edge cannot inflate the
graph). networkx is a project ``[topology]`` extra; the metrics return
0.0 when it is unavailable rather than raising — that fallback is
covered by monkeypatching the import.

Also covers the ``_score_claim`` dispatch for the ``inter-community``
and ``type homogeneity`` metric keys, which the existing suite skips
(it only drives modularity / cat7 / cat3 / cat2b / temporal /
provenance).
"""
from __future__ import annotations

import builtins

import pytest

from sme.adapters.base import Edge, Entity
from sme.categories.ontology_coherence import (
    _compute_inter_community_edge_density,
    _compute_modularity,
    _compute_type_homogeneity,
    _score_claim,
    load_claim_library,
)

pytest.importorskip("networkx", reason="Cat 8 topology metrics need the [topology] extra")


# ── Fixtures: graphs with a known community structure ──────────────


def _two_pure_triangles() -> tuple[list[Entity], list[Edge]]:
    """Two disjoint triangles; each triangle is a single entity_type.

    Louvain on this graph recovers exactly the two triangles as
    communities — a clean, deterministic structure to pin metrics on.
    """
    ents = [
        Entity("a1", "a1", "person"),
        Entity("a2", "a2", "person"),
        Entity("a3", "a3", "person"),
        Entity("b1", "b1", "place"),
        Entity("b2", "b2", "place"),
        Entity("b3", "b3", "place"),
    ]
    edges = [
        Edge("a1", "a2", "rel"),
        Edge("a2", "a3", "rel"),
        Edge("a3", "a1", "rel"),
        Edge("b1", "b2", "rel"),
        Edge("b2", "b3", "rel"),
        Edge("b3", "b1", "rel"),
    ]
    return ents, edges


# ── Modularity ─────────────────────────────────────────────────────


def test_modularity_two_disjoint_communities_is_positive():
    ents, edges = _two_pure_triangles()
    mod = _compute_modularity(ents, edges)
    # Two equal disjoint triangles → Q = 0.5 exactly under the standard
    # modularity definition (each community keeps half the edge mass).
    assert mod == pytest.approx(0.5)


def test_modularity_no_entities_is_zero():
    assert _compute_modularity([], []) == 0.0


def test_modularity_entities_but_no_edges_is_zero():
    ents, _ = _two_pure_triangles()
    assert _compute_modularity(ents, []) == 0.0


def test_modularity_dangling_edges_are_dropped():
    """An edge whose endpoints aren't in the entity list must not be added
    to the graph — otherwise a phantom node would distort modularity."""
    ents, _ = _two_pure_triangles()
    dangling = [Edge("a1", "GHOST", "rel"), Edge("NOPE", "ALSO_GONE", "rel")]
    # a1 exists but GHOST does not, and the second edge has two missing
    # endpoints — none of these become graph edges, so there are zero
    # real edges and modularity falls back to 0.0.
    assert _compute_modularity(ents, dangling) == 0.0


# ── Type homogeneity ───────────────────────────────────────────────


def test_homogeneity_pure_type_communities_is_one():
    ents, edges = _two_pure_triangles()
    assert _compute_type_homogeneity(ents, edges) == pytest.approx(1.0)


def test_homogeneity_fully_mixed_community():
    """A single 3-node community with three distinct types: the dominant
    type covers 1 of 3 nodes → weighted homogeneity 1/3."""
    ents = [
        Entity("a1", "a1", "person"),
        Entity("a2", "a2", "place"),
        Entity("a3", "a3", "thing"),
    ]
    edges = [Edge("a1", "a2", "rel"), Edge("a2", "a3", "rel"), Edge("a3", "a1", "rel")]
    assert _compute_type_homogeneity(ents, edges) == pytest.approx(1 / 3)


def test_homogeneity_no_entities_is_zero():
    assert _compute_type_homogeneity([], []) == 0.0


def test_homogeneity_no_edges_is_zero():
    """No edges → no community structure to measure → 0.0 (not 1.0)."""
    ents, _ = _two_pure_triangles()
    assert _compute_type_homogeneity(ents, []) == 0.0


# ── Inter-community edge density ───────────────────────────────────


def test_inter_community_density_zero_when_communities_isolated():
    ents, edges = _two_pure_triangles()
    density, inter = _compute_inter_community_edge_density(ents, edges)
    assert inter == 0
    assert density == 0.0


def test_inter_community_density_counts_crossing_edge():
    """Adding one edge between the two triangles → 1 inter-community edge
    out of 7 total → density 1/7."""
    ents, edges = _two_pure_triangles()
    edges = edges + [Edge("a1", "b1", "rel")]
    density, inter = _compute_inter_community_edge_density(ents, edges)
    assert inter == 1
    assert density == pytest.approx(1 / 7)


def test_inter_community_density_no_entities_is_zero():
    assert _compute_inter_community_edge_density([], []) == (0.0, 0)


def test_inter_community_density_no_edges_is_zero():
    ents, _ = _two_pure_triangles()
    assert _compute_inter_community_edge_density(ents, []) == (0.0, 0)


# ── networkx-absent fallback ───────────────────────────────────────


def _hide_networkx(monkeypatch):
    """Make ``import networkx`` raise ImportError, simulating a runner
    without the [topology] extra installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "networkx" or name.startswith("networkx."):
            raise ImportError("networkx hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_modularity_returns_zero_without_networkx(monkeypatch):
    ents, edges = _two_pure_triangles()
    _hide_networkx(monkeypatch)
    assert _compute_modularity(ents, edges) == 0.0


def test_homogeneity_returns_zero_without_networkx(monkeypatch):
    ents, edges = _two_pure_triangles()
    _hide_networkx(monkeypatch)
    assert _compute_type_homogeneity(ents, edges) == 0.0


def test_inter_community_returns_zero_without_networkx(monkeypatch):
    ents, edges = _two_pure_triangles()
    _hide_networkx(monkeypatch)
    assert _compute_inter_community_edge_density(ents, edges) == (0.0, 0)


# ── _score_claim dispatch over the topology metric keys ────────────
#
# These two metric branches ("inter-community", "type homogeneity") are
# untested in the existing suite, which only drives the modularity /
# cross-category branches. They route the real claim patterns from
# sme/claims/structural_claims.yaml, so the dispatch + threshold logic
# is part of the measurement surface.


@pytest.fixture(scope="module")
def claim_library():
    return load_claim_library()


def test_score_claim_inter_community_passes_with_crossing_edge(claim_library):
    ents, edges = _two_pure_triangles()
    edges = edges + [Edge("a1", "b1", "rel")]
    claim = {"id": "x_connect", "text": "cross-wing tunnels connect domains"}
    res = _score_claim(claim, ents, edges, {}, claim_library)
    assert res.status == "pass"  # density > 0
    assert res.metrics["inter_community_edges"] == 1
    assert res.metrics["inter_community_density"] == pytest.approx(1 / 7)
    # default substrate is the structural projection
    assert res.metrics["graph"] == "structural"


def test_score_claim_inter_community_fails_when_isolated(claim_library):
    ents, edges = _two_pure_triangles()  # no crossing edge
    claim = {"id": "x_connect", "text": "cross-wing tunnels connect domains"}
    res = _score_claim(claim, ents, edges, {}, claim_library)
    assert res.status == "fail"  # density == 0, threshold is strictly > 0
    assert res.metrics["inter_community_edges"] == 0


def test_score_claim_type_homogeneity_passes_on_pure_communities(claim_library):
    ents, edges = _two_pure_triangles()
    claim = {"id": "partition", "text": "entities partitioned by type"}
    res = _score_claim(claim, ents, edges, {}, claim_library)
    assert res.status == "pass"  # homogeneity 1.0 >= 0.9 threshold
    assert res.metrics["type_homogeneity"] == pytest.approx(1.0)


def test_score_claim_type_homogeneity_fails_on_mixed_community(claim_library):
    ents = [
        Entity("a1", "a1", "person"),
        Entity("a2", "a2", "place"),
        Entity("a3", "a3", "thing"),
    ]
    edges = [Edge("a1", "a2", "rel"), Edge("a2", "a3", "rel"), Edge("a3", "a1", "rel")]
    claim = {"id": "partition", "text": "entities partitioned by type"}
    res = _score_claim(claim, ents, edges, {}, claim_library)
    assert res.status == "fail"  # 1/3 < 0.9
    assert res.metrics["type_homogeneity"] == pytest.approx(1 / 3)
