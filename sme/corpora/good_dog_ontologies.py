"""Three ontology projections of the good-dog corpus graph (upstream #45).

The good-dog corpus is authored at a *moderate* granularity: 8 entity
types (breed, person, organization, publication, concept, event, location,
product) and 10 edge types. This module defines the flat and fine-grained
projections of that same graph so the ontology-sensitivity sweep can read
Cat 4 / Cat 5 under all three without changing the corpus or the topology.

Design principles:

- **Deterministic.** Every projection is a pure function of the entity /
  edge fields already in the graph (type, name, properties). No
  randomness, no model calls — the sweep must reproduce byte-for-byte.
- **Topology-preserving.** Only ``entity_type`` / ``edge_type`` change.
  The node set and edge set are identical across conditions, so any Cat 5
  movement comes from type-driven signals (isolate-by-type), not a
  different graph.
- **Plausible.** The fine-grained split mirrors distinctions a real
  schema designer might make (researcher vs journalist; study vs
  standard vs bylaw; kennel-club vs regulatory body), grounded in the
  corpus's own source-domain structure rather than invented.
"""

from __future__ import annotations

from sme.adapters.base import Edge, Entity
from sme.eval.ontology_sensitivity import OntologyCondition

# --- flat: collapse everything to one type ----------------------------


def _flat_entity(_e: Entity) -> str:
    return "node"


def _flat_edge(_x: Edge) -> str:
    return "related"


# --- fine-grained: split the moderate types into sub-types ------------
#
# Splits are driven by the entity's source_note domain (the corpus
# organizes notes under six domains: behavioral_research,
# veterinary_research, breed_standards, community_journalism,
# municipal_policy, nutrition_safety) plus light name heuristics. This
# turns 8 entity types into 15+ without touching topology.


def _domain_of(e: Entity) -> str:
    """The corpus source-domain folder for an entity, or '' if unknown."""
    note = str(e.properties.get("source_note", ""))
    # source_note is like "behavioral_research/dominance_theory.md"
    return note.split("/", 1)[0] if "/" in note else ""


def _fine_entity(e: Entity) -> str:
    t = e.entity_type
    domain = _domain_of(e)

    if t == "person":
        # Researchers vs journalists vs officials, by the domain that
        # introduced them.
        if domain in ("behavioral_research", "veterinary_research"):
            return "researcher"
        if domain == "community_journalism":
            return "journalist"
        if domain == "municipal_policy":
            return "official"
        return "person_other"

    if t == "organization":
        if domain == "breed_standards":
            return "kennel_club"
        if domain in ("behavioral_research", "veterinary_research"):
            return "research_org"
        if domain == "municipal_policy":
            return "regulatory_body"
        return "organization_other"

    if t == "publication":
        # Standards, bylaws, and studies are distinct authority classes.
        if domain == "breed_standards":
            return "breed_standard"
        if domain == "municipal_policy":
            return "bylaw"
        if domain == "community_journalism":
            return "article"
        return "study"

    # breed, concept, event, location, product stay as-is — already
    # single-purpose in this corpus.
    return t


def _fine_edge(x: Edge) -> str:
    # Split the catch-all "mentions" by the kind of thing it connects,
    # using the reserved evidence string when present. This is the one
    # edge type the corpus uses as a sink, so splitting it is the
    # highest-leverage fine-grained move.
    if x.edge_type == "mentions":
        ev = str(x.properties.get("evidence", "")).lower()
        if "cite" in ev or "reference" in ev:
            return "mentions_citation"
        if "study" in ev or "research" in ev or "found" in ev:
            return "mentions_finding"
        return "mentions_general"
    return x.edge_type


# --- the three conditions ---------------------------------------------

GOOD_DOG_FLAT = OntologyCondition(
    name="flat",
    description="all nodes one type, all edges one type",
    entity_mapper=_flat_entity,
    edge_mapper=_flat_edge,
)

GOOD_DOG_MODERATE = OntologyCondition(
    name="moderate",
    description="the corpus as-authored (8 entity types, 10 edge types)",
    # identity mappers (defaults)
)

GOOD_DOG_FINE_GRAINED = OntologyCondition(
    name="fine_grained",
    description="person/org/publication split by source domain; mentions split by evidence",
    entity_mapper=_fine_entity,
    edge_mapper=_fine_edge,
)

GOOD_DOG_CONDITIONS = [
    GOOD_DOG_FLAT,
    GOOD_DOG_MODERATE,
    GOOD_DOG_FINE_GRAINED,
]
