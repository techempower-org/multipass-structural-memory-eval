"""Detectors that map a defect type to the SME signal that catches it.

Two of the three first-slice defects are caught by existing categories
(Cat 4a dedup, Cat 5 isolated nodes); this module wraps them in a
uniform ``(found_ids, report)`` shape so the harness can score recall
the same way for every defect. The third (broken_ref) has no existing
category, so a small referential-integrity check lives here.

A detector takes the corrupted ``(entities, edges)`` and returns the set
of ids it flags as defective, of the kind the harness will compare
against the injection manifest:

  - duplicate_entity → ids that appear in a canonical-collision group
  - orphan_node      → ids with no incident edges
  - broken_ref       → target ids referenced by an edge but absent from
                       the entity set (dangling references)
"""
from __future__ import annotations

from sme.adapters.base import Edge, Entity


def detect_duplicate_entities(
    entities: list[Entity], edges: list[Edge]
) -> set[str]:
    """Ids flagged by Cat 4a canonical-collision dedup.

    Returns every id inside a collision group EXCEPT one representative
    per group (the originals), i.e. the set of *extra* duplicate ids —
    matching what ``InjectionResult.expected_duplicate_ids`` records.
    The kept representative is the highest-degree id (the dedup
    convention: the canonical entity others converged on); ties broken
    by id for determinism.
    """
    from sme.categories.ingestion_integrity import score_ingestion_integrity

    report = score_ingestion_integrity(entities, edges)
    flagged: set[str] = set()
    for group in report.collision_groups:
        if len(group.ids) < 2:
            continue
        degrees = group.id_degrees or {}
        keep = max(group.ids, key=lambda i: (degrees.get(i, 0), i))
        flagged.update(i for i in group.ids if i != keep)
    return flagged


def detect_orphan_nodes(entities: list[Entity], edges: list[Edge]) -> set[str]:
    """Ids with no incident edges — the structural isolates Cat 5 counts.

    Cat 5's report gives the *count* of isolated nodes, not their ids,
    so we recompute the isolate set directly (an entity id that is
    neither a source nor a target of any edge). This is the same
    membership test Cat 5 uses to build its isolated-node count, exposed
    at id granularity for recall scoring.
    """
    incident: set[str] = set()
    for e in edges:
        incident.add(e.source_id)
        incident.add(e.target_id)
    return {ent.id for ent in entities if ent.id not in incident}


def detect_broken_refs(entities: list[Entity], edges: list[Edge]) -> set[str]:
    """Edge target ids that are not present in the entity set.

    A referential-integrity violation: an edge points at a node that
    does not exist. Source endpoints are checked too, but the first-slice
    injector only breaks targets, so callers compare against
    ``expected_dangling_target_ids``.
    """
    entity_ids = {ent.id for ent in entities}
    dangling: set[str] = set()
    for e in edges:
        if e.target_id not in entity_ids:
            dangling.add(e.target_id)
        if e.source_id not in entity_ids:
            dangling.add(e.source_id)
    return dangling


# defect_type → detector function. The harness uses this to pick the
# detector for a given injection without a branch per defect type.
DETECTORS = {
    "duplicate_entity": detect_duplicate_entities,
    "orphan_node": detect_orphan_nodes,
    "broken_ref": detect_broken_refs,
}
