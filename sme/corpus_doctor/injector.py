"""Inject KNOWN structural defects into a clean graph snapshot.

The injector is deterministic given a seed, never mutates its inputs
(it deep-copies the entities/edges first), and records a manifest of
exactly what it injected. That manifest is the ground truth the
verification harness scores a category's detection against — without it,
"the category found 3 collisions" is unscored; with it, "the category
recalled 3/3 injected duplicates" is.

Defect types (first slice — see package docstring for scope):

  duplicate_entity
      Clone an existing entity under a fresh ID with a name that
      canonicalizes to the SAME key (a case/whitespace variant), so a
      dedup pass should collapse the pair. Targets Cat 4a.

  orphan_node
      Add a brand-new entity with NO incident edges — a structural
      isolate. Targets Cat 5 isolated-node detection.

  broken_ref
      Repoint an existing edge's ``target_id`` (or add a new edge) to an
      entity id that does not exist in the entity set — a dangling
      reference. Targets a referential-integrity check. Distinct from a
      phantom edge (no source support): a broken ref points at a MISSING
      node, which a referential check catches by endpoint membership.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

from sme.adapters.base import Edge, Entity

# The defect types this first slice supports, each mapped (in the
# harness) to the category that should detect it.
DEFECT_TYPES = ("duplicate_entity", "orphan_node", "broken_ref")


@dataclass(frozen=True)
class InjectedDefect:
    """One injected defect — the ground truth for detection scoring.

    Fields are a superset across defect types; only the relevant ones
    are populated per type (documented inline). ``defect_type`` selects
    which fields are meaningful.
    """

    defect_type: str
    # duplicate_entity: the new duplicate id and the original it clones.
    duplicate_id: Optional[str] = None
    original_id: Optional[str] = None
    canonical_key: Optional[str] = None
    # orphan_node: the id of the isolated entity that was added.
    orphan_id: Optional[str] = None
    # broken_ref: the dangling target id and the edge that now points at it.
    dangling_target_id: Optional[str] = None
    edge_source_id: Optional[str] = None
    edge_type: Optional[str] = None


@dataclass
class InjectionResult:
    """The corrupted graph plus the manifest of what was injected."""

    entities: list[Entity]
    edges: list[Edge]
    defects: list[InjectedDefect] = field(default_factory=list)

    @property
    def defect_type(self) -> Optional[str]:
        """The single defect type injected (None if mixed/empty)."""
        types = {d.defect_type for d in self.defects}
        return next(iter(types)) if len(types) == 1 else None

    def expected_duplicate_ids(self) -> set[str]:
        """Ids that a dedup pass should flag as extra duplicates."""
        return {d.duplicate_id for d in self.defects if d.duplicate_id}

    def expected_orphan_ids(self) -> set[str]:
        """Ids that should read as isolated nodes."""
        return {d.orphan_id for d in self.defects if d.orphan_id}

    def expected_dangling_target_ids(self) -> set[str]:
        """Target ids referenced by an edge but absent from the entity set."""
        return {
            d.dangling_target_id for d in self.defects if d.dangling_target_id
        }


# --- canonical-variant helpers ---------------------------------------

# Variants that change the surface name but preserve the canonical key
# under ingestion_integrity.default_canonical_key (lowercase +
# whitespace-collapse, scoped by type). An uppercased or
# double-spaced name canonicalizes to the same key as the original.
def _case_variant(name: str) -> str:
    """Upper-case the name — collapses to the same canonical key."""
    up = name.upper()
    # If the name has no lowercase letters to flip, pad whitespace
    # instead so the surface form still differs from the original.
    return up if up != name else f"{name}  "


class CorpusDoctor:
    """Inject known defects into a clean ``(entities, edges)`` snapshot.

    Stateless apart from the RNG; each ``inject_*`` returns a fresh
    ``InjectionResult`` over deep-copied inputs, leaving the caller's
    graph untouched.
    """

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    # -- internal -----------------------------------------------------

    def _copy(
        self, entities: list[Entity], edges: list[Edge]
    ) -> tuple[list[Entity], list[Edge]]:
        return copy.deepcopy(list(entities)), copy.deepcopy(list(edges))

    def _sample(self, items: list, n: int) -> list:
        """Sample up to n items without replacement, deterministically."""
        if n >= len(items):
            return list(items)
        return self._rng.sample(items, n)

    # -- duplicate_entity ---------------------------------------------

    def inject_duplicate_entity(
        self,
        entities: list[Entity],
        edges: list[Edge],
        *,
        count: int = 1,
    ) -> InjectionResult:
        """Clone ``count`` entities under fresh ids with canonical-key-
        preserving name variants — Cat 4a collisions."""
        ents, eds = self._copy(entities, edges)
        if not ents:
            raise ValueError("cannot inject duplicate_entity into an empty entity set")
        from sme.categories.ingestion_integrity import default_canonical_key

        targets = self._sample(ents, count)
        defects: list[InjectedDefect] = []
        for orig in targets:
            dup_id = f"{orig.id}__dup_{self._rng.randrange(1_000_000):06d}"
            dup_name = _case_variant(orig.name)
            dup = Entity(
                id=dup_id,
                name=dup_name,
                entity_type=orig.entity_type,
                properties={**orig.properties, "_injected_duplicate_of": orig.id},
            )
            ents.append(dup)
            defects.append(
                InjectedDefect(
                    defect_type="duplicate_entity",
                    duplicate_id=dup_id,
                    original_id=orig.id,
                    canonical_key=default_canonical_key(orig.name, orig.entity_type),
                )
            )
        return InjectionResult(entities=ents, edges=eds, defects=defects)

    # -- orphan_node --------------------------------------------------

    def inject_orphan_node(
        self,
        entities: list[Entity],
        edges: list[Edge],
        *,
        count: int = 1,
        entity_type: str = "InjectedOrphan",
    ) -> InjectionResult:
        """Add ``count`` entities with no incident edges — Cat 5 isolates."""
        ents, eds = self._copy(entities, edges)
        defects: list[InjectedDefect] = []
        for _ in range(count):
            orphan_id = f"orphan_{self._rng.randrange(1_000_000):06d}"
            ents.append(
                Entity(
                    id=orphan_id,
                    name=f"Injected orphan {orphan_id}",
                    entity_type=entity_type,
                    properties={"_injected_orphan": True},
                )
            )
            defects.append(
                InjectedDefect(defect_type="orphan_node", orphan_id=orphan_id)
            )
        return InjectionResult(entities=ents, edges=eds, defects=defects)

    # -- broken_ref ---------------------------------------------------

    def inject_broken_ref(
        self,
        entities: list[Entity],
        edges: list[Edge],
        *,
        count: int = 1,
        edge_type: str = "injected_broken_ref",
    ) -> InjectionResult:
        """Add ``count`` edges whose ``target_id`` is a non-existent
        entity — dangling references. Source endpoints are real (sampled
        from the entity set) so the only integrity violation is the
        missing target, isolating the defect from a phantom edge."""
        ents, eds = self._copy(entities, edges)
        if not ents:
            raise ValueError("cannot inject broken_ref into an empty entity set")
        existing_ids = {e.id for e in ents}
        sources = self._sample(ents, count)
        # If count > len(ents), pad by reusing sources cyclically.
        while len(sources) < count:
            sources.append(ents[self._rng.randrange(len(ents))])
        defects: list[InjectedDefect] = []
        for src in sources[:count]:
            # Generate a target id guaranteed absent from the entity set.
            dangling = f"missing_{self._rng.randrange(1_000_000):06d}"
            while dangling in existing_ids:
                dangling = f"missing_{self._rng.randrange(1_000_000):06d}"
            eds.append(
                Edge(
                    source_id=src.id,
                    target_id=dangling,
                    edge_type=edge_type,
                    properties={"_injected_broken_ref": True},
                )
            )
            defects.append(
                InjectedDefect(
                    defect_type="broken_ref",
                    dangling_target_id=dangling,
                    edge_source_id=src.id,
                    edge_type=edge_type,
                )
            )
        return InjectionResult(entities=ents, edges=eds, defects=defects)

    # -- dispatch -----------------------------------------------------

    def inject(
        self,
        defect_type: str,
        entities: list[Entity],
        edges: list[Edge],
        *,
        count: int = 1,
    ) -> InjectionResult:
        """Dispatch to the injector for ``defect_type``."""
        if defect_type == "duplicate_entity":
            return self.inject_duplicate_entity(entities, edges, count=count)
        if defect_type == "orphan_node":
            return self.inject_orphan_node(entities, edges, count=count)
        if defect_type == "broken_ref":
            return self.inject_broken_ref(entities, edges, count=count)
        raise ValueError(
            f"unknown defect_type {defect_type!r}; expected one of {DEFECT_TYPES}"
        )
