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

  phantom_edge
      Add an edge between two REAL, existing entities, stamped with a
      ``source_note`` whose prose body does NOT name either endpoint — an
      assertion with no source support. Targets the phantom-edge detector
      (``sme.categories.phantom_edge.score_phantom_edges``, upstream #4).
      Distinct from ``broken_ref``: both endpoints EXIST (so a referential
      check passes), but the source text doesn't ground the relation.

  edge_type_monoculture
      Add many edges of a single dominant type between real entities,
      collapsing the edge-type distribution toward one type. Targets
      Cat 4c's monoculture signal (``dominant_edge_type_fraction`` up,
      normalized edge-type entropy down). Detection here is a SCALAR
      signal (the distribution moved), not per-id recall — the harness
      grades it by direction-of-signal vs the clean baseline.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

from sme.adapters.base import Edge, Entity

# The defect types corpus-doctor supports, each mapped (in the harness)
# to the category that should detect it. The first three are id-recall
# defects (a set of entity/target ids); phantom_edge recalls edge keys;
# edge_type_monoculture is a scalar-signal defect (see harness).
DEFECT_TYPES = (
    "duplicate_entity",
    "orphan_node",
    "broken_ref",
    "phantom_edge",
    "edge_type_monoculture",
)

# Source-note id used for injected phantom edges. Its body (supplied to
# the phantom-edge detector via the harness) is deliberately empty, so an
# edge stamped with it can never ground — both endpoints are absent from
# the body by construction. Kept distinct from any real good-dog note.
PHANTOM_SOURCE_NOTE = "__corpus_doctor_phantom__.md"


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
    # phantom_edge: the (source, target, type, note) of an ungrounded edge.
    # Both endpoints are real; the source_note body does not name them.
    phantom_source_id: Optional[str] = None
    phantom_target_id: Optional[str] = None
    phantom_source_note: Optional[str] = None
    # edge_type_monoculture: the dominant type the injected edges collapse
    # toward. Scalar-signal defect — there are no per-id "defective" ids to
    # recall; the manifest records the type + how many edges were added.
    monoculture_type: Optional[str] = None


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

    def expected_phantom_edge_keys(self) -> set[tuple[str, str, str]]:
        """``(source_id, target_id, edge_type)`` of injected phantom edges —
        edges between real entities that no source body grounds."""
        return {
            (d.phantom_source_id, d.phantom_target_id, d.edge_type)
            for d in self.defects
            if d.defect_type == "phantom_edge"
        }

    def phantom_source_bodies(self) -> dict[str, str]:
        """``{source_note: ""}`` for every note an injected phantom edge
        was stamped with. Empty bodies guarantee the endpoints cannot
        ground. Merge into the corpus's real source-bodies map before
        running the phantom-edge detector so the injected edges are
        *checkable* (a note absent from the map is skipped, not flagged)."""
        return {
            d.phantom_source_note: ""
            for d in self.defects
            if d.defect_type == "phantom_edge" and d.phantom_source_note
        }

    def monoculture_type(self) -> Optional[str]:
        """The single edge type injected monoculture edges collapse toward
        (None if no monoculture defect was injected)."""
        for d in self.defects:
            if d.defect_type == "edge_type_monoculture":
                return d.monoculture_type
        return None


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

    # -- phantom_edge -------------------------------------------------

    def inject_phantom_edge(
        self,
        entities: list[Entity],
        edges: list[Edge],
        *,
        count: int = 1,
        edge_type: str = "injected_phantom",
    ) -> InjectionResult:
        """Add ``count`` edges between REAL entities whose ``source_note``
        body does not name either endpoint — ungrounded assertions.

        Both endpoints are sampled from the existing entity set (so a
        referential check passes — this is NOT a broken ref), and each
        edge is stamped with :data:`PHANTOM_SOURCE_NOTE`. The harness
        supplies that note an empty body (via
        ``InjectionResult.phantom_source_bodies``), so the phantom-edge
        detector finds neither endpoint present and flags the edge. Needs
        at least two distinct entities to form a non-self edge."""
        ents, eds = self._copy(entities, edges)
        if len(ents) < 2:
            raise ValueError(
                "cannot inject phantom_edge: need at least 2 entities to "
                "form an edge between distinct real endpoints"
            )
        defects: list[InjectedDefect] = []
        for _ in range(count):
            src = ents[self._rng.randrange(len(ents))]
            dst = src
            # Pick a distinct target so the edge isn't a self-loop.
            while dst.id == src.id:
                dst = ents[self._rng.randrange(len(ents))]
            eds.append(
                Edge(
                    source_id=src.id,
                    target_id=dst.id,
                    edge_type=edge_type,
                    properties={
                        "_injected_phantom": True,
                        "source_note": PHANTOM_SOURCE_NOTE,
                    },
                )
            )
            defects.append(
                InjectedDefect(
                    defect_type="phantom_edge",
                    phantom_source_id=src.id,
                    phantom_target_id=dst.id,
                    phantom_source_note=PHANTOM_SOURCE_NOTE,
                    edge_type=edge_type,
                )
            )
        return InjectionResult(entities=ents, edges=eds, defects=defects)

    # -- edge_type_monoculture ----------------------------------------

    def inject_edge_type_monoculture(
        self,
        entities: list[Entity],
        edges: list[Edge],
        *,
        count: int = 1,
        dominant_type: Optional[str] = None,
    ) -> InjectionResult:
        """Add ``count`` edges of a single type between real entities,
        collapsing the edge-type distribution toward that type.

        ``dominant_type`` defaults to the corpus's CURRENT most-common
        edge type, so the injection *amplifies* the existing skew and can
        only drive ``dominant_edge_type_fraction`` up / normalized entropy
        down. (Forcing a minority or brand-new type can REDISTRIBUTE share
        and lower the dominant fraction — a footgun; the default avoids
        it. Ties broken by name for determinism.) Detection is a scalar
        signal, so no per-id manifest is produced beyond the type + count;
        the harness compares the Cat 4c reading on the corrupted graph
        against the clean baseline."""
        ents, eds = self._copy(entities, edges)
        if len(ents) < 2:
            raise ValueError(
                "cannot inject edge_type_monoculture: need at least 2 "
                "entities to form edges between distinct real endpoints"
            )
        if dominant_type is None:
            counts: dict[str, int] = {}
            for e in eds:
                if e.edge_type:
                    counts[e.edge_type] = counts.get(e.edge_type, 0) + 1
            dominant_type = (
                max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
                if counts
                else "RELATED"
            )
        defects: list[InjectedDefect] = []
        for _ in range(count):
            src = ents[self._rng.randrange(len(ents))]
            dst = src
            while dst.id == src.id:
                dst = ents[self._rng.randrange(len(ents))]
            eds.append(
                Edge(
                    source_id=src.id,
                    target_id=dst.id,
                    edge_type=dominant_type,
                    properties={"_injected_monoculture": True},
                )
            )
            defects.append(
                InjectedDefect(
                    defect_type="edge_type_monoculture",
                    monoculture_type=dominant_type,
                    edge_source_id=src.id,
                    edge_type=dominant_type,
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
        if defect_type == "phantom_edge":
            return self.inject_phantom_edge(entities, edges, count=count)
        if defect_type == "edge_type_monoculture":
            return self.inject_edge_type_monoculture(entities, edges, count=count)
        raise ValueError(
            f"unknown defect_type {defect_type!r}; expected one of {DEFECT_TYPES}"
        )
