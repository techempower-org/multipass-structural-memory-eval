"""corpus-doctor — synthetic defect injection for stress-testing adapters.

SME's public corpora are hand-authored and integrity-clean by
construction: across every shipped corpus the structural probes report
zero collisions, zero orphans, full edge-type diversity. That makes
Cat 4 (The Threshold) and Cat 5 (The Missing Room) un-calibrated — they
always read clean on clean input, so a *passing* reading proves nothing
about the detector's sensitivity. corpus-doctor closes that gap: it
takes a clean ``(entities, edges)`` snapshot, injects a KNOWN defect,
and emits a PROV-O-shaped ``defects`` manifest stating exactly what was
done. The verification harness (``sme.categories.corpus_doctor_harness``)
then runs the relevant category and asserts it RECOVERS the injected
defect — turning the cats' clean readings into a falsifiable claim.

This is the FIRST SLICE of upstream issue #27. Three pathologies are
implemented, each mapping to an existing detector with no new dependency:

  * ``duplicate_evidence``  → Cat 4a canonical-collision dedup. Clones N
    entities under fresh IDs but identical name+type, so they collapse
    onto one canonical key. The clones are the "extra duplicate IDs"
    Cat 4 counts.
  * ``orphan_inflation``    → Cat 5 isolated-node count. Strips every
    edge touching a sampled set of entities, leaving them as single-node
    connected components (orphans).
  * ``monoculture_edge_type`` → Cat 4c edge-type monoculture. Rewrites a
    sampled fraction of edges to a single dominant type, collapsing the
    edge-type distribution toward an entropy of ~0.

The remaining issue-#27 pathologies (``zipfian_degree``,
``hotspot_entity``, ``stale_facts``, plus a ``phantom_edge`` pathology
once Cat #4 phantom-edge detection lands) are intentionally deferred —
see ``PATHOLOGY_BACKLOG`` and the module-level note. The design keeps the
injectors as pure ``(entities, edges) -> (entities, edges, defects)``
transforms over the core dataclasses, so they compose and require no
file IO, no corpus rewriting, and no backend.

Manifest schema (PROV-O-aligned, one record per injected defect)::

    {
      "defect_id":        "dup::e1::0001",       # stable, unique
      "pathology":        "duplicate_evidence",
      "prov:activity":    "inject_defect",
      "prov:wasAttributedTo": "corpus-doctor/0.1",
      "severity":         0.3,
      "seed":             1234,
      "target":           {"kind": "entity", "ids": ["e1"]},
      "expect": {                                # what the cat should read
        "category": "cat4",
        "field":    "canonical_collisions",
        "delta":    1                            # +1 extra duplicate ID
      },
      "detail":           {...}                  # pathology-specific
    }

``severity`` is interpreted per-pathology (it scales the count/fraction
of injected defects). Injection is deterministic given ``seed``.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from sme.adapters.base import Edge, Entity

# Provenance tag stamped on every manifest record (PROV-O wasAttributedTo).
DOCTOR_AGENT = "corpus-doctor/0.1"
PROV_ACTIVITY = "inject_defect"

# Pathologies named in issue #27 that this first slice does NOT implement.
# Kept here so the gap is discoverable in code, not just in the issue.
PATHOLOGY_BACKLOG = (
    "zipfian_degree",      # YCSB power-law degree re-weight (param s)
    "hotspot_entity",      # YCSB N% entities answer M% of queries
    "stale_facts",         # later-dated contradicting fact (Cat 3 + Cat 6)
    "phantom_edge",        # edge with no source support (blocked on Cat #4)
)


@dataclass
class Defect:
    """One injected defect, PROV-O-shaped ground truth.

    ``expect`` states the category/field the defect should move and by
    how much, so the harness can grade detection against the manifest
    without re-deriving the expectation from the pathology name.
    """

    defect_id: str
    pathology: str
    severity: float
    seed: int
    target: dict
    expect: dict
    detail: dict = field(default_factory=dict)
    prov_activity: str = PROV_ACTIVITY
    prov_was_attributed_to: str = DOCTOR_AGENT

    def to_manifest_record(self) -> dict:
        """Serialize to the PROV-O-aligned manifest shape (prov: keys)."""
        d = asdict(self)
        d["prov:activity"] = d.pop("prov_activity")
        d["prov:wasAttributedTo"] = d.pop("prov_was_attributed_to")
        return d


@dataclass
class DoctorResult:
    """The output of an injection pass: a dirtied snapshot + manifest."""

    entities: list[Entity]
    edges: list[Edge]
    defects: list[Defect]

    @property
    def pathologies(self) -> list[str]:
        return sorted({d.pathology for d in self.defects})


# --- helpers ---------------------------------------------------------


def _clone_snapshot(
    entities: list[Entity], edges: list[Edge]
) -> tuple[list[Entity], list[Edge]]:
    """Deep-copy the snapshot so injection never mutates the caller's
    clean corpus in place. Entity embeddings (numpy arrays) are copied by
    reference — they are not relevant to the structural cats and a deep
    copy of large arrays would be wasteful."""
    new_entities = [
        Entity(
            id=e.id,
            name=e.name,
            entity_type=e.entity_type,
            properties=copy.deepcopy(e.properties),
            embedding=e.embedding,
        )
        for e in entities
    ]
    new_edges = [
        Edge(
            source_id=e.source_id,
            target_id=e.target_id,
            edge_type=e.edge_type,
            properties=copy.deepcopy(e.properties),
        )
        for e in edges
    ]
    return new_entities, new_edges


def _count_for_severity(n_candidates: int, severity: float, *, minimum: int = 1) -> int:
    """Map a 0..1 severity onto a count over ``n_candidates``.

    Severity is clamped to [0, 1]. A positive severity always injects at
    least ``minimum`` defect (so a smoke run at low severity still
    produces a gradeable manifest), capped at the candidate population.
    """
    if n_candidates <= 0:
        return 0
    sev = max(0.0, min(1.0, severity))
    if sev <= 0.0:
        return 0
    count = round(sev * n_candidates)
    count = max(minimum, count)
    return min(count, n_candidates)


# --- pathology: duplicate_evidence (→ Cat 4a) ------------------------


def inject_duplicate_evidence(
    entities: list[Entity],
    edges: list[Edge],
    *,
    severity: float = 0.3,
    seed: int = 0,
) -> DoctorResult:
    """Clone entities under fresh IDs but identical name+type.

    Each clone canonicalizes to the same key as its source, so Cat 4a
    counts it as an EXTRA duplicate ID (``canonical_collisions`` rises by
    one per clone). The clone is edge-less by default — it is the
    low-degree straggler that "missed canonicalization", exactly the
    Cat 4 collision-group shape (the source keeps its degree; the clone
    has 0). ``severity`` scales how many distinct entities get cloned.
    """
    new_entities, new_edges = _clone_snapshot(entities, edges)
    rng = random.Random(seed)

    # Only clone entities that have a usable canonical key (non-empty
    # name + type) — cloning an empty-name entity would be a Cat 4b gap,
    # not a 4a collision, muddying the manifest's expectation.
    candidates = [e for e in new_entities if e.name and e.entity_type]
    candidates.sort(key=lambda e: e.id)  # deterministic order pre-shuffle
    rng.shuffle(candidates)
    k = _count_for_severity(len(candidates), severity)
    chosen = candidates[:k]

    defects: list[Defect] = []
    existing_ids = {e.id for e in new_entities}
    for i, src in enumerate(sorted(chosen, key=lambda e: e.id)):
        clone_id = f"{src.id}__dupe{i:04d}"
        # Guard against a (vanishingly unlikely) collision with a real ID.
        while clone_id in existing_ids:
            clone_id += "_x"
        existing_ids.add(clone_id)
        clone = Entity(
            id=clone_id,
            name=src.name,
            entity_type=src.entity_type,
            properties={
                **copy.deepcopy(src.properties),
                "_corpus_doctor": "duplicate_evidence",
                "_dupe_of": src.id,
            },
        )
        new_entities.append(clone)
        defects.append(
            Defect(
                defect_id=f"dup::{src.id}::{i:04d}",
                pathology="duplicate_evidence",
                severity=severity,
                seed=seed,
                target={"kind": "entity", "ids": [clone_id], "source_id": src.id},
                expect={
                    "category": "cat4",
                    "field": "canonical_collisions",
                    "delta": 1,
                },
                detail={"name": src.name, "entity_type": src.entity_type},
            )
        )

    return DoctorResult(new_entities, new_edges, defects)


# --- pathology: orphan_inflation (→ Cat 5) ---------------------------


def inject_orphan_inflation(
    entities: list[Entity],
    edges: list[Edge],
    *,
    severity: float = 0.2,
    seed: int = 0,
) -> DoctorResult:
    """Strip every edge touching a sampled set of entities.

    Each chosen entity loses all incident edges, becoming a single-node
    connected component — an orphan Cat 5 counts in ``isolated_nodes``.
    Only entities that currently HAVE at least one edge are candidates
    (orphaning an already-isolated node is a no-op, not a defect). The
    removed edges are recorded on each defect so the manifest is fully
    reversible. ``severity`` scales how many connected entities are
    orphaned.

    Collateral isolation: stripping a node's edges can also strand a
    degree-1 NEIGHBOUR that only connected through it. The manifest lists
    exactly the nodes corpus-doctor TARGETED, so Cat 5 may legitimately
    report MORE isolates than there are defects. The verification harness
    treats this as full recovery (it caps the observed delta at the
    expected one) rather than penalising the detector for being right
    about the collateral — see ``corpus_doctor_harness`` over-shoot note.
    """
    new_entities, new_edges = _clone_snapshot(entities, edges)
    rng = random.Random(seed)

    touched: dict[str, int] = Counter()
    for e in new_edges:
        touched[e.source_id] += 1
        touched[e.target_id] += 1

    valid_ids = {e.id for e in new_entities}
    candidates = sorted(eid for eid in touched if eid in valid_ids)
    rng.shuffle(candidates)
    k = _count_for_severity(len(candidates), severity)
    chosen = set(candidates[:k])

    removed_for: dict[str, list[dict]] = {cid: [] for cid in chosen}
    kept_edges: list[Edge] = []
    for e in new_edges:
        if e.source_id in chosen or e.target_id in chosen:
            rec = {
                "source_id": e.source_id,
                "target_id": e.target_id,
                "edge_type": e.edge_type,
            }
            for endpoint in (e.source_id, e.target_id):
                if endpoint in removed_for:
                    removed_for[endpoint].append(rec)
        else:
            kept_edges.append(e)

    defects: list[Defect] = []
    for i, cid in enumerate(sorted(chosen)):
        defects.append(
            Defect(
                defect_id=f"orphan::{cid}::{i:04d}",
                pathology="orphan_inflation",
                severity=severity,
                seed=seed,
                target={"kind": "entity", "ids": [cid]},
                expect={
                    "category": "cat5",
                    "field": "isolated_nodes",
                    "delta": 1,
                },
                detail={"removed_edges": removed_for[cid]},
            )
        )

    return DoctorResult(new_entities, kept_edges, defects)


# --- pathology: monoculture_edge_type (→ Cat 4c) ---------------------


def inject_monoculture_edge_type(
    entities: list[Entity],
    edges: list[Edge],
    *,
    severity: float = 0.5,
    seed: int = 0,
    dominant_type: Optional[str] = None,
) -> DoctorResult:
    """Collapse a sampled fraction of edges onto a single dominant type.

    Rewrites ``edge_type`` to ``dominant_type`` for ``severity`` fraction
    of edges whose type is NOT already dominant. This drives Cat 4c's
    ``dominant_edge_type_fraction`` up and the normalized edge-type
    entropy down — the monoculture signature. Each rewritten edge gets a
    defect recording its original type so the manifest is reversible.

    ``dominant_type`` defaults to the corpus's CURRENT most-common edge
    type. Amplifying the existing skew is the honest monoculture: it can
    only push the dominant fraction up. Passing a type that is currently
    a minority (or absent) would instead REDISTRIBUTE share away from the
    true dominant type and can lower the dominant fraction — a real bug
    this default avoids. Pass an explicit ``dominant_type`` only when you
    want to force a specific collapse target (e.g. a synthetic graph
    whose intended dominant type you control). Ties are broken by name so
    the choice is deterministic.

    ``entities`` is carried through untouched (this pathology only
    rewrites edge types) so every injector shares the same
    ``(entities, edges, *, severity, seed)`` signature the dispatcher
    relies on.
    """
    new_entities, new_edges = _clone_snapshot(entities, edges)
    rng = random.Random(seed)

    if dominant_type is None:
        type_counts = Counter(e.edge_type for e in new_edges if e.edge_type)
        if type_counts:
            # most_common ties are insertion-ordered; sort for determinism.
            top = max(type_counts.items(), key=lambda kv: (kv[1], kv[0]))
            dominant_type = top[0]
        else:
            dominant_type = "RELATED"

    candidate_idx = [
        i for i, e in enumerate(new_edges) if e.edge_type != dominant_type
    ]
    candidate_idx.sort()
    rng.shuffle(candidate_idx)
    k = _count_for_severity(len(candidate_idx), severity)
    chosen_idx = sorted(candidate_idx[:k])

    defects: list[Defect] = []
    for i, idx in enumerate(chosen_idx):
        e = new_edges[idx]
        original = e.edge_type
        e.edge_type = dominant_type
        e.properties = {
            **e.properties,
            "_corpus_doctor": "monoculture_edge_type",
            "_original_edge_type": original,
        }
        defects.append(
            Defect(
                defect_id=f"mono::{idx:05d}::{i:04d}",
                pathology="monoculture_edge_type",
                severity=severity,
                seed=seed,
                target={
                    "kind": "edge",
                    "edge_index": idx,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                },
                expect={
                    "category": "cat4",
                    "field": "dominant_edge_type_fraction",
                    "direction": "increase",
                },
                detail={
                    "original_edge_type": original,
                    "rewritten_to": dominant_type,
                },
            )
        )

    return DoctorResult(new_entities, new_edges, defects)


# --- dispatch + manifest IO ------------------------------------------

# name -> injector. Each injector has signature
#   (entities, edges, *, severity, seed, **kw) -> DoctorResult
PATHOLOGIES: dict[str, Callable[..., DoctorResult]] = {
    "duplicate_evidence": inject_duplicate_evidence,
    "orphan_inflation": inject_orphan_inflation,
    "monoculture_edge_type": inject_monoculture_edge_type,
}


def inject(
    entities: list[Entity],
    edges: list[Edge],
    pathology: str,
    *,
    severity: float = 0.3,
    seed: int = 0,
    **kwargs,
) -> DoctorResult:
    """Inject a single named pathology into a clean snapshot.

    Raises ``KeyError`` (with the backlog noted) for an unknown name.
    """
    if pathology not in PATHOLOGIES:
        known = ", ".join(sorted(PATHOLOGIES))
        backlog = ", ".join(PATHOLOGY_BACKLOG)
        raise KeyError(
            f"unknown pathology {pathology!r}. implemented: {known}. "
            f"deferred (issue #27 backlog): {backlog}."
        )
    return PATHOLOGIES[pathology](
        entities, edges, severity=severity, seed=seed, **kwargs
    )


def inject_many(
    entities: list[Entity],
    edges: list[Edge],
    pathologies: list[str],
    *,
    severity: float = 0.3,
    seed: int = 0,
) -> DoctorResult:
    """Compose several pathologies, threading the dirtied snapshot through
    each in turn. Defect manifests accumulate. Each pathology gets a
    distinct derived seed so the runs don't correlate."""
    cur_entities, cur_edges = entities, edges
    all_defects: list[Defect] = []
    for offset, name in enumerate(pathologies):
        result = inject(
            cur_entities,
            cur_edges,
            name,
            severity=severity,
            seed=seed + offset * 7919,  # prime stride to decorrelate
        )
        cur_entities, cur_edges = result.entities, result.edges
        all_defects.extend(result.defects)
    return DoctorResult(cur_entities, cur_edges, all_defects)


def defects_to_jsonl(defects: list[Defect]) -> str:
    """Render the manifest as newline-delimited PROV-O records."""
    return "\n".join(
        json.dumps(d.to_manifest_record(), sort_keys=True) for d in defects
    )


def write_manifest(defects: list[Defect], path: str | Path) -> Path:
    """Write ``defects.jsonl`` to ``path`` and return the resolved path."""
    out = Path(path)
    out.write_text(defects_to_jsonl(defects) + ("\n" if defects else ""))
    return out


def load_manifest(path: str | Path) -> list[Defect]:
    """Read a ``defects.jsonl`` manifest back into ``Defect`` records.

    Tolerates blank lines. Maps the PROV-O ``prov:`` keys back onto the
    dataclass fields so a written-then-read manifest round-trips.
    """
    text = Path(path).read_text()
    out: list[Defect] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out.append(
            Defect(
                defect_id=rec["defect_id"],
                pathology=rec["pathology"],
                severity=rec["severity"],
                seed=rec["seed"],
                target=rec["target"],
                expect=rec["expect"],
                detail=rec.get("detail", {}),
                prov_activity=rec.get("prov:activity", PROV_ACTIVITY),
                prov_was_attributed_to=rec.get(
                    "prov:wasAttributedTo", DOCTOR_AGENT
                ),
            )
        )
    return out
