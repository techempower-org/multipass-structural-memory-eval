"""good-dog-corpus vault → graph loader.

Parses the ``entities:`` and ``edges:`` YAML frontmatter blocks of every
note under ``sme/corpora/good-dog-corpus/vault/`` and projects them into
the SME ``(list[Entity], list[Edge])`` shape. The vault already carries
the seeded ``contradicts`` (publication→publication) and ``supersedes``
(publication→publication) edges that Cat 3 (The Dissonance) and Cat 6
(The Archive) probe — see ``good-dog-corpus/ontology.yaml`` for the
schema and the seeded-pair registry.

This is the corpus-side ground truth: a deterministic, in-tree graph
carrying real typed contradiction/supersession edges, with no external
service, no daemon, and no production-palace contamination risk. It is
the structural counterweight to the flat-baseline reading in
``docs/good_dog_cat3_cat6_findings.md`` — running Cat 3 / Cat 6 against
this graph yields the ``(structural − flat)`` headline the matrix wants.

The loader is adapter-agnostic. ``GoodDogGraphAdapter`` (in
``sme.adapters.good_dog_graph``) wraps it for the ``get_graph_snapshot``
/ ``get_contradiction_pairs`` contract; the raw ``load_graph`` function
is also usable directly by tests and the category scorers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from sme.adapters.base import (
    Edge,
    Entity,
    annotate_superseded_edges,
)

# Repo-relative corpus root: sme/corpora/good-dog-corpus/
CORPUS_ROOT = Path(__file__).resolve().parent / "good-dog-corpus"
VAULT_ROOT = CORPUS_ROOT / "vault"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def _parse_frontmatter(text: str) -> Optional[dict]:
    """Return the YAML frontmatter dict for a note, or None if absent."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _find_notes(vault_root: Path) -> list[Path]:
    return sorted(vault_root.rglob("*.md"))


def load_graph(
    vault_root: Path | str = VAULT_ROOT,
) -> tuple[list[Entity], list[Edge]]:
    """Build the good-dog corpus graph from vault frontmatter.

    Every ``entities:`` row becomes an ``Entity`` (id, canonical name,
    ontology type). Every ``edges:`` row becomes an ``Edge`` whose
    ``edge_type`` is the declared ontology edge type — including
    ``contradicts`` and ``supersedes``, which carry the evidence string
    in ``properties['evidence']``. After projection the reserved
    ``_superseded_by`` property is stamped on edges originating from a
    superseded entity (Cat 6 plumbing).

    Notes are processed in sorted path order for deterministic output.
    Entities are de-duplicated on id (a note may re-declare an entity
    introduced in another note — e.g. the FDA org reused across the DCM
    chain); the first declaration wins.
    """
    root = Path(vault_root)
    entities_by_id: dict[str, Entity] = {}
    edges: list[Edge] = []

    for note_path in _find_notes(root):
        try:
            fm = _parse_frontmatter(note_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if not fm:
            continue
        source_note = str(note_path.relative_to(root))

        for ent in fm.get("entities") or []:
            if not isinstance(ent, dict):
                continue
            eid = ent.get("id")
            if not eid or eid in entities_by_id:
                continue
            entities_by_id[eid] = Entity(
                id=eid,
                name=ent.get("canonical") or eid,
                entity_type=ent.get("type") or "unknown",
                properties={
                    "_table": "good_dog_entity",
                    "aliases": ent.get("aliases") or [],
                    "source_note": source_note,
                    "timestamp": ent.get("timestamp"),
                    "status": ent.get("status"),
                },
            )

        for edge in fm.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            src = edge.get("from")
            dst = edge.get("to")
            etype = edge.get("type")
            if not (src and dst and etype):
                continue
            edges.append(
                Edge(
                    source_id=src,
                    target_id=dst,
                    edge_type=etype,
                    properties={
                        "_table": "good_dog_edge",
                        "evidence": edge.get("evidence") or "",
                        "source_note": source_note,
                    },
                )
            )

    entities = list(entities_by_id.values())
    # Cat 6 plumbing — derive `_superseded_by` from supersedes edges.
    annotate_superseded_edges(edges)
    return entities, edges
