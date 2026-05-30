"""good-dog-corpus structural graph adapter.

Wraps ``sme.corpora.good_dog_graph.load_graph`` in the SME adapter
contract so Cat 3 (The Dissonance) and Cat 6 (The Archive) can take a
*structural* reading against the corpus's seeded ``contradicts`` /
``supersedes`` edges.

This adapter is the structural counterpart to ``FlatBaselineAdapter``
on the good-dog corpus. The flat baseline reads only the drawer text
(substring recall) and surfaces no contradictions or supersession links;
this adapter reads the typed-edge layer the corpus declares and surfaces:

  * ``QueryResult.contradictions`` — populated on a query whose retrieved
    entities sit on a ``contradicts`` edge (Cat 3 structured channel).
  * the reserved ``_superseded_by`` edge property on the snapshot
    (Cat 6 supersession channel).

The retrieval here is a deliberately minimal lexical match over entity
names / aliases / edge evidence — the point of the adapter is the
*structured-field surfacing*, not a sophisticated retriever. The
``(structural − flat)`` Cat 3 / Cat 6 delta is the headline; absolute
retrieval quality is not the claim.

It is diagnostic-only (Mode B): the corpus is fixed in-tree, so
``ingest_corpus`` is a no-op that returns the loaded counts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sme.adapters.base import (
    ContradictionPair,
    Edge,
    Entity,
    QueryResult,
    SMEAdapter,
    contradiction_pairs_from_edges,
    is_contradicts_edge,
)
from sme.corpora.good_dog_graph import VAULT_ROOT, load_graph

log = logging.getLogger(__name__)


class GoodDogGraphAdapter(SMEAdapter):
    """Structural adapter over the good-dog-corpus vault graph.

    Args:
        vault_dir: path to the vault root. Defaults to the in-tree
            ``sme/corpora/good-dog-corpus/vault/``. Named ``vault_dir``
            for CLI parity with ``full-context``.
        n_results: max entities returned per ``query()``.
        read_only: accepted for CLI parity; the corpus is read-only.
    """

    def __init__(
        self,
        *,
        vault_dir: str | Path | None = None,
        n_results: int = 10,
        read_only: bool = True,
    ) -> None:
        self.vault_dir = Path(vault_dir) if vault_dir else VAULT_ROOT
        self.n_results = n_results
        self._read_only = read_only
        self._entities, self._edges = load_graph(self.vault_dir)
        self._by_id = {e.id: e for e in self._entities}
        log.info(
            "good-dog graph: %d entities, %d edges (%d contradicts)",
            len(self._entities),
            len(self._edges),
            sum(1 for e in self._edges if is_contradicts_edge(e.edge_type)),
        )

    # --- required SMEAdapter methods ----------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """No-op: the corpus is fixed in-tree. Returns the loaded counts
        so a caller treating ingest as a load-confirmation gets honest
        numbers rather than a NotImplementedError."""
        return {
            "entities_created": len(self._entities),
            "edges_created": len(self._edges),
            "errors": [],
            "warnings": [
                "GoodDogGraphAdapter is diagnostic-only; the vault graph "
                "is loaded from disk, not from the passed corpus."
            ],
        }

    def query(self, question: str, *, n_results: Optional[int] = None) -> QueryResult:
        """Lexical retrieval over entity names/aliases + edge evidence.

        Matches query tokens against each entity's canonical name and
        aliases; ranks by token-overlap count. Builds a context string
        from the matched entities and the evidence strings of edges that
        touch them, then populates ``contradictions`` with any
        ContradictionPair whose endpoints are both in the retrieved set
        (the Cat 3 structured channel).
        """
        k = n_results or self.n_results
        q_tokens = {
            t.strip(".,?!;:()[]\"'").lower()
            for t in question.split()
            if len(t) > 2
        }

        scored: list[tuple[int, Entity]] = []
        for ent in self._entities:
            haystack = " ".join(
                [ent.name] + (ent.properties.get("aliases") or [])
            ).lower()
            hay_tokens = {
                t.strip(".,?!;:()[]\"'")
                for t in haystack.replace("-", " ").replace("_", " ").split()
                if len(t) > 2
            }
            score = len(q_tokens & hay_tokens)
            if score > 0:
                scored.append((score, ent))

        scored.sort(key=lambda s: (-s[0], s[1].id))
        top = [ent for _, ent in scored[:k]]
        top_ids = {ent.id for ent in top}

        context_parts: list[str] = []
        for i, ent in enumerate(top):
            note = ent.properties.get("source_note", "?")
            context_parts.append(
                f"[{i + 1}] [{ent.entity_type}] {ent.name} ({note})"
            )
            for edge in self._edges:
                if edge.source_id == ent.id and edge.properties.get("evidence"):
                    context_parts.append(
                        f"    -[{edge.edge_type}]-> {edge.target_id}: "
                        f"{edge.properties['evidence']}"
                    )

        # Cat 3 channel: surface contradictions whose endpoints are both
        # in the retrieved set. A contradiction that involves a retrieved
        # entity but whose counterpart wasn't retrieved is still surfaced
        # (the system flags it even if the other framing didn't rank) —
        # that is exactly the consolidation behaviour Cat 3 rewards.
        relevant_edges = [
            e
            for e in self._edges
            if is_contradicts_edge(e.edge_type)
            and (e.source_id in top_ids or e.target_id in top_ids)
        ]
        node_names = {e.id: e.name for e in self._entities}
        contradictions = contradiction_pairs_from_edges(
            relevant_edges, node_names=node_names
        )

        context_string = "\n".join(context_parts)
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=top,
            contradictions=contradictions,
            retrieval_path=[f"matched {len(top)} entities by lexical overlap"],
            error=None if top else "NO_RESULTS",
        )

    def get_graph_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        return list(self._entities), list(self._edges)

    def get_contradiction_pairs(self) -> list[ContradictionPair]:
        node_names = {e.id: e.name for e in self._entities}
        return contradiction_pairs_from_edges(self._edges, node_names=node_names)

    def get_ontology_source(self) -> dict:
        """The good-dog ontology is declared, not inferred — point Cat 8
        at the corpus ontology.yaml schema."""
        return {
            "type": "declared",
            "schema": [
                {
                    "kind": "edge_types",
                    "values": [
                        "mentions", "alias_of", "supersedes", "contradicts",
                        "cites", "authored_by", "affiliated_with", "regulates",
                        "subject_of", "member_of", "located_in",
                    ],
                },
            ],
            "documentation": (
                "good-dog-corpus declares an 8-entity / 11-edge ontology in "
                "ontology.yaml. The supersedes (publication->publication) and "
                "contradicts (publication->publication) edge types seed Cat 6 "
                "and Cat 3 respectively."
            ),
        }

    def close(self) -> None:
        pass
