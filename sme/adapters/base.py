"""Adapter interface for SME (spec v5).

Every memory system under test implements SMEAdapter. The benchmark suite
never touches a database directly — it talks to this thin interface.

Three required methods: ingest_corpus, query, get_graph_snapshot.
Three optional: get_flat_retrieval, get_ontology_source, get_harness_manifest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None  # shape (dim,) when present


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    # Reserved property keys for Cat 6b provenance:
    #   _created_by:    str  — extraction pattern or process that created this edge
    #   _created_at:    str  — ISO timestamp
    #   _superseded_by: str  — edge id that replaced this one (if applicable)


@dataclass
class ContradictionPair:
    """Structured response for Cat 3. Systems that don't surface
    contradictions leave this empty and score 0."""

    claim_a: str
    claim_b: str
    source_a: str  # entity/session id
    source_b: str


# --- Structured-field derivation (Cat 3 / Cat 6 plumbing) -------------
#
# Cat 3 (contradiction detection) and Cat 6 (temporal supersession) both
# read structured fields off the graph snapshot that the raw adapter
# projection does not populate by default:
#
#   * Cat 3 reads ``QueryResult.contradictions`` and the adapter's
#     ``get_contradiction_pairs()`` — derived from edges whose type
#     normalizes to ``contradicts``.
#   * Cat 6 reads the reserved ``_superseded_by`` edge property — derived
#     from edges whose type normalizes to ``supersedes``. An edge
#     ``A --supersedes--> B`` means "B is superseded by A", so B's edge
#     (or B as a node) carries ``_superseded_by = <A>``.
#
# Both palace adapters (direct ChromaDB + daemon HTTP) and OMEGA store
# the relationship as a typed edge with the predicate/relation_type
# string already correct (palace-daemon projects ``predicate =
# relation_type``; OMEGA reads ``edge_type`` straight off its edges
# table). The plumbing these helpers add is purely SME-side: normalize
# the predicate, stamp ``_superseded_by`` on the superseded edge, and
# extract ContradictionPair[] from contradicts edges. No backend schema
# change is required.

# Predicate strings (case-insensitive, with common backend variants)
# that mean "this edge supersedes / replaces an earlier one".
_SUPERSEDES_PREDICATES = frozenset({
    "supersedes", "superseded_by", "supersede", "replaces", "replaced_by",
})
# Predicate strings that mean "these two make incompatible claims".
_CONTRADICTS_PREDICATES = frozenset({
    "contradicts", "contradicted_by", "contradict", "conflicts_with",
})


def normalize_predicate(edge_type: Optional[str]) -> str:
    """Lower-case, strip, and collapse separators on an edge-type string
    so backend variants (``CONTRADICTS``, ``contradicted_by``,
    ``conflicts-with``) map to a stable comparison key."""
    if not edge_type:
        return ""
    return str(edge_type).strip().lower().replace("-", "_").replace(" ", "_")


def is_supersedes_edge(edge_type: Optional[str]) -> bool:
    return normalize_predicate(edge_type) in _SUPERSEDES_PREDICATES


def is_contradicts_edge(edge_type: Optional[str]) -> bool:
    return normalize_predicate(edge_type) in _CONTRADICTS_PREDICATES


def annotate_superseded_edges(edges: list["Edge"]) -> int:
    """Stamp the reserved ``_superseded_by`` property on superseded edges.

    Walks the edge list once. For every ``A --supersedes--> B`` edge, it
    records that B is superseded by A. Any other edge *originating from*
    B (B is no longer current) then gets ``_superseded_by = A`` set in
    its properties, and the supersedes edge itself records its
    ``_supersedes_target`` for symmetry. This is the spec v8 §6 channel
    that Cat 6 reads to tell the current state from the historical one.

    Mutates ``edges`` in place. Returns the number of edges annotated so
    callers can log/assert coverage. Idempotent — re-running does not
    double-stamp.
    """
    superseded_by: dict[str, str] = {}
    for e in edges:
        if is_supersedes_edge(e.edge_type):
            # A (source) supersedes B (target): B is the older one.
            superseded_by[e.target_id] = e.source_id
            # Record the linkage on the supersedes edge itself too, so a
            # consumer holding only the typed edge can see the direction.
            e.properties.setdefault("_supersedes_target", e.target_id)

    annotated = 0
    for e in edges:
        # Mark every edge whose source is a superseded entity. This lets
        # a temporal reader drop the historical framing's outgoing claims
        # in favour of the superseding entity's.
        replacement = superseded_by.get(e.source_id)
        if replacement is not None and not is_supersedes_edge(e.edge_type):
            if e.properties.get("_superseded_by") != replacement:
                e.properties["_superseded_by"] = replacement
                annotated += 1
        # The supersedes edge's target is itself superseded — stamp it so
        # the reserved field is present on the canonical supersession edge.
        if is_supersedes_edge(e.edge_type):
            if e.properties.get("_superseded_by") != e.source_id:
                # The target is superseded by the source; expose that on
                # the edge for direct lookups keyed on the supersedes edge.
                e.properties.setdefault("_superseded_target_by", e.source_id)
    return annotated


def contradiction_pairs_from_edges(
    edges: list["Edge"],
    *,
    node_names: Optional[dict[str, str]] = None,
) -> list["ContradictionPair"]:
    """Extract ContradictionPair[] from edges typed ``contradicts``.

    ``node_names`` maps entity id → human-readable name; when provided,
    the pair's ``claim_a`` / ``claim_b`` use the evidence string (if the
    edge carries one) or the resolved node name, and ``source_a`` /
    ``source_b`` carry the entity ids. The pairs are de-duplicated on the
    unordered ``frozenset({source_a, source_b})`` so a corpus that
    declares the contradiction from both directions yields one pair.
    """
    names = node_names or {}
    seen: set[frozenset[str]] = set()
    pairs: list[ContradictionPair] = []
    for e in edges:
        if not is_contradicts_edge(e.edge_type):
            continue
        key = frozenset({e.source_id, e.target_id})
        if key in seen:
            continue
        seen.add(key)
        evidence = e.properties.get("evidence") or e.properties.get(
            "_evidence"
        )
        name_a = names.get(e.source_id, e.source_id)
        name_b = names.get(e.target_id, e.target_id)
        pairs.append(
            ContradictionPair(
                claim_a=str(evidence) if evidence else name_a,
                claim_b=name_b,
                source_a=e.source_id,
                source_b=e.target_id,
            )
        )
    return pairs


@dataclass
class ProbeResult:
    """Outcome of probing a single HarnessDescriptor.

    Minimum viable shape for Cat 9b (call-through success). A future
    Cat 9a/9c implementation that involves a real model API will likely
    extend this with ``reply_text``, ``model_invoked``, ``context_used``
    etc. — treat this as a stable floor, not a frozen schema.
    """

    success: bool
    latency_ms: float = 0.0
    error: Optional[str] = None
    # Free-form for diagnostics; not parsed by the scorecard.
    output: Optional[str] = None


@dataclass
class HarnessDescriptor:
    """Declaration of one way an external caller can reach this memory system.

    Adapters return a list of these from ``get_harness_manifest()``. The
    ``kind`` field follows the spec v8 § Cat 9 taxonomy. For the current
    minimum-viable 9b, SME only needs ``probe_fn`` to do an end-to-end
    dry call and report success/failure.

    ``kind`` values:
      - ``"tool_call"``       — a generic tool-call surface (OpenAI
                                tool-calls, Anthropic tool-use, etc.)
      - ``"mcp_resource"``    — an MCP server method the client calls
                                over stdio/http
      - ``"claude_code_hook"``— a Claude Code hook (Stop, PreCompact,
                                UserPromptSubmit, SessionStart,
                                PreToolUse, PostToolUse)
      - ``"slash_command"``   — a harness-level slash command
      - ``"custom_action"``   — arbitrary shape; the adapter owns the
                                invocation semantics

    Future sub-tests (9a/9c/9d/9e/9f/9g) will need the schema/URI/hook-
    type info on top of ``probe_fn``; put them in ``properties`` for now.
    """

    name: str
    kind: str
    probe_fn: Callable[[], ProbeResult]
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    answer: str
    # The exact text the adapter would inject into the LLM prompt.
    # SME tokenizes this with tiktoken (cl100k_base) to compute Cat 7
    # metrics. Adapters cannot game the token count.
    context_string: str = ""
    retrieved_entities: list[Entity] = field(default_factory=list)
    retrieved_edges: list[Edge] = field(default_factory=list)
    retrieval_path: list[Any] = field(default_factory=list)
    contradictions: list[ContradictionPair] = field(default_factory=list)
    # If query() fails, set this instead of raising. SME distinguishes
    # "errored" from "answered wrong" in the scorecard.
    error: Optional[str] = None


class SMEAdapter(ABC):
    """Implement this for your database/memory system.

    Three required methods. Three optional.
    """

    # --- Required ------------------------------------------------------

    @abstractmethod
    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """Load the seeded test corpus.

        Returns a dict with at least:
            entities_created: int
            edges_created: int
            errors: list[str]
            warnings: list[str]
        """

    @abstractmethod
    def query(self, question: str) -> QueryResult:
        """Run a natural language query through the full pipeline.

        Must populate `context_string` with the exact text the adapter
        would send to the LLM.
        """

    @abstractmethod
    def get_graph_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        """Return the full graph state for topology analysis.

        For systems without a graph, return ([], []).
        """

    # --- Optional (have sensible defaults) -----------------------------

    def get_flat_retrieval(self, question: str) -> QueryResult:
        """Vector-only retrieval with no graph traversal.

        Used as Cat 7 Condition A. If not implemented, SME falls back to
        its built-in FlatBaseline adapter using the same corpus and
        embedding model.
        """
        raise NotImplementedError

    def get_ontology_source(self) -> dict:
        """Return ontology documentation for Category 8.

        Priority order:
          1. declared — declared schema, ONTOLOGY.md, typed tables
          2. readme   — documentation claims (pre-extracted YAML)
          3. inferred — no docs, analyze graph directly

        Returns:
            {'type': 'declared'|'readme'|'inferred',
             'schema': list,
             'documentation': str}
        """
        return {"type": "inferred", "schema": [], "documentation": ""}

    def get_contradiction_pairs(self) -> list[ContradictionPair]:
        """Return the contradictions the system surfaces about its store.

        Used by Category 3 (Contradiction Detection — The Dissonance).
        A system that explicitly tracks conflicting facts returns one
        ContradictionPair per detected conflict; a system that only
        retrieves (no contradiction model) returns ``[]`` and scores 0
        on Cat 3's structured-detection metric.

        The default implementation derives pairs from the graph snapshot
        — any edge whose type normalizes to ``contradicts`` becomes a
        pair. Adapters whose backend stores contradictions as typed edges
        get correct Cat 3 behaviour for free; adapters with a richer
        native contradiction API can override this.
        """
        entities, edges = self.get_graph_snapshot()
        node_names = {e.id: e.name for e in entities}
        return contradiction_pairs_from_edges(edges, node_names=node_names)

    def get_harness_manifest(self) -> list[HarnessDescriptor]:
        """Return the invocation surfaces this memory system exposes.

        Used by Category 9 (Harness Integration). Each descriptor describes
        one surface through which an external caller (a model, a hook, a
        tool) can reach the memory system. Systems that don't expose any
        harness surface — pure library APIs — return ``[]``.

        The current minimum-viable consumer (Cat 9b call-through success)
        only invokes ``probe_fn`` on each descriptor. Future sub-tests
        will need the ``kind`` + ``properties`` metadata to run model
        calls and compose with Cat 7.
        """
        return []

    def get_introspection_report(self) -> Optional[dict]:
        """Return the system's OWN declared-vs-effective ontology drift.

        Used by Category 8's introspection sub-test. Most memory systems
        have no self-report capability — they cannot tell you what entity
        types / edge vocabulary they actually hold versus what they claim —
        so the default returns ``None`` and Cat 8 reports introspection 0.0.
        That zero is meaningful: it says "this system can't audit its own
        ontology," distinct from "this system has a bad graph."

        Adapters whose backend exposes a self-report surface override this.
        The expected shape (see ``MemPalaceDaemonAdapter`` / palace-daemon's
        ``GET /ontology``)::

            {
              "declared":  {entity_types, edge_types, hall_vocabulary, ...},
              "effective": {edge_types, entity_kinds, entities, ...},
              "drift":     {declared_edge_types_present,
                            declared_edge_types_absent,
                            entity_kinds_undeclared,
                            structure_claim, structure_observed,
                            drift_score, ...},
            }

        ``score_cat8`` credits introspection by counting which of the three
        capability dimensions (effective entity-kind reporting, effective
        edge-vocabulary reporting, declared-vs-effective drift reporting) the
        report actually populates — so the score reflects a genuine
        capability, never a hand-set number.
        """
        return None

    # --- Lifecycle -----------------------------------------------------

    def close(self) -> None:
        """Release any resources. Safe to call multiple times."""
