"""Category 6: Temporal Reasoning — The Archive.

Scores supersession-chain integrity: does the system track which fact
replaced which, so a reader can tell the *current* state from the
*historical* one (spec v8 §6)? LongMemEval tests time-point queries; SME
adds the structured supersession channel — the reserved ``_superseded_by``
edge property (spec v8 §6 / §6b provenance).

Two readings:

  * **structural** — supersession completeness: of the seeded
    ``supersedes`` edges in the snapshot, how many resolve into a
    ``_superseded_by`` linkage that lets a reader drop the superseded
    framing. Also reports how many distinct entities are marked
    superseded (i.e. have a known replacement).
  * **flat** — the substring-recall floor. A flat retriever has no
    edges and no ``_superseded_by`` field, so structural completeness is
    0; carried so the headline is ``(structural − flat)``.

A ``supersedes`` edge ``A --supersedes--> B`` is *resolved* when B is
recorded as superseded by A (``_superseded_by`` stamped, via
``annotate_superseded_edges``). The completeness metric is resolved /
seeded supersedes edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sme.adapters.base import (
    Edge,
    Entity,
    is_supersedes_edge,
)


@dataclass
class Cat6Report:
    """Cat 6 scorecard. ``flat_completeness`` is 0 by construction; the
    headline is the ``(structural − flat)`` supersession-completeness
    delta."""

    seeded_supersedes_edges: int
    resolved_supersedes_edges: int  # those that produced a _superseded_by
    superseded_entities: int  # distinct entities with a known replacement
    completeness: float  # resolved / seeded
    flat_completeness: float = 0.0
    chains: list[list[str]] = field(default_factory=list)
    superseded_by: dict[str, str] = field(default_factory=dict)
    detection_level: str = "structured"

    @property
    def completeness_delta(self) -> float:
        return self.completeness - self.flat_completeness

    def to_dict(self) -> dict:
        return {
            "category": "cat_6_supersession",
            "seeded_supersedes_edges": self.seeded_supersedes_edges,
            "resolved_supersedes_edges": self.resolved_supersedes_edges,
            "superseded_entities": self.superseded_entities,
            "structural_completeness": self.completeness,
            "flat_completeness": self.flat_completeness,
            "completeness_delta": self.completeness_delta,
            "detection_level": self.detection_level,
            "superseded_by": self.superseded_by,
            "chains": self.chains,
        }


def _build_chains(superseded_by: dict[str, str]) -> list[list[str]]:
    """Reconstruct supersession chains from the ``superseded_by`` map.

    ``superseded_by[B] = A`` means A supersedes B. A chain is the longest
    path C ← B ← A (newest last). Returns chains with ≥ 2 nodes, oldest
    first, sorted for determinism.
    """
    # Reverse map: A supersedes -> [B, ...] (entities A replaces)
    replaced: dict[str, list[str]] = {}
    for older, newer in superseded_by.items():
        replaced.setdefault(newer, []).append(older)

    # Heads = entities never themselves superseded (the current state).
    superseded = set(superseded_by.keys())
    heads = [n for n in replaced if n not in superseded]

    chains: list[list[str]] = []
    for head in sorted(heads):
        # Walk back from the current state through what it replaced.
        chain = [head]
        frontier = sorted(replaced.get(head, []))
        seen = {head}
        while frontier:
            node = frontier.pop(0)
            if node in seen:
                continue
            seen.add(node)
            chain.append(node)
            frontier = sorted(replaced.get(node, [])) + frontier
        if len(chain) >= 2:
            # Reverse so oldest is first, current state last.
            chains.append(list(reversed(chain)))
    return chains


def score_cat6(
    entities: list[Entity],
    edges: list[Edge],
    *,
    flat_completeness: float = 0.0,
) -> Cat6Report:
    """Produce a Cat 6 scorecard from a structural graph snapshot.

    The snapshot's ``supersedes`` edges are the seeded ground truth.
    Resolution is read off the reserved ``_superseded_by`` property that
    ``annotate_superseded_edges`` stamps — so this scorer verifies the
    Cat 6 plumbing actually fired, not just that the edges exist.
    """
    seeded = [e for e in edges if is_supersedes_edge(e.edge_type)]
    n_seeded = len(seeded)

    # superseded_by[B] = A  (A supersedes B)
    superseded_by: dict[str, str] = {}
    resolved = 0
    for e in seeded:
        # annotate_superseded_edges records the target's superseder on the
        # supersedes edge as `_supersedes_target` / `_superseded_target_by`.
        replacement = e.properties.get("_superseded_target_by") or e.source_id
        target = e.properties.get("_supersedes_target") or e.target_id
        if replacement and target:
            superseded_by[target] = replacement
            resolved += 1

    # Also count any edge that carried a derived `_superseded_by` (an
    # outgoing claim of a superseded entity) as corroborating evidence
    # that the plumbing reached the claim layer, not just the chain edge.
    for e in edges:
        sb = e.properties.get("_superseded_by")
        if sb and e.source_id not in superseded_by:
            superseded_by[e.source_id] = sb

    completeness = (resolved / n_seeded) if n_seeded else 0.0
    chains = _build_chains(superseded_by)

    return Cat6Report(
        seeded_supersedes_edges=n_seeded,
        resolved_supersedes_edges=resolved,
        superseded_entities=len(superseded_by),
        completeness=completeness,
        flat_completeness=flat_completeness,
        chains=chains,
        superseded_by=dict(sorted(superseded_by.items())),
    )


def format_report(report: Cat6Report) -> str:
    lines = [
        "Cat 6 — The Archive (Temporal Supersession)",
        "═" * 56,
        "",
        f"  Seeded supersedes edges:      {report.seeded_supersedes_edges}",
        f"  Resolved (_superseded_by):    {report.resolved_supersedes_edges}",
        f"  Distinct superseded entities: {report.superseded_entities}",
        "",
        f"  Structural completeness:      {report.completeness:.2f}",
        f"  Flat-baseline completeness:   {report.flat_completeness:.2f}",
        f"  (structural − flat) delta:    {report.completeness_delta:+.2f}",
        "",
        "Reading",
        "─" * 56,
    ]
    if report.seeded_supersedes_edges == 0:
        lines.append(
            "  No seeded supersedes edges in this snapshot — Cat 6 is "
            "not exercised by this corpus/graph."
        )
        return "\n".join(lines)
    if report.completeness >= 0.99:
        lines.append(
            "  ● Full supersession plumbing: every seeded supersedes "
            "edge resolved into a _superseded_by linkage, so a reader "
            "can tell the current state from the historical one. The "
            "flat baseline has no edges and surfaces 0 — the entire "
            "reading is the structural channel."
        )
    elif report.completeness > 0:
        lines.append(
            "  ● Partial supersession plumbing: some supersedes edges did "
            "not resolve into _superseded_by. Check annotate_superseded_"
            "edges coverage on this snapshot."
        )
    else:
        lines.append(
            "  ● No supersession plumbing: the snapshot has supersedes "
            "edges but none resolved into the reserved _superseded_by "
            "field — the structured channel is not wired."
        )
    if report.chains:
        lines.append("")
        lines.append(f"  Reconstructed chains ({len(report.chains)}):")
        for chain in report.chains[:5]:
            lines.append("    " + " → ".join(chain) + "  (current: last)")
    return "\n".join(lines)
