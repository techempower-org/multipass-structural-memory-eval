"""Shared /graph payload to (Entity, Edge) mapping.

palace-daemon's GET /graph returns a single payload shape (wings,
rooms, tunnels, kg_entities, kg_triples, kg_stats). Both the
MemPalaceDaemonAdapter and the FamiliarAdapter consume this shape.
Familiar's GET /api/familiar/graph proxies the daemon's response
unchanged (with a 5-minute cache), so both adapters share this
projection function rather than re-implementing it.

Extracted verbatim from MemPalaceDaemonAdapter._project_graph 2026-04-26.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sme.adapters.base import Edge, Entity, annotate_superseded_edges


def project_graph(
    body: dict[str, Any], *, kg_only: bool = False
) -> tuple[list[Entity], list[Edge]]:
    """Turn the daemon's /graph response into (entities, edges).

    Mirrors the wing/room/tunnel projection in
    ``sme.adapters.mempalace.MemPalaceAdapter.get_graph_snapshot``,
    minus drawer-level surface (impractical at 151K-drawer scale
    through the HTTP API).

    ``kg_only`` (default False) controls which graph the structural
    categories measure:

    * **False** — the full projection: wings + rooms + tunnels +
      member_of structural edges *plus* the KG layer. Correct for
      callers that want the whole palace surface (e.g. a connectivity
      overview).
    * **True** — the **real knowledge graph only**: KG entities and
      their entity→entity ``RELATION`` edges (``kg_triples``), with the
      wing/room/tunnel/member_of *structural projection excluded
      entirely*. This is the honest substrate for Cat 4 (edge-type
      monoculture), Cat 5 (isolates / fragmentation) and Cat 8
      (modularity / type coverage).

    Why ``kg_only`` exists (the #147 measurement fix): the daemon's
    ``/graph`` caps the KG sample at ``2*limit`` triples (default
    1000) but always returns the *full* structural projection. On the
    live palace the structural tunnel + member_of edges then swamp the
    tiny KG sample — 98.98% of edges in the default snapshot are
    structural, not semantic — so Cat 4/5/8 end up measuring the
    wing/room scaffold instead of the 1.92M-edge RELATION graph. The
    structural readings (modularity ≈ 0.009, edge monoculture, 2-of-6
    type coverage) were artifacts of that mix. Pulling ``/graph`` at a
    high ``limit`` and projecting ``kg_only=True`` measures the real KG,
    whose ``relation_type`` distribution is in fact a healthy tail
    (other / contains / depends_on / created_by / is_a / uses / …).

    The CTE-bounded RELATION read on the daemon side (``kg_reader``)
    survives at ``limit=5000`` where a raw Cypher ``MATCH`` walk OOMs,
    so the caller raises the limit rather than walking the graph.
    """
    kg_ents: list[dict] = body.get("kg_entities") or []
    kg_trips: list[dict] = body.get("kg_triples") or []

    entities: list[Entity] = []
    edges: list[Edge] = []

    # Structural projection (wings / rooms / tunnels / member_of) — skipped
    # under kg_only so Cat 4/5/8 measure the real KG, not the scaffold.
    if not kg_only:
        wings: dict[str, int] = body.get("wings") or {}
        rooms_by_wing: list[dict] = body.get("rooms") or []
        tunnels: list[dict] = body.get("tunnels") or []

        # Wings
        for wing in sorted(wings):
            entities.append(
                Entity(
                    id=f"wing:{wing}",
                    name=wing,
                    entity_type="wing",
                    properties={"_table": "wing", "drawer_count": wings[wing]},
                )
            )

        # Rooms — collect wings-per-room across the per-wing lists
        room_wings: dict[str, set[str]] = defaultdict(set)
        room_count: dict[str, int] = defaultdict(int)
        for entry in rooms_by_wing:
            wing = entry.get("wing", "")
            for room, n in (entry.get("rooms") or {}).items():
                if not room or room == "general":
                    continue
                room_wings[room].add(wing)
                room_count[room] += int(n or 0)

        for room in sorted(room_wings):
            wings_list = sorted(room_wings[room])
            entities.append(
                Entity(
                    id=f"room:{room}",
                    name=room,
                    entity_type="room:untyped",
                    properties={
                        "_table": "room",
                        "wings": wings_list,
                        "drawer_count": room_count[room],
                    },
                )
            )
            for wing in wings_list:
                edges.append(
                    Edge(
                        source_id=f"room:{room}",
                        target_id=f"wing:{wing}",
                        edge_type="member_of",
                        properties={
                            "_table": "structural",
                            "drawer_count": room_count[room],
                        },
                    )
                )

        # Tunnels — wing<->wing for each shared room
        for t in tunnels:
            room = t.get("room", "")
            t_wings = sorted(t.get("wings") or [])
            for i, wa in enumerate(t_wings):
                for wb in t_wings[i + 1:]:
                    edges.append(
                        Edge(
                            source_id=f"wing:{wa}",
                            target_id=f"wing:{wb}",
                            edge_type="tunnel",
                            properties={
                                "_table": "structural",
                                "via_room": room,
                            },
                        )
                    )

    # KG layer
    for ke in kg_ents:
        ent_id = ke.get("id")
        if not ent_id:
            continue
        props = dict(ke.get("properties") or {})
        props["_table"] = "kg_entity"
        entities.append(
            Entity(
                id=f"kg:{ent_id}",
                name=ke.get("name") or ent_id,
                entity_type=f"kg:{ke.get('type') or 'unknown'}",
                properties=props,
            )
        )
    for tr in kg_trips:
        subj, obj = tr.get("subject"), tr.get("object")
        if not subj or not obj:
            continue
        edges.append(
            Edge(
                source_id=f"kg:{subj}",
                target_id=f"kg:{obj}",
                edge_type=tr.get("predicate") or "kg_related",
                properties={
                    "_table": "kg_triple",
                    "_created_at": tr.get("valid_from"),
                    "valid_to": tr.get("valid_to"),
                    "confidence": tr.get("confidence"),
                    "source_file": tr.get("source_file"),
                },
            )
        )

    # #147 — endpoint consistency under kg_only. The daemon samples
    # kg_entities (LIMIT N) and kg_triples (LIMIT 2N) *independently* off the
    # AGE label tables, so most RELATION endpoints fall outside the entity
    # sample. Measuring fragmentation (Cat 5) over the entity sample alone
    # would then count the bulk of nodes as isolates — a sampling artifact,
    # not a real disconnection. Synthesize a minimal node for every edge
    # endpoint missing from the entity sample so the kg_only graph is
    # self-consistent: every node carries at least one in-sample edge and
    # connectivity reflects the real RELATION wiring, not the sampling gap.
    # (Skipped under the full projection, where the structural scaffold
    # already supplies the node set the metrics expect.)
    if kg_only:
        present = {e.id for e in entities}
        for ed in edges:
            for endpoint in (ed.source_id, ed.target_id):
                if endpoint not in present:
                    present.add(endpoint)
                    entities.append(
                        Entity(
                            id=endpoint,
                            name=endpoint[3:] if endpoint.startswith("kg:") else endpoint,
                            entity_type="kg:entity",
                            properties={"_table": "kg_entity", "_endpoint_only": True},
                        )
                    )

    # Cat 6 plumbing: stamp the reserved ``_superseded_by`` property on
    # edges originating from a superseded entity, derived from edges whose
    # predicate normalizes to ``supersedes``. The daemon already projects
    # these predicates verbatim (kg_reader: predicate = relation_type), so
    # no backend change is needed — only this SME-side annotation.
    annotate_superseded_edges(edges)

    return entities, edges
