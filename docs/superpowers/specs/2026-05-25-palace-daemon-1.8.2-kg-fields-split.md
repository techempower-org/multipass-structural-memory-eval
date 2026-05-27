# palace-daemon 1.8.2 — `/graph` KG fields split (heads-up for SME)

**Status:** Heads-up. SME adapter currently works against 1.8.2 (tests
don't read the removed keys), but the meaning of `kg_triples` has
changed — most of the graph SME used to ingest is now under a new
field name.

**Date written:** 2026-05-25
**Source:** palace-daemon commit `3c0135f` (origin/main on
`techempower-org/palace-daemon`)

## What changed in `/graph`

palace-daemon 1.8.0–1.8.1 labelled the Drawer→Entity `MENTIONS` edges
as "triples". A triple is an entity→entity *semantic fact* (the
`RELATION` label); a mention is a *provenance link* from a drawer
to an entity it names. They are not the same thing, and the live
corpus exposes this — ~1 RELATION row vs. ~5.66M MENTIONS edges. The
old field overstated the size of the actual knowledge graph by six
orders of magnitude.

### Response shape, before vs after

| Field | 1.8.0–1.8.1 | 1.8.2 |
|---|---|---|
| `kg_entities` | entities | entities (unchanged) |
| `kg_triples` | **all MENTIONS edges (~5.66M source rows)** | **RELATION edges only (~1 row in current corpus)** |
| `kg_mentions` | — (didn't exist) | **drawer→entity MENTIONS rows (new)** |
| `kg_stats.entities` | int | int (unchanged) |
| `kg_stats.triples` | MENTIONS count (mislabelled) | RELATION count (real) |
| `kg_stats.mentions` | — | MENTIONS count (new) |
| `kg_stats.current_facts` | int | **gone** |
| `kg_stats.expired_facts` | int | **gone** |
| `kg_stats.relationship_types` | derived from MENTIONS | `["RELATION", "MENTIONS"]` when both populated |

`?limit=N` now applies ×1 to entities, ×2 to triples, ×2 to mentions.

## SME impact assessment (as of 2026-05-25)

SME's `MemPalaceDaemonAdapter` does NOT read the dropped
`current_facts` / `expired_facts` keys — checked via grep across the
SME tree:

```
$ grep -rn 'current_facts\|expired_facts' \
    ~/Projects/multipass-structural-memory-eval
# (no matches)
```

So no key-error breakage. But: SME's `test_graph_mapping.py` builds
edges from `kg_triples`. Before 1.8.2 that fed ~5.66M MENTIONS rows
into the structural map. After 1.8.2, `kg_triples` carries ~1
RELATION row, and the bulk of the graph is in `kg_mentions` — which
the adapter currently ignores.

**This is a silent semantic change, not a crash.** SME runs that
depended on the dense MENTIONS layer for connectivity (Cat 4/5/8/9
structural diagnostics?) will see a near-empty graph after pointing
at a 1.8.2+ daemon, and won't fail loudly — they'll just produce
quiet, useless scores.

## When SME next touches palace-daemon integration

Pick one of these (lightest-touch first):

1. **Map both fields to edges.** Concatenate `kg_triples ∪
   kg_mentions` in the graph-mapping layer. Cheapest: SME's
   structural diagnostics get the same edge density they had before
   1.8.2, plus the (currently-tiny) real RELATION facts.

2. **Treat the two as semantically different.** RELATION edges
   become "semantic facts" (high weight, sparse). MENTIONS edges
   become "co-occurrence provenance" (lower weight, dense). Two
   separate edge types in the structural map. Better for Cat
   diagnostics that distinguish facts from mentions.

3. **Drop MENTIONS entirely.** If SME's structural model was always
   meant to be entity→entity, the 1.8.2 split exposes that the
   pre-1.8.2 graph was actually drawer→entity and the diagnostics
   were measuring the wrong thing. Walk away from MENTIONS, accept
   the ~1-RELATION-row truth, and either wire a real RELATION
   extraction pipeline upstream of the daemon or shrink scope.

The lone RELATION row in the current corpus is literally
`(A)-[r]->(B)` — an AGE setup placeholder. The semantic-triples
extraction pipeline upstream of mempalace isn't wired yet; until it
is, `kg_triples` will be a placeholder-only field.

## Verification

```bash
set -a; source ~/.config/palace-daemon/env; set +a
curl -sS -H "X-Api-Key: $PALACE_API_KEY" \
    https://your-palace-host/graph?limit=1 | jq '.kg_stats'
# {
#   "entities": 267544,
#   "triples": 1,
#   "mentions": 5660070,
#   "relationship_types": ["RELATION", "MENTIONS"]
# }
```

## Cross-reference

- palace-daemon changelog: `CHANGELOG.md` 1.8.2 entry
- palace-daemon docs: `docs/graph-endpoint.md` (1.8.2 banner added)
- palace-daemon commit: `3c0135f`
- Roadmap drawer: `palace_daemon/planning` (id `1e3d68fae0f7ae5e9908119e`)
