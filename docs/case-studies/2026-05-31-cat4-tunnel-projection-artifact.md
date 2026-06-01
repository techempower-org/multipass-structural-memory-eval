# Case Study 1 — The tunnel-projection measurement artifact (Cat 4)

**The framework catching itself.** SME's Cat 4 reading on MemPalace's
knowledge graph said "severe edge-type monoculture." Investigating the
finding revealed the monoculture was an artifact of *how SME sampled the
graph*, not a property of the graph — the single most important honesty
result of the campaign.

- **Category:** Cat 4 — The Threshold / Ingestigation (edge-type entropy)
- **System:** MemPalace (techempower-org fork), production `familiar` palace
- **Scale:** ~1.9M `:RELATION` edges, ~1.16M entities
- **Status:** RESOLVED — `kg_only` measurement mode + a genuine
  underlying defect fixed; cross-validated from three vantage points.

---

## The finding

The first Cat 4 reading, taken over a capped `/graph` HTTP projection:

> **98.98% one edge type, normalized entropy 0.020.**
> Edge-type monoculture — a single generic relation absorbs nearly
> every edge.

Read at face value, this says the extractor collapsed the entire typed
vocabulary into one relation — a serious ingestion defect.

## The fix (two layers)

The finding was **half artifact, half real defect**, and untangling them
is the whole lesson:

1. **The measurement artifact (the larger half).** The `/graph` endpoint
   returned a capped projection that mixed the *structural scaffold*
   (wing/room/`tunnel` navigation edges) with the *semantic KG*. The
   scaffold generates `tunnel` edges **combinatorially** — O(k²) per room
   shared across wings — so popular rooms alone produced 167,645 of
   169,372 sampled edges (98.98%). The entropy was being computed over a
   surface dominated by a combinatorial scaffold, not the knowledge graph.
   **Fix:** measure the real `:RELATION` set directly (`kg_only` mode,
   mempalace#147 / PR #210), and feed the scorer the *exact* population
   distribution via `edge_type_counts_override` (the daemon's full
   `MATCH ()-[r:RELATION]->() RETURN r.relation_type, count(*)`
   aggregation) instead of a capped sample.

2. **The genuine defect underneath (the smaller, real half).** Once the
   scaffold was excluded, the true KG still had **55% of `:RELATION`
   edges in an `other` fallback bucket** — a real predicate-extraction
   gap. **Fix:** an expanded deterministic predicate pre-pass
   (mempalace#336) relabeled the `other` sink 55% → 27%, plus a
   `--drop-code-tokens` DELETE of 48,135 junk edges.

This is precisely the shape of the new `remediation` field: the Cat 4
report now ships a "check whether the dominant type is a generic fallback
bucket; add deterministic predicate rules; report WITH the type count"
fix alongside the monoculture finding.

## The re-run (before → after)

| Reading | Capped projection (artifact) | Real full KG (FINAL) |
|---|---|---|
| Normalized edge-type entropy | **0.020** | **0.645** |
| Dominant edge type | one type @ 98.98% | `other` @ 26.83% |
| Distinct edge types | (swamped) | 40 |
| Verdict | "severe monoculture" | "diverse vocabulary" |

The companion Cat 5 / Cat 8 cells flipped the same way (61.87% giant
component, modularity 0.796 — see the synthesis §5.4). The artifact said
*monoculture / fragmented / flat*; the truth is *diverse / connected /
hierarchical*.

## Verification

Three independent vantage points make the corrected numbers trustworthy
(`docs/benchmarks/2026-05-31-cat458-real-kg-crossvalidation.md`):

1. A direct-cypher full-graph aggregate reproduced the pre-fix `other`
   55.05% / entropy 0.3402 **to the digit**.
2. The corrected numbers were verified read-only against the live AGE graph.
3. The honesty cut both ways: deleting junk edges *raised* isolates
   20.4% → 22.2%, because ~20k entities whose only edge was junk became
   honestly isolated — the fix did not flatter the graph.

## The lesson

**Validate that the adapter reads the real graph before trusting any
structural reading.** A capped or projection-based snapshot can
manufacture a defect that the underlying data doesn't have. Cat 4 in
particular is **ontology-granularity-sensitive** (synthesis §5.5): the
same graph re-typed flat → moderate → fine moves normalized entropy
0.000 → 0.842 → 0.856, so the 0.020 → 0.34 → 0.645 progression is partly
the *real* de-monoculturing of a genuine defect and partly the metric
tracking a changed vocabulary. **Always report Cat 4 with the entity/edge
type counts**, and compare systems on Cat 5's topology (which the
sensitivity sweep shows is byte-identical under re-typing) instead.

**Artifacts:** `docs/benchmarks/2026-05-31-cat458-real-kg-crossvalidation.md`,
`baselines/mempalace_cat4_realkg_*_2026-05-31.json`, synthesis §5.4–§5.5.
