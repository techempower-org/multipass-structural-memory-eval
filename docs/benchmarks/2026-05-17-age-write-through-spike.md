# AGE write-through spike — graph signal adds +9pp R@5 over vector

**Date:** 2026-05-17
**Branch:** `feat/rlm-adapter`
**Commits:** `28ae3f1` (spike), `abe4095` (chunked follow-up)

## Result

| Mode | R@5 | hits | Δ vs vector_only |
|---|---:|---:|---:|
| vector_only (file-level pgvector + MiniLM base) | 0.1850 | 37/200 | — |
| graph_only (AGE entity-overlap) | **0.2350** | 47/200 | **+5.0pp** |
| fusion (RRF combine) | **0.2750** | 55/200 | **+9.0pp** |

**The graph signal isn't noise — it adds 5pp on its own AND composes with vector for an additional 4pp on top.** Smallest-possible test bed (regex extractor, file-level corpus, vanilla pgvector) and the architectural validation still lands.

## Setup

- **Corpus:** 238 files (77 `.md` + 161 `.py`) from `techempower-org/mempalace` HEAD. One drawer per file, file-shaped IDs to match the n=200 git-derived probe set's `expected_sources` shape.
- **Extractor:** Two-pass regex — capitalized proper nouns + technical identifiers (hyphenated lowercase, version strings, `owner/repo` handles, `FT-XXX` patterns).
- **AGE write-through:** Each drawer's entities `CREATE` `:Entity` nodes and `[:MENTIONED_IN {count: N}]` edges in the `sme_spike_kg` AGE graph on postgres.
- **Modes:**
  - `vector_only` — pgvector cosine against MiniLM-L6-v2 base embeddings (no FT).
  - `graph_only` — Cypher `MATCH (e:Entity {name})-[:MENTIONED_IN]->(d:Drawer)` against entities extracted from the query.
  - `fusion` — Reciprocal Rank Fusion of vector + graph rankings.

## Why the absolute numbers are low

`vector_only = 0.185` is well below the chunked-substrate daemon's 0.280 R@5 on the same corpus. The reason is intentional: this spike used **file-level** embeddings (one vector per file) instead of the daemon's paragraph-chunked vectors. The spike's purpose was to test whether the *graph layer adds signal*, not to compete with chunked retrieval. The relative graph contribution is what matters here; the absolute numbers shouldn't be compared to chunked baselines.

A follow-up bench (`age_chunked_bench.py`, committed `abe4095`) repeats the experiment with paragraph-chunked vectors to test whether the +9pp graph lift survives at production-tier vector quality.

## AGE Cypher dialect gaps discovered

The Apache AGE 1.6.0 implementation has nontrivial gaps from what a Neo4j-shaped programmer would expect. Three that bit during this spike, all needing workarounds:

1. **No multi-column `RETURN` inside `cypher()`** — `RETURN d.id, r.count` errors with "syntax error at end of input" when wrapped in `cypher('graph', $$ ... $$) AS (col1 agtype, col2 agtype)`. *Workaround:* `RETURN d.id` only; run a separate query for any second projection.
2. **No list literals** — `RETURN [a, b]` errors with "syntax error at or near ']'". *Workaround:* return columns one at a time.
3. **No `MERGE ... ON CREATE SET` / `ON MATCH SET`** — the standard upsert idiom fails. *Workaround:* truncate the graph fresh and use plain `CREATE` in three passes (entities → drawers → edges); skip MERGE entirely on bulk ingest.

These workarounds are encoded in `sme/adapters/postgres_age_ingest.py`. They constrain the *effective* expressiveness of AGE for retrieval queries to a narrower subset than the docs imply. Any upstream architectural conversation about adopting write-through to AGE in mempalace needs to plan around them.

## Mechanism on a worked example

Probe: `"Post-mortem section in pgvector-cutover-runbook"`.

- **Regex extractor pulls:** `pgvector-cutover-runbook` (TECH_IDENT), `post` (PROPER_NOUN).
- **Graph query for `pgvector-cutover-runbook`:** one match — `CHANGELOG.md` mentions the filename literally.
- **The target file `pgvector-cutover-runbook.md`** uses natural-language prose like "Pgvector cutover runbook" — never the hyphenated identifier — so it has no graph edge to the query entity.
- **Vector retrieval** finds the target file via semantic similarity; **graph retrieval** finds `CHANGELOG.md` via literal mention.
- **Fusion** combines both candidate sets, ranks by RRF.

This is the right behavior pattern for a graph layer: prioritize *literal coreference* (where named entities appear) rather than re-implementing semantic similarity. The signals compose precisely because they index different things.

## Architectural implication

Validates the "write-through to AGE on every drawer + read-side fusion" architecture at the smallest possible test bed. The path forward:

1. Fill out `mempalace.knowledge_graph_age.KnowledgeGraphAGE` — currently has `add_triple`, `query_triples`, `stats`, `clear`, `_run_cypher`. Missing vs. the SQLite KG: `add_entity`, `invalidate`, `query_entity`, `query_relationship`, `timeline`, `seed_from_entity_facts`.
2. Add write-through middleware in `mempalace.backends.postgres._insert_rows`: every drawer write extracts entities + calls `add_triple`. Gated by config; configurable extractor (regex → spacy → LLM).
3. Add palace structure as AGE nodes: `Wing → CONTAINS → Room → CONTAINS → Drawer → MENTIONS → Entity`. The `/graph` SQL-aggregation today (39K co-membership edges) becomes Cypher-queryable.
4. Production backfill of the 274K-drawer canonical palace.
5. Read-side fusion in palace-daemon `/search`.
6. Expose via MCP — `mempalace_walk_palace(start_wing, depth)` makes the metaphor real for agents.

## Cross-reference

- Atakan ran a similar entity-graph baseline against 500q LongMemEval today, hitting R@1=0.354 / R@5=0.406 with in-memory entity extraction (no AGE). Both reach the same finding: entity-graph adds real signal, *independently of substrate*. See [`MemPalace/mempalace/discussions/1384#discussioncomment-16951344`](https://github.com/MemPalace/mempalace/discussions/1384#discussioncomment-16951344) for his full per-category breakdown and the router-vs-replacement framing.

## Artifacts

- Bench script: `scripts/age_writethrough_bench.py`
- Adapter: `sme/adapters/postgres_age_ingest.py`
- Extractor: `sme/extractors/regex.py`
- Results: `baselines/age_writethrough_spike_2026-05-17.json`
- Chunked follow-up bench (pending result): `scripts/age_chunked_bench.py`
