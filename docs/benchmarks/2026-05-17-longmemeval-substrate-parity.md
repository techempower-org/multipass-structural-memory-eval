# LongMemEval substrate-floor: postgres+pgvector parity with upstream chromadb

**Date:** 2026-05-17
**Branch:** `feat/rlm-adapter`
**Result:** **R@5 = 0.9660 exact match** between upstream chromadb and our postgres+pgvector backends.

## Setup

Same dataset, same embedding model, same scoring protocol:

| | Upstream chromadb (reference) | Our postgres+pgvector (this run) |
|---|---|---|
| Dataset | `longmemeval_s_cleaned.json` (500q) | same |
| Embedding | ChromaDB default = `all-MiniLM-L6-v2` (384-dim) | same (via `mempalace.backends.postgres._embed`) |
| Distance | cosine | cosine (`vector_cosine_ops` index) |
| Ingest content | concatenated user turns per session (no metadata, no assistant turns) | same |
| Scoring | recall_any@5 on `answer_session_ids` | same |
| Reproduction tool | `memorypalace/benchmarks/longmemeval_bench.py --mode raw` | `scripts/lme_substrate_parity_bench.py` |

## Results (per-category R@5)

| Question type | n | R@5 (both backends) |
|---|---:|---:|
| knowledge-update | 78 | 1.0000 |
| multi-session | 133 | 0.9925 |
| single-session-assistant | 56 | 0.9643 |
| single-session-preference | 30 | 0.9667 |
| single-session-user | 70 | 0.9143 |
| temporal-reasoning | 133 | 0.9474 |
| **OVERALL** | **500** | **0.9660** |

Both backends agree to four decimal places on every question_type category, including overall.

## Why this matters

The MemPalace fork (`techempower-org/mempalace` 3.3.5) migrated the substrate from ChromaDB to PostgreSQL+pgvector+Apache AGE between 2026-05-13 and 2026-05-14 (see `MemPalace/mempalace#665`, fork-side rebase in `techempower-org/mempalace#17`). This benchmark establishes that the swap is **substrate-equivalent** on the standard memory-systems retrieval test — i.e., no recall regression from the backend migration alone.

## Methodology note (gap discovered + closed)

An earlier reading via SME's standard `cross_validate_longmemeval.py` harness reported R@5 = 0.9440 on the same data + same postgres adapter (-2.2pp vs upstream). Investigation showed the gap was 100% attributable to SME's loader rendering each session as a markdown file with YAML frontmatter, `## role` headers, and **both user and assistant turns** — which produces different embeddings than upstream's "user-turns-only, no metadata" protocol.

When the ingest is matched exactly to upstream's content rules (this run), every per-question ranking is byte-identical, including the questions that miss top-5 in both backends. **Postgres+pgvector with MiniLM produces the same retrievals as ChromaDB with MiniLM.**

The 2.2pp loader-cost finding is a separate signal — useful as a baseline for "what does SME's default materialization buy or cost." It's not a backend-swap regression.

## Artifacts

- Bench script: `scripts/lme_substrate_parity_bench.py`
- Postgres result: `baselines/lme_substrate_postgres_2026-05-17.json` (per-question rankings + summary)
- Upstream reference: `baselines/lme_substrate_chroma_upstream_2026-05-17.jsonl` (raw output from upstream bench)
- Adapter: `sme/adapters/postgres_ingest.py` (per-question TRUNCATE+upsert wrapper around `mempalace.backends.postgres.PostgresCollection`)

## Next legs

This is the **substrate-floor** reading (35.a). Pipeline-ceiling (35.b familiar over LongMemEval) and orchestrator-mediated (35.c RLM over LongMemEval) build on this validated foundation. The 35.a number is what every other LongMemEval reading on the fork should be compared against — equality validates a substrate-equivalent pipeline; a gap quantifies what that pipeline adds or loses vs the raw substrate.
