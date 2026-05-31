# Field retrieval-cluster adapters (`ai-memory`, `agentmemory`)

Two not-yet-benched **retrieval-only** field systems from epic #234's
"(a)-class" cluster (Nebula's scoping doc): both are **$0, fully local, no
write-time LLM, no cloud key**, and both publish **R@5** on LongMemEval — SME's
headline metric — so they slot directly against the mempalace / OMEGA /
postgres_ingest R@5 cells with no metric mixing.

Both adapters subclass `SMEAdapter`, talk to a **local HTTP daemon**, ingest one
corpus row per LongMemEval haystack session tagged with its `session_id`, and
surface that `session_id` on every retrieved `Entity`. The shared runner
`scripts/run_longmemeval_field.py` then computes **session-level hit@K** against
each question's `expected_sources_session_level()` — the identical topology used
by `scripts/run_longmemeval_omega.py` and the mempalace-daemon's
`session_id → drawer_id` map, so the field rows are apples-to-apples with the
existing matrix on the **same strat150 subset** (6 question types × 25, n=150,
`--content-rules upstream-exact`). Both return `([], [])` from
`get_graph_snapshot` (retrieval-only, like `flat_baseline` / `postgres_ingest`).

## ai-memory (`sme/adapters/ai_memory.py`)

[ai-memory](https://github.com/alphaonedev/ai-memory-mcp) (alphaonedev), Rust,
Apache-2.0. Pure SQLite **FTS5 + local MiniLM** (`all-MiniLM-L6-v2`, 384-dim,
Candle). No write-time LLM (the optional LLM query-expansion is a `smart+` tier
feature requiring Ollama and stays OFF at the default `semantic` tier).

### Provision ($0 / local)

```bash
cargo install ai-memory                      # ~10 min (heavy crypto/onnx deps)
ai-memory serve --host 127.0.0.1 --port 9077 --db <isolated.db>
# config ~/.config/ai-memory/config.toml: tier="semantic" + mini_lm_l6_v2
# (semantic is the default tier; serve binds plain HTTP, no auth)
```

### Adapter contract (verified against v0.6.4)

| op | endpoint | notes |
|----|----------|-------|
| reset | `POST /api/v1/forget` `{namespace}` | per-question wipe of the working namespace |
| ingest | `POST /api/v1/memories/bulk` (array, cap 1000) | `session_id` stored in BOTH `metadata.session_id` and `title` |
| query | `POST /api/v1/recall` `{context, limit, namespace}` | query field is **`context`** NOT `query`; resp `{"memories":[{...,"score","metadata"}]}` |

Recall is a "hybrid FTS5 + semantic blend" with **no mode toggle**. recall
mutates the DB (auto-promote/touch) — harmless here because the namespace is
forgotten before the next ingest.

### Measured R@5 (strat150, session-level)

```
category          n     R@1     R@5    R@10
cat_1 (single)   75  74.67%  85.33%  85.33%
cat_2c           25  76.00%  96.00%  96.00%
cat_3_partial    25  96.00% 100.00% 100.00%
cat_6            25  84.00% 100.00% 100.00%
overall         150  80.00%  92.00%  92.00%
```

**R@5 = 0.92** (0 adapter errors). vs the project's **published 0.978** — a
defensible 5.8pp gap: different subset, `upstream-exact` content rendering, and
per-question forget/reload isolation. Single-session questions (cat_1, 85.3%)
are where it loses ground. Baseline:
`baselines/longmemeval_s_strat150_ai_memory_2026-05-31.json`.

CLI alias: `--adapter ai-memory` (`ai_memory`, `aimemory`), accepts `api_url`,
`namespace`, `tier`, `n_results`, `api_timeout`, `read_only`.

## agentmemory (`sme/adapters/agentmemory.py`)

[agentmemory](https://github.com/rohitg00/agentmemory) (rohitg00), Node/TS,
Apache-2.0. **Hybrid BM25 + vector + graph RRF** over *observations*; local
MiniLM via `@xenova/transformers`; runs on an `iii-engine` runtime
(`ws://localhost:49134`), REST on `:3111`. Auto-compress LLM hook OFF by default.

### Provision ($0 / local)

```bash
npm install -g @agentmemory/agentmemory     # ~1.3 GB (bundles MiniLM)
agentmemory init                            # writes ~/.agentmemory/.env
# MUST set EMBEDDING_PROVIDER=local in .env, else it falls back to BM25-ONLY
# (no vector leg). AGENTMEMORY_AUTO_COMPRESS=false (default) => synthetic
# (no-LLM) compression. No LLM key => noop mode (synthetic compression).
agentmemory --port 3111
```

### Adapter contract (verified against v0.9.24 source)

| op | endpoint | notes |
|----|----------|-------|
| ingest | `POST /agentmemory/observe` `{hookType:"prompt_submit", sessionId, project, cwd, timestamp, data:{prompt}}` | one observation per chunk |
| query | `POST /agentmemory/search` `{query, limit, project, format:"compact"}` | resp `{results:[{obsId, sessionId, title, score}]}` |

Two load-bearing source facts that shape the adapter:

1. **No-LLM synthetic compression truncates each observation's searchable body
   to ~400 chars** (`compress-synthetic.ts buildSyntheticCompression`:
   `narrative = truncate(prompt, 400)`, title = `"observation"` for a
   `prompt_submit`). The published 95.2% R@5 was measured with agentmemory's own
   compression pipeline. To keep the full session text searchable WITHOUT an
   LLM, the adapter **chunks each document to ≤380 chars**, one observation per
   chunk, all tagged with the same `sessionId`. Session-level R@K is unaffected
   because hits carry `sessionId`.
2. **`smart-search` has NO project filter** (it searches the global observation
   index) — so it can't isolate per-question haystacks. `mem::search` (the
   `/agentmemory/search` endpoint) is the only retrieval path that post-filters
   hits by `project`. The adapter uses `/search` with the rotated project name;
   project-rotation + the project search filter is the per-question isolation
   mechanism (the agentmemory analogue of the daemon's per-question wing).

### Status: provisioned + adapter verified; full bench THROUGHPUT-WALLED

The adapter is correct and fast on small loads (a 5-session ingest + query is
**instant**, `session_id` round-trips, project-filtered isolation works — see
`tests/test_field_retrieval_adapters.py`). But a full LongMemEval question is
~48 large sessions → ~70 chunked `observe` calls, and the per-observation
`iii-engine` ingest of large `upstream-exact` sessions crawls (~0.15–0.3
observes/s) and intermittently wedges as the observation index grows within a
question. Extrapolated, the full strat150 (~7,200 sessions) is ~15+ hours and
unstable — past the bounded-effort threshold (flag-don't-thrash, epic #234).

This is itself a finding: agentmemory's hook-lifecycle / per-observation REST
ingest is not built for bulk corpus loading at LongMemEval scale. A bounded
reduced-n R@5 reading is recorded where it completed; see the baseline JSON if
present (`baselines/longmemeval_s_*_agentmemory_2026-05-31.json`).

CLI alias: `--adapter agentmemory` (`agent-memory`, `agent_memory`), accepts
`api_url`, `project`, `n_results`, `api_timeout`, `include_lessons`, `read_only`.

## Tests

`tests/test_field_retrieval_adapters.py` — 13 HTTP-mocked unit tests (no live
daemon): construction, per-question reset, request-shape assertions
(ai-memory's `context` field; agentmemory's `/search` + project filter),
session_id round-trip, chunking, empty-result handling, and empty graph
snapshot for both.
