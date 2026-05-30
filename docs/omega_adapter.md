# OMEGA adapter (`sme/adapters/omega.py`)

First competitor adapter for the independent head-to-head
(`M0nkeyFl0wer/multipass-structural-memory-eval#178`). Runs
[OMEGA](https://pypi.org/project/omega-memory/) on SME's **identical**
corpus + reader + canonical judge, so the matrix stops citing OMEGA's own
leaderboard and starts measuring it under the same conditions as every
other substrate.

## What OMEGA is

OMEGA (`pip install omega-memory`, import name `omega`) is a local-only
persistent memory system for AI coding agents. It stores typed memories
in a SQLite database under `$OMEGA_HOME/omega.db` (default `~/.omega/`)
and serves retrieval via 384-dim `bge-small-en-v1.5` ONNX embeddings +
`sqlite-vec`, with an FTS5 full-text fallback. Memories carry an
`event_type` (`decision`, `lesson`, `error`, `summary`,
`user_preference`, …) and are linked into a typed `edges` graph
(`related`, `supersedes`, `contradicts`) by a background auto-relate
pass.

Verified against **omega-memory 1.4.15**.

## Install + run

```bash
# install the optional extra (adds omega-memory + sqlite-vec)
./venv/bin/pip install 'sme-eval[omega]'

# LongMemEval retrieval smoke (SME-only, no API key needed)
./venv/bin/python scripts/cross_validate_longmemeval.py \
  --dataset sme/corpora/longmemeval/data/longmemeval_oracle.json \
  --adapter omega --max-questions 3 --skip-judge \
  --out /tmp/omega_lme_smoke.json

# full E2E QA (reader + canonical judge) — needs OPENAI_API_KEY / a reader
./venv/bin/python scripts/cross_validate_longmemeval.py \
  --dataset sme/corpora/longmemeval/data/longmemeval_oracle.json \
  --adapter omega --reader-model gpt-4.1-mini
```

The CLI also registers the adapter under the `omega` alias
(`sme-eval … --adapter omega`), accepting `omega_home`, `db_path`,
`default_memory_type`, `n_results`, `read_only`.

## Tests

- `tests/test_omega_adapter.py` — mocked unit coverage (no install
  needed). The fake `omega` module mirrors the **verified** 1.4.x
  surface.
- `tests/test_omega_live.py` — `@pytest.mark.live` end-to-end smoke
  against the real package on an isolated `OMEGA_HOME`. Auto-skips when
  omega-memory is absent. Deselect with `-m "not live"`.

## API facts that are NOT obvious from the README

The adapter that previously shipped (PR #18) was written against a
*guessed* API — its docstrings literally said "return shape is not
documented." Driving the real package surfaced three load-bearing
corrections:

1. **`omega.query()` returns a formatted `str`**, not a list/dict — it's
   meant for an LLM to read ("Results: N\n## 1. [event_type] …"). The
   machine-readable path is **`omega.query_structured(text, limit=…)`**,
   which returns `list[dict]` with `id` / `content` / `event_type` /
   `relevance` / `strength` fields. The adapter uses `query_structured`;
   it falls back to parsing the string form only if `query_structured`
   is missing (older OMEGA). *Against the old guessed-list normaliser,
   every real query returned `NO_RESULTS`.*

2. **OMEGA resolves its store location from `OMEGA_HOME` (a directory
   env var), not a db-file path.** `OMEGA_DB_PATH` is ignored. The
   adapter sets `OMEGA_HOME` to an isolated dir before the first store
   write and restores the prior value on `close()`, so a benchmark run
   never pollutes the user's real `~/.omega`. `SQLiteStore` re-reads
   `OMEGA_HOME` each time it is constructed, so dropping OMEGA's cached
   store singleton (`omega.bridge.reset_memory()`) re-targets it.

3. **Schema field names**: the type column on the `memories` table is
   `event_type` (not `type`); retrieval scores are `relevance` /
   `strength` (not `score`). `get_graph_snapshot()` reads `event_type`
   so the typed vocabulary survives into SME's `Entity.entity_type`.

## Caveat: in-process re-isolation is racy

OMEGA caches a **process-global** `SQLiteStore` singleton and runs
auto-relate on a `daemon=True` thread. Re-binding it to a fresh
`OMEGA_HOME` repeatedly within a single Python process — e.g. building a
new adapter per question — can race: a prior adapter's background thread
can re-trip the singleton between our env-set and the next write, so a
`store()` occasionally lands in the wrong db.

Mitigations in the adapter:

- `_bind_store_to_home()` joins outstanding background threads, drops the
  singleton, rebuilds it, and **verifies** `store.db_path` matches ours,
  retrying a few times.
- `get_graph_snapshot()` reads through OMEGA's **own live store
  connection** (race-free for its own writes) rather than a parallel
  connection (a `mode=ro` second connection can't map the WAL `-shm` and
  transiently reads zero rows).

The **benchmark path is unaffected and deterministic**: the LongMemEval
runner rebuilds an adapter per question but only calls `query()`, which
reads through OMEGA's freshly-bound live store (verified 5/5 identical
`sme_recall` runs). The only place the race ever surfaced was two
ingest+snapshot lifecycles in one process, which is why the live test
exercises ingest → query → snapshot in a **single** adapter lifecycle
(verified 40/40 deterministic). Heavy multi-question topology benches
that need many snapshots in one process should run OMEGA in a subprocess
per sample.

## Embedding model availability

In a minimal env OMEGA's ONNX embedding model may not load; it logs a
"hash-fallback" warning and degrades to FTS5 keyword matching. Under
that mode a question whose tokens don't overlap the stored text returns
nothing — an OMEGA property, not an adapter bug. The head-to-head
benches should run with the ONNX model present (semantic retrieval) for
a fair comparison; the smoke test queries with in-corpus terms so it
passes in either mode.

## Follow-ups

- `…#178` — Hindsight (MCP) adapter: verify against the real
  `hindsight-client` surface (currently scaffolded like OMEGA was).
- `…#178` — Mem0-OSS adapter: same — verify `mem0ai` ingest/recall.
