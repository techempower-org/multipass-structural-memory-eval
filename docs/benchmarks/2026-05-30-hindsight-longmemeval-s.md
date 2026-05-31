# Hindsight on LongMemEval-S — verified adapter + the extraction-throughput finding

**Date:** 2026-05-30
**Author:** Iris (SME dream-team)
**Branch:** `bench/run-hindsight-longmemeval`
**Issue:** [techempower-org/multipass-structural-memory-eval#184](https://github.com/techempower-org/multipass-structural-memory-eval/issues/184) (sub-issue of M0nkeyFl0wer/multipass-structural-memory-eval#178 — independent multi-system head-to-head)

## Bottom line

Hindsight (vectorize-io) is the **second** competitor wired onto SME's harness
after OMEGA (#183). The adapter is now **verified against the real client + a
live server** and is runnable on our identical corpus + reader + canonical
judge. But the headline result of this investigation is **architectural, not a
QA number**:

> **Extraction-based memory systems are benchmark-throughput-bound.** Hindsight
> runs an LLM fact-extraction on *every session ingest* (~60–96 s/session here,
> local phi4 on CPU). LongMemEval-S averages ~48 sessions/question, so the
> strat150 subset = ~7,200 ingests ≈ **150 hours** to benchmark. A
> verbatim-first system (mempalace) ingests raw text at ~zero marginal compute.
> The two architectures differ by orders of magnitude in ingest/benchmark cost
> — a tradeoff the public leaderboards hide.

Hindsight self-reports **91.4%** LongMemEval QA on its own leaderboard. We did
**not** produce an our-harness QA number: a full strat150 run is infeasible
CPU-locally, and a tiny-`n` QA number would be misleading to publish (the
"don't conclude from partial-N" rule). The matrix row therefore keeps the
field's self-reported figure, annotated *"adapter verified + runnable; full
bench deferred on extraction throughput."*

## What we did produce

- **A verified, reconciled adapter** against the real `hindsight-client` 0.7.1
  (recall has no `top_k` → `max_tokens`/`budget`; Pydantic `RecallResponse` not
  a dict; `document_id` wired for session-level R@K). 22 mocked unit tests + 2
  live smoke tests, green.
- **A dedicated runner** (`scripts/run_longmemeval_hindsight.py`) +
  `--adapter hindsight` in the cross-validate harness, mirroring Solara's OMEGA
  runner (stratified subset, per-question isolation, canonical reader/judge,
  `--status` for detached runs).
- **An attempted n=12 indicative QA run — INVALIDATED by infra, not published.**
  The run ingested + scored 12 stratified LongMemEval-S questions through the
  live phi4-backed server. Partway through, the scratch Hindsight container was
  terminated (SIGTERM / exit 143 — a box cleanup, not OOM). The runner kept
  POSTing to a now-dead `localhost:8888`: questions 5–12 stored **zero** sessions
  ("Cannot connect to host"), and questions 1–4 were partial (30–40 ingest
  errors each). The resulting QA 0.0 / R@5 0.083 measures a **dead server
  mid-run, not Hindsight**, so it is discarded — publishing it would be the exact
  "benchmark a broken system" trap this investigation exists to avoid. (The early
  questions' partial ingests confirm the pipeline itself works; the live smoke
  tests are the clean proof.) A valid on-harness QA row needs a stable box for
  the full ~12 h, a GPU extractor, or a fast cloud extractor — see below.

## The phi4-vs-qwen extraction catch (headline framing, like OMEGA's bge catch)

Hindsight's `retain` runs an LLM to extract structured facts. **The extractor
must emit strict JSON.** Run with ollama `qwen3.5:4b`, every extraction failed
with a JSON parse error (the model emitted markdown prose), **0 facts were
stored, and recall silently returned nothing** — while `retain` still reported
`success=True`. A naive bench would have published ~0, measuring a crippled
extractor, not Hindsight.

Switched to `phi4` (valid JSON) and **verified before benchmarking**: a
paraphrased query ("What kind of *pet* does Maria have?" — "pet" never appears
in the stored "golden retriever" text) surfaces the right fact, and a
disjoint-vocab temporal query ("When is the *dog* seeing the *doctor*?") ranks
the vet-appointment fact #1 with "next Friday" resolved to an absolute date.
Real semantic + temporal extraction, confirmed live.

Page sentence: *"Hindsight's fact-extractor needs a model that emits strict
JSON; a smaller model silently extracted 0 facts (empty memory) — we verified
clean extraction + semantic/temporal recall before benchmarking."*

This is the second competitor where **"is it actually working?"** was the real
story (OMEGA shipped without its embedding model → FTS5 fallback; Hindsight ran
with too weak an extractor → 0 facts). SME's diagnostic posture in action.

## Disclosure (comparison-readiness §3.4)

- **System:** Hindsight `ghcr.io/vectorize-io/hindsight` v0.7.1 (Docker),
  embedded Postgres + pgvector, local `BAAI/bge-small-en-v1.5` embeddings
  (384-dim), local `ms-marco-MiniLM-L-6-v2` cross-encoder rerank. Extraction
  LLM: ollama `phi4` (local) — **distinct** from the SME reader/judge.
- **Dataset:** LongMemEval-S (`longmemeval_s_cleaned.json`), the same stratified
  subset definition as the mempalace + OMEGA baselines (`question_type`
  round-robin). Content rules `upstream-exact` (user turns only) — identical to
  the mempalace baseline.
- **Retrieval metric:** Hindsight session-level hit@K **via `document_id`**, and
  it is **extraction-mediated** — "did a fact *extracted from* the evidence
  session rank top-K", which is *softer* than mempalace's raw-chunk session R@K
  or OMEGA's raw-memory hit@K. Comparable in spirit, not identical. Not
  substring `sme_recall` (structurally ~0 under upstream-exact + extraction).
- **Reader / judge (when a QA run is feasible):** o4-mini reader + `gpt-5.3-chat`
  canonical type-specific judge — same as the mempalace + OMEGA rows.
- **Isolation:** per-question Hindsight `bank_id`; throwaway local Docker server;
  prod familiar / palace-daemon untouched throughout. Torn down after.

## To get a full QA row later (JP's cost call)

Point Hindsight's extractor at a fast cloud model (e.g. Azure `gpt-5.3-chat` via
`HINDSIGHT_API_LLM_PROVIDER`) instead of local phi4. This makes strat150
feasible overnight but (a) incurs ~7,200 reasoning-model extraction calls — a
real $ cost not run unsupervised — and (b) trades off the local-only isolation
posture. Deferred to a human greenlight. The runner is ready:

```bash
HINDSIGHT_BASE_URL=http://localhost:8888 AZURE_API_KEY=... AZURE_API_BASE=... \
  python scripts/run_longmemeval_hindsight.py \
    --questions sme/corpora/longmemeval/data/longmemeval_s_cleaned.json \
    --max-questions 150 --stratify-by question_type --content-rules upstream-exact \
    --json baselines/longmemeval_hindsight_strat150_qa_<date>.json \
    --status /tmp/hindsight.STATUS
```

Adapter docs: `docs/hindsight_adapter.md`.
