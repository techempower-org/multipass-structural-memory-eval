# Hindsight adapter

`sme/adapters/hindsight.py` runs SME against [Hindsight](https://github.com/vectorize-io/hindsight)
(vectorize-io) — an MIT-licensed agent-memory server that stores **LLM-extracted
facts** (biomimetic World / Experiences / Mental-Models structures) rather than
raw text, and recalls them via a fused semantic + BM25 + graph + temporal
strategy with cross-encoder reranking.

Verified against the **real** `hindsight-client` 0.7.1 + a live server
(techempower-org/multipass-structural-memory-eval#184), the same way the OMEGA
adapter was (#183): confirm the actual API surface, prove retrieval end-to-end
on the real corpus, isolate state so a bench can't pollute a user store.

## Standing up a local server

Hindsight is Docker-hosted and needs an LLM provider for its inline fact
extraction. A fully-local, isolated setup (no cloud key, never touches prod):

```bash
docker run -d --name hindsight-scratch \
  --add-host=host.docker.internal:host-gateway \
  -p 127.0.0.1:8888:8888 -p 127.0.0.1:9999:9999 \
  -e HINDSIGHT_API_LLM_PROVIDER=ollama \
  -e HINDSIGHT_API_LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  -e HINDSIGHT_API_LLM_MODEL=phi4:latest \
  -e HINDSIGHT_API_LLM_API_KEY=ollama \
  -e HINDSIGHT_API_EMBEDDINGS_PROVIDER=local \
  -v "$PWD/hindsight-pg0":/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest

pip install hindsight-client     # the Python SDK the adapter prefers
curl localhost:8888/health       # {"status":"healthy","database":"connected"}
```

Embedded Postgres + pgvector, local `BAAI/bge-small-en-v1.5` embeddings (dim
384), local `ms-marco-MiniLM-L-6-v2` cross-encoder rerank. No API key required
for the self-hosted API; only the extraction LLM provider is configured.

### Extractor model matters — the silent-degradation trap

Hindsight's `retain` runs the configured LLM to extract structured facts. **The
model must reliably emit strict JSON.** A weak model (observed: ollama
`qwen3.5:4b`) emits markdown prose, every extraction fails with a JSON parse
error, **0 facts are stored, and recall silently returns nothing** — `retain`
still reports `success=True`. A bench against that state scores ~0 and looks
like "Hindsight is bad" when really the extractor is crippled.

`phi4:latest` produces valid JSON and works. **Always run the live smoke
(`tests/test_hindsight_live.py`) first** — it ingests a tiny corpus and asserts
a paraphrased query (disjoint vocabulary) returns on-topic results, which only
passes if extraction + semantic recall actually work.

## API surface (verified, hindsight-client 0.7.1)

- `Hindsight(base_url, api_key=None, timeout=300.0)`
- `retain(bank_id, content, timestamp=None, context=None, document_id=None, metadata=None, ...)`
  — the adapter passes `document_id` so recall hits trace back to the ingest unit.
- `recall(bank_id, query, max_tokens=4096, budget='mid', include_source_facts=False, ...)`
  — **no `top_k`**; budgeted by tokens. The adapter over-fetches and slices to
  `n_results`. Returns a Pydantic `RecallResponse` with `.results`
  (`RecallResult`: `id`, `text`, `type`, `document_id`, `source_fact_ids`, …),
  not a dict.
- `reflect(bank_id, query, budget='low', ...)` — deeper analysis (opt-in via
  `use_reflect=True`).

## Isolation

`bank_id` namespaces memories. The harness factory and the LongMemEval runner
give each question a unique bank (`sme_<vault>` / `lme_<qid>_<rand>`), so no
cross-question contamination. The whole stack is a throwaway local container;
prod familiar / palace-daemon are never touched.

## Running the benchmark

```bash
HINDSIGHT_BASE_URL=http://localhost:8888 AZURE_API_KEY=... AZURE_API_BASE=... \
  python scripts/run_longmemeval_hindsight.py \
    --questions sme/corpora/longmemeval/data/longmemeval_s_cleaned.json \
    --max-questions 150 --stratify-by question_type \
    --content-rules upstream-exact \
    --json baselines/longmemeval_hindsight_strat150_qa_<date>.json \
    --status /tmp/hindsight.STATUS
```

Or `--adapter hindsight` via `scripts/cross_validate_longmemeval.py`.

## Methodology caveat — extraction-mediated R@K

Hindsight stores **extracted facts, not raw sessions**. Recall returns
fact-units; each carries the `document_id` of the session it was extracted from.
Session-level R@K is therefore a membership test of *"did a fact extracted from
the evidence session rank top-K"* — **softer and extraction-mediated** vs the
raw-chunk R@K used for mempalace (drawer hits) or OMEGA (raw memories). It is
comparable *in spirit* to the daemon's `drawer_hit_at_K`, not identical. The QA
number (canonical reader + judge over recalled facts) is the cleaner
apples-to-apples metric across substrates.

## Throughput note (the #184 bench wall)

Fact extraction runs the LLM on **every session ingest** (LongMemEval-S
averages ~48 sessions/question; the strat150 subset = ~7,200 session ingests).
With the local phi4 extractor on CPU, measured extraction is ~60–96 s/session,
i.e. the full strat150 run is ~150 hours — impractical for a CPU-local
extractor. A defensible Hindsight QA row needs either a small indicative `n`, a
GPU-accelerated extractor, or a fast cloud extraction provider (which trades off
the local-only isolation posture). See the run's baseline JSON `run_metadata`
for the actual `n`, extractor, and timing of any committed result.
