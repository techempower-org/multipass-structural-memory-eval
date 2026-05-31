# Mem0 adapter

`sme/adapters/mem0.py` runs SME against [Mem0](https://github.com/mem0ai/mem0)
(`pip install mem0ai`) **OSS library mode** (`from mem0 import Memory`) — an
Apache-2.0 memory layer that stores **LLM-extracted facts** (not raw text) and
recalls them via fused semantic + BM25 + entity matching over an on-disk
Qdrant vector store.

Verified against the **real** `mem0ai` 2.0.4
(techempower-org/multipass-structural-memory-eval#185), the same way the OMEGA
(#183) and Hindsight (#184) adapters were: confirm the actual API surface,
prove ingest + retrieve end-to-end on a live local stack, and characterize the
extraction cost so a full bench is planned, not stumbled into.

## Verdict (the deliverable)

- **Adapter VERIFIED** against real `mem0ai` 2.0.4 — `from_config` / `add` /
  `search` / `get_all` / `delete_all` signatures match the shipped adapter; no
  code change was needed.
- **Smoke green on a local, $0-cloud stack** (ollama phi4 extractor +
  nomic-embed-text embedder). A paraphrased query ("What pet does Maria have?",
  zero token overlap) retrieves the right memory ("Biscuit", score 0.65) — the
  extractor is healthy, not crippled.
- **Extraction-throughput-bound**, the same wall Hindsight hit. A full
  strat150 QA bench is deferred; the 2nd extraction-cost data point next to
  Hindsight is the real finding (see below).

## Local, $0-cloud setup (never touches prod, never bills OpenAI)

Mem0 OSS **defaults to OpenAI** for both extraction (`gpt-*`) and embeddings
(`text-embedding-3-*`) — it will silently bill if pointed at the default. Wire
everything to local ollama instead:

```python
config = {
  "llm":      {"provider": "ollama",
               "config": {"model": "phi4:latest",
                          "ollama_base_url": "http://localhost:11434"}},
  "embedder": {"provider": "ollama",
               "config": {"model": "nomic-embed-text:v1.5",
                          "ollama_base_url": "http://localhost:11434"}},
  "vector_store": {"provider": "qdrant",
                   "config": {"path": "<isolated tmp dir>",
                              "collection_name": "...",
                              "embedding_model_dims": 768}},  # nomic dim
}
Mem0Adapter(config=config, user_id="...")
```

`OPENAI_API_KEY` should be **unset** for a bench run so a misconfigured stack
fails loudly rather than billing.

### Two real-API findings the stub didn't have

1. **The `ollama` Python lib is a hard dependency of mem0's ollama provider.**
   If it's absent, mem0's `embeddings/ollama.py` and `llms/ollama.py` fall back
   to an interactive `input("Install it now? [y/N]")` prompt that **`EOFError`s
   in any non-interactive / bench run**. Pinned into the `mem0` extra
   (`pip install 'sme-eval[mem0]'`) so the local stack works headless.
2. **On-disk Qdrant locks `~/.mem0/migrations_qdrant` to a single process.** A
   second `Memory` instance against the default location raises
   `RuntimeError: Storage folder ... is already accessed by another instance`.
   So you **cannot run two mem0 instances concurrently** on the default path —
   a per-question-isolation bench must give each question its own isolated
   store path (the live smoke uses a fresh `tmp_path`), or serialize. This is
   the mem0 analogue of OMEGA's process-global `SQLiteStore` singleton.

### Extractor model matters (the silent-degradation trap)

Like Hindsight, mem0's `add()` runs the LLM to extract facts; a model that
can't emit clean structured output silently stores nothing. `phi4:latest`
produces valid extractions (it even *normalized* "March" → "March 2026" and
"next Friday" → a concrete date). **Always run the live smoke
(`tests/test_mem0_live.py`) first** — it asserts a paraphrased query returns the
ingested memory, which only passes when extraction + embedding actually work.
(Per the Hindsight finding, `qwen3.5:4b` emits markdown prose and fails this
class of test; use phi4.)

## API surface (verified, mem0ai 2.0.4)

- `Memory.from_config(config_dict)` — single positional dict.
- `add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None,
  infer=True, memory_type=None, prompt=None)` — `infer=True` (default) is the
  per-ingest LLM extraction. `messages` is a list of `{role, content}` dicts.
- `search(query, *, top_k=20, filters=None, threshold=0.1, rerank=False)` —
  `filters={"user_id": ...}` is the post-graph-removal scoping; returns
  `{"results": [{id, memory, hash, score, user_id, created_at, ...}]}`.
- `get_all(*, filters=None, top_k=20)` — same result shape.
- `delete_all(user_id=None, agent_id=None, run_id=None)`.

**Graph memory was removed from mem0 OSS**, so `relations` are not populated and
`get_graph_snapshot()` returns isolated entities with **zero edges** — Cat 5/6
score zero against mem0, which is the honest reading of a system that dropped
its graph layer.

## Extraction is lossy by design (verbatim-vs-extraction, illustrated)

In the n=5 smoke, mem0 stored only **3 of 5** facts — its `add()` reasons about
whether each input is worth storing/updating and returned `{"results": []}`
(stored nothing) for the standup-time and one other fact. This is the
extraction paradigm's lossiness made concrete: a verbatim-first store (MemPalace)
keeps everything; an extraction-first store (mem0, Hindsight) keeps what its
LLM judges salient — trading recall completeness for compression and the
extraction cost below.

## The cost-wall (2nd data point next to Hindsight)

`add()` runs phi4 extraction (plus mem0's store/update/skip reasoning) on every
ingest. Measured local, CPU, n=5:

| | per-ingest |
|---|---|
| cold start (first-ever phi4 load + Qdrant migration init) | ~7,000 s (one-time, not representative) |
| **warm steady-state** | **~9 s/ingest** (5-ingest run: 20.7 / 6.4 / 6.2 / 5.5 / 5.6 s, avg 8.9 s) |

LongMemEval-S averages ~48 sessions/question; the strat150 subset = ~7,200
session ingests → **≈18 hours CPU-local** at the warm rate. **Throughput-bound**,
the same class of wall Hindsight hit (~60–96 s/session there; mem0 is faster
per-ingest but a full bench is still many hours). A defensible mem0 QA row needs
a GPU extractor, a fast cloud extraction provider (trades off the local-only
posture — flag the spend), or a small indicative `n`. **Full QA is deferred**;
the cost characterization is the contribution.

## Tests

- `tests/test_mem0_adapter.py` — 22 mocked unit tests (no install needed),
  mirroring the verified 2.0.4 surface.
- `tests/test_mem0_live.py` — `@pytest.mark.live` end-to-end smoke against the
  real package on a throwaway on-disk Qdrant + local ollama. Auto-skips when
  `mem0ai` is absent or ollama (phi4 + nomic-embed-text) isn't reachable.
  Deselect with `-m "not live"`.

## Follow-ups

- A full QA row would need the cost-wall mitigations above — JP's call on spend
  vs a GPU extractor; not run here.
- `…#185` companion to OMEGA (#183, verified + benched) and Hindsight (#184,
  verified + cost-wall). With this, all three Tier-3 competitor adapters are
  real-API-verified on identical local infrastructure.
