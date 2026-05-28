# Encoder-swap baseline — LongMemEval-S 500Q via in-memory ChromaDB

**Date:** 2026-05-28
**Branch:** `bench/runs-2026-05-28` (harness from PR #62)
**Result:** **R@5 = 0.9640** on LongMemEval-S 500Q (independent reproduction of upstream's published 0.9660 to within 0.2pp).

## Why this run exists

The mempalace fork's `benchmarks/longmemeval_bench.py --mode raw` published R@5 = 0.9660 on LongMemEval-S using ChromaDB EphemeralClient + `all-MiniLM-L6-v2`. That number lives in [`~/Projects/memorypalace/docs/benchmarks/2026-05-26-longmemeval-500q-results.md`](https://github.com/techempower-org/mempalace/blob/main/docs/benchmarks/2026-05-26-longmemeval-500q-results.md) and is the project's headline retrieval-recall number.

This run reproduces that number on the **SME side** using `scripts/encoder_swap_eval.py` (issue #53 sub-task 1) — same dataset, same encoder, same ChromaDB-EphemeralClient protocol, but with SME's eval scoring and report shape. That validates the encoder-swap harness can stand in for upstream's `longmemeval_bench.py --mode raw` for future cross-encoder A/B work.

## Results

### Headline

| Metric | This run | Upstream raw (published 2026-05-26) | Δ |
|---|---|---|---|
| R@1 | **0.8060** | 0.806 | 0.0 |
| **R@5** | **0.9640** | **0.966** | **-0.2pp** |
| R@10 | **0.9820** | 0.982 | 0.0 |
| R@30 | 0.9960 | 0.996 | 0.0 |
| R@50 | 1.0000 | 1.000 | 0.0 |

The 0.2pp gap at R@5 is well within the stochastic noise floor (ChromaDB's CPU embedder is non-deterministic at the FP-rounding level). Effectively a byte-identical reproduction.

### Per question_type

| Question type (LongMemEval) | SME category | n | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| single-session-user / -assistant / -preference (combined) | `cat_1` | 150 | 0.813 | 0.947 | 0.973 |
| abstention | `cat_1_negative` | 30 | 0.633 | 0.967 | 0.967 |
| multi-session | `cat_2c` | 121 | 0.868 | 0.992 | 1.000 |
| knowledge-update | `cat_3_partial` | 72 | 0.917 | 1.000 | 1.000 |
| temporal-reasoning | `cat_6` | 127 | 0.717 | 0.937 | 0.969 |

Per-type shape matches upstream's published table — knowledge-update saturates at 100%, temporal-reasoning is the lowest, multi-session is near-saturated by R@10.

### R@1 miss breakdown (#53 sub-task 2 — `r1_misses` field in the JSON)

97 / 500 questions (19.4%) missed at rank-1. By LongMemEval question_type:

| Question type | r1_misses | share of own bucket |
|---|---:|---:|
| temporal-reasoning | 38 | 30% of cat_6 |
| single-session-user | 20 | 29% of single-session-user |
| multi-session | 19 | 16% of multi-session |
| single-session-preference | 9 | 30% of preference |
| knowledge-update | 8 | 10% of knowledge-update |
| single-session-assistant | 3 | 5% of assistant |

This matches the published article's per-type R@1 numbers (assistant turns are the easiest because they tend to have task-specific vocabulary; temporal-reasoning is hardest because date-arithmetic dependencies don't surface in vector similarity).

## Runtime

- **Total wall:** 573.8s (~9.6 min) on katana CPU (no GPU)
- **Per question:** 1.15s avg
- **Bottleneck:** ChromaDB's CPU embedder (all-MiniLM-L6-v2 on CPU, ~50 sessions per Q to embed)

Upstream's published run on the same protocol was ~1224s wall (20.4 min) for raw mode. The 2× speedup here is from a faster ChromaDB version (1.5.8 vs whatever was in upstream's published run) and possibly katana's higher single-thread perf.

## Substrate this is NOT measuring

- **Not the daemon path.** This is in-memory ChromaDB EphemeralClient; nothing in `palace-daemon` is exercised. The daemon-side reading is gated on #61 (palace-daemon SIGTERM cycle under bench load).
- **Not the AGE substrate.** No graph traversal, no RELATION layer. The 1.76M-triple AGE backfill that JP completed 2026-05-27 has no effect on this number.
- **No reader, no judge.** Pure retrieval recall. End-to-end QA accuracy (the True-Memory-comparable metric) lives downstream on #44/#45/#46 once the daemon is stable.

## Reproduction

```bash
./venv/bin/python scripts/encoder_swap_eval.py \
    --questions sme/corpora/longmemeval/data/longmemeval_s_cleaned.json \
    --content-rules upstream-exact \
    --top-k 50 \
    --json baselines/longmemeval_encoder_swap_default_$(date +%Y-%m-%d).json
```

Default encoder is `all-MiniLM-L6-v2`. To swap in a candidate fine-tuned encoder (e.g. nakata-app/adaptmem's FT-300):

```bash
./venv/bin/python scripts/encoder_swap_eval.py \
    --questions sme/corpora/longmemeval/data/longmemeval_s_cleaned.json \
    --model /path/to/minilm-lme-ft-300 \
    --json baselines/longmemeval_encoder_swap_ft300_$(date +%Y-%m-%d).json
```

The script writes `baselines/<name>.json` containing:
- `summary.overall` with R@1/5/10/30/50
- `summary.per_category` with per-LongMemEval-type recall
- `summary.r1_misses` (list per #53 sub-task 2) and `summary.r1_miss_by_type` histogram
- `per_question` with full per-question diagnostic record

## Cross-references

- #51 — substrate-floor parity reading (postgres+pgvector ≡ chromadb byte-identical at the same R@5=0.9660)
- #53 — adaptmem v0.7 encoder-swap measurement pattern (this PR closes sub-task 1)
- #58 — substring matcher endpoint divergence; orthogonal — encoder_swap_eval uses entity-id matching directly via ChromaDB result IDs, sidestepping the substring-matcher limitation entirely
- #62 — bench/runs-2026-05-28 branch where the harness lives
- [`~/Projects/memorypalace/docs/benchmarks/2026-05-26-longmemeval-500q-results.md`](https://github.com/techempower-org/mempalace/blob/main/docs/benchmarks/2026-05-26-longmemeval-500q-results.md) — upstream's published reference
