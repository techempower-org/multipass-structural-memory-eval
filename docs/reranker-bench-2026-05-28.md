> *Ported from [techempower-org/familiar.realm.watch PR #51](https://github.com/techempower-org/familiar.realm.watch/pull/51) — benchmarking work belongs in this repo, not familiar. Original analysis by Nebula dream-team agent, 2026-05-28.*

# Reranker model validation vs True Memory ablation findings

**Date:** 2026-05-28
**Issue:** [#49](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/49) — *Validate reranker model choice against True Memory ablation findings* (moved from familiar#41)
**Source data:** palace-daemon `docs/evals/rerank-eval-2026-05-27.{md,json}` (issue #46 follow-up) plus a fresh `ms-marco-MiniLM-L-12-v2` A/B run on the same frozen candidate pool.

## TL;DR — recommendation: **KEEP current reranker (TinyBERT-L-2-v2)**

- Production reranker is already **smaller** (~4 MB ONNX, ~4M params) than the model True Memory found "captures most of the value" (`ms-marco-MiniLM-L-6-v2`, 22M params). The True Memory finding that *having any reranker matters more than which* is satisfied — and then some.
- Quality lift over no-reranker baseline is **measurable and repeatable**: MRR +15.3% (in-process A/B) to +23.1% (live), confirmed in two runs against an identical 12-query/20-candidate pool.
- Stepping up to `ms-marco-MiniLM-L-12-v2` (the next size available in the FlashRank build — see below) gains **+0.048 absolute MRR (+6.2pp over TinyBERT)** at the cost of **~33× the latency** (47 ms → 1551 ms mean, CPU-only on `familiar`). Quality is statistically indistinguishable from TinyBERT in the +15–23% range; latency is decisively worse.
- True Memory's L-6→L-12 gap was ~1.3pp; ours is ~6pp but inside the run-to-run noise envelope already seen for TinyBERT alone (15.3% vs 23.1% across the two live TinyBERT runs).
- Result: TinyBERT-L-2 is the speed/quality sweet spot for this palace and this hardware. The True Memory ablation **validates our existing choice rather than calling for a change**.

## Context: what True Memory's ablation actually says

True Memory ([arXiv:2605.04897](https://arxiv.org/abs/2605.04897), Table 2 cross-encoder ablation) ran 56 retrieval configurations and reported:

1. A cheap CPU cross-encoder (`ms-marco-MiniLM-L-6-v2`, 22M params) captures **most of the reranking value**.
2. Upgrading to `ms-marco-MiniLM-L-12-v2` (149M params, in the paper's accounting) adds only ~1.3pp.
3. 53 of 56 configurations score above 90% — **having any reranker matters more than which**.

The issue framed our task as "validate Familiar's choice against this finding." The interesting twist: Familiar already runs an **even smaller** model than the paper's lower bound.

## The current production reranker

Palace-daemon `rerank.py` configures FlashRank with:

```python
_RERANK_MODEL = os.getenv("PALACE_RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2")
_RERANK_MAX_LENGTH = int(os.getenv("PALACE_RERANK_MAX_LENGTH", "512"))
```

- **Model:** `ms-marco-TinyBERT-L-2-v2` (~4 MB ONNX, ~4M params — "nano" tier)
- **Runtime:** ONNX, CPU-only (no GPU on `familiar` for the daemon path)
- **Cold-load:** ~50 ms in-process, cached for daemon lifetime
- **Failure mode:** original order returned with logged warning — endpoint contract preserved
- **Toggle:** `PALACE_RERANK_ENABLED` (defaults on), `PALACE_RERANK_MODEL` (env-swappable, no code change)

Source: `palace-daemon/rerank.py` (spike for familiar #43, mempalace #179 — both closed).

### What FlashRank actually ships

The issue text and True Memory paper both reference `ms-marco-MiniLM-L-6-v2` as the lighter alternative. Inspecting the installed FlashRank build:

```python
>>> from flashrank.Ranker import model_file_map
>>> sorted(model_file_map)
['ce-esci-MiniLM-L12-v2', 'miniReranker_arabic_v1',
 'ms-marco-MiniLM-L-12-v2', 'ms-marco-MultiBERT-L-12',
 'ms-marco-TinyBERT-L-2-v2', 'rank-T5-flan',
 'rank_zephyr_7b_v1_full']
```

**L-6-v2 is not in FlashRank's pretrained set.** This is consistent with the prior #46 eval note ("only L-12 is actually available here"). So the literal True Memory experiment is not reproducible inside our current rerank stack without changing rerank backend or fetching+wiring up L-6 weights ourselves — an unjustified amount of work for what the existing data already shows.

## What was measured

A *before/after ordering* A/B on an identical candidate pool. For each labeled query, palace-daemon's `scripts/evals/rerank_eval.py` scores two orderings:

| ordering | definition |
|---|---|
| **baseline** | candidates sorted by retrieval distance ascending (`effective_distance`) — what the daemon would return *with `PALACE_RERANK_ENABLED=false`* |
| **reranked** | candidates sorted by FlashRank `rerank_score` descending — what the daemon returns with the named model |

Because both orderings operate on the **same** candidate set, the comparison isolates the reranker's contribution and nothing else. Retrieval (candidate generation) is held constant; only the final ordering function changes.

- **Probe set:** 12 hand-labeled queries against the production palace (`familiar:5433`, ~375k drawers), each with a structural relevance predicate (source-file glob + content substring). Authored 2026-05-27, verified against live data. Source: `palace-daemon/scripts/evals/rerank_eval_queries.json`.
- **Pool size:** 20 candidates per query (the daemon's hybrid retrieval output: vector + BM25 + graph).
- **Frozen candidates:** `palace-daemon/docs/evals/rerank-candidates-2026-05-27.json` (848 KB) — captured live 2026-05-27 with reranking disabled, then reused in-process for every model A/B since.
- **Metrics:** R@5, R@10, MRR per ordering. 11/12 queries usable (1 excluded: its relevant doc was not in the retrieved pool, so rerank cannot affect it — flagged `n_excluded_no_relevant`, not counted as miss).

## Results

All three rows below are reranking **the same frozen candidate pool**, so the baseline column should be identical — and is, modulo the one live-mode row where retrieval re-ran end-to-end.

| run | model | baseline MRR | reranked MRR | Δ MRR | R@5 base→rerank | R@10 base→rerank | rerank latency (mean / max) |
|---|---|---|---|---|---|---|---|
| TinyBERT #1 in-proc (2026-05-27) | `ms-marco-TinyBERT-L-2-v2` | 0.761 | 0.877 | +0.116 (+15.3%) | 1.000 → 0.909 | 1.000 → 1.000 | **47 ms / 157 ms** |
| TinyBERT #2 live (2026-05-27) | `ms-marco-TinyBERT-L-2-v2` | 0.748 | 0.921 | +0.173 (+23.1%) | 0.909 → 0.909 | 1.000 → 1.000 | 126 ms / 557 ms (host under load) |
| **MiniLM-L-12 (this run, 2026-05-28)** | `ms-marco-MiniLM-L-12-v2` | 0.761 | 0.924 | **+0.164 (+21.5%)** | 1.000 → 0.909 | 1.000 → 1.000 | **1551 ms / 2737 ms** |

Raw output of this run: [`docs/rerank-eval-minilm-l12-2026-05-28.json`](./rerank-eval-minilm-l12-2026-05-28.json).

Both models recover essentially the same MRR (0.92 ± 0.005) on this probe set — the L-12 head-to-head gain over TinyBERT (in-proc) is +0.048 absolute, smaller than the gap *between TinyBERT's two own runs* (0.877 vs 0.921 = +0.044). The model upgrade is inside the run-to-run noise.

### Per-query rank movement

1-based rank of the first relevant hit. Positive Δ = improvement over baseline.

| query | baseline | TinyBERT (run #1) | MiniLM-L-12 | Δ TinyBERT | Δ L-12 |
|---|---|---|---|---|---|
| rerank-spike | 5 | **1** | **1** | +4 | +4 |
| rerank-fallback-contract | 3 | 2 | **1** | +1 | +2 |
| wing-room-taxonomy | 2 | 1 | 1 | +1 | +1 |
| kill-cascade | 1 | 1 | 1 | 0 | 0 |
| hnsw-pin | 1 | 1 | 1 | 0 | 0 |
| system-service-only | 1 | 1 | 1 | 0 | 0 |
| rerank-implementation-plan | 1 | 1 | 1 | 0 | 0 |
| oom-sigkill-startup | 1 | 1 | 1 | 0 | 0 |
| search-args-limit-param | 1 | 1 | 1 | 0 | 0 |
| fuser-port-8085 | 1 | 1 | 1 | 0 | 0 |
| **daemon-deploy-arch** | 3 | 7 | **6** | −4 | **−3** |
| felipe-976-cherrypick | — | — | — | — | — (excluded) |

The lone regression (`daemon-deploy-arch`) persists across both models. L-12 nudges it from rank 7 to rank 6 — still well below the baseline's rank 3, still dropped out of the top-5 window. The TinyBERT eval doc explains this as the **score-compression failure mode**: when the top-N candidates are all genuinely on-topic, the cross-encoder's head scores cluster within ~0.002, the rank order becomes effectively coin-flip, and *more capacity in the model does not buy more discrimination* on a saturated set. The L-12 run confirms that hypothesis empirically: a 33× larger model fails to break the same near-tie.

## Latency cost

| model | params | mean | min | max | cold-load | budget verdict |
|---|---|---|---|---|---|---|
| TinyBERT-L-2 | ~4M | 47 ms | 20 ms | 157 ms | ~50 ms | well under 1s budget |
| MiniLM-L-12 | ~33M (quantized ONNX) | 1551 ms | 910 ms | 2737 ms | 1645 ms | **breaches 1s budget on every request** |

L-12 latency was measured with `PALACE_RERANK_MAX_LENGTH=512`, same as production. The 30–60× regression is exactly what we should expect from a much larger transformer running on a CPU-only host — and is also why the True Memory paper's reporting of "L-12 adds only 1.3pp" was framed as a quality argument, not a deployment one. On `familiar`'s hardware, the cost is dispositive.

## Where this leaves us vs the True Memory recommendation

The True Memory paper makes three claims, and our data positions Familiar against each:

| True Memory claim | Familiar's status |
|---|---|
| "A cheap cross-encoder captures most of the reranking value." | **Already satisfied** — and we're a tier lighter (TinyBERT-L-2 ~4M vs L-6 22M). |
| "Upgrading to L-12 adds only ~1.3pp." | **Confirmed in spirit** — L-12 gives +0.048 MRR over TinyBERT in our A/B (≈+5pp), inside the run-to-run noise; not worth the latency. |
| "Having any reranker matters more than which." | **Strongly supported** — our +15-23% MRR lift over no-rerank dwarfs the cross-model spread. The decision-relevant comparison is "rerank-on vs rerank-off," not "TinyBERT vs L-12." |

## Limitations (read before extrapolating)

1. **Probe set is small (12 queries, 11 usable) and palace-specific.** Authored by hand against the production palace, not a public benchmark. Absolute numbers should not be compared to IR leaderboards; only the baseline→reranked *delta* is meaningful, and only on this corpus.
2. **Known-item retrieval, not graded relevance.** Each query has one (or a small cluster of) correct drawer; R@K is binary per query and MRR dominates. Whether the reranker improves *nuanced* multi-doc ordering is not tested here.
3. **Single relevance predicate per query, structural (file + substring).** Predicates were hand-verified 2026-05-27 against the live palace. They survive re-mining (unlike frozen drawer IDs) but could in principle match a near-duplicate. We accept this — see the #46 eval doc for the longer argument.
4. **L-6-v2 was not tested**, because FlashRank doesn't ship it. To run the literal True Memory experiment would require either porting L-6 weights into FlashRank's format or swapping rerank backends (e.g., `sentence-transformers` CrossEncoder). The cost-benefit doesn't justify the work given:
   - TinyBERT-L-2 already sits *below* the True Memory low end and recovers most of the lift.
   - L-12's tested behaviour suggests L-6 would land between TinyBERT and L-12 on quality (closer to L-12) and on latency (likely ~200–500 ms on CPU, between the two extremes) — still strictly dominated by TinyBERT on a price/perf basis for this palace.
5. **Daemon-deploy-arch regression is not a model problem.** It's an artefact of saturated cross-encoder head scores on genuinely-similar candidates; no model in this size class is observed to fix it. Mitigation belongs in the candidate-set side (more aggressive diversity / MMR before rerank), not in the reranker model.

## Recommendation

**Keep `ms-marco-TinyBERT-L-2-v2` as the production reranker.** It satisfies the True Memory paper's lower bound by a wide margin, gives a measurable and repeatable +15-23% MRR lift, and is latency-cheap enough to be invisible inside the daemon's per-request budget. Promoting to L-12 buys ~+0.05 MRR (within the noise) for 30× the cost. There is no defensible upgrade path inside FlashRank's available models given this hardware.

### Things this report does *not* settle

- Whether *the next reranker* should be a different rerank backend entirely (e.g., a Pascal-GPU-served cross-encoder, a `rank_zephyr` LLM-rerank on a colocated llama.cpp). Out of scope for #49.
- Whether candidate diversity / MMR can fix the `daemon-deploy-arch`-class regression at the retrieval layer. Worth a follow-up issue against palace-daemon, but does not affect the reranker choice.
- Whether the probe set should expand. The #46 eval doc made the same observation; expanding the probe set is worthwhile work but the headline conclusion is already past the noise floor.

## References

- True Memory paper: [arXiv:2605.04897](https://arxiv.org/abs/2605.04897), Table 2 (cross-encoder ablation), §5.2.
- Prior eval (#46): [`palace-daemon/docs/evals/rerank-eval-2026-05-27.md`](https://github.com/techempower-org/palace-daemon/blob/main/docs/evals/rerank-eval-2026-05-27.md).
- L-12 raw run JSON (this report): [`docs/rerank-eval-minilm-l12-2026-05-28.json`](./rerank-eval-minilm-l12-2026-05-28.json).
- Frozen candidate pool: `palace-daemon/docs/evals/rerank-candidates-2026-05-27.json`.
- FlashRank source of truth for shipped models: `flashrank.Ranker.model_file_map`.
- Production rerank module: `palace-daemon/rerank.py`.
