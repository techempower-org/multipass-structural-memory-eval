# Eval probe sweep — 2026-05-28

4-config matrix over (`PALACE_RERANK_ENABLED` × `PALACE_MODALITY_WEIGHT`) against
`https://familiar.jphe.in/api/familiar/eval` using
`familiar.realm.watch/tests/eval/run_paraphrase_probe.py` (51-question
paraphrase-stressed corpus, 7 shapes). Closes
techempower-org/multipass-structural-memory-eval#88.

## Headline

**All four toggle configurations returned bit-identical retrieval on this
corpus.** Overall recall@5 = 18/51 (35.3%), MRR = 0.277. Per-shape recall,
per-row `matched_via`, and per-row rank are identical across all 4 runs.

Neither `PALACE_RERANK_ENABLED` (palace-daemon's FlashRank gate, controls
`/search/hybrid` reranking) nor `PALACE_MODALITY_WEIGHT` (familiar-api's
TS `modalityWeight()` gate) moved a single retrieved entity in the top-5
returned by `/api/familiar/eval`.

This is a real null result, not a measurement bug — see "Validation" below.

## Overall recall@5 + MRR

| Config | HyDE off recall | HyDE on recall | MRR off | MRR on |
|---|---|---|---|---|
| Baseline (rerank=off, mod=off) | 35.3% (18/51) | 35.3% (18/51) | 0.277 | 0.277 |
| Modality only (rerank=off, mod=on) | 35.3% (18/51) | 35.3% (18/51) | 0.277 | 0.277 |
| FlashRank only (rerank=on, mod=off) | 35.3% (18/51) | 35.3% (18/51) | 0.277 | 0.277 |
| Both ON (rerank=on, mod=on) | 35.3% (18/51) | 35.3% (18/51) | 0.277 | 0.277 |

## Δ vs. baseline

| Config | ΔRecall HyDE off | ΔRecall HyDE on | ΔMRR off | ΔMRR on |
|---|---|---|---|---|
| Modality only | +0.00pp | +0.00pp | +0.000 | +0.000 |
| FlashRank only | +0.00pp | +0.00pp | +0.000 | +0.000 |
| Both ON | +0.00pp | +0.00pp | +0.000 | +0.000 |

## Per-shape recall (HyDE off, identical across all configs)

| Shape | n | hits | recall |
|---|---|---|---|
| canary | 4 | 2 | 50% |
| contradiction | 4 | 2 | 50% |
| cross_project | 11 | 5 | 45% |
| recency | 4 | 1 | 25% |
| temporal | 8 | 4 | 50% |
| topical_mismatch | 9 | 2 | 22% |
| vocab_mismatch | 11 | 2 | 18% |

## Validation: the toggles really did apply

Both env vars require a service restart to take effect (each is read at
process start, not per-request). The sweep script restarted the owning
service between every config:

- `palace-daemon` restarted 4× during the sweep window
  (11:41:10 → modality_only start, 11:45:24 → flashrank_only start,
   11:49:13 → both start). Confirmed via `journalctl -u palace-daemon`.
- `familiar-api` restarted 4× (11:34, 11:41, 11:45, 11:49) — each pickup
  of a new `PALACE_MODALITY_WEIGHT` value.
- FlashRank loader log line (`FlashRank loaded model=ms-marco-TinyBERT-L-2-v2`)
  appears in the daemon journal at 11:34:05 (initial), 11:45:29
  (flashrank_only start), 11:49:51 (both start) — and is **absent** at
  11:41:40 (modality_only start, where rerank=off was applied). The
  daemon honoured the toggle.

So the env did flip. The pipeline simply did not produce different output.

## Why the toggles had no effect on this corpus

`retrieveAndGround` in familiar-api (`src/memory-protocol.ts`) re-ranks
palace's response through several stages downstream of palace-daemon's
FlashRank:

```
palace.search/hybrid           # PALACE_RERANK_ENABLED gates rerank here
  → drawers.filter null/diary
  → domainRerank(wing + recency) # always-on TS reorder
  → modalityWeight(intent)        # PALACE_MODALITY_WEIGHT gates this
  → temporalDecay (half-life)
  → drawers.sort by similarity     # FINAL ordering
```

Two observations:

1. **Palace-daemon's rerank order is overwritten by `domainRerank +
   temporalDecay + sort`**. Even if FlashRank produces a meaningfully
   different ranking, `temporalDecay` multiplies by `exp(-λ · age_days)`
   and the final `sort` re-orders by the decayed score — which dominates
   on a corpus where drawer ages span months.

2. **`modalityWeight` adjusts similarity by ±10% at most** (lookup table
   in `src/retrieval/modality.ts` — factors 0.8 .. 1.1). On this corpus,
   the resulting score change isn't large enough to swap any drawer in or
   out of top-5 once temporal decay has been applied.

The net is that palace-daemon-side and familiar-api-side modality/rerank
gates are dwarfed by the final temporal-decay sort. They're effectively
no-ops from the eval endpoint's vantage point.

## Per-stage latency (across both HyDE arms, ms)

Stages with measurable latency. All four configs were within 30% of each
other on `palace_search_ms`, dominated by palace-daemon serving from
warm disk + pgvector + AGE.

| Config | palace_search p50 | palace_search p95 | total p50 | total p95 |
|---|---|---|---|---|
| baseline | 1495 | 4533 | 1552 | 4535 |
| modality_only | 1287 | 3772 | 1340 | 3836 |
| flashrank_only | 1397 | 4179 | 1450 | 4181 |
| both | 1365 | 3341 | 1450 | 3342 |

`rerank_ms` and `modality_ms` are ~0 across all configs (sub-millisecond
TS reorders — they ran, they just don't take time).

## Recommendations

Before re-running this matrix, fix one of:

1. **Move temporal-decay BEFORE rerank**, so palace's reranked order has
   a chance to influence the final top-5 — currently decay+sort is the
   load-bearing step.
2. **Increase modality factors** beyond ±10% so they can plausibly move
   drawers across the top-5 boundary.
3. **Pick a corpus where age dispersion is narrow** (drawers all within
   the same week), neutralising temporal decay and letting rerank /
   modality show through. The current paraphrase corpus is age-diverse.

Until one of those changes lands, this sweep will continue to return
identical numbers across all four configs.

## Reproduction

```
# In a familiar.realm.watch worktree:
bash tests/eval/sweep_probe_47.sh
# Generates probe-results-2026-05-28-{baseline,modality_only,flashrank_only,both}.json
# Then aggregate:
python3 tests/eval/aggregate_probe_sweep.py \
  --date 2026-05-28 \
  --in-dir tests/eval \
  --out-json probe-results-2026-05-28-sweep.json \
  --out-md   probe-results-2026-05-28-sweep.md
```

Both scripts (`sweep_probe_47.sh`, `aggregate_probe_sweep.py`) and the four
per-config result JSONs were generated in
`familiar.realm.watch/tests/eval/` and are reproducible from
`paraphrase_questions.yaml`. The aggregator is committed alongside this
summary; the sweep driver script remains in `familiar.realm.watch` (it
shells out to that repo's `run_paraphrase_probe.py`).
