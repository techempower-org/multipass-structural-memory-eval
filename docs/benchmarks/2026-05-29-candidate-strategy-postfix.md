# Candidate-strategy re-bench — post-#292/#293 (2026-05-29)

**Issue:** techempower-org/multipass-structural-memory-eval#75
**Baseline JSON:** `baselines/candidate-strategy-2026-05-29-postfix.json`
**Query set:** `scripts/evals/rerank_eval_queries.json` (12 labeled queries, vendored from palace-daemon PR #64)
**Params:** `--strategies vector union hybrid --limit 20`, daemon `PALACE_RERANK_ENABLED` at default (on)

## Results (n=12)

| strategy | R@5 | R@10 | MRR | p50 | p95 | vs 2026-05-28 |
|---|---:|---:|---:|---:|---:|---|
| vector | 0.917 | 1.000 | 0.808 | 372 ms | 515 ms | MRR 0.870→0.808, p50 400→372 ms |
| union  | 0.917 | 1.000 | 0.808 | 366 ms | 473 ms | MRR 0.870→0.808, p50 229→366 ms |
| hybrid | **1.000** | 1.000 | 0.785 | **2181 ms** | 5473 ms | **p50 3330→2181 ms**, MRR 0.847→0.785, R@5 1.0 held |

## Reading (with caveats)

- **Latency (hypothesis 1): partially confirmed.** Hybrid p50 dropped 3330→2181 ms — the mempalace#292 directional-graph-expand speedup. Not the ~1.5 s single-call figure, because this bench measures the full end-to-end search path (incl. the now-default rerank stage), not the isolated graph call.
- **R@5 (hypothesis 3): holds.** Hybrid R@5 = 1.000 — graph candidates still surface drawers neither vector nor BM25 ranks top-5.
- **MRR (hypothesis 2): NOT closed, but confounded.** Hybrid MRR (0.785) is below union (0.808) and below the 2026-05-28 hybrid (0.847). On its face this *falsifies* mempalace#297's hope that the cross-encoder reorder closes the graph-induced MRR loss without a convex-weight rebalance. BUT:
  - **All three strategies dropped ~6pp MRR uniformly** vs 2026-05-28. A uniform drop points to a shared confound (rerank globally on now, or index drift from continued mining changing the relevant-set ranking), not a strategy-specific effect.
  - **n=12 is below the n≥25 threshold** for claiming conclusions.
  - **The bench doesn't record the daemon's rerank state** (no `rerank` field in `run_metadata`) — so we can't even confirm these queries went through the cross-encoder. Gap worth fixing.

## Blocked on for a clean verdict

The leg-A (hybrid, rerank-OFF) vs leg-B (hybrid, rerank-ON) isolation needs a per-request rerank toggle — **palace-daemon#189**. Until that lands, the rerank-on-vs-off comparison can't run in one pass without globally toggling `PALACE_RERANK_ENABLED` (perturbs concurrent consumers). The uniform MRR drop is un-attributable without it.
