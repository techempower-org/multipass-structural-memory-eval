# Hybrid scorer-weight tuning — closes the MRR regression (2026-05-31)

**Issue:** techempower-org/multipass-structural-memory-eval#111 (weight-tuning half)
**Baseline JSON:** `baselines/hybrid-scorer-weight-tuning-2026-05-31.json`
**Query set:** `scripts/evals/rerank_eval_queries.json` (12 labeled queries, vendored from palace-daemon PR #64)
**Code:** mempalace `_hybrid_weights()` env knob (`PALACE_HYBRID_VECTOR_WEIGHT` / `PALACE_HYBRID_BM25_WEIGHT`)
**Method:** in-process `mempalace.searcher.search_memories` on familiar against the live postgres+AGE palace, FlashRank OFF (matches the `/mcp tools/call` path the candidate-strategy harness drives). The prod daemon was not restarted or reconfigured — weights are set per sweep point in the sweep process only.

## Acceptance: SATISFIED

The #111 criterion — *a documented hybrid weight achieving R@5 ≈ 1.000 WITHOUT regressing MRR vs union/vector* — is met by **`vector_weight=0.85, bm25_weight=0.15`**:

| config | R@5 | R@10 | MRR | felipe | ddarch |
|---|---:|---:|---:|---:|---:|
| vector / convex (default wts) | 0.917 | 1.000 | 0.808 | 1 | 9 |
| union / convex (default wts) | 1.000 | 1.000 | 0.785 | 2 | 3 |
| hybrid / convex (default 0.6/0.4) | 1.000 | 1.000 | 0.785 | 2 | 3 |
| **hybrid / convex (0.85/0.15)** | **1.000** | **1.000** | **0.833** | 3 | 3 |

vs the union/vector MRR floor (0.808): **+2.5pp**. vs the default-hybrid regression (0.785): **+4.8pp**. R@5 holds at 1.000.

## Why the default 0.6/0.4 regressed, and why 0.85/0.15 fixes it

**MRR is non-monotonic in `vector_weight`.** The 2026-05-28 baseline read the hybrid MRR drop as graph promoting one query and demoting another. That attribution is wrong (see below). The real mechanism:

- The default 0.4 BM25 weight was *burying a strong vector match* — `rerank-implementation-plan` sat at rank 4. Raising `vector_weight` to 0.85 surfaces it to **rank 1** (RR +0.75).
- `daemon-deploy-arch` (the R@5 win) holds **rank 3** across the whole `vw ∈ [0.55, 0.85]` band — its lift is robust.
- `felipe-976-cherrypick` slips **rank 2 → 3** (RR −0.167) — a genuine near-tie (convex scores 0.715 vs 0.714 at default).
- Net: 0.785 → 0.833.

Past `vw=0.9` the blend over-commits to vector, `felipe` collapses to rank 9, and R@5 drops to 0.917. So 0.85/0.15 is a real local optimum, not an edge artifact.

| vw | bw | R@5 | R@10 | MRR | felipe | ddarch |
|---:|---:|---:|---:|---:|---:|---:|
| 0.55–0.70 | 0.45–0.30 | 1.000 | 1.000 | 0.785 | 2 | 3 |
| 0.75–0.80 | 0.25–0.20 | 1.000 | 1.000 | 0.771 | 3 | 3 |
| **0.85** | **0.15** | **1.000** | **1.000** | **0.833** | 3 | 3 |
| 0.90 | 0.10 | 0.917 | 1.000 | 0.815 | 9 | 3 |

## Correction to the 2026-05-28 baseline: the graph leg is inert here

On all 12 golden queries the hybrid candidate-strategy surfaced **zero graph candidates** (every `trace.sources` = `drawer` + `bm25_postgres` only). `candidate_strategy="hybrid"` is byte-identical to `"union"` on this set. The hybrid-vs-vector difference is **BM25 pool-widening**: injecting BM25 candidates recomputes corpus-relative BM25 IDF over a larger candidate set (`mempalace.searcher._bm25_scores`), which shifts every drawer's BM25 score and reshuffles the convex blend. There is no graph promotion/demotion on this set — the convex weight is the correct and only lever, which is why the tuning works.

## Cat 7b latency — post-index reading (two facts, not one "win")

The graph-walk *latency* half of #111 was resolved separately (Haze's profile + 4 AGE edge-endpoint indexes — `idx_mentions_end_id/start_id`, `idx_relation_start_id/end_id` — applied to prod familiar 2026-05-30). This run captures a **post-index** reading on the 12-query golden set, measured on the daemon HTTP path (consumer-facing), prod-default FlashRank ON, warm steady-state, 12 queries × 4 reps (48 calls/strategy), limit=20.

| strategy | p50 post | p95 post |
|---|---:|---:|
| vector | 719 ms | 4584 ms |
| union | 505 ms | 590 ms |
| hybrid | 746 ms | 4586 ms |

Read this as **two distinct facts** — *do not* report a single "2064 → 746 ms on the same set" speedup, which would conflate two different query mixes:

1. **On this golden set, the graph leg is inert, so hybrid ≈ union.** Post-index hybrid p50 (746 ms) sits right next to union (505 ms) and vector (719 ms) — consistent with the weight-tuning finding above that these 12 drawer-anchored queries surface zero graph candidates. There is no graph walk to accelerate here, so the indexes do little on this set.
2. **The pre-index 2064 ms hybrid p50 came from a graph-*firing* query set** (entity-anchored queries that actually walk AGE). The AGE-index speedup lands on *those* queries — the ones that traverse the graph — not on these drawer queries. So the index fix is validated where the graph walk is on the critical path; this golden set just isn't that workload.

**p95 caveat:** vector & hybrid show a ~4.5 s p95, but it is an *intermittent per-query FlashRank rerank tail*, not the graph walk — a single same-query ×8 repeat stays ~640 ms with no tail, and `union` (which reranks via the same endpoint, no graph) keeps a tight 590 ms p95. The tail lives in the larger vector-candidate-pool rerank, addressable independently of #111.

## Shipped

- **mempalace:** `_hybrid_weights()` reads `PALACE_HYBRID_VECTOR_WEIGHT` / `PALACE_HYBRID_BM25_WEIGHT` live from the environment (mirrors the rating/recency gates), wired into `_hybrid_rank` — landed as **techempower-org/mempalace#342**. **Defaults unchanged at 0.6/0.4** — behavior is identical unless an operator opts in. Unparseable/negative values fall back to default. 5 new unit tests; all hybrid/rrf/closet tests green.
- **Operating recommendation:** set `PALACE_HYBRID_VECTOR_WEIGHT=0.85` and `PALACE_HYBRID_BM25_WEIGHT=0.15` on the daemon to adopt the tuned operating point. Default left at 0.6/0.4 because n=12 is below the n≥25 threshold for flipping a global default; the knob lets the operating point move without a code change.

## Caveat

n=12 golden queries. The recommended weight is a documented, reproducible operating point on this set, not a corpus-general claim — consistent with SME's diagnostic posture (deltas under controlled conditions, not absolute scores). A larger labeled set would be needed to flip the mempalace default.
