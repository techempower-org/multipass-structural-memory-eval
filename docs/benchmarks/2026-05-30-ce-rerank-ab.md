# Cross-encoder rerank A/B — BLOCKED on corpus availability (2026-05-30)

> **SUPERSEDED 2026-05-31** by `docs/benchmarks/2026-05-31-ce-rerank-corpus-seeded.md` — the A/B was
> re-run on a scratch daemon seeded with the git/docs corpus (all 49/49 targets present). Real verdict:
> rerank is neutral-to-slightly-negative (R@10 flat at 0.60 across legs, MRR slightly drops, bigger CE
> ~3× slower) → keep rerank OFF / opt-in. Note also: this run's rerank-ON model label
> `ms-marco-MiniLM-L-6-v2` was stale — that model isn't in this flashrank build; the daemon default
> `ms-marco-TinyBERT-L-2-v2` actually ran. (Neither correction changes the BLOCKED conclusion below.)

**Issue:** techempower-org/multipass-structural-memory-eval#103 (routed from techempower-org/mempalace#301)
**Status:** **BLOCKED — target corpus absent from prod familiar.** Not a rerank verdict. (Superseded — see banner above.)
**Evidence JSON:** `baselines/ce_rerank_ab_2026-05-30.json` (`status: blocked_corpus_mismatch`)
**Probe set:** `probes_v2_git_derived.json` (200 probes; same set as the #162 fusion A/B), generated 2026-05-23 from ~14 months of git history
**Method:** daemon `POST /search/hybrid` per-request `rerank` flag — `rerank:false` vs `rerank:true`, read-only, against prod familiar (`familiar:8085`, Postgres/pgvector + Apache AGE)

## TL;DR

The A/B ran clean (EXIT=0, n=200, both legs) but is **not publishable as a rerank result**: the
200 git-derived probes target mempalace's own **git/docs corpus** (e.g. `pgvector-cutover-runbook.md`,
`postgres.py`, `searcher.py`), and that corpus is **no longer present in prod familiar** — familiar is
now conversational-only. Only **17/200** probe targets appeared anywhere in the top-10 (Recall@10 =
0.085). With the relevant set absent, the rerank has nothing to reorder, so the delta (R@5 +0.005,
MRR +0.0073) is corpus-floor noise, not signal. **Do not flip the rerank default on this evidence.**

This is a measurement-validity finding, not a failed run: a clean exit code masked a near-empty corpus.

## The numbers (floored — do not cite as a rerank verdict)

| leg | MRR | Recall@5 | Recall@10 | found | mean lat | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rerank OFF (baseline) | 0.0368 | 0.055 | 0.085 | 17/200 | 2717 ms | 2165 ms | 6337 ms |
| rerank ON (L-6) | 0.0441 | 0.060 | 0.085 | 17/200 | 2569 ms | 2102 ms | 5858 ms |
| **delta (ON − OFF)** | +0.0073 | +0.005 | 0.000 | — | **−148 ms** | — | — |

Per-probe: 7 improved, 3 regressed under rerank — all within the 17-doc floor.

The **−148 ms "overhead"** is physically impossible for an additive rerank stage and confirms the run is
not interpretable: it's a sequential-leg artifact (leg B ran after leg A, warm caches / writethrough
drift), not a measurement of rerank cost.

## Root cause (verified)

1. **Probe targets are mempalace git/docs.** probe[0] expects `pgvector-cutover-runbook.md`; the set is
   dominated by `.md`/`.py`/`.json` source files from the mempalace tree.
2. **Those files are absent from familiar.** Three independent doc-targeted probes, queried directly
   against familiar `/search/hybrid` (rerank off):
   - `postgres.py` — not in top-10 (hits: all `.jsonl` conversation logs + bench `.json`)
   - `searcher.py` — not in top-10 (hits: all `.jsonl` conversation logs)
   - `migrate_to_postgres.py` — not in top-10 (hits: `.jsonl` + the probe file itself, not the source tree)
3. **Historical regression confirms the corpus dropped out.** The same `probes_v2` corpus_version scored
   **mean_recall 0.47** against familiar on 2026-05-16
   (`baselines/mempalace_git_probes_v2_familiar_2026-05-16_n20.json`). Today: **0.085**. The git/docs
   corpus that was in familiar in mid-May is gone — consistent with familiar being cleaned to
   conversational-only (the #121 `lme_*` purge era; see `familiar_palace_conversational_only`).
4. **Daemon is healthy.** `available_in_scope = 409190`, daemon v1.9.1, 0 db errors in window. This is
   genuine corpus-content absence, not a daemon hiccup.

## Why the shipped harness couldn't have caught this either

The mempalace harness (`scripts/eval_cross_encoder_rerank.py`) runs **in-process** against a local
ChromaDB palace and toggles rerank via `MEMPALACE_RERANK_CROSS_ENCODER`. The local chroma palace was
retired 2026-05-14 (pgvector cutover); its surviving snapshot has **2/200** probe coverage. So neither
the local nor the prod path currently holds the corpus #103 needs. The per-request `rerank` flag on
`/search/hybrid` (the palace-daemon#189 toggle that `2026-05-29-candidate-strategy-postfix.md` was
blocked on) works correctly — the blocker is corpus, not apparatus.

## Recommendation

**Do NOT flip the rerank default.** Insufficient data: the A/B measured reordering over a 17-doc floor.

To actually run #103, two paths forward:
1. **Seed a scratch daemon** with the mempalace git/docs corpus indexed (mine the mempalace tree into an
   isolated palace), then run this exact runner (`scripts/ce_rerank_daemon_ab.py`) against it. Faithful to
   the 200-probe set; no prod contact. Preferred.
2. **Locate an existing git-indexed palace.** If a palace still holds the 2026-05-16-era git/docs corpus,
   point the runner at it. Unverified that one survives.

The 12-query golden set (`rerank_eval_queries.json`, #75) DOES live in familiar, but n=12 is far below the
n≥25 threshold for a 200-probe-class A/B and can't substitute here.

## Apparatus (reusable once a corpus exists)

`scripts/ce_rerank_daemon_ab.py` — daemon-routed, read-only, stdlib-only. Loads the dict-shaped probe
file directly, A/Bs `rerank:false`/`rerank:true` over `/search/hybrid`, and scores with the exact
contract of `mempalace/scripts/eval_fusion_ab.py` (1-indexed basename match; MRR averages `1/rank`;
Recall@k = fraction `<= k`). Point `--api-url` at a corpus-complete daemon to get a real verdict.
