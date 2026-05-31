# Cross-encoder rerank A/B — corpus-seeded scratch daemon (2026-05-31)

**Issue:** techempower-org/multipass-structural-memory-eval#103 (routed from techempower-org/mempalace#301)
**Status:** **RESOLVED — real verdict on a corpus-complete daemon.** Supersedes the 2026-05-30 BLOCKED run (`docs/benchmarks/2026-05-30-ce-rerank-ab.md`).
**Baseline JSON:** `baselines/ce_rerank_corpus_seeded_2026-05-31.json`
**Probe set:** `probes_v2_git_derived.json` (200 probes → 49 unique target files; same set as #225 / the #162 fusion A/B)
**Method:** 3-leg rerank-model sweep over `POST /search/hybrid` (`candidate_strategy=hybrid`, `fusion_mode=rrf`, `n_results=10`), read-only, against an **isolated scratch palace-daemon** (Postgres/pgvector + Apache AGE) on `localhost:8086`, **seeded with the mempalace git/docs corpus the probes target**.

## TL;DR

With the corpus present (all **49/49** unique target files seeded; `found=120/200` vs #225's 17/200 floor),
the cross-encoder rerank stage is **neutral-to-slightly-negative** on this git/docs corpus:

- **Recall@10 is identical (0.60) across all three legs** — the relevant doc is already in the hybrid
  candidate set; reranking only reorders *within* it, it doesn't recover misses.
- **MRR slightly *drops* under rerank** (0.299 off → 0.293 small-CE → 0.284 large-CE).
- **The bigger cross-encoder is worse AND ~3× slower** (MiniLM-L-12 p50 1523 ms vs the 555 ms baseline;
  the nano TinyBERT-L-2 is actually slightly *faster* than baseline here and the least harmful).

**Recommendation: keep cross-encoder rerank OFF by default / opt-in only** (`mempalace search --rerank`,
or the per-request `rerank:true` flag on `/search/hybrid`). It does not improve recall on this corpus and
carries a real latency cost at the larger model, with a small MRR regression. This is the verdict the
BLOCKED #225 run could not give — there the corpus was absent, so the A/B measured reordering over a
17-doc floor.

## The numbers (n=200, corpus present)

| leg | rerank | model | MRR | Recall@5 | Recall@10 | found | p50 lat | p95 lat |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | off | — | 0.2994 | 0.510 | 0.600 | 120/200 | 555 ms | 754 ms |
| B | on | ms-marco-TinyBERT-L-2-v2 (nano) | 0.2927 | 0.515 | 0.600 | 120/200 | 475 ms | 735 ms |
| C | on | ms-marco-MiniLM-L-12-v2 (large CE) | 0.2840 | 0.505 | 0.600 | 120/200 | 1523 ms | 1869 ms |

Deltas vs baseline (leg A): **B** ΔMRR −0.0067, ΔR@5 +0.005, ΔR@10 0.000, Δp50 −79 ms; **C** ΔMRR −0.0154,
ΔR@5 −0.005, ΔR@10 0.000, Δp50 +968 ms.

R@10 holding constant while MRR moves is the signature of a rerank stage operating on an already-correct
candidate set: it shuffles the top-k order without adding or losing relevant docs. On this corpus that
shuffle is, on net, very slightly *worse* — the hybrid vector+BM25+AGE fusion ranking is already strong.

## Correction to the prior (#225) run's model label

#225's doc/JSON labeled the rerank-ON leg `ms-marco-MiniLM-L-6-v2`. **That model is not present in this
flashrank build** — the installed zoo is exactly `ms-marco-TinyBERT-L-2-v2` (the daemon default),
`ms-marco-MiniLM-L-12-v2`, and `ms-marco-MultiBERT-L-12`. The daemon's `PALACE_RERANK_MODEL` default is
TinyBERT-L-2-v2 (`rerank.py`), so #225's rerank-ON leg actually ran **TinyBERT-L-2**, not MiniLM-L-6. This
did not change #225's BLOCKED conclusion (recall floored regardless of model), so #225 is not reopened —
but the record should be accurate. Labels in this report are the models that actually ran.

## Seeding recipe (reproducible)

1. Scratch Postgres+pgvector+AGE in Docker (reuse the prod `mempalace-db:0.1` image — apache/age PG16 +
   pgvector), `127.0.0.1:5434`, DB `mempalace_103_scratch`, Docker **named volume** (no host bind-mount →
   cannot collide with prod). Adapted from Iris's #176 throwaway-daemon template.
2. Scratch daemon `main.py --manual --port 8086 --palace <scratch dir>` with
   `MEMPALACE_BACKEND=postgres`, `MEMPALACE_KG_BACKEND=age`, `MEMPALACE_KG_WRITETHROUGH=1`,
   `MEMPALACE_POSTGRES_DSN=...localhost:5434/mempalace_103_scratch`. Ingest-time isolation guard:
   localhost + empty palace.
3. `POST /mine` (mode=projects) over the mempalace repo dirs holding all 49 targets — `mempalace/`,
   `docs/`, `scripts/`, `tools/`, `.claude-plugin/`, `.github/`, root files, and the 6 benchmark target
   files (staged into a temp dir; the rest of `benchmarks/` is ~45 MB of result-dump noise, excluded).
   `mempalace mine` sets `metadata.source_file` per file deterministically (local-embedding chunking, no
   LLM extraction), and the scorer matches on basename — so the real repo files line up with the probes.
   Result: **6510 drawers, 249 distinct source files, all 49/49 target basenames present.**
4. `POST /backfill-age` once → AGE entity graph (76,025 entities, 0 errors) for hybrid's graph-fusion
   candidate path.
5. 3 legs, daemon restart per rerank model (`PALACE_RERANK_MODEL`), cross-encoder warmed before each
   timed leg so the cold model-load stays out of the latency distribution.
6. Teardown: `docker compose -f docker-compose.scratch-103.yml down -v` + stop the scratch daemon. Prod
   familiar (familiar:5433 / familiar:8085) never touched.

## Apparatus

- `scripts/ce_rerank_daemon_ab.py` (Nyx, #225) — the read-only daemon A/B runner; scoring contract matches
  `mempalace eval_fusion_ab.py` (1-indexed basename match, MRR = mean 1/rank, Recall@k = fraction ≤ k).
- `scripts/ce_rerank_3leg_sweep.py` (this run) — drives 3 legs that differ by rerank **model**, restarting
  the daemon per model (palace-daemon pins `PALACE_RERANK_MODEL` at startup; the per-request flag only
  toggles on/off). Reuses `ce_rerank_daemon_ab`'s `run_leg`/`evaluate_ranking`/`load_probes` verbatim, so
  the numbers are directly comparable to the #225 2-leg pass and the #162 fusion A/B.
