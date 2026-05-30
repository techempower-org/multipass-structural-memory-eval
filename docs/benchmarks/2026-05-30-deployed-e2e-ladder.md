# Deployed LongMemEval QA — Retrieval-Breadth Ladder (limit 5 → 20 → 50)

**Date:** 2026-05-30
**Author:** Echo (SME dream-team)
**Branch / PR:** `bench/117-deployed-e2e-ladder`
**Addresses:** `techempower-org/multipass-structural-memory-eval#117` (re-run the deployed pipeline at higher retrieval breadth) / `#116` (the reader-sweep / pinned-context program this builds on)

## Bottom line

The published "deployed E2E QA = 0.610" was **retrieval-limited at `/search` limit=5**, and the limit was baked into a *pinned-context reader sweep*, never a live variable-breadth retrieval. Re-querying the same strat150 wings at **limit=20 and limit=50** — read-only, same reader, same judge — recovers most of the gap to the reader ceiling:

| retrieval limit | QA-acc | n | CORRECT | INCORRECT | ERROR |
|---|---|---|---|---|---|
| **5** (re-measured anchor) | **0.5667** | 150 | 85 | 65 | 0 |
| **20** | **0.7400** | 150 | 111 | 39 | 0 |
| **50** | **0.7600** | 150 | 114 | 35 | 1 |

- **+17.3pp from limit=5→20**, then **+2.0pp from 20→50** — a clear **plateau at ~20**. Widening the retrieval window is the single biggest deployed-substrate lever measured to date.
- Against the true-oracle reader ceiling (**0.868**, gold present — see `2026-05-29-true-oracle-floor.md`), limit=20 closes **~58%** of the 0.5667→0.868 gap and limit=50 **~64%**. The remaining ~11pp is the reader/substrate floor (single-session-assistant + the residual KU/temporal synthesis), not retrieval breadth.
- This is the **first true variable-breadth deployed retrieval number** for mempalace. The prior 0.610 was a fixed-window artifact.

## The load-bearing methodology finding

The published 0.610 (precisely **0.6133**, `docs/benchmarks/2026-05-29-canonical-judge-passA.md` §8) was produced by **replaying a pinned-context JSON** (`scratch/sme-bench-2026-05-29/pinned_search-default_strat150.json`) through the reader sweep. That pinned context was captured by a one-shot retrieval pass at **`/search` limit=5**. So:

> "deployed E2E 0.610" was really **limit=5-pinned-context reader QA**, not a live, variable-breadth retrieval measurement.

The reader sweep can only re-read whatever the pinned capture froze; it cannot retrieve *more*. To vary retrieval breadth you must re-run retrieval. This note does that. The framework exists to surface exactly this kind of "the published number was optimistic / under-characterized" distinction — here it sharpens *what* 0.610 measured rather than just reporting a bigger number.

## How it was run (read-only — zero new writes to prod)

The strat150 haystacks were **already ingested** in JP's live prod familiar palace (`http://familiar:8085`, DB `mempalace_2026_05_13`) under `lme_<question_id>` wings — the original #116/#91 runs persisted them via `POST /memory` and never cleaned up. So no re-ingest was needed: the ladder re-queries those existing wings **read-only via `GET /search`** at limit=5/20/50, assembling the context_string with the identical wing-scoped adapter the deployed pipeline uses (`scripts/run_longmemeval_mempalace.py::_make_wing_scoped_daemon_adapter`). The only changed variable is `n_results`.

Each leg's freshly-captured context is then replayed through the **same reader and judge as the 0.6133 baseline**:

- **Reader:** `claude-opus-4-8` + `preference` prompt, full context (AWS Bedrock)
- **Judge:** `gpt-5.3-chat` + canonical type-specific LongMemEval prompts (Azure) — NOT `gpt-4o-2024-08-06` (not deployed on this resource); the categories are un-collapsed by the *prompts*, per `2026-05-29-canonical-judge-passA.md`
- **Subset:** strat150 — LongMemEval-S oracle, 25/type × 6 types, 0 abstention records

> **Prod-pollution flag (separate issue):** 611 `lme_*` wings / 27,487 drawers (~6.3% of the palace's 436,646) are standing LongMemEval pollution from the May-29 ingests. Tracked for cleanup as `familiar#92` / task #121. This run added **zero** new drawers.

## Content-filter accounting (gates comparability)

Azure's content filter tripped on exactly **one** question — qid `95228167` (single-session-preference), `hate: severity=medium` (a known false-positive the canonical-judge doc already documents) — at **limit=50 only**. The judge retried 3× then returned `ERROR`.

- **The ERROR is counted in the denominator** as effectively-wrong (`qa_acc = CORRECT / n`, `n=150` includes the ERROR row) → a **conservative floor**, not an exclusion.
- **limit=5 and limit=20 had zero filter hits.** All three legs share an **identical denominator of 150** with zero abstentions and zero exclusions → the ladder is **strictly comparable**.
- Sensitivity: if that one ERROR were actually CORRECT, limit=50 = 115/150 = 0.7667 (+0.7pp) — does not touch the plateau conclusion.

## Per-category breakdown

| question_type | n | limit=5 | limit=20 | limit=50 |
|---|---|---|---|---|
| knowledge-update | 25 | 0.560 | 0.760 | 0.840 |
| multi-session | 25 | 0.480 | 0.760 | 0.720 |
| single-session-assistant | 25 | 0.160 | 0.280 | 0.280 |
| single-session-preference | 25 | 0.800 | 0.800 | 0.840 |
| single-session-user | 25 | 0.880 | 0.920 | 1.000 |
| temporal-reasoning | 25 | 0.520 | 0.920 | 0.880 |
| **OVERALL** | **150** | **0.5667** | **0.7400** | **0.7600** |

The breadth lift is concentrated in the **synthesis-heavy categories**: temporal-reasoning (+40pp at limit=20), multi-session (+28pp), knowledge-update (+20pp). These are precisely the question types that need evidence spread across *several* sessions — limit=5 starves them; limit=20 feeds them. single-session-assistant stays the floor (0.16→0.28) — it is a reader/grounding problem, not a retrieval-breadth problem, so widening the window barely moves it.

## Reconciliation: re-measured limit=5 (0.5667) vs published 0.6133 (−4.7pp)

Same reader, same judge, same subset, same prompt — the **only** difference is the retrieval source: my live re-query of the prod wings vs passA's cached pinned context from 2026-05-29. The delta localizes almost entirely to two categories:

- **knowledge-update: 0.720 → 0.560 (−16pp)** — the dominant driver
- single-session-assistant: 0.240 → 0.160 (−8pp); single-session-preference: −4pp
- multi-session, single-session-user, temporal-reasoning: **identical** at limit=5

**Mechanism: daemon retrieval drift, quantified.** Diffing the 2026-05-29 cached limit=5 context against the 2026-05-30 live limit=5 re-query, per question (n=150):

| | count | share |
|---|---|---|
| identical top-5 context | 76 | 51% |
| **changed** | **74** | **49%** |
| &nbsp;&nbsp;— same hits, fuller/reordered chunk text | 10 | 7% |
| &nbsp;&nbsp;— **different hit-set in the top-5** | **64** | **43%** |

Mean context grew +32% (2594 → 3421 chars). So **half the subset's top-5 retrieval moved** between the cached capture and the live daemon — the dominant component (64 questions) is a genuinely *different set of chunks* surfacing in the top-5.

**Mechanism — a daemon-deploy boundary, traced.** The deployed familiar daemon process restarted at **2026-05-30 00:16 UTC** (now v1.9.1), squarely between passA's cached capture (2026-05-29 17:37 UTC) and this live re-query (2026-05-30 15:57 UTC). Both runs hit the **plain `/search`** endpoint (`kind=all`), whose top-5 is vector candidates **reordered by a FlashRank cross-encoder** (`ms-marco-TinyBERT-L-2-v2`, `PALACE_RERANK_ENABLED=true`) — a singleton reloaded on every daemon restart. A reranker reorder over an unchanged candidate pool produces exactly this signature: same wings, different top-5 selection, fuller chunks surfacing.

Two alternative causes were checked and **ruled out**:
- **Not the #202 age-fused hydration fix** (deployed at the same 00:16 restart): that fix lives inside the `/search/age-fused` POST handler (graph-only drawer hydration); neither run used age-fused — both used plain `/search`, which never calls that code.
- **Not a corpus/data change**: the strat150 `lme_*` drawers were filed 2026-05-25 (predating both runs) and the live palace shows 0 empty `document` / 0 null `doc_tsv` across all 27,487 lme drawers — the candidate text was fully populated for both runs, so the +32% is a *selection* change (which chunks reach the top-5), not a hydration/repopulation change.

The shift is therefore a **one-time deploy-boundary rerank/ranking change, not an ongoing regression**: the live 0.5667 reflects the *current* deployed daemon and supersedes the 0.6133 measured on the pre-restart state. (Same haystack, same sessions, same `upstream-exact` rendering throughout — only the daemon process changed.)

4 of 6 categories are unaffected at the QA level, so this is not globally worse retrieval — it concentrates in knowledge-update (−16pp), the category that punishes a top-5 reorder hardest: KU needs the *latest* value, and if the reranker surfaces an older revision of the fact inside a now-different top-5, the reader answers stale. The breadth ladder itself corroborates this — at limit=20, KU recovers 0.560 → 0.760: widening the window pulls *both* the old and the updated revision into context regardless of rerank order, and the reader correctly prefers the update.

**The finding:** the published 0.6133 was a **cached-context snapshot of a since-drifted daemon**. The self-consistent ladder (all three legs re-queried against identical *current* daemon state) is the methodologically clean comparison, and the 5→20→50 deltas are unconfounded by drift because every leg sees the same daemon.

## Reproducing

The strat150 subset is the same question SET the published 0.6133 used, deterministically derivable from the committed corpus (the QA aggregator is order-independent):

```bash
# read-only re-query at a given breadth (self-contained subset derivation)
./venv/bin/python scripts/requery_deployed_breadth.py \
    --from-oracle sme/corpora/longmemeval/data/longmemeval_oracle.json \
    --limit 20 --api-url http://familiar:8085 \
    --out baselines/pinned_strat150_limit20.json
# replay through the same reader + judge as 0.6133
./venv/bin/python scripts/reader_sweep_eval.py reader-sweep \
    --pinned baselines/pinned_strat150_limit20.json --headline \
    --reader-models claude-opus-4-8 --prompts preference --context-widths full \
    --judge gpt-5.3-chat --concurrency 6 \
    --json baselines/longmemeval_deployed_qa_strat150_limit20_2026-05-30.json
```

`--from-oracle` derives the subset via the same `_stratified_cap(oracle, 150, question_type)` the deployed pipeline uses, so no scratch dependency. (The actual ladder run used `--source-pinned` against the #116 capture; both yield the identical question set.)

## Artifacts

- Baselines: `baselines/longmemeval_deployed_qa_strat150_limit{5,20,50}_2026-05-30.json`
- Read-only re-query harness: `scripts/requery_deployed_breadth.py` (+ `tests/test_requery_deployed_breadth.py`)
- Reader-sweep replay: `scripts/reader_sweep_eval.py` (unchanged)
- Published anchor (limit=5 pinned, 0.6133): `baselines/reader_sweep_stacked_opus-pref_canonical-judge_2026-05-29.json`
- True-oracle ceiling (0.868): `docs/benchmarks/2026-05-29-true-oracle-floor.md`
- Judge provenance: `docs/benchmarks/2026-05-29-canonical-judge-passA.md` §8
