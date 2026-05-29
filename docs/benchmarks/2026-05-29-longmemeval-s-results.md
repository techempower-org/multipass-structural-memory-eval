# LongMemEval-S: `/search` vs `/search/age-fused` — retrieval A/B

**Date:** 2026-05-29
**Issue:** [techempower-org/multipass-structural-memory-eval#91](https://github.com/techempower-org/multipass-structural-memory-eval/issues/91)
**Prep:** [`2026-05-29-longmemeval-s-prep.md`](2026-05-29-longmemeval-s-prep.md) (loader + ingest-cost sizing, Somnia)
**Adapter:** `mempalace-daemon` @ familiar:8085 · **retrieval-only** (`--skip-judge`) · R@K via the #98 chunk-suffix drawer matcher · no Azure.

## TL;DR

On a **representative, category-stratified** LongMemEval-S sample (n=150, 25 per
question_type), **age-fusion shows no significant retrieval gain over plain
`/search`:**

| Leg | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| `/search` | 0.8533 | **0.9267** | 0.9267 |
| `/search/age-fused` | 0.8667 | **0.9200** | 0.9200 |
| **age-fusion Δ** | **+0.0134** | **−0.0067** | −0.0067 |

- ΔR@5 = **−0.0067 = −1 question of 150** — statistically indistinguishable from `/search`.
- ΔR@1 = **+0.0134 = +2 questions** — a small, non-significant edge.
- **The +2.0pp R@5 "win" seen on an earlier n=100 slice did NOT replicate** — that
  slice was single-session-dominated (see below). On representative data the win vanishes.

**Diagnostic posture (per CLAUDE.md): this is a delta under controlled conditions,
not a leaderboard score.** The honest read: age-fusion is a *targeted* re-ranker,
not a blanket retrieval improvement.

## The methodology trap this run exists to avoid

A first run capped at n=100 (`--max-questions 100`, no stratification) reported
`/search` R@5 = 0.940, age-fused R@5 = 0.960 — an apparent **+2.0pp age-fusion win**.
But the S corpus is **sorted by `question_type`**, so the first 100 questions were
**70 single-session-user + 30 multi-session** — zero temporal-reasoning,
knowledge-update, or single-session-{assistant,preference}. The "win" was a
**category-composition artifact**, not a robust result. This is filed as
[techempower-org/multipass-structural-memory-eval#122](https://github.com/techempower-org/multipass-structural-memory-eval/issues/122)
and fixed by the new `--stratify-by question_type` flag, which this run uses:
25 questions per question_type → 150 total, all 6 categories represented.

The n=100 absolute is not wrong as a *data point* (it measures single-session
retrieval), but it is **not representative**, and the A/B delta computed on it
does not generalize. Same trap hit #116 Phase-1 (worked around by running the
full oracle 500).

## Per-category (suggestive only — n=25/category)

| SME category (question_type) | n | `/search` R@5 | age-fused R@5 | Δ R@5 |
|---|---:|---:|---:|---:|
| cat_1 — single-session (user/asst/pref) | 75 | 0.907 | 0.880 | −0.027 |
| cat_2c — multi-session | 25 | 0.960 | 0.920 | −0.040 |
| cat_3_partial — knowledge-update | 25 | 0.960 | **1.000** | +0.040 |
| cat_6 — temporal-reasoning | 25 | 0.920 | 0.960 | +0.040 |

**Directional pattern (hypothesis, NOT a finding):** age-fusion *helps* the
categories where graph/temporal structure is the discriminator — knowledge-update
(facts that change over time) and temporal-reasoning — and *hurts* the recall-style
categories — single-session and multi-session — where strong vector+BM25 matches
get diluted by graph neighbours.

**Why it's only a hypothesis:** at n=25/category, every ±0.04 delta is **±1
question**. These deltas are within sampling noise and must not be reported as
effects. A full-500 stratified run (≈83/category) would be needed to confirm the
pattern. The pattern is *plausible* and *directionally sensible*, which is exactly
why it deserves a larger run rather than a conclusion.

## Validations carried by this run

- **Daemon stability (palace-daemon#190).** The bench-lock + hard mine kill-switch
  held across the entire campaign — n=100 leg-1 (4.7K-POST ingest), leg-2, #116
  Phase-1 oracle (500), and this stratified run — with `restart_count = 0` and no
  crash-loop throughout. This is the same sustained write load that **killed
  Somnia's n=3 smoke** before #190 (the prep doc's "Deliverable 4"). The fix holds.
- **#98 chunk-suffix matcher.** Raw `RESULT` lines on the pre-#98 checkout read
  R@5 ≈ 0.01 (the daemon returns `<parent>_chunk_NNNNNN` ids; the old matcher
  compared them exact-string against parent ids). Re-aggregation with
  `_drawer_parent_id` recovered the true R@5 — no re-bench, just re-scoring stored
  ids. All numbers here are post-#98.
- **Warm-cache second leg.** Each leg-2 (`/search/age-fused`) reused the wings
  ingested by leg-1 (idempotent ingest), so it ran retrieval-only in a fraction of
  the time — and retrieval differed on ~100% of questions, confirming age-fusion
  was genuinely engaged, not silently falling back to `/search`.

## Reproduce

```bash
# stratified A/B, retrieval-only, both endpoints (supervisor handles bench-lock + reagg)
./venv/bin/python scripts/run_longmemeval_mempalace.py \
  --adapter mempalace-daemon --api-url http://familiar:8085 \
  --questions sme/corpora/longmemeval/data/longmemeval_s_cleaned.json \
  --max-questions 150 --stratify-by question_type \
  --skip-judge --content-rules upstream-exact \
  --search-endpoint /search --json baselines/longmemeval_s_strat150_search_2026-05-29.json
# repeat with --search-endpoint /search/age-fused
./venv/bin/python scripts/reaggregate_drawer_hits.py baselines/longmemeval_s_strat150_*_2026-05-29.json
```

## Next

- **[techempower-org/multipass-structural-memory-eval#116](https://github.com/techempower-org/multipass-structural-memory-eval/issues/116)** —
  the reader sweep (Pass A): retrieval is near-ceiling (oracle R@5 = 0.974); the
  open question is the reader's R@5→QA gap (#98's ~38pp). That, not retrieval, is
  where the jp-realm/LongMemEval gap lives.
- A **full-500 stratified** run to confirm (or kill) the per-category age-fusion
  pattern, if it proves worth the ~5h ingest.

## Artifacts (committed)

```
baselines/longmemeval_s_strat150_{search,age_fused}_2026-05-29.json{,.reagg.json}   # representative n=150
baselines/longmemeval_s_{search,age_fused}_n100_2026-05-29.json{,.reagg.json}        # n=100 slice (non-representative)
```
