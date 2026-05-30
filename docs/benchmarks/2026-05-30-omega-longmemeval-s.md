# OMEGA on LongMemEval-S — first independent competitor run

**Date:** 2026-05-30
**Author:** Solara (SME dream-team)
**Branch / PR:** `bench/run-omega-longmemeval`
**Addresses:** [techempower-org/multipass-structural-memory-eval#178](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/178) — Tier 3 (independent multi-system head-to-head), the "real moat" from `docs/research/2026-05-29-comparison-readiness.md` §4.

## TL;DR

The first time [OMEGA](https://pypi.org/project/omega-memory/) is measured on **our**
conditions rather than its own leaderboard. On a category-stratified LongMemEval-S
sample (n=150, 25 per `question_type`) — the **same subset, content rendering, and
session-level R@K metric** as the published mempalace-daemon baseline — OMEGA's
**retrieval is strong and mid-pack, ~2.7pp below the daemon on R@5:**

| System (retrieval-only) | R@1 | R@5 | R@10 | n |
|---|---:|---:|---:|---:|
| mempalace-daemon `/search` | 0.8533 | **0.9267** | 0.9267 | 150 |
| **OMEGA (on our harness)** | 0.8000 | **0.9000** | 0.9000 | 150 |
| **Δ (OMEGA − mempalace)** | −0.053 | **−0.027** | −0.027 | |

- ΔR@5 = **−2.67pp = 4 questions of 150** — both substrates retrieve the evidence
  session for ~90% of questions.
- This is a **diagnostic delta under controlled conditions, not a leaderboard score**
  (per CLAUDE.md). The value is the *on-harness OMEGA-vs-mempalace* comparison, which
  no one else has run.

**E2E QA** (o4-mini reader + canonical `gpt-5.3-chat` judge, same n=150, `sme-rich` =
dates present): OMEGA QA = **0.593 macro** (headline; micro 0.653 secondary) — ≈ at parity
with the same-reader mempalace-daemon retrieved-context comparator (**0.580 macro**, both
labeled "o4-mini reader, retrieved context"). The mempalace 0.610/0.868 oracle figures are
a different axis (the mempalace gradient), **not** the OMEGA comparator. See the *E2E QA
leg* section below, including the temporal-reasoning content-rendering finding (cat_6) that
cross-validates SME's "what reaches the reader is the lever" ingest thesis.

## The embedding-model catch (verify the system is doing what it claims)

OMEGA installs via `pip` **without its `bge-small-en-v1.5` ONNX embedding model
present.** On first use it logs a *"Embedding model is None, circuit-breaker tripped.
Using hash fallback"* warning and **silently degrades to FTS5 keyword matching.** Under
that mode a paraphrased natural-language question whose tokens don't overlap the stored
text returns **nothing** — so a naive benchmark would have published an unfairly low
OMEGA number that measured a *broken install*, not OMEGA.

We restored semantic mode before benchmarking (`omega setup --download-model`, a 133MB
ONNX download) and **verified** it: the paraphrase *"What pet does Maria have?"* now
surfaces *"a golden retriever named Biscuit"* with zero token overlap (keyword-only
returned `NO_RESULTS` for the same query). All numbers here are post-restore, in OMEGA's
intended semantic configuration.

This is a concrete instance of SME's whole diagnostic posture — *is the system actually
doing what it claims?* — applied to a competitor. A fair head-to-head requires verifying
each system is in its intended configuration first.

## Method / disclosure (per comparison-readiness §3.4)

- **System:** `omega-memory` 1.4.15 (pip), local SQLite + `sqlite-vec` with 384-dim
  `bge-small-en-v1.5` ONNX embeddings (semantic mode, **not** FTS5 hash-fallback).
  SME adapter at merge commit `954a9c2` (includes
  [#186](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/186)
  Gemini-review fixes on top of #183).
- **Dataset:** LongMemEval-S (`longmemeval_s_cleaned.json`), **n=150 stratified** (25
  each × 6 `question_type`s) via the shared `_stratified_cap` round-robin. The subset is
  **deterministic** (RNG-free) and **identical to the mempalace strat150 baseline** —
  the first five `question_id`s match byte-for-byte. **0 abstention questions** in this
  subset, so QA (next leg) is pure CORRECT/INCORRECT.
- **Content rules:** `upstream-exact` (user turns only) — same rendering as the daemon
  baseline.
- **Ingest granularity:** one OMEGA memory per session (avg 48 sessions/question),
  matching the daemon's one-drawer-per-session topology.
- **Retrieval metric:** OMEGA **session-level hit@K** — each session is stored tagged
  with its `session_id`; OMEGA returns that `session_id` on every `query_structured`
  hit, so hit@K is a set-membership test against the question's `expected_sources`. This
  is the OMEGA analogue of the daemon's `session→drawer_id` map (`drawer_hit_at_K`,
  #58/#98). It is **not** the substring `sme_recall` matcher, which is structurally 0
  under `upstream-exact` (session ids never appear in user-turn text) — a metric
  artifact, not an OMEGA deficiency.
- **Isolation:** per-question `OMEGA_HOME` tempdir (local SQLite); the production
  familiar / palace-daemon were never touched.
- **Errors:** 0 adapter errors across all 150 questions.

## Per-category R@5 (suggestive only — n=25/category)

| SME category (question_type) | n | mempalace `/search` R@5 | OMEGA R@5 | Δ |
|---|---:|---:|---:|---:|
| cat_1 — single-session (user/asst/pref) | 75 | 0.9067 | 0.8400 | −0.0667 |
| cat_2c — multi-session | 25 | 0.9600 | 0.9200 | −0.0400 |
| cat_3_partial — knowledge-update | 25 | 0.9600 | **1.0000** | +0.0400 |
| cat_6 — temporal-reasoning | 25 | 0.9200 | **0.9600** | +0.0400 |

**Directional pattern (hypothesis, NOT a finding):** OMEGA matches or beats the daemon
on the structure-discriminated categories (knowledge-update, temporal-reasoning) and
trails on single-session recall. At n=25/category every ±0.04 is ±1 question — within
sampling noise; do not report as effects. A full-500 stratified run would be needed to
confirm.

## Comparability caveats (label the matrix row with all three)

1. **R@K parity is real.** OMEGA's session-level hit@K and the daemon's `drawer_hit_at_K`
   both answer "did retrieval surface the evidence *session*?" on the identical subset.
2. **Ingest granularity matches** (one memory/drawer per session).
3. **NOT comparable to OMEGA's self-reported 95.4%.** OMEGA's leaderboard number is
   **E2E QA with GPT-4.1 as the answer model**; answer-model variation is ~24pp
   (`2026-05-24-memory-system-benchmarks.md` §3). The gap between our on-harness OMEGA
   number and OMEGA's 95.4% is **not** a substrate delta and must **not** be framed as
   "OMEGA beats/underperforms its claim." The defensible comparison is
   **OMEGA-vs-mempalace on this harness** (identical corpus, reader, judge) — the moat.

## E2E QA leg (follow-up — reader + canonical judge)

Reader **o4-mini**, judge **`gpt-5.3-chat` + canonical type-specific prompts**, same n=150
stratified subset. **The headline OMEGA QA number is the `sme-rich` run** (dates present);
the `upstream-exact` run is reported only as a date-stripped diagnostic (see below).

**Headline: macro 0.593.** Macro (unweighted mean of the four equal-n categories) is the
published number because it matches the stratified-equal-n philosophy *and* the averaging of
the 0.580 comparator. Micro (per-question) is a clearly-labeled secondary — it runs higher
only because cat_1 is half the sample and scores well.

| OMEGA QA (sme-rich, fair) | n | QA |
|---|---:|---:|
| cat_1 — single-session | 75 | 0.773 |
| cat_2c — multi-session | 25 | 0.480 |
| cat_3_partial — knowledge-update | 25 | 0.760 |
| cat_6 — temporal-reasoning | 25 | 0.360 |
| **OVERALL — macro (headline)** — unweighted mean of the 4 cats | 150 | **0.593** |
| OVERALL — micro (secondary; per-question, cat_1-weighted, 98/150) | 150 | 0.653 |

sme-rich also lifts OMEGA's *own* retrieval (R@5 0.953 vs its upstream-exact 0.900 — the
documented +date-frontmatter effect). **This 0.953 is NOT a comparator to mempalace's
published R@5 0.927.** The published mempalace R@5 was measured `upstream-exact`
(`docs/benchmarks/2026-05-29-longmemeval-s-results.md` reproduce command), so the fair,
same-rendering R@5 pair is **OMEGA 0.900 vs mempalace 0.927** (both upstream-exact, mempalace
+2.7pp) — the figure on the live matrix. No mempalace `sme-rich` strat150 R@5 baseline
exists, so pairing OMEGA's 0.953 against 0.927 would be a rendering-mismatch in OMEGA's
favour; don't.

### The temporal-reasoning content-rendering finding (cat_6)

The **`upstream-exact`** QA run (user-turns-only rendering — the R@5-parity rendering)
floored **cat_6 (temporal-reasoning) at 0.04 (1/25)**, dragging its overall to 0.38. Root
cause, spot-checked: retrieval was **fine** (cat_6 hit@5 = 0.96), but `upstream-exact`
**strips the session dates**, and temporal questions need date arithmetic ("how many weeks
ago", "days between X and Y", "order these three events"). The date-starved reader answered
*"today"*, *"0 weeks ago"*, or *"I don't know."* Restoring dates (`sme-rich` frontmatter)
lifted cat_6 to **0.36** — a **+0.32** swing from a rendering change, not a substrate change.

**This cross-validates SME's own headline finding.** The published mempalace ingest-fidelity
gradient says *what reaches the reader is the lever* — `upstream-exact` ingest starves the
reader. OMEGA, an independent system, hit the **same wall**: retrieval intact, reader
date-starved. A competitor confirming the ingest-fidelity axis **strengthens** the thesis;
it is not an OMEGA weakness. The `upstream-exact` 0.38 / cat_6 0.04 numbers are a
**date-stripped diagnostic**, NOT OMEGA's headline.

### Comparator (read carefully — the obvious comparison is a category error)

The mempalace **0.610** figure (`docs/benchmarks/2026-05-29-canonical-judge-passA.md`) is
**NOT** an apples comparison to OMEGA's 0.593: it used reader **`claude-opus-4-8`** (not
o4-mini) on **`ctx=full` ORACLE** context (retrieval bypassed) with an abstention-credit
adjustment. Both a stronger reader and oracle context favor 0.610; pitting OMEGA's
o4-mini/retrieved number against it understates OMEGA.

The **clean same-reader comparator** is the mempalace-daemon run with the **identical
reader (o4-mini) + judge (gpt-5.3-chat) on retrieved context**
(`baselines/longmemeval_mempalace_daemon_2026-05-28-rerun.reagg.json`): macro over the four
matching categories = **0.580** (cat_1 0.51, cat_2c 0.74, cat_3 0.65, cat_6 0.41). Against
that, **OMEGA (0.593 macro) is ≈ at parity (+1.3pp)** with a different per-category profile
(OMEGA leads cat_1/cat_3; the daemon leads cat_2c/cat_6). Residual caveat: that daemon run
was `longmemeval_oracle` n=500, not the strat150-S subset, so haystack size differs — even
this comparator is not pixel-perfect, and both numbers are **diagnostic deltas, not
leaderboard scores**.

## Reproduce

```bash
./venv/bin/pip install 'sme-eval[omega]'
./venv/bin/omega setup --download-model          # restore semantic mode (133MB ONNX)

# R@5 (retrieval-only)
./venv/bin/python scripts/run_longmemeval_omega.py \
  --questions sme/corpora/longmemeval/data/longmemeval_s_cleaned.json \
  --max-questions 150 --stratify-by question_type --content-rules upstream-exact \
  --skip-judge --json baselines/longmemeval_omega_strat150_r5_2026-05-30.json

# E2E QA (reader + canonical judge) — sme-rich = dates present = the fair QA number
./venv/bin/python scripts/run_longmemeval_omega.py \
  --questions sme/corpora/longmemeval/data/longmemeval_s_cleaned.json \
  --max-questions 150 --stratify-by question_type --content-rules sme-rich \
  --answer-model o4-mini --judge gpt-5.3-chat \
  --json baselines/longmemeval_omega_strat150_qa_smerich_2026-05-30.json
```

## Artifacts

```
baselines/longmemeval_omega_strat150_r5_2026-05-30.json              # R@5 (PR #188)
baselines/longmemeval_omega_strat150_qa_smerich_2026-05-30.json      # E2E QA — fair (headline)
baselines/longmemeval_omega_strat150_qa_upstreamexact_2026-05-30.json # E2E QA — date-stripped diagnostic
scripts/run_longmemeval_omega.py                                     # the runner
tests/test_run_longmemeval_omega.py                                  # session-level scoring tests
```

## The definitive single-config head-to-head (#119) — spec for whoever runs it

The numbers above are honest and **publishable now** (the QA cell is 0.593 vs the
same-reader 0.580, ≈ parity, caveated "comparator is oracle-n500 not strat150-S"). What
would make the apples-cell *pixel-perfect* and retire every cross-config caveat
(0.580 / 0.593 / 0.610 / 0.900 / 0.953) is a **single run where every variable below is
held identical across both systems**. This is **infra, not a flag-swap** — the deployed
mempalace `lme_*` wings are `upstream-exact`, so the mempalace leg needs a **fresh
`sme-rich` strat150 ingest on a SCRATCH daemon** (Iris's isolated-palace pattern — **never
prod**; prod already carries ~6.3% LME pollution awaiting cleanup, familiar#92).

**The five constants that MUST match on both legs (get any wrong and the cell is invalid):**

1. **Reader** — ONE model, both legs. Choice (opus vs o4-mini) + the ~$14 opus spend is
   **JP's call** (surfaced, not picked unilaterally):
   - `claude-opus-4-8` via Bedrock (`us.anthropic.claude-opus-4-8` inference profile;
     `_BedrockOpenAIShim` in `sme/eval/answer_generator.py`; verified reachable) — Tau2
     leader, ~$14 total, ~90 min/leg.
   - `o4-mini` (Azure) — ~$0; OMEGA's `sme-rich` o4-mini number already exists (**0.593**),
     so only the mempalace scratch-daemon leg would be new.
2. **Judge** — `gpt-5.3-chat` + canonical type-specific LongMemEval prompts. Hold constant
   (do **not** swap to a Bedrock judge — that adds a variable for no gain).
3. **Rendering** — `--content-rules sme-rich` on **both** legs (dates present). The deployed
   mempalace wings are `upstream-exact`, which is **why a fresh scratch ingest is required**
   — this is the load-bearing constant the temporal (cat_6) finding exposed.
4. **Retrieval breadth** — one fixed `limit` on both legs (e.g. top-5; or coordinate with
   the #117 breadth-ladder if a wider K is chosen). Same K both sides.
5. **Subset** — `--max-questions 150 --stratify-by question_type` on
   `longmemeval_s_cleaned.json` (the deterministic `_stratified_cap` strat150 — **not** the
   `longmemeval_oracle` n=500 the current 0.580 comparator used). Same subset both sides.

Reader/judge are wired into both runners (`scripts/run_longmemeval_omega.py` and
`scripts/run_longmemeval_mempalace.py`) via the shared `generate_answer` / `grade_answer`
path, so once the scratch daemon is ingested `sme-rich`, the run is two pinned invocations.
**Until then the published apples cell stays 0.593 vs 0.580 with its honest caveat.**

## Next

- **#119** — the definitive run above, gated on a scratch-daemon `sme-rich` mempalace ingest
  + JP's reader/spend choice. Tracked, not launched.
- Optional: Hindsight (MCP) and Mem0-OSS adapters on the same harness (`#178` Tier 3
  follow-ups) to widen the independent multi-system table — same five constants apply.
