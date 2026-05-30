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
dates present): OMEGA QA = **0.593 macro / 0.653 micro** — ≈ at parity with the
same-reader mempalace-daemon retrieved-context comparator (0.580 macro). See the *E2E QA
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

| OMEGA QA (sme-rich, fair) | n | QA |
|---|---:|---:|
| cat_1 — single-session | 75 | 0.773 |
| cat_2c — multi-session | 25 | 0.480 |
| cat_3_partial — knowledge-update | 25 | 0.760 |
| cat_6 — temporal-reasoning | 25 | 0.360 |
| **OVERALL — macro** (unweighted mean of the 4 cats) | 150 | **0.593** |
| **OVERALL — micro** (per-question, 98/150) | 150 | **0.653** |

(Macro and micro differ because cat_1 is half the sample and scores high; cite the same
averaging the comparator uses.) sme-rich also lifts retrieval (R@5 0.953 vs upstream-exact
0.900 — the documented +date-frontmatter retrieval effect).

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

## Next

- **E2E QA leg** (in flight): o4-mini reader + canonical `gpt-5.3-chat` judge, same
  subset → first independent OMEGA QA number on our harness, directly comparable to the
  mempalace E2E QA row.
- Optional: Hindsight (MCP) and Mem0-OSS adapters on the same harness (`#178` Tier 3
  follow-ups) to widen the independent multi-system table.
