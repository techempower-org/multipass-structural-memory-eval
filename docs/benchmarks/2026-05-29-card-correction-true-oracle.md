# Card Correction Draft — retrieval-limited vs true-oracle oracle-QA

**Date:** 2026-05-29
**Author:** Nyx (SME dream-team)
**Status:** DRAFT for the "mempalace vs the field" comparison card. NOT the live
site card — to be ported to `docs/site/index.html` after JP approves the reframe.
**Issue:** `techempower-org/multipass-structural-memory-eval#59` / PR #162, #164

## The correction in one line

The published oracle-QA headline (**0.61**) was **not true oracle** — it was
**retrieval-limited**: the "oracle" context was built by running `/search` at
`limit=5` over the oracle haystack, so chunking + the top-5 cut dropped or
fragmented the gold for FOUR of six categories. With the gold made
**definitionally present** (evidence sessions, verbatim), the oracle-QA reader
ceiling is **0.840 CORRECT-only / 0.868 abstention-credited** (rigorous full-six
Pass A, n=500) — and the gap from 0.61 to that ceiling is **"what reaches the
reader" (ingest mode + retrieval), not reader reasoning.** The true-oracle
ceiling (0.868) lands **within ~0.2pp of the published GPT-4o oracle (0.870)**:
once the gold is actually in context, our reader essentially matches the
published oracle — the entire ~26pp apparent "gap" was substrate, with
near-zero genuine reader deficit.

## Two headline numbers (state both)

| Number | Value (credited / CORRECT-only) | What it measures |
|---|---|---|
| **Retrieval-limited oracle** (as published) | **0.610 / 0.568** | reader handed `/search@limit=5` top-5 chunks of the oracle haystack — bounded by chunking + the top-5 cut |
| **True-oracle reader ceiling** | **0.868 / 0.840** | reader handed the evidence sessions verbatim (gold definitionally present) — rigorous full-six Pass A, n=500, all categories re-run on true-oracle |
| Published GPT-4o oracle (LongMemEval) | 0.870 | external reference |

**The true-oracle reader ceiling (0.868) ≈ the published GPT-4o oracle (0.870).**
The full-six Pass A is the rigorous number (not a swap-estimate): all six
categories were re-run on true-oracle. A swap-estimate that *carried over* the
three non-floor categories at their retrieval-limited values gave 0.828 — but
the rigorous run shows knowledge-update (+20pp) and single-session-preference
(+13pp) were ALSO retrieval-limited, so the true ceiling is 0.868. Only
single-session-user genuinely carried over (0.929→0.929).

## Per-category: retrieval-limited vs true-oracle

Reader = `claude-opus-4-8`; judge = `gpt-5.3-chat` + canonical type-specific
prompts (#146); abstention-credited via `is_abstention` (#148). "RL" =
retrieval-limited (`/search@limit=5`, the published substrate); "TO" =
true-oracle (evidence sessions present).

| category | n | RL (credited) | TO (credited) | Δ | dominant cause of the RL→TO gap |
|---|---|---|---|---|---|
| single-session-assistant | 56 | 0.321 | **0.982** | +66pp | **ingest mode** — upstream-exact stripped assistant turns (gold never ingested) |
| temporal-reasoning | 133 | 0.361 | 0.752 | +39pp | **retrieval** — limit=5 dropped a date-bearing session |
| knowledge-update | 78 | 0.705 | 0.910 | +21pp | **retrieval** — limit=5 dropped/fragmented the updated value |
| multi-session | 133 | 0.714 | 0.865 | +15pp | **retrieval** — limit=5 dropped some scattered mentions |
| single-session-preference | 30 | 0.800 | 0.933 | +13pp | **retrieval** — limit=5 dropped some preference-evidence turns |
| single-session-user | 70 | 0.929 | 0.929 | +0 | gold in user turns, single salient mention → present on both |
| **OVERALL** | **500** | **0.610** | **0.868** | **+26pp** | **what reaches the reader** |

The TO column is the rigorous full-six Pass A (every category re-run on
true-oracle, preference reader). FIVE of six categories lifted — only
single-session-user (single salient user-turn fact, reliably in the top-5)
carried over. The +26pp overall is the entire apparent oracle gap, and it is
substrate (ingest + retrieval), not reader reasoning.

## The honest decomposition (the reframe)

The oracle-QA gap from the published 0.61 up to the ~0.87 published GPT-4o
ceiling splits into two parts, and they are not the same kind of problem:

1. **0.61 → ~0.87 (the big part, ~22pp): WHAT REACHES THE READER.** Not reader
   reasoning — the gold simply wasn't in the context the reader received.
   Sub-causes:
   - **Ingest mode** (single-session-assistant): the pinned context used
     `upstream-exact` ingest (`loader.py:333`, user-turns-only — an R@5-parity
     mode, #51), which strips assistant turns; for assistant-authored answers
     the gold was dropped at ingest. Fix: `sme-rich` ingest.
   - **Retrieval breadth** (temporal, multi-session): `/search@limit=5` +
     chunking dropped/fragmented gold sessions. Fix: higher limit / all-session
     retrieval for these categories.
2. **~0.87 → ~0.87 (the small part): GENUINE READER residual.** Real reasoning
   error even with the gold present:
   - temporal date arithmetic / relative-time anchoring (~+4pp recoverable by a
     date-reasoning reader clause, `temporal_cot`),
   - multi-session aggregation / dedup (~+1.5pp recoverable by `dedup_count`),
   - single-session-assistant: ~0 residual (true-oracle hits 1.00).

So the SME decomposition cleanly separates **substrate-limited** (recoverable by
ingest + retrieval fixes) from a **small genuine reader residual** — a sharper,
more defensible story than "the reader leaves 26pp on the table."

## Reader-prompt clauses (validated on true-oracle, where they can fire)

| clause | category | true-oracle lift vs preference |
|---|---|---|
| assistant_trust | single-session-assistant | +1.8pp (preference 0.982 → 1.000) |
| temporal_cot (v1) | temporal-reasoning | +3.8pp (0.729 → 0.767) |
| dedup_count | multi-session | +1.5pp |

(Lucid's refined `temporal_cot_v2` relative-anchoring clause, #160, did NOT beat
v1 — net −2 flips; keep v1. The clauses are ~flat on the retrieval-limited
substrate because the gold often isn't present to reason over.)

## Caveat — front and center

**True oracle = evidence-sessions-present = the reader ceiling once retrieval is
perfect.** A real deployed system must still *retrieve* those sessions: the
**DEPLOYED** number (real chunked `/search` against the production daemon) is a
separate measurement that still needs a daemon re-test (held while the daemon
lane is in use). The two numbers answer different questions:
- **0.61 (retrieval-limited):** what our *current pinned* substrate delivered.
- **~0.87 (true-oracle ceiling):** what the reader achieves when retrieval is
  perfect — the upper bound the deployed system trends toward as ingest +
  retrieval improve.

Neither is the canonical `gpt-4o-2024-08-06` judge (we use gpt-5.3-chat +
canonical prompts; disclosed per #146).

## Artifacts

- True-oracle floor baselines: `baselines/reader_trueoracle_{ss-assistant,temporal,multi-session}_2026-05-29.json`
- ss-assistant sme-rich (ingest-fixed): `baselines/reader_floorlift_ss-assistant_smerich_2026-05-29.json`
- Non-floor true-oracle: `baselines/reader_trueoracle_{ss-user,ss-preference,knowledge-update}_2026-05-29.json`
- Findings: `docs/benchmarks/2026-05-29-true-oracle-floor.md`, `docs/benchmarks/2026-05-29-reader-floor-lift-results.md`
- As-published baseline: `baselines/reader_sweep_passA_canonical-judge_opus-preference_search-default_2026-05-29.json`
