# LoCoMo-10: first end-to-end QA — flat retrieval + canonical judge

**Date:** 2026-05-29
**Issue:** [M0nkeyFl0wer/multipass-structural-memory-eval#175](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/175)
**Author:** Cassia (SME dream-team)
**Builds on:** Zephyr's LoCoMo-10 flat *retrieval* row (R@5 = 0.440, n=1986; clean rank-aware, adversarial included).
**Adapter:** `flat` (ChromaDB baseline) · **end-to-end** (reader + judge) · reader = judge = `gpt-5.3-chat` (Azure Foundry).

## TL;DR

First LoCoMo **end-to-end QA accuracy** through SME's flat baseline: retrieved
context (top-5 sessions) → reader → canonical LongMemEval type-specific judge.
This is the QA number that sits on top of Zephyr's R@5 = 0.440 retrieval row —
*what the system actually answers*, not just whether the evidence session was
retrieved.

| overall (stratified, n=250) | QA accuracy |
|---|---:|
| **unweighted** (mean of 5 equal-n categories) | **0.384** |
| **proportion-weighted** (est. full LoCoMo-10) | **0.4255** |

The two overalls differ because the equal-per-type sample over-represents the
weakest categories. Reweighting each category's accuracy by its **natural share
of the 1986-QA set** (open-domain 42%, adversarial 22%, single-hop 16%,
multi-hop 14%, temporal 5%) lifts the estimate to **0.4255** — open-domain
(0.42) and adversarial (0.62) dominate the natural distribution, while the
floor categories (multi-hop 0.22, temporal 0.26) are small shares. The
proportion-weighted figure is the better estimate of "what the full LoCoMo-10
overall would be for this adapter+reader+judge."

**Diagnostic posture (per CLAUDE.md): this is a delta under controlled
conditions, not a leaderboard score.** The published LoCoMo leaderboard numbers
(EverOS 93.05% / True Memory 93.0% / Mem0 92.5% / Hindsight 89.61%) use
purpose-built memory systems **with their own retrieval + reasoning stacks and
stronger answer models**, so they are **not directly comparable** to this row.
LoCoMo QA accuracy swings heavily with the *answer model* — a flat ChromaDB
baseline with a generic reader prompt is the **floor** SME's smarter adapters
(and stronger readers) get measured against, not a competitor entry. The
honest read is the *delta* this enables: mempalace-vs-flat on the identical
subset/reader/judge (the #176 daemon comparison), where the only changed
variable is the retrieval substrate.

## Comparability contract (the pinned subset — MUST be stated)

| Constant | Value |
|---|---|
| Source | LoCoMo-10 (`locomo10.json`), full release |
| Full subset size | `SUBSET_QA_COUNT = 1986` |
| **This run** | **stratified subset, 50 / question_type, n=250, seed=1729** |
| Adversarial | **INCLUDED** (judged abstention-aware) |
| Samples covered | all 10 conversations (conv-26 … conv-50) |
| Adapter | `flat` (ChromaDB), `n_results=5`, sme-rich rendering |
| Reader | `gpt-5.3-chat`, baseline reader prompt, full retrieved context |
| Judge | `gpt-5.3-chat` + **canonical LongMemEval type-specific prompts** |

Why a stratified subset and not full n=1986: gpt-5.3-chat reader+judge runs
~14 s/question serial (reasoning-model latency), so the full set is ~7.7 h.
Per-type stratification (50/type) gives a strong per-category n while completing
in ~1 h, and is the disclosed-subset standard for LoCoMo cross-comparisons
(mirrors the LongMemEval comparison-card's stratified-150 precedent). The seed
is pinned so the subset is reproducible. Temporal has only 96 questions total in
LoCoMo-10, so 50 is a ~52% sample of that category.

## How LoCoMo question types map onto the canonical judge

LoCoMo's native question types are not LongMemEval types, so the harness now
maps them onto the canonical type-specific grading templates
(`scripts/cross_validate_longmemeval.py::_LOCOMO_TO_JUDGE_TYPE`). The recorded
`question_type` stays the LoCoMo-native label (so the per-category table reads
naturally); only the *judge template* is mapped:

| LoCoMo type | judge template | rationale |
|---|---|---|
| single-hop | base correctness | plain factual retrieval |
| multi-hop | base correctness | multi-session synthesis, still factual |
| open-domain | base correctness | factual + world knowledge |
| temporal | **temporal-reasoning** | off-by-one-days tolerance LoCoMo temporal Qs assume |
| adversarial | **abstention** | correct behavior is refusal; `is_adversarial` drives this |

Before this fix every LoCoMo type silently fell to the base template — temporal
lost its off-by-one tolerance, which would have understated temporal accuracy.
126 harness/judge tests pass with the mapping in place.

`qa_accuracy` counts a judge `CORRECT` **or** a correctly-refused adversarial
(`ABSTAIN`) as right (matching `judge_label_to_correct`); `ERROR` rows (judge
call failures after retries) are excluded from the denominator and reported
separately.

## Results — per LoCoMo question type

n = 50 per type, 0 ERROR rows (every judge call returned a verdict; clean run).

| LoCoMo type | n | QA accuracy | label breakdown | natural share of 1986 |
|---|---:|---:|---|---:|
| adversarial | 50 | **0.62** | 31 ABSTAIN (correct refusal) / 19 INCORRECT | 22.5% |
| open-domain | 50 | **0.42** | 21 CORRECT / 29 INCORRECT | 42.3% |
| single-hop | 50 | **0.40** | 20 CORRECT / 30 INCORRECT | 16.2% |
| temporal | 50 | **0.26** | 13 CORRECT / 37 INCORRECT | 4.8% |
| multi-hop | 50 | **0.22** | 11 CORRECT / 39 INCORRECT | 14.2% |

(`adversarial` accuracy = fraction correctly refused, i.e. `ABSTAIN`/n —
abstention-aware judging, since the correct behavior on a baited adversarial
question is to refuse.)

## Results — overall

| overall | QA accuracy |
|---|---:|
| unweighted (mean of 5 equal-n categories) | **0.384** |
| proportion-weighted (natural LoCoMo-10 shares; est. full-set) | **0.4255** |

Judge token usage for the whole n=250 run: 46,165 total tokens (39,715 prompt /
6,450 completion) — the judge is cheap; the cost is the reader latency.

## Interpretation

1. **Retrieval is the dominant bottleneck, not the reader.** Zephyr's flat
   retrieval row is R@5 = 0.440 (n=1986) — i.e. the correct evidence session is
   in the top-5 only ~44% of the time. A reader cannot answer from context it
   never received, so the QA ceiling for this adapter is bounded near its
   retrieval recall. The 0.384 unweighted QA sitting *below* the 0.440 R@5 is
   exactly the expected shape: retrieval gates, then the reader loses a bit more
   to synthesis/temporal failures even when the evidence is present.
2. **multi-hop (0.22) and temporal (0.26) are the floor** — the
   multi-session-synthesis and date-arithmetic categories, the same primitives
   that floor the LongMemEval oracle. multi-hop also has LoCoMo's highest
   evidence-reference count (3.13 refs/Q), so a top-5 flat retrieval frequently
   misses one of the required sessions.
3. **adversarial (0.62) is the strongest category** — the abstention-aware
   judge credits the 31/50 questions the reader correctly refused. A flat
   baseline refusing baited questions ~62% of the time is a reasonable floor;
   the 19 INCORRECT are cases where the reader fabricated the baited answer.
4. **open-domain (0.42)** carries the proportion-weighted overall because it is
   42% of the natural set; its accuracy being near the unweighted mean is why
   the weighted overall (0.4255) lands close to but above the unweighted (0.384).
5. **This is a floor, and the comparison it unlocks is the point.** The next
   lever is the #176 daemon run: mempalace retrieval on the *identical*
   subset/reader/judge, so the mempalace-vs-flat QA delta is attributable to the
   retrieval substrate alone. The throwaway-daemon provisioning plan for that is
   `docs/benchmarks/2026-05-29-locomo-throwaway-daemon-plan.md`.

## Artifacts

- Baseline JSON: `baselines/locomo10_flat_e2e_stratified_2026-05-29.json`
- Retrieval row this builds on (Zephyr): `locomo10_flat_rk.json` (R@5 = 0.440, n=1986)
- Harness: `scripts/cross_validate_longmemeval.py` (`--corpus locomo --adapter flat`)
- Judge: `sme/eval/longmemeval_judge.py` (canonical type-specific prompts, judge = gpt-5.3-chat)
- LoCoMo loader + pinned-subset contract: `sme/corpora/locomo/loader.py`
- Throwaway-daemon plan for the mempalace comparison (#176): `docs/benchmarks/2026-05-29-locomo-throwaway-daemon-plan.md`
