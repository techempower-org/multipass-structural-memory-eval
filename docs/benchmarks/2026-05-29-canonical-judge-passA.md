# Canonical LongMemEval Judge — Corrected Pass A Oracle

**Date:** 2026-05-29
**Author:** Nyx (SME dream-team)
**Branch / PR:** `feat/canonical-longmemeval-judge`
**Addresses:** Selene Tier-1 #1 blocker (`docs/research/2026-05-29-comparison-readiness.md` §2.2 / §3.1) — the confounded Pass A oracle topped out at ~0.52 vs published GPT-4o oracle 87.0%, a 35pp deficit Selene diagnosed as a **judge-prompt mismatch**, not retrieval or reader capability.

## Bottom line

Porting LongMemEval's canonical type-specific judge prompts **un-collapsed the preference category exactly as predicted** (+20.0pp search-default, +26.7pp age-fused with the held-constant gpt-5.3-chat reader) and **removed the spurious-ABSTAIN noise** (the old judge invented 34 ABSTAIN labels, 21 of them on *non-abstention* temporal questions; the canonical binary judge can't). **Stacked with the #116 winning reader config** (`claude-opus-4-8|preference`), preference *explodes* — +66.7pp (0.133→0.800) search-default, +76.7pp (0.100→0.867) age-fused — and the stacked, abstention-credited overall reaches **0.610** (search-default) / **0.556** (age-fused), clearing the #116 0.593 reader-only floor on search-default. But even stacked it **does not reach the published 87% oracle**: the residual ~26–31pp is **reader/substrate** (multi-session synthesis + temporal reasoning are the floor), not the scorer. **The judge-prompt mismatch was a real, fixable measurement artifact (§4); it is necessary for a defensible QA row but not sufficient to close the oracle gap (§7).**

## What was wrong, and the fix

The prior judge (`sme/eval/longmemeval_judge.py`) used a **paraphrased** 4-way JSON-label rubric (`CORRECT / PARTIAL / INCORRECT / ABSTAIN`) — its own docstring admitted "primary-source verification was not available." It also mapped **all three `single-session-*` types — including `single-session-preference` — to the same strict Information-Extraction rubric.** LongMemEval grades preference with a **rubric-based** template: the gold "answer" is a *rubric*, and a response passes if it "recalls and utilizes the user's personal information correctly" (need not satisfy every point).

The five canonical templates were ported **verbatim** from `xiaowu0162/LongMemEval` `src/evaluation/evaluate_qa.py::get_anscheck_prompt` (MIT): a shared base correctness template (single-session-user / -assistant / multi-session); temporal-reasoning = base + off-by-one-days tolerance; knowledge-update accepts the prior value as long as the *updated* answer is present; preference = rubric-based; abstention treats the gold as an *explanation* and asks whether the model correctly identifies the question as unanswerable. Decision is **binary**: `'yes' in reply.lower()` at `temperature=0`, `max_tokens=10`, no system prompt.

## Model / prompt disclosure (per comparison-readiness §3.4)

- **Judge model: `gpt-5.3-chat`** + canonical type-specific prompts. This is **NOT** the canonical `gpt-4o-2024-08-06` snapshot — that model is not deployed on this Azure resource. The fix that un-collapses the categories is the **prompts**, not the model.
- **Reader model (held constant): `gpt-5.3-chat`**, baseline answer prompt, full context. Matches the confounded run's reader so the delta is attributable to the judge prompts alone. Reader == judge model, but the prompts differ entirely (answer-generation vs canonical grading).
- **Dataset:** LongMemEval-S oracle, n=500, all six question types + 30 abstention (`_abs`) questions. Same pinned context (`baselines/pinned_context_{search-default,age-fused}.json`) the confounded run replayed.
- **2 ERROR rows per run:** Azure content-filter (`ResponsibleAIPolicyViolation`, hate severity medium) tripped on ~2 questions per endpoint; the judge retried 3× then returned `ERROR` (graceful degradation by design). These count as wrong in the harness denominator. The confounded run's paraphrased prompt did not trip the filter, so its denominator had 0 ERROR — a 0.4% asymmetry, disclosed for parity.

## Harness scoring caveat (pre-existing, affects both runs equally)

The reader-sweep aggregator's `qa_acc` **never credits a correct `ABSTAIN`**, because per-question rows carry the *original* `question_type` (e.g. `multi-session`), never literally `"abstention"`, so the aggregator's `ABSTAIN and question_type=='abstention'` condition is dead code. This depresses both the confounded and the corrected absolute numbers identically, so the **delta is valid**, but the absolute corrected number is understated. The tables below report both the raw harness `qa_acc` (apples-to-apples with the confounded file) **and** an `corr+abs` column that credits `ABSTAIN` on the 30 true abstention questions (joined on `question_id` against the pinned `is_abstention` flag). This harness limitation should be fixed separately (out of scope for this PR).

## Results — search-default context

| question_type | n | confounded | corrected (harness) | delta | corrected +abstention |
|---|---|---|---|---|---|
| single-session-preference | 30 | 0.1333 | 0.3333 | **+0.2000** | 0.3333 |
| single-session-assistant | 56 | 0.2500 | 0.2500 | +0.0000 | 0.2500 |
| single-session-user | 70 | 0.8429 | 0.8429 | +0.0000 | 0.9286 |
| multi-session | 133 | 0.6767 | 0.6391 | −0.0376 | 0.7143 |
| knowledge-update | 78 | 0.6795 | 0.6282 | −0.0513 | 0.6923 |
| temporal-reasoning | 133 | 0.3083 | 0.2857 | −0.0226 | 0.3233 |
| **OVERALL** | **500** | **0.5220** | **0.5100** | **−0.0120** | **0.5620** |

## Results — age-fused context

| question_type | n | confounded | corrected (harness) | delta | corrected +abstention |
|---|---|---|---|---|---|
| single-session-preference | 30 | 0.1000 | 0.3667 | **+0.2667** | 0.3667 |
| single-session-assistant | 56 | 0.2143 | 0.2857 | +0.0714 | 0.2857 |
| single-session-user | 70 | 0.8714 | 0.8857 | +0.0143 | 0.9714 |
| multi-session | 133 | 0.4662 | 0.4060 | −0.0602 | 0.4812 |
| knowledge-update | 78 | 0.6795 | 0.5897 | −0.0897 | 0.6538 |
| temporal-reasoning | 133 | 0.3158 | 0.2932 | −0.0226 | 0.3383 |
| **OVERALL** | **500** | **0.4660** | **0.4560** | **−0.0100** | **0.5100** |

## Interpretation

1. **Preference un-collapsed (the predicted win).** +20.0pp / +26.7pp — the largest single-category delta, driven entirely by the rubric-based template the old judge was missing. This validates Selene's diagnosis: the category-collapse was a prompt artifact.
2. **The labeling noise is gone.** Old judge: 34 ABSTAIN labels, 21 on non-abstention temporal questions. Corrected: 26–27 ABSTAIN, landing on the 30 true abstention questions. The canonical binary judge cannot emit a stray ABSTAIN.
3. **KU and multi-session went *down* in the raw harness number** (−5 to −9pp). This is the canonical templates being **stricter and more faithful** than the loose paraphrase, not a regression — the old judge was over-crediting these via its softer rubric. With abstention crediting (`corr+abs`) they recover most of the apparent loss.
4. **Overall did not move toward 87%.** Corrected, abstention-credited: 0.562 / 0.510. The judge fix is necessary for a *defensible* QA row (the preference collapse and the ABSTAIN noise were genuine measurement artifacts) but it is **not sufficient** to close the oracle gap. The residual ~31–36pp vs the published 87% oracle is a **reader/substrate** gap, not a scorer gap — consistent with the #116 unified finding that the reader is the bottleneck. The next lever is reader-prompt / reader-model work on the oracle context, not further judge tuning.

## 7. Stacked reader-fix + judge-fix (the headline-defensible number)

The #116 prompt-axis sweep (PR #144) found the reader-prompt fix took
`claude-opus-4-8` from worst (0.360 baseline) to best (0.593) — but that sweep
still used `gpt-5.3-chat` with the *generic* judge prompt, so 0.593 is a FLOOR.
Stacking the canonical type-specific judge on top of the winning reader config
(`claude-opus-4-8|preference`) is the real headline. Same pinned context, n=500,
judge = gpt-5.3-chat + canonical prompts.

### 7.1 search-default

| question_type | n | confounded | stacked (harness) | delta | stacked +abstention |
|---|---|---|---|---|---|
| single-session-preference | 30 | 0.1333 | 0.8000 | **+0.6667** | 0.8000 |
| single-session-assistant | 56 | 0.2500 | 0.3214 | +0.0714 | 0.3214 |
| single-session-user | 70 | 0.8429 | 0.8571 | +0.0143 | 0.9286 |
| multi-session | 133 | 0.6767 | 0.6466 | −0.0301 | 0.7143 |
| knowledge-update | 78 | 0.6795 | 0.6667 | −0.0128 | 0.7051 |
| temporal-reasoning | 133 | 0.3083 | 0.3308 | +0.0226 | 0.3609 |
| **OVERALL** | **500** | **0.5220** | **0.5680 (+4.6pp)** | | **0.6100** |

### 7.2 age-fused

| question_type | n | confounded | stacked (harness) | delta | stacked +abstention |
|---|---|---|---|---|---|
| single-session-preference | 30 | 0.1000 | 0.8667 | **+0.7667** | 0.8667 |
| single-session-assistant | 56 | 0.2143 | 0.2679 | +0.0536 | 0.2679 |
| single-session-user | 70 | 0.8714 | 0.8857 | +0.0143 | 0.9571 |
| multi-session | 133 | 0.4662 | 0.4135 | −0.0526 | 0.4812 |
| knowledge-update | 78 | 0.6795 | 0.6795 | +0.0000 | 0.7051 |
| temporal-reasoning | 133 | 0.3158 | 0.3684 | +0.0526 | 0.3835 |
| **OVERALL** | **500** | **0.4660** | **0.5200 (+5.4pp)** | | **0.5560** |

### 7.3 Reading

- **Preference exploded** — the reader-fix and the judge-fix *compound*:
  +66.7pp (search-default) and +76.7pp (age-fused), preference reaching
  0.80 / 0.87. Neither fix alone gets there; the rubric-respecting reader
  prompt produces answers the canonical rubric-based judge can actually credit.
- **Stacked overall, abstention-credited, clears the 0.593 floor on
  search-default (0.610).** The stacked *harness* number for search-default
  (0.568) sits just under the 0.593 floor only because the canonical judge is
  stricter on KU/MR than the generic judge Solace's floor used — and because the
  harness still doesn't credit the 26 correct abstentions. Crediting them puts
  it at 0.610, above the floor.
- The honest takeaway stands: even stacked, the oracle tops out at ~0.56–0.61,
  ~26–31pp below the published 87% oracle ceiling. The remaining gap is
  reader/substrate (multi-session synthesis + temporal reasoning are the floor),
  not the scorer.

Stacked baselines:
`baselines/reader_sweep_passA_canonical-judge_opus-preference_{search-default,age-fused}_2026-05-29.json`.

## 8. Stacked best-config (n=150 stratified) — the comparison-card number

The single cleanest "best defensible oracle QA" reading: the #116 winning
reader (`claude-opus-4-8` + `preference` prompt) with the canonical judge,
on the **same strat150 `/search` pinned context** the reader-only 0.593 number
used. The ONLY changed variable vs that 0.593 reference is the judge
(generic gpt-5.3-chat → canonical type-specific prompts). n=150 (25/type),
**0 abstention records** in the stratified sample (so abstention-credited ==
harness), **0 ERROR rows**.

| question_type | n | reader-only (generic judge) | stacked (canonical judge) | delta |
|---|---|---|---|---|
| single-session-preference | 25 | 0.7600 | 0.8400 | +0.0800 |
| single-session-user | 25 | 0.8800 | 0.8800 | +0.0000 |
| single-session-assistant | 25 | 0.2400 | 0.2400 | +0.0000 |
| multi-session | 25 | 0.4400 | 0.4800 | +0.0400 |
| knowledge-update | 25 | 0.7600 | 0.7200 | −0.0400 |
| temporal-reasoning | 25 | 0.4800 | 0.5200 | +0.0400 |
| **OVERALL** | **150** | **0.5933** | **0.6133** | **+0.0200** |

**Reading.** Stacking the canonical judge onto the best reader adds **+2.0pp**
(0.593 → 0.613): preference +8pp (the rubric template credits more correct
personalizations even on top of the preference-tuned reader), temporal +4pp
(off-by-one clause), multi-session +4pp, knowledge-update −4pp (canonical KU
template stricter). **0.613 is our best defensible oracle QA number.** That the
judge swap moves only +2pp on the best reader confirms the thesis: the residual
gap vs the published 87% oracle is **reader/substrate, not the scorer** — the
floor is single-session-assistant (0.24), multi-session (0.48), and temporal
(0.52). Verification gate (before the run): a captured-prompt test confirmed the
reader-sweep judge path fires the canonical rubric/unanswerable templates, not
the old generic rubric.


## Artifacts

- Judge: `sme/eval/longmemeval_judge.py`
- Tests: `tests/test_longmemeval_judge.py`, `tests/test_longmemeval_judge_replicates.py`
- Corrected baselines: `baselines/reader_sweep_passA_canonical-judge_{search-default,age-fused}_2026-05-29.json`
- Stacked baselines: `baselines/reader_sweep_passA_canonical-judge_opus-preference_{search-default,age-fused}_2026-05-29.json`
- Stacked best-config (n=150): `baselines/reader_sweep_stacked_opus-pref_canonical-judge_2026-05-29.json`
- Reader-only reference (0.593, generic judge): `baselines/reader_sweep_passB_promptfix_2026-05-29.json`
- Confounded baselines (corrected here): `baselines/reader_sweep_passA_{search-default,age-fused}_2026-05-29.json`
- Open question #2 resolved: `docs/related_work/longmemeval.md`
