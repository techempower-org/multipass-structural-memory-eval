# LoCoMo temporal: date-confound checked and RULED OUT

**Date:** 2026-05-30
**Issue:** [M0nkeyFl0wer/multipass-structural-memory-eval#175](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/175) (task #108)
**Author:** Cassia (SME dream-team)
**Relates to:** the flat LoCoMo E2E QA row (`docs/benchmarks/2026-05-29-locomo-flat-e2e-qa.md`, temporal = 0.26) and the OMEGA date-confound finding (Solara) cross-referenced below.

## TL;DR

LoCoMo temporal QA scored **0.26** on the flat adapter. That is the same
"smell" as the **OMEGA date-confound** Solara found, where OMEGA's
*upstream-exact* rendering stripped session timestamps and temporal QA
collapsed (cat_6 **0.04 → 0.36** once dates were restored). We checked whether
LoCoMo's flat run was hit by the same artifact — and **ruled it out**.

> **100% of reader contexts (50/50) carried a session date. Zero
> date-stripped.** The temporal failures are **retrieval misses + genuine
> reasoning failures**, not date-starvation. So **0.26 temporal (and the
> 0.384 / 0.4255 overall) is a genuine retrieval-ceiling + reasoning result,
> not a fixable rendering artifact. No inflating re-run was performed** — the
> honest number stands.

This is the diagnostic posture working as intended: OMEGA *was* date-confounded
and we fixed it; LoCoMo we *checked* and ruled out. Distinguishing artifact
from real is the point.

## Why the LoCoMo flat path structurally cannot be date-stripped

Three facts, verified against the code and the materialized vault:

1. **The LoCoMo harness path does not honor `content_rules` at all.**
   `run_locomo_questions` calls the loader's `materialize_sme_corpus`, which
   always renders sessions via `_render_session_md` (**sme-rich**). The
   `upstream-exact` stripping branch that bit OMEGA exists only on the
   *LongMemEval per-question* path, not the LoCoMo per-sample path.
2. **Every session `.md` carries the date twice** — once in the frontmatter
   (`date: '1:56 pm on 8 May, 2023'`) and once in the body (`_Date: 1:56 pm on
   8 May, 2023_`). Verified on conv-26's materialized vault.
3. **The flat adapter ingests the whole `.md` file as one ChromaDB document
   per session** (`docs.append(text)` — no sub-chunking). A retrieved session
   chunk therefore *always* includes its date line; chunking cannot separate
   the date from the dialogue turns.

## The dynamic proof (not just arguing from code)

Re-ran the **exact same 50 temporal questions** (`seed=1729`, verified
identical `question_id`s to the published baseline) with `capture_context=True`,
so we could read precisely what the reader saw and classify every failure.
Same flat adapter, same reader = judge = `gpt-5.3-chat`, same canonical
temporal-reasoning judge template (off-by-one tolerance).

| metric | value |
|---|---:|
| temporal accuracy (reproduced) | **0.26** |
| reader contexts containing a date | **50 / 50 = 100%** |
| `DATE_STRIPPED` failures (the OMEGA confound) | **0** |

### Failure classification (37 INCORRECT temporal questions)

| class | n | meaning |
|---|---:|---|
| `RETRIEVAL_MISS` | 24 | evidence session not in the top-5 (`sme_recall < 0.5`) — the gold never reached the reader |
| `GENUINE_REASONING_FAIL` | 8 | dates **and** evidence present, reader still wrong (mis-inference) |
| `IDK_DESPITE_EVIDENCE` | 5 | evidence present, reader refused ("I don't know") |
| `DATE_STRIPPED` | **0** | **none** — the confound did not occur |

**65% of failures are retrieval misses** — the dominant bottleneck is the same
R@5 = 0.440 retrieval ceiling that gates every LoCoMo category, not date
absence. The 8 genuine reasoning fails are cases like *"What might John's degree
be in?"* (gold: political science; reader answered "mechanical engineering"
from a present-but-misleading job mention) — inference errors with full context,
unrelated to timestamps. (Several LoCoMo `temporal`-category questions are
actually attribute/inference questions, not date arithmetic; none of the
failures were date-starved regardless.)

## Contrast with OMEGA (the value of checking)

| system | rendering | dates in reader context | temporal result |
|---|---|---|---|
| OMEGA | upstream-exact | **stripped** | cat_6 0.04 → **0.36** after restoring dates |
| LoCoMo flat | sme-rich (always) | **100% present** | 0.26 — genuine, no lift available |

Had we *not* checked, 0.26 would be ambiguous (real, or a hidden artifact like
OMEGA's). The check removes the ambiguity: it is real.

## Conclusion

`0.384` (unweighted) / `0.4255` (proportion-weighted) overall and `0.26`
temporal are **published as-is**. The lever for improving them is **retrieval
breadth / a smarter retrieval substrate** (the #176 mempalace-vs-flat
comparison), not date rendering — which is already correct on this path.

## Artifacts

- Diagnostic JSON (per-question class + date-presence): `baselines/locomo10_temporal_date_diagnostic_2026-05-30.json`
- Diagnostic script: `scripts/locomo_temporal_date_confound.py`
- Flat E2E QA row this checks: `docs/benchmarks/2026-05-29-locomo-flat-e2e-qa.md`
- OMEGA date-confound (the confounded counterexample): OMEGA cat_6 0.04→0.36 (Solara; see `docs/site/index.html` LoCoMo explainer + OMEGA notes)
