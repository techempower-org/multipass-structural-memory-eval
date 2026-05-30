# BEAM — end-to-end QA pass-rate at token buckets (flat-local substrate)

**Date:** 2026-05-30
**Issue:** [techempower-org/multipass-structural-memory-eval#177](https://github.com/techempower-org/multipass-structural-memory-eval/issues/177)
**Loader/runner:** Reverie's [#181](https://github.com/techempower-org/multipass-structural-memory-eval/pull/181) (`sme/corpora/beam/` + `cross_validate_longmemeval.py --corpus beam`)
**Adapter:** `flat` — ephemeral **local** ChromaDB per conversation, sentence-transformer embeddings. This is the **mempalace retrieval core run locally** (no familiar daemon touched, no prod ingest).
**Reader:** `gpt-5.3-chat` · **Judge:** `o4-mini` (Azure Foundry `claud-assistant-resource` — the only two models deployed there; `gpt-4o*` 404). Both are reasoning-class, so temperature is dropped per the harness.

## TL;DR

BEAM grades the **same conversation at multiple token buckets**; a number is
meaningless without its bucket. E2E QA pass-rate (judge: CORRECT, plus a correct
ABSTAIN on abstention items, over judged items):

| Bucket | regime | n | E2E QA pass-rate | notes |
|---|---|---:|---:|---|
| **100K** | full-conversation context (`n_results=5` ≥ 3-5 sessions ⇒ whole conv) | 400 | **0.649** | 235 CORRECT + 24 ABSTAIN / 399 judged (1 judge ERROR) |
| 500K | retrieval (`n_results=3` of 10 sessions) | 700 | _running_ | top-3 ≈ 145K-tok ctx |
| 1M | retrieval (`n_results=2` of 10 sessions) | 700 | _chained_ | top-2 ≈ 175-330K-tok ctx |
| 10M | — | — | **deferred** | only Mem0/Hindsight publish it; out of scope tonight |

**Diagnostic posture (per CLAUDE.md): this is a delta under controlled
conditions, not a leaderboard score.** For reference, Mem0 reports BEAM QA
**64.1 / 48.6 at the 1M / 10M** buckets with <7K tokens per retrieval call —
i.e. an aggressive-retrieval system, a different regime from the full-context
100K reading below.

## The bucket-name trap (load-bearing)

BEAM bucket names are **nominal token budgets, not conversation sizes**. Measured
per-conversation chat size (chars/4):

| bucket | sessions/conv | median tok/conv | max tok/conv |
|---|---:|---:|---:|
| 100K | 3-5 | ~127K | ~225K |
| 500K | 10 | ~538K | ~857K |
| 1M | 10 | ~1.09M | ~1.84M |

Consequence for the flat adapter (one ingested doc **per session**):

- **100K**: with `n_results=5` and only 3-5 sessions, retrieval returns the
  **whole conversation**. So the 100K reading is **full-conversation-context QA**
  on the local substrate — retrieval is *not* doing subsetting work at this
  granularity. `gpt-5.3-chat` was verified to ingest the largest 100K
  conversation (182K prompt tokens) without truncation.
- **500K / 1M**: a single conversation (median 538K / 1.09M tokens) **exceeds any
  reader window**, so full-context is infeasible — these buckets *require* real
  retrieval subsetting (`n_results` 2-3 of 10 sessions) so the retrieved top-K
  fits the reader. They are therefore a genuine **retrieval** regime, reported
  with their `n_results`. The largest 1M conversations may still overflow at
  `n_results=2`; when the reader 400s, the answer is empty and the item is judged
  **INCORRECT** — a real, honest outcome, never fabricated.

## 100K per-ability (n=40 each) and per-SME-category

| BEAM ability | SME cat | n | pass-rate |
|---|---|---:|---:|
| information_extraction | cat_1 | 40 | 0.850 |
| contradiction_resolution | cat_3 | 40 | 0.875 |
| knowledge_update | cat_3 | 40 | 0.425 |
| multi_session_reasoning | cat_2c | 40 | 0.600 |
| temporal_reasoning | cat_6 | 40 | 0.750 |
| **event_ordering** | cat_6 | 40 | **0.000** ⚠ |
| abstention | cat_1_neg | 40 | 0.600 (24 ABSTAIN / 16 fabricated) |
| preference_following | unmapped | 40 | 0.875 |
| instruction_following | unmapped | 39 | 0.769 (1 judge ERROR) |
| summarization | unmapped | 40 | 0.750 |

### ⚠ event_ordering = 0.000 is a grader mismatch, not a substrate failure

All 40 event_ordering items scored INCORRECT. This is **not** the memory system
failing to retrieve — it is a **judge/metric mismatch**. Upstream BEAM scores
event_ordering with **Kendall tau-b** over the reconstructed sequence (a graded
ordering correlation), whereas this harness reuses the LongMemEval binary
CORRECT/INCORRECT judge. A full chronological reconstruction is almost never a
verbatim match to the gold ordering string, so the binary judge floors it at
zero. The honest read: **event_ordering needs the tau-b grader to produce a
meaningful number**; until then it should be excluded from the headline (the
overall 0.649 *includes* it, so the substrate-only pass-rate is higher — 235/359
≈ 0.654 excluding the event_ordering floor, before re-adding its true tau-b
score). Filed as a follow-up.

**knowledge_update (0.425)** is the expected KU divergence (see
`docs/related_work/longmemeval.md`): BEAM rewards returning the *revised* value,
and a full-context reader sees both the old and new statements with no recency
signal, so it often answers with the stale value.

## What changed in the harness for the larger buckets

- `--beam-n-results` (default 5 = the 100K full-context regime; set 2-3 for the
  500K/1M retrieval regime). Threaded into `run_beam_questions`; recorded in
  `run_metadata.beam_n_results`. (The committed 100K baseline shows
  `beam_n_results: null` because that run predated the flag — its effective depth
  was the hardcoded 5.)
- Fixed the flat-adapter **hit-metric id mismatch**: the adapter labels retrieved
  entities `chunk:S0` while `expected_sources` are bare `S0`, so per-session
  retrieval recall (`hit_at_k`) read as all-miss. Now normalized before the
  comparison (raw ids preserved for provenance). This affects the SME retrieval
  diagnostics only, **not** the E2E QA pass-rate (the judge grades the reader's
  answer, not the session ids).
- 3 new tests; full relevant suite 34/34 green, ruff clean.

## Reproduce

```bash
# Fetch a bucket (gitignored data/ dir); see sme/corpora/beam/README.md.
# Then, per bucket:
PYTHONPATH=. ./venv/bin/python scripts/cross_validate_longmemeval.py \
  --dataset sme/corpora/beam/data/beam_100K.json \
  --corpus beam --bucket 100K --beam-n-results 5 \
  --adapter flat \
  --reader-model gpt-5.3-chat --judge-model o4-mini \
  --out baselines/beam_100K_qa_2026-05-30.json
# 500K: --bucket 500K --beam-n-results 3 ; 1M: --bucket 1M --beam-n-results 2
```

## Artifacts (committed)

```
baselines/beam_100K_qa_2026-05-30.json        # full per-question report (400q)
baselines/beam_100K_summary_2026-05-30.json   # headline + per-ability/-category rates
# 500K / 1M appended as they land
```

## Next

- **event_ordering tau-b grader** — replace the binary judge for this ability so
  its number is meaningful (currently a hard 0.0 floor dragging the overall down).
- 500K / 1M numbers (retrieval regime) appended to this note + baselines as they
  complete; 10M remains deferred (cost).
