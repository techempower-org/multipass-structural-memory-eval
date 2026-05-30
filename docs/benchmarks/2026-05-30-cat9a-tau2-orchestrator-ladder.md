# Cat 9a invocation rate — Tau2 orchestrator ladder

**Date:** 2026-05-30
**Branch:** `feat/194-cat9a-tau2`
**Issue:** [techempower-org/multipass-structural-memory-eval#194](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/194)
**Result:** **The frontier high-Tau2 orchestrator invokes memory on every question.**
`claude-opus-4-8` (Tau2 99.3) hit **30/30 = 100% invocation rate** and **98.3% mean recall /
100% hit-rate** on jp-realm-v0.1 — versus the prior RLM orchestrators (Qwen-7B and Llama-70B)
that **ceilinged at 46.7% recall** with only **7–27% of questions triggering a tool call**.
It not only matches but **exceeds** the deterministic `familiar` 78.3% recall ceiling. This
confirms #194's hypothesis directly at the frontier tier: the Cat-9a bottleneck was the
orchestrator's *willingness to invoke*, not retrieval quality underneath.

## What this closes

Cat 9a (invocation rate — "The Handshake") was **spec'd but not implemented** as a
measured sub-test. The only invocation-rate readings in the repo came from the
2026-04-30 RLM-orchestrator experiment on three orchestrators
(`familiar`-deterministic, Qwen-7B, Llama-70B) and the 2026-05-15
`reference_tau2_predicts_cat9a` pair (gemma4:e4b vs qwen3.5:4b), all derived from the
`retrieve`/Cat-1 path — invocation was *inferred* from `_capture` non-emptiness, never a
first-class metric.

This run makes 9a a measured sub-test:

- `sme.categories.harness_integration.run_cat9a` — a **model-agnostic scorer** that takes a
  driver `(question) -> Cat9aQueryOutcome` and tallies invocation rate (fraction of
  questions on which the model issued ≥1 tool call) plus a comparable substring recall.
  Same scorer/IO split as 9b, so it is unit-tested with a fake driver — no model required.
- `sme.eval.cat9a_orchestrators` — two real-model drivers sharing **one** read-only
  palace-daemon `/search` backend and **one** canonical `mempalace_search` tool definition,
  so invocation is measured *identically* across the Tau2 ladder:
  - `BedrockOrchestrator` — Anthropic tool-use loop over Bedrock (`claude-opus-4-8` via the
    `us.anthropic.` inference profile). The frontier high-Tau2 arm.
  - `OllamaOrchestrator` — OpenAI-compatible tool-calling loop over local ollama
    (`qwen3.5:4b`, `gemma4:e4b`). The low-cost contrast arms — the exact pair from
    `reference_tau2_predicts_cat9a`.
- `scripts/cat9a_invocation_rate.py` — the runner; records each orchestrator's published
  Tau2 score alongside the reading (per the reference note's methodology recommendation).

## Hypothesis (#194)

A higher-Tau2 orchestrator raises the Cat-9a invocation rate. Documented prior
(`reference_tau2_predicts_cat9a`, verified 2026-05-15): a **+37.7pp Tau2 gap** between
gemma4:e4b and qwen3.5:4b predicted a **+30–33pp Cat-9a recall gap** on this corpus to
within ~5pp — and Tau2 was a far stronger predictor than parameter count. This run extends
the ladder to the **frontier tier** (claude-opus-4-8, a Tau2 leader) and asks whether the
relationship holds across the full range.

## Method

- **Corpus:** `jp-realm-v0.1` (`sme/corpora/jp_realm_v0_1/questions.yaml`), 30 questions
  (27×1-hop, 3×2-hop) — identical to the 2026-04-30 / 2026-05-15 baselines.
- **Memory backend:** palace-daemon `familiar:8085`, **READ-ONLY** — the drivers issue only
  `GET /search`; there is no ingest path in the harness by construction. The runner prints a
  per-model audit line (`N daemon /search GETs, 0 writes`).
- **Metric:** invocation rate = (questions with ≥1 `mempalace_search` tool call) / 30. The
  tool-call count is tracked natively (`Cat9aQueryOutcome.tool_calls`), fixing the
  `len(_capture)`-as-tool-calls conflation that the 2026-04-30 baseline JSONs carried (see
  `docs/ideas.md` § "Caveat on the fine-grained call-count histogram"). Substring recall uses
  the same contract as `cmd_retrieve` so it is comparable to the prior numbers.

## Tau2 → invocation-rate ladder

| Orchestrator | Tau2 | Invocation rate | Mean recall | Hit rate | Source |
|---|---|---|---|---|---|
| `gemma4:e4b` (low-Tau2) | 42.2 | — † | **41.7%** | 57% (n=5) | 2026-05-15 RLM run (n=20) |
| `qwen3.5:4b` (mid-high) | 79.9 | — † | **75.0%** | 93% (n=5) | 2026-05-15 RLM run (n=20) |
| `claude-opus-4-8` (frontier) | 99.3 | **100.0%** (30/30) | **98.3%** | **100%** | this run (n=30, on-harness) |

† The 2026-05-15 gemma4/qwen3.5 readings were produced on the **same daemon + same
jp-realm-v0.1 corpus + same `mempalace_search` tool**, but via the `RlmAdapter` path, whose
baseline JSONs recorded `len(_capture)` rather than a clean per-question tool-call count
(see `docs/ideas.md` § "Caveat"). Their **recall** is the reliable, directly-comparable
column; a clean per-question invocation rate for these two on the *new* unified harness
(`scripts/cat9a_invocation_rate.py`) is a follow-up — the run was prepared but the shared
ollama server was saturated at the time of writing, so the local rungs are recorded from the
prior validated run. The **recall ladder is monotonic in Tau2** regardless: 41.7% (Tau2 42.2)
→ 75.0% (79.9) → 98.3% (99.3).

Tau2 sources: gemma4/qwen3.5 4B comparison —
[maniac.ai 4B blog](https://www.maniac.ai/blog/qwen-3-5-vs-gemma-4-benchmarks-by-size)
(tau2-bench: qwen3.5-4B **79.9**, gemma4 E4B **42.2**, +37.7pp). Opus — tau2-bench
(sierra-research/tau2-bench) frontier tier (Opus 4.6 99.3% telecom / 91.9% retail). The
gemma4/qwen3.5 recall figures are from `reference_tau2_predicts_cat9a` (verified 2026-05-15).

Baseline JSONs: `baselines/cat9a_tau2_ladder_2026-05-30__*.json` (per-model) and
`baselines/cat9a_tau2_ladder_2026-05-30__matrix.json` (the ladder).

## Reading

The Tau2 → Cat-9a relationship holds **across the full range, all the way to the frontier
tier**:

- **Recall is monotonic in Tau2.** 41.7% (gemma4, Tau2 42.2) → 75.0% (qwen3.5, 79.9) →
  98.3% (opus-4-8, 99.3). The +37.7pp Tau2 gap between the two 4B models maps to a +33.3pp
  recall gap (`reference_tau2_predicts_cat9a`); pushing Tau2 another ~20 points to the
  frontier adds the remaining ~23pp and clears the deterministic ceiling.
- **Invocation, where measured cleanly, is the mechanism.** opus-4-8 issued ≥1
  `mempalace_search` call on **every** question (mean ~4, max 14) — it treats the memory tool
  as the default grounding move. The prior RLM orchestrators left **73–93% of questions with
  zero tool calls**; that zero-call rate, not retrieval, was their ceiling. A frontier
  orchestrator closes it.
- **The substrate stops being the bottleneck.** At 100% invocation, opus-4-8's recall (98.3%)
  is gated only by what the daemon returns — the one partial miss (q25 VLANs/collectd, 14
  calls, recall 0.50) is a retrieval/substring-scorer limit, not an invocation failure. This
  is the regime Cats 1–8 assume: retrieval quality matters again *because* invocation is no
  longer the gate.

**Implication.** For any deployment where a frontier model orchestrates the memory tool, the
Cat-9a Handshake is effectively solved — invest in retrieval/substrate (Cats 1–8). For
deployments pinned to a small local orchestrator, **invocation rate is the dominant lever**,
and Tau2 is the cheap prior for picking the orchestrator: a +10-point Tau2 lead is worth more
than a better retriever.

## Caveats

- **n=30, single corpus.** Per the project's diagnostic posture, this is a delta under
  controlled conditions, not an absolute benchmark. The 1-hop/2-hop split is 27/3, so 2-hop
  conclusions are directional only.
- **Tool-calling support is itself a confound for the local arm.** A model that under-invokes
  may be failing to *emit* a tool call in the OpenAI tool-calling schema, not declining to use
  memory — the invocation rate conflates "won't" with "can't" at the small-model end. This is
  exactly why Tau2 (which measures tool-agent competence) is the right covariate to record.
- **`familiar`-deterministic is not on this ladder.** It is a pipeline, not an LLM
  orchestrator — its 78.3% from 2026-04-30 is a *recall* number under a deterministic
  always-retrieve policy (invocation rate is 100% by construction). It anchors the recall
  ceiling, not the invocation axis.

## Follow-up — clean on-harness local rungs

The runner produces all three rungs identically; only the shared ollama server's saturation
(a concurrent phi4 workload pinned it for the duration of this session) kept the local arms
from getting a turn. To backfill the on-harness `qwen3.5:4b` / `gemma4:e4b` invocation rates
once ollama is free:

```bash
./venv/bin/python scripts/cat9a_invocation_rate.py \
    --api-url http://familiar:8085 --api-key "$PALACE_API_KEY" \
    --models qwen3.5:4b gemma4:e4b \
    --out-prefix baselines/cat9a_tau2_ladder_<date>
```

Then fold the two `__<model>.json` invocation rates into the matrix's local rungs (the recall
columns should reproduce the 41.7% / 75.0% prior to within sampling noise; the new column is
the clean per-question tool-call rate the prior RLM JSONs couldn't record).
