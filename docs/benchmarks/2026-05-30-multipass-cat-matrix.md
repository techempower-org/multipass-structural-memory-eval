# Multipass / SME Cat 1–9 Structural Matrix — MemPalace reading

**Date:** 2026-05-30
**Author:** Muse (sme-dreamteam)
**Issue:** [techempower-org/multipass-structural-memory-eval#115](https://github.com/techempower-org/multipass-structural-memory-eval/issues/115)
**Status:** Consolidated. Most-recent defensible result per category against the
mempalace substrate (palace-daemon, populated AGE knowledge graph), plus an
honest gap list and the Cat-9 orchestrator-model callout.

---

## What this is

This is the **third** matrix for the benchmarks page, alongside the field
leaderboard (where SME sits vs Zep / Mem0 / etc. on public corpora) and the
apples-to-apples head-to-head (on-harness, identical conditions). Where those
two answer *"how does this system rank?"*, this matrix answers a different
question: **what does the system know about its own structure?**

These nine categories are SME's unique contribution. **No competitor framework
has an analogue** — LongMemEval, LoCoMo, BEAM, Mem0's benchmarks, Zep's
evaluations all measure end-to-end QA or retrieval recall. None of them
diagnose canonical-collision dedup, edge-type monoculture, structural holes,
ontology drift, or harness invocation rate. The `competitor-analogue?` column
in the matrix below is **`none`** for every row, by construction.

The posture is diagnostic, not a benchmark: each cell is a controlled reading
of a single substrate, not a leaderboard score. Cross-system comparison on
these categories carries the ontology-sensitivity confound the spec flags
(`docs/sme_spec_v8.md`, "Ontology design quality").

---

## The matrix

Substrate under test: **mempalace** via the `mempalace-daemon` adapter against
the live palace-daemon at `familiar:8085`, reading the **populated AGE
knowledge graph** (1,106 entities / 169k edges in the 2026-05-29 snapshot).
Cat 9a is the older `familiar` vs `rlm` handshake reading on jp-realm-v0.1.

| Cat | Name | What it measures (one line) | MemPalace reading | Corpus | Adapter | Date | Competitor analogue? |
|---|---|---|---|---|---|---|---|
| **1** | The Lookup | Can it find a specific memory from a natural-language query? | **R@5 = 0.867** (full-recall 22/30; by-hop 1: 0.889, 2: 0.667) | jp-realm-v0.1 | mempalace-daemon `/search/age-fused` | 2026-05-29 | none |
| **2c** | The Stairway | Multi-hop retrieval recall by hop depth — does structure scale with depth? | **mean recall 0.933** (Condition B only; hop-1: 0.944 / n27, hop-2: 0.833 / n3) | jp-realm-v0.1 | mempalace-daemon (AGE) | 2026-05-29 | none |
| **3** | The Dissonance | Does it detect and surface conflicting facts? | **gap** — flat floor 90.3% (substring); structured `ContradictionPair[]` path not wired for the daemon adapter | good-dog-corpus | flat baseline | 2026-05-24 | none |
| **4** | The Threshold (Ingestigation) | Is the extraction pipeline producing a clean graph (dedup, field coverage, monoculture)? | 4a collisions **0** / 1,106 keys · 4b coverage **1.000** · 4c normalized entropy **0.020** (tunnel = 98.98% — severe monoculture, expected for a tunnel-dense palace) | live palace (AGE) | mempalace-daemon | 2026-05-29 | none |
| **5** | The Missing Room | Can it identify what's structurally missing (components, holes, gaps)? | **498 components**, largest 606 (54.8%), isolates 496 (44.8%), bridges 2, Betti-1 **0**, candidate gaps 0/1 | live palace (AGE) | mempalace-daemon | 2026-05-29 | none |
| **6** | The Archive | Current vs. historical state, supersession tracking | **gap** — flat floor 100% (substring can't distinguish consolidation-aware from flat RAG); real signal is `_superseded_by` / `valid_from`–`valid_to`, not read by the matcher | good-dog-corpus | flat baseline | 2026-05-24 | none |
| **7** | The Abacus | Does structure earn its token overhead? (graph vs no-graph) | tokens-per-correct **52.9k** on jp-realm Cat 1 (age-fused). Token-efficiency A/B/C deltas need a flat Condition-A run (gap, see below) | jp-realm-v0.1 | mempalace-daemon | 2026-05-29 | none |
| **7b** | Latency | Query latency distribution (YCSB p50/p95) | p50 **vector 626ms / union 429ms / hybrid 2064ms**; p95 4.7s / 636ms / 5.6s | live palace (AGE) | mempalace-daemon (candidate-strategy) | 2026-05-30 | none |
| **8** | The Blueprint | Does the actual graph match what the system claims to do? | 8a type coverage **0.333** (2/6 declared types found) · 8b edge vocab **0.667** · 8d drift **0.556** · 8e claims **0.50** (2/4 testable pass; "hierarchical" claim FAILS — modularity 0.009) · introspection **0.0** | live palace (AGE) | mempalace-daemon | 2026-05-29 | none |
| **9a** | The Handshake (invocation) | Does the model actually invoke memory when it has access? | **familiar 78.3%** vs **RLM 46.7%** (recall under invocation). Both RLM orchestrators (Qwen-7B, Llama-70B) plateau at 46.7% — ceiling is the orchestrator's *willingness to invoke*, not retrieval | jp-realm-v0.1 | familiar / rlm | 2026-04-30 | none |
| **9b** | Call-through success | Given an invocation, does the tool call complete and return a valid result? | live surfaces reachable (clean floor; mock-model probe path) | — | — | — | none |

---

## Per-category descriptions

- **Cat 1 — The Lookup (baseline).** Factual retrieval: can the system find a
  specific memory given a natural-language query? This is the floor every other
  category sits on. SME adopts LongMemEval's GPT-4o judge methodology here.
- **Cat 2c — The Stairway (multi-hop).** Retrieval recall broken down by hop
  depth. The spec expectation: graph advantage is modest at 1-hop (~1.5x) and
  dramatic at 3-hop (5–10x). If recall doesn't scale with depth, the traversal
  isn't working. SME's reading is currently Condition-B-only (no flat baseline
  on this corpus — see gaps), so it shows the absolute by-hop recall, not the
  B−A structural delta.
- **Cat 3 — The Dissonance.** Contradiction detection — does the system flag
  conflicting facts (old framing vs new framing) rather than silently returning
  one? The real signal lives in the structured `ContradictionPair[]` field of
  `QueryResult`, which the substring matcher does not read.
- **Cat 4 — The Threshold (Ingestigation).** An *investigation* into what
  ingestion preserved: 4a canonical-collision dedup (distinct IDs that
  canonicalize to the same key), 4b required-field coverage, 4c edge-type
  monoculture (normalized entropy over the edge-type distribution). Generalizes
  the DUP monitor from vault-rag.
- **Cat 5 — The Missing Room.** External topology reading: connected
  components, isolated nodes, structural bridges, Betti-1 persistence (loops
  that survive the Vietoris–Rips filtration = stable structural holes), and
  candidate cross-component gaps (pairs of components holding the same
  entity_type but disconnected).
- **Cat 6 — The Archive.** Temporal reasoning: current vs historical state and
  supersession-chain tracking. The consolidation signal lives in
  `_superseded_by` edges and `valid_from`/`valid_to` properties — not in
  substring overlap, which is why the flat floor is 100% and uninformative.
- **Cat 7 — The Abacus.** Token efficiency: does structure earn its overhead?
  Three-condition design (A flat / B full pipeline / C structure-disabled),
  pairwise judge (BenchmarkQED AutoE) to remove verbosity bias. 7d breaks
  efficiency down by hop depth.
- **Cat 7b — Latency.** YCSB-standard wall-clock latency distribution per
  `query()` call: p50, p95, p99, p99.9, max.
- **Cat 8 — The Blueprint.** Ontology coherence: does the actual graph match
  the declared schema? 8a type coverage, 8b edge-vocabulary coverage, 8c
  schema-data alignment, 8d declared-vs-effective drift, 8e structural-claim
  verification. Introspection (does the system surface its own drift?) reported
  separately — most systems, including mempalace, score 0 (no health-check API).
- **Cat 9a — The Handshake (invocation).** Of every category here, this is the
  only one that measures the layer *between* retrieval and a running model. On
  questions the memory provably contains (verified by the offline Cat 1 run),
  how often does the model actually invoke the memory tool? A 95%-Cat-1 system
  invoked 20% of the time is a 19% effective memory.
- **Cat 9b — Call-through success.** Given an invocation, does the call
  complete and return a valid result? Isolates integration breakage (bad
  schema, timeout, tool not registered, MCP unreachable) from model behaviour.
  The one Cat 9 sub-test scoreable against a mock model that always invokes.

---

## Cat-9 special note: orchestrator-model selection (Tau2 prior)

**Cat-9a Handshake recall TRACKS the orchestrator's Tau2 tool-agent score, not
its parameter count.** This is the single biggest lever on the 9a numbers, and
it is empirically validated, not speculative.

The prior (see `reference_tau2_predicts_cat9a`): a **+37.7pp Tau2 gap** between
two orchestrators predicted a **+30–33pp Cat-9a recall gap** on jp-realm-v0.1,
to within ~5pp. Tau2 (tool-agent benchmark) is a far better predictor than
parameter count — the live data bears this out: RLM with **Qwen-7B and
Llama-70B plateau at the same 46.7%** recall despite a ~10× parameter
difference and a ~4× difference in raw tool-invocation rate (7% vs 27%). Both
ceiling at *willingness to invoke the tool*, not at retrieval quality
underneath. The deterministic `familiar` pipeline (retrieve → rerank →
temporal-decay → extractive-compression → grounding) consistently calls the
retrieval system and lands at **78.3%**.

**Implication for raising the 9a numbers:** swap the orchestrator for a current
Tau2 leader. As of this writing the best invocation-path candidates are **Opus
4.6 (99.3% Tau2 telecom)**, **GPT-5.4 (98.9%)**, and **GLM-5 (~98%)**. The
prediction is that orchestrating familiar's retrieval through a high-Tau2 model
lifts RLM's 46.7% toward — and the deterministic 78.3% past — the offline Cat 1
ceiling, because the bottleneck is the invocation decision, not the substrate.
This is the recommended next experiment for closing the harness-integration gap.

---

## Honest gaps

These are *not* category failures — they are "not cleanly runnable against the
mempalace substrate right now" markers. Flagged per the diagnostic posture
(deltas under controlled conditions, never absolute scores presented as more
than they are).

1. **Cat 3 / Cat 6 structural reading on mempalace.** Only flat good-dog
   baselines exist (Cat 3 90.3%, Cat 6 100% on the substring matcher). The
   structured-field plumbing — `ContradictionPair[]` for Cat 3, `_superseded_by`
   / `valid_from`–`valid_to` for Cat 6 — is **not wired through the
   `mempalace-daemon` adapter**, so a true structural reading can't be produced
   yet. The flat floors are corpus-side measurements, not system readings; the
   honest headline for these cats is `(structural − flat)`, which is currently
   uncomputable on mempalace.

2. **Cat 2c Condition A (flat baseline).** The jp-realm-v0.1 corpus ships only
   `questions.yaml` — the source notes live in JP's private palace, not the
   repo. The flat adapter needs a pre-built ChromaDB of those notes to embed,
   which would require re-ingesting the vault (out of scope, not flat-safe in
   spirit). So Cat 2c stays **Condition-B-only**: a defensible single-system
   by-hop reading (0.933), but not the B−A structural delta the spec wants.

3. **Cat 7 A/B/C token-efficiency deltas.** Same root cause as #2 — no flat
   Condition-A run on jp-realm. The tokens-per-correct figure (52.9k) is a
   single-condition reading; the graph-vs-no-graph pairwise win-rate that is
   Cat 7's headline metric needs the flat baseline that the corpus can't supply
   locally.

4. **Cat 9a model coverage.** The 78.3 / 46.7 readings are from 2026-04-30 on
   three orchestrators (familiar-deterministic, Qwen-7B, Llama-70B). Tau2
   cross-validation across the current high-Tau2 model families (Opus 4.6,
   GPT-5.4, GLM-5) is the pending experiment from the Cat-9 note above.

---

## Provenance

| Reading | Baseline file |
|---|---|
| Cat 1 | `baselines/jp_realm_v0_1_daemon_age_fused_2026-05-29.json` |
| Cat 2c | `baselines/cat2c_daemon_age_2026-05-29.json` |
| Cat 4 | `baselines/cat4_daemon_age_2026-05-29.json` |
| Cat 5 | `baselines/cat5_daemon_age_2026-05-29.json` |
| Cat 7b latency | `baselines/candidate_strategy_age_2026-05-29.json` |
| Cat 8 | `baselines/cat8_daemon_age_2026-05-29.json` |
| Cat 9a | `docs/ideas.md` §"Live benchmark answers (2026-04-30)" |
| Cat 3 / Cat 6 (flat floor) | `docs/good_dog_cat3_cat6_findings.md` |

A 2026-05-30 read-only Cat 5 re-run against the live daemon reproduced the
reading cleanly (the live palace has grown to 1,217 entities / 717-node largest
component since the 05-29 snapshot — expected, it's JP's working palace). The
05-29 snapshot remains the labeled defensible figure.
