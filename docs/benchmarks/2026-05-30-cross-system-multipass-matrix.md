# Cross-System Structural Matrix — systems × SME Cat 1–9

**Date:** 2026-05-30
**Author:** Cassia (sme-dreamteam)
**Issue:** [techempower-org/multipass-structural-memory-eval#178](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/178) (Tier 3 — independent multi-system head-to-head) + [#115](https://github.com/techempower-org/multipass-structural-memory-eval/issues/115) (the multipass matrix)
**Status:** First cross-system structural matrix. mempalace (done, Muse #115) + OMEGA (this pass) populated; Hindsight (#107, Iris) + Mem0 (#185, Solara) slots reserved.
**Matrix data:** `baselines/cross_system_multipass_matrix_2026-05-30.json` (Luna renders on the site)

---

## What this is

Muse's matrix ([`2026-05-30-multipass-cat-matrix.md`](2026-05-30-multipass-cat-matrix.md))
answered *"what does **mempalace** know about its own structure?"* across Cat 1–9.
This extends that to the **cross-system** question: run the **other** memory systems
through the same categories and build a systems × Cat 1–9 grid. JP scoped this to the
**adapter-ready 3**: OMEGA (adapter merged), Hindsight (#107) and Mem0 (#185) landing.

This pass populates the **OMEGA** column end-to-end. The headline is *not* a leaderboard
— it is the set of **diagnostic deltas under controlled conditions** that fall out of
putting two substrates through the identical categories.

## Posture (read before the numbers)

- **Diagnostic, not benchmark.** Each cell is one substrate under stated conditions.
- **N/A is a finding, not a blank.** "N/A — no graph to evaluate" / "N/A — no harness
  surface" are real readings about a system's shape.
- **The structural cats carry an ontology-sensitivity confound** (`sme_spec_v8.md`,
  "Ontology design quality"). Cross-system numbers on Cat 3/4/5/6/8 are *suggestive
  contrasts*, not rankings.
- **The single most load-bearing distinction this pass surfaced:**
  **corpus-declared (ceiling) vs emergent (generated) structure.** mempalace's Cat 3/6
  +1.00 came from the good-dog-**graph** adapter, which loads the corpus's *hand-declared*
  `contradicts`/`supersedes` edges — a ceiling case. OMEGA does **not** read corpus edges;
  it stores note **text** and a background auto-relate pass **generates** typed edges. So
  OMEGA's Cat 3/6 measure *emergent* structure, which is a fundamentally harder and more
  honest reading. Comparing OMEGA's emergent 0.50/0.00 against mempalace's declared-ceiling
  +1.00 is a category contrast, not "OMEGA is worse" — the two cells answer different
  questions and the matrix labels both.

---

## The matrix (mempalace + OMEGA columns)

| Cat | Name | mempalace | OMEGA | Notes / comparability |
|---|---|---|---|---|
| **1** | The Lookup (R@5) | **0.927** | **0.900** | Identical LongMemEval-S strat150 subset + rendering. ΔR@5 = −2.7pp = 4 questions of 150. |
| **2c** | The Stairway (multi-hop R@5) | **0.960** | **0.920** | cat_2c on the same subset; n=25/cat, ±0.04 = ±1 question (sampling noise). |
| **3** | The Dissonance (contradiction) | **+1.00** *(declared ceiling)* | **recall 0.50 / prec 0.25** *(emergent)* | mempalace: 6/6 corpus-declared `contradicts` edges via good-dog-graph (ceiling). OMEGA: auto-relate **generated** 4 contradicts edges from text — caught 1 of 2 ground-truth themes (grain-free DCM ✓, dominance-theory ✗), 1 of 4 edges a true positive. **Not the same measurement.** |
| **4** | The Threshold (ingestion) | collisions 0 · coverage 1.00 · **entropy 0.020** | collisions 0 · coverage 1.00 · **entropy 0.78** | OMEGA's emergent edge vocabulary is far more **balanced** (3 types, normalized entropy 0.78) than mempalace's tunnel-monoculture (0.020). Different corpora (good-dog vs live palace) — directional, not ranked. |
| **5** | The Missing Room (topology) | 498 components · 44.8% isolates · Betti-1 0 | **1 component · 0 isolates · Betti-1 1** | OMEGA on a small clean corpus = one fully-connected component. mempalace = a fragmented *working* palace. **Required an adapter fix** (see below) for OMEGA's edges to resolve. |
| **6** | The Archive (supersession) | **+1.00** *(declared ceiling)* | **+0.00** *(emergent)* | mempalace: 8/8 declared `supersedes` edges → `_superseded_by`, 5 chains. OMEGA: auto-relate emitted **0 supersedes** — its temporal analogue is the `evolution` edge type (17 edges), which doesn't carry directional supersession semantics, so SME Cat 6 reads 0.00. Honest finding: OMEGA *has* a temporal edge type but no SME-recognized supersession-completeness. |
| **7** | The Abacus (E2E QA) | **0.580** (same-reader) | **0.593** (same-reader) | o4-mini reader + gpt-5.3-chat judge, sme-rich. ≈ parity (+1.3pp), different per-cat profile. NOT comparable to OMEGA's self-reported 95.4% (GPT-4.1 reader). |
| **8** | The Blueprint (ontology) | type-cov 0.333 · edge-vocab 0.667 · drift 0.556 · introspection 0.0 | type-cov 0.0* · edge-vocab 0.333 · **drift 0.875** · introspection 0.0 | OMEGA drift 87.5%: its **documented** edge vocab (`related`/`supersedes`/`contradicts`) and its **emergent** vocab (`evolution`/`temporal_cluster`/`contradicts`) diverge substantially. *8a type-cov 0% is a corpus-coverage artifact (all good-dog notes ingested as the default `summary` event_type) — caveat, not a failure. |
| **9a** | The Handshake (invocation) | **0.983** (opus-4-8 orch, Tau2 99.3) | **N/A — no harness** | Cat 9a is an **orchestrator-model** property, not a substrate property. mempalace's 0.983 is really opus-4-8's invocation rate in front of it. OMEGA was driven via its library API (no model-in-the-loop), so substrate-level 9a is N/A. |
| **9b** | Call-through success | reachable (clean floor) | **N/A — no harness** | OmegaAdapter declares no `get_harness_manifest()` (library usage). Empty-manifest = does-not-apply (real finding: OMEGA ships an MCP server, but the SME adapter uses the library path). |

---

## What the OMEGA column actually says (system character, not score)

Running OMEGA through every applicable category produces a coherent **character sketch**
of the system, which is exactly what the multipass matrix is for:

1. **Retrieval is strong and mid-pack** — R@5 0.900 vs mempalace 0.927, QA 0.593 ≈ 0.580.
   OMEGA is a real retrieval competitor on identical conditions (Solara, #178).

2. **OMEGA generates emergent structure where mempalace consumes declared structure.**
   This is the deepest cross-system finding. Fed the good-dog vault *text*, OMEGA's
   auto-relate produced **54 typed edges** across 3 emergent types
   (`temporal_cluster` 61%, `evolution` 31%, `contradicts` 7%) — a connected,
   balanced-entropy graph it built **itself**. mempalace's structural cats, by contrast,
   were scored on the corpus's *hand-declared* edges (good-dog-graph). OMEGA's emergent
   contradiction detection (theme recall 0.50, edge precision 0.25) is **noisy but real**;
   it caught the grain-free-DCM contradiction (2022 Freeman ↔ 2018 Tufts) and missed
   dominance-theory, with 3 false-positive edges from embedding-similarity. That is a far
   more *interesting* Cat 3 reading than a corpus-declared 1.00.

3. **OMEGA's documented ontology and its emergent ontology diverge (Cat 8 drift 87.5%).**
   OMEGA's README declares `related`/`supersedes`/`contradicts`; its auto-relate emits
   `evolution`/`temporal_cluster`/`contradicts`. It never emitted a single `supersedes`
   edge on this corpus (Cat 6 = 0.00) — the system's self-description and its behaviour
   are out of sync, which is precisely the kind of thing Cat 8 exists to catch.

4. **No harness surface through the adapter (Cat 9a/9b N/A).** OMEGA is consumed as a
   library; invocation-rate is an orchestrator property, so 9a/9b don't apply to the
   substrate.

---

## Adapter bug found + fixed during this pass (Cat 5 was silently broken)

OMEGA's `edges` table references memories by **`node_id`** (`mem-<hash>`), but
`OmegaAdapter._read_omega_memories` was projecting entity ids from the integer
autoincrement **`id`** column. Result: entity ids (`omega:3`) never matched edge endpoints
(`omega:mem-abc…`) — **every one of the 54 emergent edges dangled.** Cat 5 read the graph
as 18 isolates / 18 components (the all-isolates artifact), and Cat 4's per-edge-type
component counts were inflated.

**Fix** (`sme/adapters/omega.py`): prefer `node_id` over `id` as the entity-id column, so
edge endpoints resolve against the node set. After the fix all 54/54 edges resolve and
Cat 5 reads the true topology (1 component, 0 isolates, Betti-1 1). Locked in with a
regression test (`tests/test_omega_adapter.py::test_snapshot_entity_ids_use_node_id_so_edges_resolve`)
that builds the real OMEGA schema shape (integer `id` + `node_id`, edges via `node_id`).
This is a genuine adapter-correctness fix the cross-system pass surfaced — without it,
**any** topology/structural reading of OMEGA (Cat 4c, Cat 5) would have been wrong.

---

## Method / disclosure

- **OMEGA:** omega-memory 1.4.15 (pip), local SQLite + sqlite-vec, semantic ONNX mode.
- **Retrieval cats (1/2c/7):** the existing LongMemEval-S strat150 baselines (Solara,
  #178) — same subset, rendering, reader, judge as the mempalace comparator. Reused, not
  re-run (the conditions are already pinned and documented).
- **Structural cats (3/4/5/6/8):** good-dog-corpus vault **text** (24 notes, frontmatter
  kept — same content the flat baseline got) ingested into an **isolated** `OMEGA_HOME`
  scratch store; OMEGA's auto-relate ran; SME's `cat3/cat4/cat5/cat6/cat8` CLIs scored the
  resulting snapshot. **Same corpus mempalace used for Cat 3/6** (good-dog) for
  comparability — but via OMEGA's *emergent* path, not the declared-edge good-dog-graph
  adapter, which is the central caveat above.
- **Cat 3 emergent-vs-ground-truth:** the plain `sme-eval cat3 --adapter omega` reading is
  tautological for OMEGA (it scores the system's surfaced pairs against the system's own
  contradicts edges → trivial 1.00). The comparable reading scores OMEGA's emergent
  contradicts edges against good-dog's **ground-truth** contradiction themes
  (`ontology.yaml` `cat_3_contradiction.seeded_pairs`), matching on source-file content
  because OMEGA's `mem-<hash>` ids and good-dog's semantic ids can't be matched directly.
  Both JSONs are kept; the matrix cell uses the ground-truth read.
- **Isolation:** per-run scratch `OMEGA_HOME` under
  `~/.claude/projects/<slug>/scratch/cassia-crosssystem/`; the production familiar / palace
  were never touched.

## Honest gaps / caveats

1. **Corpus differs across the structural cats.** OMEGA's Cat 4/5/8 are on good-dog (n=18
   memories); mempalace's are on the live palace AGE (1,106 entities). These are
   *directional contrasts*, not controlled A/B — a clean head-to-head would run **both**
   systems' structural cats on the **same** corpus snapshot. good-dog is the shared corpus
   for Cat 3/6; extending mempalace's Cat 4/5/8 to a good-dog ingest (or OMEGA's to a
   palace-sized corpus) is the follow-up.
2. **Cat 8 8a type-coverage 0% is a corpus artifact**, flagged in-cell — all good-dog notes
   landed as the default `summary` event_type, so the 5 declared event_types weren't
   exercised. A corpus that seeds decisions/lessons/errors would exercise 8a properly.
3. **n=25/category on the retrieval cats** — every ±0.04 is ±1 question; per-cat patterns
   are hypotheses, not effects (Solara's #178 caveat carries through).

## Coordination — Hindsight (#107) + Mem0 (#185)

The matrix scaffold (`baselines/cross_system_multipass_matrix_2026-05-30.json`) reserves a
`hindsight` and a `mem0` slot in every category with `scoreability: "pending-adapter"`.
When those adapters merge to main, the same pass runs:
- retrieval cats (1/2c/7) on the LongMemEval-S strat150 subset (identical conditions);
- structural cats (3/4/5/6/8) on good-dog **only if** the system exposes a graph — the
  first question to each adapter owner is **"does this system expose typed edges, or is it
  retrieval-only?"** A retrieval-only system records **N/A — no graph to evaluate** on
  3/4/5/6/8 (a finding); a graph-bearing system runs them like OMEGA.
- Cat 9a/9b: N/A unless the system exposes a harness surface via `get_harness_manifest()`.
