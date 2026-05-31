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
| **3** | The Dissonance (contradiction) | **0.00** *(emergent, real KG)* | **recall 0.50 / prec 0.25** *(emergent)* | **RE-SCORED** (see below): mempalace's live palace KG has **0 contradicts edges** — its enrichment pipeline generates no emergent contradiction structure on real content. The prior **+1.00 was a corpus-declared ceiling** (good-dog-graph reads back hand-seeded edges), now superseded. OMEGA: auto-relate **generated** 4 contradicts edges from good-dog text — caught 1 of 2 ground-truth themes (grain-free DCM ✓, dominance-theory ✗). Both are now emergent reads, but on **different corpora** (live palace vs good-dog). |
| **4** | The Threshold (ingestion) | collisions **248** (1.9%) · coverage 1.00 · **entropy 0.340** (`other` 55.1%, 237 types) | collisions 0 · coverage 1.00 · **entropy 0.78** | **RE-ANCHORED to the full-graph EXACT** (#211 server-side cypher over the 1.92M-edge RELATION set): mempalace norm entropy **0.340**, dominant `other` 55.1%, 2.68 bits, 237 edge types, 248 collisions. This **overturns the prior sampled 0.020/98.98%-tunnel** reading (a capped-projection artifact). The real monoculture is the generic `other` relation (55.1%); the real graph has 237 types + 1.9% collisions the sample hid. OMEGA's good-dog emergent vocab is balanced (0.78). Different corpora — directional. |
| **5** | The Missing Room (topology) | **qualitatively well-connected** *(verdict; exact pending)* | **1 component · 0 isolates · Betti-1 1** | **VERDICT ONLY** (see below): the prior 498-component / 44.8%-isolate reading was a **capped-projection artifact**. The adapter's `--real-kg` topology numbers are limit-dependent (no feasible full-graph path — edge-pull OOMs), so no absolute counts published: sampling under-states connectivity → real graph is **≥ as connected**. Exact WCC pending server-side. OMEGA on a small clean corpus = one fully-connected component (**required an adapter fix** for its edges to resolve). |
| **6** | The Archive (supersession) | **0.00** *(emergent, real KG)* | **0.00** *(emergent)* | **RE-SCORED**: mempalace's live palace KG has **0 supersedes edges** → completeness 0.00; the prior +1.00 was the good-dog-graph declared ceiling (8/8 hand-seeded edges, 5 chains), now superseded. OMEGA emitted **0 supersedes** too — its temporal analogue is `evolution` (17 edges), which doesn't normalize to supersedes. **Both land at 0.00 emergent supersession from different roots** — mempalace emits no supersedes edge at all; OMEGA emits a non-canonical temporal type. |
| **7** | The Abacus (E2E QA) | **0.580** (same-reader) | **0.593** (same-reader) | o4-mini reader + gpt-5.3-chat judge, sme-rich. ≈ parity (+1.3pp), different per-cat profile. NOT comparable to OMEGA's self-reported 95.4% (GPT-4.1 reader). |
| **8** | The Blueprint (ontology) | hierarchical claim **PASS** *(verdict; modularity ≫0.5, exact pending)* · introspection **0.0 deployed / 1.0 capability merged** | type-cov 0.0* · edge-vocab 0.333 · **drift 0.875** · introspection 0.0 | **VERDICT** (see below): the structural 'hierarchical' claim **PASSES** — modularity ≫0.5 across every sample, refuting the prior **FAIL (modularity 0.009)**, a capped-projection artifact. No specific modularity number published (limit-dependent; exact pending server-side). Introspection: **0.0 on the deployed prod daemon (familiar v1.9.1), but the capability IS merged** (the `/ontology` endpoint palace-daemon#205 + SME scorer #208) — pending prod deploy, NOT "no API". OMEGA drift 87.5%: its **documented** edge vocab ≠ its **emergent** vocab. *OMEGA 8a type-cov 0% is a corpus-coverage artifact (good-dog notes ingested as default `summary`). |
| **9a** | The Handshake (invocation) | **0.983** (opus-4-8 orch, Tau2 99.3) | **N/A — no harness** | Cat 9a is an **orchestrator-model** property, not a substrate property. mempalace's 0.983 is really opus-4-8's invocation rate in front of it. OMEGA was driven via its library API (no model-in-the-loop), so substrate-level 9a is N/A. |
| **9b** | Call-through success | reachable (clean floor) | **N/A — no harness** | OmegaAdapter declares no `get_harness_manifest()` (library usage). Empty-manifest = does-not-apply (real finding: OMEGA ships an MCP server, but the SME adapter uses the library path). |

---

## Re-score over the REAL KG (post-#147 / #210 / #211)

The mempalace structural column above was re-measured on 2026-05-31 (cassia-2, #148) after
three fixes landed. The original column had two measurement problems the re-score corrects,
and the corrected cells split cleanly into **publishable numbers** vs **verdicts**:

**Why the originals were wrong.** (1) The mempalace Cat 3/6 +1.00 came from the
`good-dog-graph` adapter, which reads back the corpus's *hand-declared* `contradicts` /
`supersedes` edges — a **declared ceiling**, not emergent detection. (2) The Cat 4/5/8
numbers read the capped, tunnel-dominated daemon `/graph` projection (#147), so Cat 4
entropy looked like 0.020 (98.98% tunnel) and Cat 8 "hierarchical" FAILED at modularity
0.009 — both **capped-projection artifacts**, not the real graph.

**Cat 3 / Cat 6 — emergent, publishable (both 0.00).** Re-scored with `--real-kg`
(`cat3/cat6 --adapter mempalace-daemon --api-url http://familiar:8085 --real-kg
--graph-limit 5000`) over the real entity→entity RELATION set: the live palace has **0
`contradicts` edges and 0 `supersedes` edges**. mempalace's enrichment pipeline generates
no emergent contradiction or supersession structure on real content — so its honest
*emergent* Cat 3/6 are **0.00**, not the declared-ceiling +1.00. This is the central
correction: a number we shipped on the site (+1.00) measured read-back-of-seeded-edges, not
detection. **Comparability caveat:** mempalace's 0.00 is on JP's *live palace*; OMEGA's
0.50/0.00 are on *good-dog*. Different corpora — mempalace can't ingest good-dog (prod-only
daemon, no ingest). So the cell is an emergent-vs-emergent *kind*-match, not a controlled
A/B. Both systems landing at 0.00 emergent supersession is itself the finding (mempalace
emits no `supersedes` edge; OMEGA emits a non-canonical `evolution` type).

**Cat 4 — exact, publishable.** Anchored to the **full-graph EXACT** distribution (#211
server-side cypher aggregation over the 1.92M-edge RELATION set — runs in seconds, not the
slow `/graph` sample): normalized entropy **0.340**, dominant `other` **55.1%**, 2.68 bits
across **237** edge types, 12,849 entities, **248** canonical collisions (1.9%). This
**overturns** the sampled 0.020/98.98%-tunnel reading. The real monoculture is the generic
`other` relation absorbing 55.1% of edges (the typed vocabulary isn't surfacing), and the
real graph has both a long 237-type tail and 1.9% collisions that the capped sample
reported as 0.

**Cat 5 / Cat 8 — VERDICT only, absolute numbers held.** Per team-lead adjudication
(green-Somnia + Sage cross-validation): the adapter's `--real-kg` topology numbers
(component count, isolate fraction, modularity) are **limit-dependent** — they scale with
`--graph-limit` — and there is **no feasible full-graph path** (pulling all 1.92M edges
OOMs). So we publish the qualitative verdict, never the sampled artifact:
- **Cat 5:** sampling under-states connectivity, so the real graph is **≥ as connected** as
  any sample shows → *qualitatively well-connected*. The prior 498-component / 44.8%-isolate
  figure was a capped-projection artifact. Exact WCC pending a server-side computation.
- **Cat 8:** the structural 'hierarchical' claim **PASSES** — modularity ≫0.5 across every
  sample, decisively refuting the prior FAIL (modularity 0.009 was the same capped-projection
  artifact). No specific modularity number is published (limit-dependent). **Introspection is
  0.0 on the *deployed* prod daemon (familiar v1.9.1 runs older code), but the capability is
  MERGED** — the `/ontology` introspection endpoint (palace-daemon#205) + the SME scorer
  (#208) exist; the cell is "capability merged, pending prod `/ontology` deploy", NOT "no
  health-check API". Exact structural metric pending server-side (palace-daemon #210-followup).

All re-score reads were **READ-ONLY GET probes** against prod `familiar:8085`; no ingest
(the `--real-kg` path issues `/graph` GETs + a `POST /cypher` *read* aggregation only).

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
