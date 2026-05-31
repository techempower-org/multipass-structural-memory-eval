# Cross-System Structural Matrix — systems × SME Cat 1–9

**Date:** 2026-05-30
**Author:** Cassia (sme-dreamteam)
**Issue:** [techempower-org/multipass-structural-memory-eval#178](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/178) (Tier 3 — independent multi-system head-to-head) + [#115](https://github.com/techempower-org/multipass-structural-memory-eval/issues/115) (the multipass matrix)
**Status:** EXPANDED — cross-system structural matrix across **ALL 8 harness-tested systems** (#163). Fully scored: mempalace (FINAL exact full-graph, post re-map + junk-DELETE + networkx) + OMEGA. Verdict rows: Hindsight (Iris #220) + Mem0-OSS (Solara #221) — verified + runnable, full QA deferred (extraction-throughput-bound). Control + baseline rows: **flat** (no-structure control — real Cat 1/2c/7, structural N/A by design), **rlm** (Cat 9a orchestrator arm — the 46.7% invocation plateau), **full_context (D1)** + **karpathy_compiled (D2)** (Karpathy baselines — wired, not yet run). Every cell is a real reading OR an explicit honest marker — no fabricated numbers.
**Matrix data:** `baselines/cross_system_multipass_matrix_2026-05-30.json` (Luna renders on the site)

---

## What this is

Muse's matrix ([`2026-05-30-multipass-cat-matrix.md`](2026-05-30-multipass-cat-matrix.md))
answered *"what does **mempalace** know about its own structure?"* across Cat 1–9.
This extends that to the **cross-system** question and then to the **whole harness**: run
**every memory system the harness actually tested** through the same categories and build
an honest systems × Cat 1–9 grid — showing not just who scored what, but who was run on
what and what each architecture even *allows*.

The roster is **8 systems**: two fully-scored substrates (mempalace, OMEGA), two
extraction competitors as verdict rows (Hindsight, Mem0-OSS), the no-structure **control**
(flat), the Cat 9a orchestrator arm (rlm), and the two Karpathy baselines (full_context D1,
karpathy_compiled D2, wired but not yet run). The headline is *not* a leaderboard — it is
the set of **diagnostic deltas under controlled conditions** that fall out of putting these
systems through the identical categories, plus the **honest coverage map** of what's been
measured vs merely wired.

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

The two fully-scored columns (mempalace + OMEGA) are the table below; the
**Hindsight + Mem0 verdict rows** follow it (those two are *verified + runnable*
but their full on-harness QA is deferred — see "The extraction-throughput cost
wall").

| Cat | Name | mempalace | OMEGA | Notes / comparability |
|---|---|---|---|---|
| **1** | The Lookup (R@5) | **0.927** | **0.900** | Identical LongMemEval-S strat150 subset + rendering. ΔR@5 = −2.7pp = 4 questions of 150. |
| **2c** | The Stairway (multi-hop R@5) | **0.960** | **0.920** | cat_2c on the same subset; n=25/cat, ±0.04 = ±1 question (sampling noise). |
| **3** | The Dissonance (contradiction) | **0.00** *(emergent, real KG)* | **recall 0.50 / prec 0.25** *(emergent)* | **RE-SCORED** (see below): mempalace's live palace KG has **0 contradicts edges** — its enrichment pipeline generates no emergent contradiction structure on real content. The prior **+1.00 was a corpus-declared ceiling** (good-dog-graph reads back hand-seeded edges), now superseded. OMEGA: auto-relate **generated** 4 contradicts edges from good-dog text — caught 1 of 2 ground-truth themes (grain-free DCM ✓, dominance-theory ✗). Both are now emergent reads, but on **different corpora** (live palace vs good-dog). |
| **4** | The Threshold (ingestion) | coverage 1.00 · **entropy 0.645** (`other` 26.83%, 40 types) — FINAL post re-map + junk-DELETE | collisions 0 · coverage 1.00 · **entropy 0.78** | **FINAL exact full-graph** (#211 cypher over the RELATION set; mempalace#45 re-map + `--drop-code-tokens` DELETE applied). Norm entropy **0.645**, dominant `other` **26.83%** (502,573 edges), **40** distinct types, 1,873,489 edges (1.92M − 48,135 junk code-token edges DELETE'd). SUPERSEDES the intermediate pre-DELETE 0.4378/28.2%/236 reading: the de-monoculture relabel (520k edges off the `other` sink) **plus** the junk-DELETE cut the monoculture 55%→27% and the type tail 237→40. RESOLVED. Both overturn the original capped-projection 0.020/98.98%-tunnel artifact. OMEGA's good-dog emergent vocab is balanced (0.78). Different corpora — directional. |
| **5** | The Missing Room (topology) | **325,965 components · largest 61.87% · isolates 22.2%** (FINAL exact full-graph) | **1 component · 0 isolates · Betti-1 1** | **FINAL exact full-graph WCC** (post-DELETE re-POST via `GET /graph/structural-stats` + SME#223): 325,965 components over 1,156,314 entities, largest **715,435 (61.87%** — well-connected giant component), isolates **256,782 (22.2%)**. Replaces the verdict-only line and the bogus capped-projection 44.8%-isolate artifact. HONEST NOTE: isolates rose vs pre-DELETE (20.4%→22.2%) because deleting 48k junk code-token edges orphaned ~20k entities whose *only* edge was junk — the cleaner graph honestly reports them isolated. OMEGA on a small clean corpus = one fully-connected component (**required an adapter fix** for its edges to resolve). |
| **6** | The Archive (supersession) | **0.00** *(emergent, real KG)* | **0.00** *(emergent)* | **RE-SCORED**: mempalace's live palace KG has **0 supersedes edges** → completeness 0.00; the prior +1.00 was the good-dog-graph declared ceiling (8/8 hand-seeded edges, 5 chains), now superseded. OMEGA emitted **0 supersedes** too — its temporal analogue is `evolution` (17 edges), which doesn't normalize to supersedes. **Both land at 0.00 emergent supersession from different roots** — mempalace emits no supersedes edge at all; OMEGA emits a non-canonical temporal type. |
| **7** | The Abacus (E2E QA) | **0.580** (same-reader) | **0.593** (same-reader) | o4-mini reader + gpt-5.3-chat judge, sme-rich. ≈ parity (+1.3pp), different per-cat profile. NOT comparable to OMEGA's self-reported 95.4% (GPT-4.1 reader). |
| **8** | The Blueprint (ontology) | hierarchical claim **PASS** · **modularity 0.7961** (218 communities) · **introspection 1.0** (live) | type-cov 0.0* · edge-vocab 0.333 · **drift 0.875** · introspection 0.0 | **FINAL exact full-graph**: the structural 'hierarchical' claim **PASSES** with a real number now — modularity **0.7961** across 218 communities (≫0.5), networkx-computed on the full graph after networkx was installed on the familiar daemon and structural-stats re-POSTed. Decisively refutes the prior **FAIL (modularity 0.009)** capped-projection artifact, and replaces the earlier "pending-networkx" verdict. Introspection **1.0** — `GET /ontology` is LIVE on the prod daemon (palace-daemon#205 deployed + restarted; SME scorer #208), self-reporting declared-vs-effective drift (was 0.0). OMEGA drift 87.5%: its **documented** edge vocab ≠ its **emergent** vocab. *OMEGA 8a type-cov 0% is a corpus-coverage artifact (good-dog notes ingested as default `summary`). |
| **9a** | The Handshake (invocation) | **0.983** (opus-4-8 orch, Tau2 99.3) | **N/A — no harness** | Cat 9a is an **orchestrator-model** property, not a substrate property. mempalace's 0.983 is really opus-4-8's invocation rate in front of it. OMEGA was driven via its library API (no model-in-the-loop), so substrate-level 9a is N/A. |
| **9b** | Call-through success | reachable (clean floor) | **N/A — no harness** | OmegaAdapter declares no `get_harness_manifest()` (library usage). Empty-manifest = does-not-apply (real finding: OMEGA ships an MCP server, but the SME adapter uses the library path). |

### Competitor verdict rows — Hindsight + Mem0

Both systems are **adapter-verified and runnable on the harness**, but their full
on-harness QA is **deferred** — both run an LLM fact-extraction on *every session
ingest*, which makes a strat150 run take many hours (see the cost-wall section). So
their rows are **verdicts**, not on-harness QA numbers.

| Cat | Hindsight (Iris, #220/#184) | Mem0-OSS (Solara, #185/#221) |
|---|---|---|
| **1 / 2c / 7** (retrieval + QA) | **verified + runnable; on-harness QA DEFERRED** — extraction-throughput-bound (~60–96 s/session → strat150 ingest ≈ 150 h). Field-reported **91.4%** LongMemEval QA is Hindsight's *own* leaderboard (GPT-4.1-class), **not** an on-harness number. The one n=12 indicative attempt was **INVALIDATED** by a mid-run container SIGTERM (box cleanup, not OOM) — it measured a dead server, so it is **excluded** (not a 0.0). | **verified + runnable; on-harness QA DEFERRED** — adapter verified vs real `mem0ai` 2.0.4 + live smoke green on a local $0 stack (ollama phi4 + nomic-embed-text). Extraction-throughput-bound: warm steady-state **~9 s/ingest → ~18 h** strat150. Extraction is **lossy by design** (smoke stored 3/5 facts). No on-harness QA number slotted. |
| **3 / 4 / 5 / 6 / 8** (structural) | **N/A — no graph endpoint.** Hindsight exposes no standalone graph API; the adapter's snapshot probes an undocumented `/stats` that current Hindsight doesn't serve → empty. It is an *extraction-then-retrieve* memory, not a queryable typed graph. | **N/A — graph memory REMOVED from mem0 OSS.** `relations` aren't populated; the snapshot returns isolated entities with **zero edges**. Distinct cause from Hindsight: Mem0 OSS *had* a graph layer and the open-source edition **dropped it** — it's now a flat/vector store. (The hosted Mem0 platform keeps graph memory; the OSS package under test does not.) |
| **9a / 9b** (handshake) | **N/A — no harness** (driven via client API; adapter declares no harness manifest). | **N/A — no harness** (Python client API; adapter declares no harness manifest). |

---

## Full system roster — honest per-cat coverage (all 8 harness systems)

The point of the expanded view is to show **the whole landscape**: who was run on what,
and what each architecture even *allows*. Every cell is a real reading **or** an explicit
marker. Legend: ✓ = real number (see the detailed table / verdict rows above); **E** =
emergent reading; **QA-def** = verified, on-harness QA deferred (extraction-throughput-bound);
**N/A-graph** = no usable graph (no endpoint, or graph layer removed); **N/A-design** =
no structure *by design* (the control); **N/A-harness** = no invocation surface; **—** =
not run (adapter wired, no baseline reading; an honest coverage gap, not a 0).

| System | role | 1 | 2c | 3 | 4 | 5 | 6 | 7 | 8 | 9a | 9b |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **mempalace** | substrate (verbatim-first) | ✓ 0.927 | ✓ 0.960 | E 0.00 | ✓ 0.645 | ✓ 61.9% | E 0.00 | ✓ 0.580 | ✓ 0.796 | ✓ 0.983 | ✓ |
| **OMEGA** | substrate (embed + auto-relate) | ✓ 0.900 | ✓ 0.920 | E 0.50 | ✓ 0.78 | ✓ 1 comp | E 0.00 | ✓ 0.593 | ✓ drift .875 | N/A-harness | N/A-harness |
| **Hindsight** | competitor (extraction) | QA-def | QA-def | N/A-graph | N/A-graph | N/A-graph | N/A-graph | QA-def | N/A-graph | N/A-harness | N/A-harness |
| **Mem0-OSS** | competitor (extraction) | QA-def | QA-def | N/A-graph | N/A-graph | N/A-graph | N/A-graph | QA-def | N/A-graph | N/A-harness | N/A-harness |
| **flat** | **control** (no-structure vector) | ✓ 0.833 | ✓ 0.833 | N/A-design | N/A-design | N/A-design | N/A-design | ✓ 0.384 | N/A-design | N/A-harness | N/A-harness |
| **rlm** | Cat 9a orchestrator arm | ✓ 0.467 | ✓ | — | — | — | — | — | — | ✓ 0.467 | N/A-harness |
| **full_context** | Karpathy baseline D1 | — | — | — | — | — | — | — | — | — | — |
| **karpathy_compiled** | Karpathy baseline D2 | — | — | — | — | — | — | — | — | — | — |

**Reading the coverage rows:**

- **flat** is the **no-structure control** — the floor every structural delta is measured
  *against*. Its Cat 1 (0.833 mean_recall, jp-realm Cond-A), Cat 2c (by-hop 0.852/0.667 —
  no depth scaling, the expected flat signature) and Cat 7 (0.384 LoCoMo QA / 665 tok-per-
  correct) are **real readings**. Its structural cats are **N/A *by design*** — a different
  N/A from the competitors: there is intentionally no graph; that's what makes it the
  baseline.
- **rlm** is the **Cat 9a orchestrator arm**, and it carries the single most important
  invocation finding: Qwen-7B and Llama-70B **both plateau at 46.7% recall with 7–27%
  tool-invocation** — a ~10× parameter difference moves nothing, because the ceiling is
  *willingness to invoke the tool*, not retrieval. It's the low rung of the Tau2 ladder
  (gemma4 41.7 → qwen3.5 75.0 → opus-4-8 98.3); cf. mempalace+opus-4-8 = 0.983 / 100%
  invocation on the same corpus. RLM was run for invocation, not the structural cats (—).
- **full_context (D1)** and **karpathy_compiled (D2)** are the **Karpathy baselines** —
  adapters wired into the harness (`sme/conditions/`) but **not yet run** through any cat
  on a shared corpus. Shown as a full not-run row so the roster is honest about what's
  wired-but-unmeasured rather than silently omitting them.
- **Controls not shown as rows:** `oracle_retrieval` (ceiling) and `random_retrieval`
  (floor) adapters are present in the registry but were **not run as matrix cells** — the
  `reader_trueoracle_*` baselines are a reader-config experiment, not an oracle-*adapter*
  Cat run, so giving them numbered rows would misrepresent what was measured.

---

## The extraction-throughput cost wall (the 2-competitor finding)

The single most reusable thing the cross-system pass surfaced about the
*competitors* isn't a QA score — it's a **cost structure** the public leaderboards
hide. Both extraction-based systems are **benchmark-throughput-bound**:

- **Hindsight** runs an LLM fact-extraction per session ingest (~60–96 s/session
  on the local reasoning model) → a strat150 ingest alone is ≈ **150 h**, and a
  full QA run needs ~7,200 reasoning-model extraction calls.
- **Mem0-OSS** (Solara, #221) is the same shape: warm steady-state ~9 s/ingest →
  ~**18 h** for a full strat150 (CPU-local, ollama phi4 extractor). Its extraction
  is also **lossy by design** (stored 3 of 5 smoke facts) — and notably, **mem0 OSS
  removed its graph-memory layer entirely**, so it's now a flat/vector store with no
  edges (Cat 3/4/5/6/8 N/A).

Against that, **mempalace's verbatim-first ingest is ~0 marginal cost** — it
stores content directly and enriches asynchronously, so a strat150 ingest is
minutes, not hours. OMEGA sits between (a 384-dim ONNX embed per memory, no
per-ingest LLM call).

This is the **cost thesis the leaderboards don't show**: a system can post a high
QA number while being *so* extraction-expensive that re-running it on a new
corpus under controlled conditions is impractical. Two independent competitors
(Hindsight, Mem0) hitting the same wall makes it a finding, not an anecdote —
and it's exactly the kind of structural property SME exists to expose, the
competitor-side analogue of the substrate diagnostics in the table above.

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

**Cat 4 — FINAL exact full-graph (re-map + junk-DELETE applied), publishable.** Anchored to
the **full-graph EXACT** distribution (#211 server-side cypher over the RELATION set — runs
in seconds, not the slow `/graph` sample), now **after both the canonical re-map (mempalace#45)
and the `--drop-code-tokens` DELETE**: normalized entropy **0.645**, dominant `other` **26.83%**
(502,573 edges), **40** distinct edge types, 1,873,489 total edges (1.92M − 48,135 junk
code-token edges DELETE'd). This SUPERSEDES the intermediate **pre-DELETE** reading
(0.4378 / 28.2% / 236 types, relabel-only): the de-monoculture relabel cut the `other` sink
55%→~28%, and the junk-DELETE then dropped the long type-tail 236→40 and lifted entropy to
0.645. RESOLVED (#45). Both overturn the original capped-projection 0.020/98.98%-tunnel
artifact.

**Cat 5 — FINAL exact full-graph WCC (post-DELETE), publishable.** Computed server-side
(palace-daemon#211 `GET /graph/structural-stats` + SME#223 consumer): union-find WCC over the
whole RELATION graph, **post-DELETE re-POST**. **325,965 components** over 1,156,314 entities,
largest **715,435 (61.87%** — a well-connected giant component), isolates **256,782 (22.2%)**.
Replaces the verdict-only line and the bogus capped-projection 44.8%-isolate artifact.
**HONEST NOTE:** isolates *rose* vs the pre-DELETE re-POST (20.4%→22.2%) because deleting the
48k junk code-token edges orphaned ~20k entities whose *only* edge was junk — pre-DELETE they
were "connected" by noise; the cleaner graph honestly reports them isolated. Not a regression,
a truer reading.

**Cat 8 — FINAL exact full-graph (modularity computed) + introspection LIVE.** The structural
'hierarchical' claim **PASSES** with a real number now: modularity **0.7961** across 218
communities (≫0.5), **networkx-computed on the full graph** after networkx was installed on the
familiar daemon and structural-stats re-POSTed (#157). This replaces the earlier
"pending-networkx" verdict and decisively refutes the prior FAIL (modularity 0.009 was a
capped-projection artifact). **Introspection is now 1.0** — the `/ontology` endpoint
(palace-daemon#205) is **deployed and live on the prod familiar daemon** (restarted),
self-reporting declared-vs-effective drift; the SME scorer (#208) credits it. Was 0.0.

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

## Coordination — Hindsight (#107/#220, slotted) + Mem0 (#185, pending)

- **Hindsight (Iris, #220/#184) — SLOTTED** as a verdict row (above): verified +
  runnable; retrieval/QA cats deferred (extraction-throughput-bound); structural cats
  N/A (no graph endpoint); field-reported 91.4% recorded as a field number, the
  SIGTERM-invalidated n=12 attempt excluded.
- **Mem0-OSS (Solara, #185/#221) — SLOTTED** as a verdict row: verified + runnable;
  retrieval/QA deferred (extraction-throughput-bound, ~9 s/ingest → ~18 h strat150);
  structural cats N/A because **graph memory was removed from mem0 OSS** (zero edges) —
  a distinct finding from Hindsight's no-endpoint (Mem0 *had* a graph layer and the OSS
  edition dropped it).

**The matrix is now EXPANDED across all 8 harness-tested systems** (mempalace, OMEGA,
Hindsight, Mem0, flat, rlm, full_context, karpathy_compiled) — see the full roster table
above for honest per-cat coverage. The pass for any future graph-bearing adapter still runs
the full structural set on good-dog; the first question to each adapter owner remains
**"does this system expose typed edges, or is it retrieval/extraction-only?"** —
retrieval/extraction-only records **N/A — no graph to evaluate** on 3/4/5/6/8 (a finding),
as both Hindsight and Mem0 did. **Next coverage steps** (honest gaps, not failures): run the
Karpathy baselines (full_context D1 / karpathy_compiled D2) through the retrieval/QA cats on
a shared corpus to fill their not-run row; the oracle/random controls likewise have adapters
but no matrix runs yet.
