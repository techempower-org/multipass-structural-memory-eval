# SME Campaign Synthesis — Structural Memory Evaluation of the 2026 Memory-System Field

**Date:** 2026-05-31
**Author:** Nebula (SME dream-team), synthesizing the full campaign
**Status:** FINAL — all inputs landed. The bootstrap-CI / FDR significance pass (#21) is folded into §8; the field retrieval wave (#234/#247) is folded into §3.1 (ai-memory R@5 0.920, benched) and §6 (agentmemory throughput-wall correction). Every number traces to a committed `baselines/` artifact (provenance index appendix).
**Posture:** Diagnostic, not a leaderboard. Every cell is a controlled reading of one substrate under stated conditions. We never blur *measured* (our harness) against *claimed* (a vendor's self-report), and we never mix R@K retrieval recall with end-to-end QA accuracy.

---

## 0. Executive summary

Over this campaign SME — the Structural Memory Evaluation framework — was driven across **nine diagnostic categories**, **six-plus benched substrates**, and a **33-system published field** drawn from the MemPalace memory-system survey. Five findings carry the report:

1. **Retrieval is near-ceiling; the reader is the bottleneck.** On LongMemEval-S, oracle retrieval R@5 is 0.974 and the deployed substrate reaches R@5 ≈ 0.927 — yet end-to-end QA tops out far lower because the *reader* (retrieve → synthesize → answer) loses the points, not the *retriever*. Widening the retrieval window from top-5 to top-20 buys **+17.3pp QA** (0.567 → 0.740) and then plateaus; the residual gap to the 0.868 reader ceiling is synthesis, not retrieval.

2. **Four independent structural levers came back NULL — the vector backbone is the lever.** Age-fusion (the AGE knowledge-graph re-ranker) showed no significant retrieval gain on *three* corpora (LongMemEval-S, LoCoMo, jp-realm) — the two with paired baselines are **CI-confirmed non-significant** under FDR correction (§8); the hybrid retriever's **graph leg is inert** on the golden query set (hybrid ≡ union, byte-identical); and **cross-encoder reranking is neutral-to-negative** on a corpus-complete daemon (R@10 flat at 0.60, MRR drops, the big CE is 3× slower). The consistent message across all four: the **dense vector + BM25 backbone already does the work**, and the structural add-ons are targeted tools, not blanket improvements.

3. **Storage substrate is equivalent on retrieval and QA — the engine isn't the variable.** Swapping ChromaDB for postgres+pgvector while holding embedding, corpus, reader, and judge fixed leaves both retrieval (R@5 0.833 == 0.833) and QA (0.392 ≈ 0.384) statistically identical. The QA equivalence is now **CI-confirmed** (§8): the paired delta's 95% CI is [−2.0, +2.8]pp with 9/250 questions discordant, p_adj 0.84 — null, not eyeballed. The substrate carries the answer; the storage engine does not.

4. **Three structural cells were measurement artifacts; the corrected readings overturn them.** A capped `/graph` projection reported Cat 4 as "98.98% one edge type, normalized entropy 0.020" — a combinatorial `tunnel`-scaffold artifact. The real knowledge graph reads **normalized entropy 0.645**, a **61.87% giant component** (Cat 5), and **Louvain modularity 0.796** (Cat 8, hierarchy PASS). The artifact said "monoculture / fragmented / flat"; the truth is "diverse / connected / hierarchical." This is the framework catching itself.

5. **The field splits on ingest cost — along TWO axes.** Verbatim/retrieval-only systems with fast bulk ingest are **$0 marginal to bench** (ai-memory benched at R@5 0.920 — pure FTS5+MiniLM, no graph, level with mempalace). The wall has two independent causes: a **write-time LLM** (Mem0 OSS ≈ 18h, Hindsight ≈ 150h of fact-extraction) *and*, separately, **slow per-item ingest throughput** — agentmemory is LLM-free at write yet still throughput-walled (~15h at ~0.15–0.3 obs/s), a scoping-vs-reality correction (§6.1). "No write-time LLM" is necessary but not sufficient for cheap benching.

---

## 1. The full field — measured vs claimed

The MemPalace survey (`memorypalace/docs/research/2026-05-24-memory-system-benchmarks.md`) catalogues the 2026 memory-system landscape. SME's full-field matrix (`baselines/cross_system_multipass_matrix_2026-05-30.json`) carries **two column-groups that are never conflated**:

- **`published_field`** — the system's *own self-reported* (or indie-reproduced) benchmark claim. NOT SME-measured. Metric, answer-model, judge, and LoCoMo subset all vary system-to-system.
- **`sme_multipass`** — our harness reading under stated, held-constant conditions.

### 1.1 Verification levels (honesty ladder)

| Level | Meaning | Systems |
|---|---|---|
| **sme_measured** | Run on our harness, conditions stated | mempalace, OMEGA, flat, rlm, postgres_ingest |
| **verified-qa-deferred** | Adapter verified + runnable; full QA deferred (extraction-throughput-bound) | Hindsight, Mem0-OSS |
| **indie** | Reproduced by a third party (not the vendor) | Hindsight (Virginia Tech + Washington Post), some LoCoMo via Hindsight repo |
| **paper** | arXiv claim, code may or may not be released | True Memory, Engram (academic), Zep, Memmachine |
| **self** | Vendor self-report only | OMEGA, Mastra, Mem0-platform-v3, Supermemory, EverOS, agentmemory, engram-2, ai-memory, mcp-memory-service |
| **none** | No published benchmark | Open Brain, Claude-Mem, Letta, Cognee, CaviraOSS, EngramX, iai-mcp, longhand, ladybugdb |

### 1.2 Published field — Appendix-B headline (self-reported, NOT comparable across rows)

Sorted by best published LongMemEval QA. **These are vendor claims under heterogeneous conditions** — the answer-model swing alone is ~24pp (see §4.2). Read down a column at your peril.

| System | LongMemEval QA | LoCoMo QA | BEAM-1M | Metric | Answer model | Verification |
|---|---|---|---|---|---|---|
| OMEGA | 95.4% | — | — | QA | GPT-4.1 | self |
| Mastra | 94.87% | — | — | QA | GPT-5-mini | self |
| Mem0 platform v3 | 94.4% | 92.5% | 70.1% | QA | undisclosed (cloud) | self |
| Hindsight | 91.4% / 89.0% (OSS-120B) | 89.61% | 73.9% | QA | Gemini-3 Pro | **indie** |
| True Memory Pro | 87.8% | 93.0% | 76.6% | QA | gpt-4.1-mini | paper |
| True Memory Base | 85.5% | 92.0% | 74.9% | QA | gpt-4.1-mini | paper |
| Supermemory | 81.6–85.2% | 65.4% | — | QA | GPT-4o → Gemini-3 | indie |
| EverOS/EverMind | 83.0% | 93.05% | — | QA | undisclosed | self |
| Engram (paper) | 71.4% | 77.55% | — | QA | GPT-4o-mini | paper |
| Zep/Graphiti | 71.2% / 63.8% (4o-mini) | 75.14% | — | QA | GPT-4o | self+paper+indie |
| Memmachine | — | 91.7% | — | QA | gpt-4.1-mini | paper |
| Memobase | — | 75.78% | — | QA | undisclosed | indie |
| Celiums | 62.3% | — | — | QA | best-of-5 | self |
| Mem0 OSS (older alg) | 67.8% | 61.4–68.5% | 64.1% | QA | undisclosed | indie |
| — *retrieval-only (R@K, NOT QA)* — | | | | | | |
| MemPalace (upstream raw) | (96.6% R@5) | (88.9% R@10) | — | R@K | n/a | indie |
| ai-memory | (97.8% R@5) | — | — | R@K | n/a | self |
| engram-2 | (99.0% R@5) | 74.5% (strict judge) | — | Mixed | GPT-5.4 | self |
| agentmemory | (95.2% R@5) | — | — | R@K | n/a | self |
| mcp-memory-service | (80.4–86.0% R@5) | (49.7% R@5) | — | R@K | n/a | self |

**No published data:** Open Brain OB1, Claude-Mem, Letta/MemGPT, Cognee, CaviraOSS OpenMemory, EngramX, iai-mcp (LongMemEval-S run exists, score withheld).

The honesty line: the only rows we *stand behind as measurements* are the `sme_measured` set in §3. Everything in this table is what the vendor (or an indie reproducer) said — recorded for context, never adopted as a finding.

---

## 2. The nine multipass categories — what each probes

SME tests what a memory system knows about *its own structure*, not merely whether it retrieves. (`docs/sme_spec_v8.md`)

| Cat | Name | Probes | Healthy signature |
|---|---|---|---|
| **1** | The Lookup | R@5 factual retrieval recall | high recall on single-fact lookups |
| **2c** | The Stairway | multi-hop / multi-session R@5 | recall holds as hop-depth rises (traversal) |
| **3** | The Dissonance | contradiction detection | emergent `contradicts` edges over the KG |
| **4** | The Threshold (Ingestigation) | ingestion integrity — dedup, field coverage, edge-type entropy | balanced edge vocab, no monoculture, no collisions |
| **5** | The Missing Room | topology — connectivity, isolates, components | one giant connected component, few isolates |
| **6** | The Archive | temporal supersession | emergent `supersedes` edges tracking fact updates |
| **7** | The Abacus | token efficiency / E2E QA (A/B/C conditions) | structure earns its tokens vs the flat floor |
| **8** | The Blueprint | ontology coherence — declared-vs-effective, modularity | low drift, high modularity, self-introspection |
| **9a/9b** | The Handshake | invocation rate / call-through success | the orchestrator actually *reaches* memory |

Categories 1, 2c, 7, 9 are **retrieval/QA/integration** cats (runnable on any system). Categories 3, 4, 5, 6, 8 are **structural** cats — they need a typed-edge graph, so they are N/A for flat vector stores *by design* and N/A for systems that expose no graph endpoint (a real finding, see §3.3).

**Cat 2c construct validity — designed → demonstrated.** Cat 2c was originally exercised on jp-realm-v0.1's 10 hand-authored multi-hop questions (construct *designed*). A HotpotQA loader (#43, `sme/corpora/hotpotqa/`) now *demonstrates* it on a public 7,405-question 2-hop corpus (the pinned `dev_distractor` split, 2 gold + 8 distractor paragraphs each). A dependency-free retrieval smoke on the real split returns **64% full multi-hop R@5 / 98% partial**, and — the signal Cat 2c is built to catch — **bridge (sequential 2-hop) ≈67% > comparison (parallel 2-hop) ≈47%**: chained-evidence questions retrieve better than parallel-compare questions, the token-overlap signature multi-hop targets. (E2E Cat 2c against the daemon is the downstream run; the loader is the prerequisite.)

---

## 3. The benched readings — per category

All numbers from `baselines/cross_system_multipass_matrix_2026-05-30.json` and the underlying per-run JSONs. `sme_measured` only.

### 3.1 Retrieval & QA cats (1, 2c, 7, 9a)

| Cat | mempalace | OMEGA | ai-memory | flat | postgres_ingest | rlm | Hindsight / Mem0 |
|---|---|---|---|---|---|---|---|
| **1** R@5 | **0.927** (daemon /search, strat150) | 0.900 (−2.7pp = 4 q) | **0.920** (strat150, session-level) | 0.833 (jp-realm Cond-A) | 0.833 (==flat) | 0.467 (invocation-capped) | deferred (field 91.4% / 67.8%) |
| **2c** R@5 | 0.960 (cat_2c) | 0.920 (−1 q noise) | 0.960 (cat_2c) | 0.833 (hop-1 0.852 / hop-2 0.667, no depth scaling) | 0.833 (==flat) | by-hop avail | deferred |
| **7** QA macro | 0.580 (same-reader) | 0.593 (+1.3pp parity) | n/a (retrieval-only) | 0.384 (LoCoMo E2E n=250) | **0.392** (==flat, Δ noise) | n/a | deferred (cost wall) |
| **9a** invocation | 0.983 @ 100% invoke (Opus, Tau2 99.3) | N/A (library) | N/A (HTTP daemon) | N/A (library) | N/A (library) | **0.467 @ 7–27% invoke** | N/A |

**Reading the rows:**
- **Cat 1/2c:** mempalace, OMEGA, and **ai-memory** land within ~2pp on their *own* R@5 metrics (descriptive, not a tested delta — their per-question hit semantics differ, so §8.2 deliberately declines a CI). ai-memory's 0.920 strat150 R@5 (R@1 0.80, R@10 0.92, n=150, **0 errors**) is the campaign's third independent brick in the §5.2 thesis: it is **pure SQLite FTS5 + MiniLM, no graph, no write-time LLM**, and it lands *level with mempalace's drawer-level 0.920* — the lexical+vector backbone carries retrieval, the machinery on top does not. Its measured 0.920 sits **5.8pp below its published 0.978** (a defensible measured-vs-claimed gap from the pinned subset + per-question isolation; single-session cat_1 0.853 is the soft spot). flat and postgres_ingest are *identical* (storage-equivalence, CI-confirmed in §5.3/§8.1). flat shows the expected verbatim signature — hop-1 0.852 collapsing to hop-2 0.667, no traversal.
- **Cat 7:** OMEGA 0.593 ≈ mempalace 0.580 (same reader). flat 0.384 ≈ postgres_ingest 0.392 on LoCoMo (the storage-equivalence QA leg).
- **Cat 9a — the RLM finding:** Qwen-7B and Llama-70B *both* plateau at 46.7% recall at 7–27% tool-invocation. A 10× parameter gap does not move recall: the ceiling is **willingness to invoke**, not retrieval quality. The orchestrator's Tau2 score (Opus 99.3 → 100% invocation → 98.3% recall) is the load-bearing variable, cross-validating the Tau2-predicts-Cat-9a relationship.

### 3.2 Structural cats (3, 4, 5, 6, 8) — mempalace + OMEGA

| Cat | mempalace (real KG, FINAL) | OMEGA (good-dog auto-relate) |
|---|---|---|
| **3** Dissonance | emergent over real KG (1.92M-edge RELATION set) | theme-recall 0.5 / edge-precision 0.25 (1 of 2 ground-truth themes) |
| **4** Ingestigation | **normalized entropy 0.645**, dominant `other` 26.83%, 40 edge types, 248 collisions, 100% field coverage, 1.87M edges | entropy 0.78, 0 collisions, 100% coverage, 24→18 notes (6 hash-deduped) |
| **5** Topology | **giant component 61.87%**, isolates 22.2%, 325,965 components, 1.156M entities | single component, 0 isolates, 1 persistent H1 hole |
| **6** Archive | 0 emergent `supersedes` edges (supersession-completeness floor) | 0 `supersedes`; `evolution` (17 edges) lacks directional semantics |
| **8** Blueprint | **modularity 0.796**, 218 communities, hierarchy **PASS**, introspection **1.0** (live /ontology) | type-coverage 0.0, edge-vocab 33.3%, drift 0.875, introspection 0.0 |

The mempalace structural numbers are the **post-correction FINAL** readings (§6). OMEGA's are emergent over the 24-note good-dog corpus, so they measure a genuinely different (much smaller) graph — the cross-system structural comparison carries the ontology-sensitivity confound the spec flags (`sme_spec_v8.md` §"What SME deliberately doesn't measure"); these are diagnostic readings of each system's own structure, not a head-to-head ranking.

### 3.3 Distinct N/A reasons — themselves findings

- **flat / postgres_ingest:** N/A-**by-design**. No graph → it is the no-structure control the structural delta is measured *against*.
- **Hindsight:** N/A-**no-graph-endpoint**. Extraction-then-retrieve; serves no standalone graph. `get_graph_snapshot()` returns empty.
- **Mem0-OSS:** N/A-**no-graph-endpoint** for a *different* reason — **graph memory was removed from mem0ai 2.0.4**. `relations` is unpopulated; the snapshot has zero edges. A real architectural regression caught by the harness.
- **rlm:** not-run on the structural cats (it was the Cat 9a invocation arm). An honest coverage gap, not a score of 0.

---

## 4. Methodology — why the deltas are trustworthy

### 4.1 A/B/C/D condition isolation (the load-bearing pattern)

Cat 7 is defined as a four-condition comparison over an identical corpus / embedding / reader / judge (`sme_spec_v8.md` §Cat 7):

- **Condition A (flat):** top-K cosine over the raw vector store, no structure. The control.
- **Condition B (full pipeline):** the system with its structural retrieval layer on — what users experience.
- **Condition C (structure disabled):** the same underlying index queried with the structural filter/router *off*.
- **Condition D (structure + topology):** PageRank topological pre-filter, reported separately as research (not in the headline).

> **The headline metric is A vs B vs C above the grep floor — not A vs B alone.** Without Condition C, the benchmark can't tell "the structural layer earned its complexity" from "the index is already fine and the structure is a tax." Without the floor, it can't tell "the retriever works" from "the filename matcher happened to match."

This is why the four NULLs (§5.2) are *informative* rather than disappointing: each is a clean A-vs-B(-vs-C) delta showing the structural layer added ~nothing on that corpus.

### 4.2 Comparability caveats (applied to every cross-row claim)

From `comparability_caveats` in the matrix:

1. **R@K ≠ QA.** Retrieval recall (mempalace upstream, agentmemory, engram-2, ai-memory, mcp-memory-service) is not comparable to E2E QA accuracy (OMEGA, Mem0, Hindsight, True Memory, …). Celiums proved 100% retrieval can coexist with 62.3% QA — the synthesis ceiling.
2. **Answer-model swing ≈ 24pp** on the *same* benchmark (GPT-4.1 95.4% vs GPT-4o-mini 71.4%). No apples-to-apples QA exists unless the answer model is held constant.
3. **Platform-vs-OSS.** Mem0's 92–94% are the *cloud platform v3*; the OSS package is 61–68% LoCoMo. Label every Mem0 number.
4. **LoCoMo subset varies** (~1,540 full vs 200/300 subsets; adversarial often skipped; engram-2 uses a non-standard strict GPT-5.4 judge). Cross-comparisons unreliable without exact subset + judge.
5. **Judge variation.** Most use a GPT-4o judge; engram-2 strict GPT-5.4; EverMind GPT-OSS-120b@temp0. Not interchangeable.
6. **Ontology granularity (structural cats).** Cat 4's monoculture/entropy signals are *definitionally* a function of the system's ontology granularity, proven by the #45 sensitivity sweep (§5.5): the same graph re-typed flat → moderate → fine moves edge-type entropy from 0.000 → 0.842 → 0.856. **Cat 4 is only cross-comparable at matched type-granularity, and must be reported with the entity/edge type counts.** Cat 5 (topology) is the structural cat that *is* ontology-robust (§5.5) and safe to compare across differently-ontologized systems.

### 4.3 Content-filter handicap accounting

Azure's content filter tripped on exactly one strat150 question (qid `95228167`, a known false-positive, `hate:medium`) at limit=50 only. It was **counted in the denominator as wrong** (conservative floor), not excluded — so the breadth ladder's three legs share an identical n=150 denominator and stay strictly comparable. The kind of handicap that silently skews leaderboards is here surfaced and absorbed into a floor.

### 4.4 Diagnostic posture, not a leaderboard

Every reading is a delta under controlled conditions. We refuse to: (a) report a single number where two query mixes were conflated (the Cat 7b latency "2064 → 746 ms" trap, §5.2); (b) conclude from a non-representative slice (the n=100 age-fusion composition artifact, §5.2); (c) flip a global default off n<25 (the hybrid-weight 0.85/0.15 operating point stays opt-in). Findings are *deltas*, not absolute scores.

---

## 5. The headline story

### 5.1 Retrieval is near-ceiling; the reader is the bottleneck

The single most important campaign result. On LongMemEval-S:

- **Oracle retrieval R@5 = 0.974** (gold sessions retrievable).
- **Deployed substrate R@5 = 0.927** (mempalace daemon /search).
- **True-oracle reader ceiling = 0.868** (gold present, reader given perfect context).
- **Deployed E2E QA** climbed the retrieval-breadth ladder (`docs/benchmarks/2026-05-30-deployed-e2e-ladder.md`):

| retrieval limit | QA-acc | n | CORRECT |
|---|---|---|---|
| 5 | 0.5667 | 150 | 85 |
| 20 | 0.7400 | 150 | 111 |
| 50 | 0.7600 | 150 | 114 |

**+17.3pp from top-5 → top-20, then +2.0pp to top-50 — a plateau at ~20.** Widening retrieval breadth is the single biggest deployed lever measured. The lift concentrates in synthesis-heavy categories (temporal-reasoning +40pp, multi-session +28pp, knowledge-update +20pp) — exactly the question types that need evidence spread across several sessions. **single-session-assistant stays the floor (0.16 → 0.28)**: it is a reader/grounding problem, not a retrieval-breadth problem, so widening the window barely moves it. The residual ~11pp to the 0.868 ceiling is the reader/synthesis floor. Retrieval is not where the points are lost.

### 5.2 The four NULLs — the vector backbone is the lever

| # | Lever tested | Result | Significance (§8) | Evidence |
|---|---|---|---|---|
| 1 | **Age-fusion on LongMemEval-S** | ΔR@5 = **−0.0067** (−1 q of 150); no significant gain | **CI-confirmed null** (CI [−5.3, +4.0], p_adj 0.84) | `2026-05-29-longmemeval-s-results.md` |
| 2 | **Age-fusion on LoCoMo** | ΔQA = **+1.2pp** (noise, n=250); drawer-R@5 Δ = **exactly 0.0** | **CI-confirmed null** (ΔQA CI [−2.0, +2.0]; R@5 byte-identical) | `2026-05-30-locomo-daemon-results.md` |
| 3 | **Age-fusion on jp-realm** | no significant retrieval gain (same pattern) | not separately CI-tested | matrix Cat 2c provenance |
| 4a | **Hybrid graph leg (#111)** | **inert** — zero graph candidates on 12 golden queries; `hybrid` byte-identical to `union`; the only real lever is convex vector/BM25 weight | descriptive only (no paired baseline) | `2026-05-31-hybrid-scorer-weight-tuning.md` |
| 4b | **Cross-encoder rerank (#103)** | **neutral-to-negative** — R@10 flat at 0.60 across all 3 legs, MRR *drops* (0.299 → 0.293 → 0.284), big CE 3× slower (1523ms vs 555ms) | descriptive only (no paired baseline) | `2026-05-31-ce-rerank-corpus-seeded.md` |

Two of the four NULLs are now **CI-confirmed** non-significant under FDR correction (§8.1); the other two are **descriptive-only** because no committed per-question paired baseline exists (§8.2) — the report keeps that distinction sharp. The convergent message: **the dense-vector + BM25 backbone already does the retrieval work.** Age-fusion is a *targeted* re-ranker (directionally plausible on temporal/knowledge-update categories, but unproven at n=25/category — never reported as an effect). The graph leg and the cross-encoder add latency, not recall. The vector backbone — not the graph, not the reranker — is where retrieval quality lives. **The field bench corroborates this from the outside:** ai-memory, a pure FTS5+MiniLM store with *no graph at all*, hits R@5 0.920 — level with mempalace's full graph-augmented stack (§3.1). A system with none of the machinery matches the one with all of it, on the same metric and subset.

Two methodology traps were caught in producing these NULLs, both worth preserving:
- **The n=100 composition artifact.** An early non-stratified n=100 slice showed a +2.0pp age-fusion "win" — but the S corpus is sorted by `question_type`, so the first 100 were 70 single-session + 30 multi-session, zero temporal/KU. The "win" was category composition, not a real effect; it vanished on the stratified n=150. (Filed as #122; fixed by `--stratify-by question_type`.)
- **The two-query-mix latency trap.** The Cat 7b "2064 → 746 ms" number conflated a graph-*firing* query set (pre-index) with a graph-*inert* golden set (post-index). Reported as **two distinct facts**, not one speedup.

### 5.3 Storage-equivalence — the engine is not the variable

Holding embedding (all-MiniLM-L6-v2), corpus (jp-realm-v0.1 Cond-A / LoCoMo-10), reader, and judge fixed, and swapping *only* the storage engine ChromaDB → postgres+pgvector:

| Metric | flat (ChromaDB) | postgres_ingest (pg+pgvector) | Δ |
|---|---|---|---|
| Cat 1 R@5 (jp-realm) | 0.833 | 0.833 | **0.000** |
| Cat 2c by-hop | hop-1 0.852 / hop-2 0.667 | hop-1 0.852 / hop-2 0.667 | **identical** |
| Cat 7 LoCoMo E2E QA (n=250) | 0.384 | 0.392 | **+0.008 (noise)** |

Both retrieval *and* QA are statistically identical. The substrate (embedding + corpus) carries the answer; the storage engine is not the variable. This validates the chroma→postgres migration and isolates "what the backend swap costs" from every other factor — equal recall confirms the migration, and the QA parity confirms it downstream too. **This is the campaign's CI-confirmed central null (§8.1):** the QA delta's 95% CI is [−2.0, +2.8]pp (strict-correct basis) with only 9/250 questions discordant — "the engine is not the variable" is now a tested statement, not an eyeballed one.

### 5.4 The measurement-artifact corrections — the framework catching itself

Three mempalace structural cells were **artifacts of a capped `/graph` projection**, since overturned by full-graph computation (`docs/benchmarks/2026-05-31-cat458-real-kg-crossvalidation.md`):

| Cat | Artifact (capped projection) | FINAL (real full graph) | What flipped |
|---|---|---|---|
| **4** | 98.98% one edge type, normalized entropy 0.020 | **entropy 0.645**, dominant `other` 26.83%, 40 types | "severe monoculture" → "diverse vocabulary" |
| **5** | 44.8% isolates (bogus) | **giant component 61.87%**, isolates 22.2% | "fragmented" → "well-connected" |
| **8** | modularity 0.009, hierarchy FAIL | **modularity 0.796**, hierarchy PASS, introspection 1.0 | "flat / no hierarchy" → "strongly modular" |

Root cause (Cat 4): `_graph_mapping.py` generated `tunnel` edges **combinatorially** — O(k²) per room shared across wings — so popular rooms produced 167,645 of 169,372 edges (98.98%), and the entropy was computed over both the structural scaffold *and* the semantic KG layer with no filter. The fix (`kg_only` mode, #147/PR #210) skips the structural projection. Underneath it sat a *genuine* defect — 55% of `:RELATION` edges in the `other` fallback bucket — fixed by an expanded deterministic predicate pre-pass (mempalace#336: `other` 55% → 27%), and a `--drop-code-tokens` DELETE of 48,135 junk edges, yielding the FINAL 0.645 entropy / 40 types.

Two cross-validations make these trustworthy: (a) a direct-cypher full-graph aggregate reproduced the pre-fix `other` 55.05% / entropy 0.3402 **to the digit**; (b) the corrected numbers were verified read-only against the live AGE graph. The honesty note runs both ways — deleting junk edges *raised* isolates (20.4% → 22.2%) because ~20k entities whose only edge was junk became honestly isolated. The framework's own measurement surface was the bug; SME found it, fixed it, and cross-checked the fix from three vantage points.

> **Read the corrected Cat 4 entropy 0.645 *as ontology-dependent*, not absolute.** The §5.5 sensitivity sweep proves Cat 4's entropy/monoculture signals move with the type vocabulary — and the prod re-map (mempalace#336) is precisely a vocabulary move (it relabeled 520k edges out of the `other` sink). So the 0.020 → 0.34 → 0.645 progression is partly the *real* de-monoculturing of a genuine defect and partly the headline metric tracking the changed ontology. The cell that is robust to all of this is **Cat 5's topology** (61.87% giant component), which the sweep shows is byte-identical under re-typing. Compare systems on Cat 5; report Cat 4 with its type-count.

### 5.5 Ontology-sensitivity — Cat 4 sensitive, Cat 5 robust (a split finding)

The #45 sweep (`docs/benchmarks/2026-05-31-ontology-sensitivity-cat45.md`, `baselines/ontology_sensitivity_good_dog_2026-05-31.json`) took **one** corpus (good-dog, 97 entities / 164 edges) and ran it through three deliberately-different ontologies — **flat** (1 type), **moderate** (8 entity / 10 edge as authored), **fine-grained** (15 entity / 12 edge, strict subtypes) — remapping *only the types*, leaving the node set, edge set, and every identity untouched (topology-preservation is the load-bearing invariant). The result is a clean split, both halves publishable:

| metric | flat | moderate | fine-grained | robust? |
|---|---|---|---|---|
| **Cat 4** edge-type entropy (norm) | 0.000 | 0.842 | 0.856 | ✗ sensitive |
| **Cat 4** dominant-edge fraction | 1.000 | 0.348 | 0.293 | ✗ sensitive |
| **Cat 4** canonical collisions | **1 (false)** | 0 | 0 | ✗ sensitive |
| **Cat 5** components | 4 | 4 | 4 | ✓ identical |
| **Cat 5** largest component | 44 | 44 | 44 | ✓ identical |
| **Cat 5** Betti-0 / Betti-1 | 1 / 9 | 1 / 9 | 1 / 9 | ✓ identical |

- **Cat 5 (topology) is ROBUST** — components, largest-component size, isolates, and both Betti numbers are **byte-identical** across all three ontologies. Not luck: Cat 5's signals are functions of graph topology, and the topology is the same graph under every type vocabulary. **Cross-system Cat 5 comparison is valid even when systems use different ontologies.**
- **Cat 4 (ingestion integrity) is SENSITIVE — by construction.** Normalized entropy is `H / log2(n_types)`, so it is **0.0 by definition** under a one-type ontology and climbs as types split; the dominant-fraction alarm is only interpretable relative to a fixed vocabulary. The sharpest illustration: a *too-coarse* ontology **manufactures a false canonical collision** (flat reports 1 collision because two distinct entities with the same name canonicalize together once `entity_type` is stripped) — an ingestion "defect" that does not exist.

This is the cleanest demonstration of why SME is diagnostic-not-leaderboard: the *same graph* yields a "severe monoculture / one false collision" Cat 4 reading or a "healthy diverse vocabulary" reading depending purely on the type granularity, while its Cat 5 topology reading does not move at all.

---

## 6. Cost-wall taxonomy — verbatim-first vs extraction-based

The #234 scoping proposed a single axis — **write-time extraction cost** — for *whether a system can be benched cheaply*. The bench wave refined it to **two axes**: no write-time LLM is *necessary but not sufficient* for cheap benching; **ingest throughput** is a second, independent wall (§6.1). Classes:

| Class | Definition | Marginal cost | Systems |
|---|---|---|---|
| **Verbatim / retrieval-only, cheap** | embedding/FTS-only at write *and* fast bulk ingest | **$0** | flat, postgres_ingest, mempalace, **ai-memory (benched 0.920)**, engram-2, mcp-memory-service |
| **Throughput-walled** | per-item ingest too slow to bench a full corpus — *whether or not* there is a write-time LLM | **hours** | Mem0-OSS (LLM extraction ~9s/ingest → **~18h** strat150), Hindsight (LLM ~60–96s/session → **~150h**), **agentmemory (NO write-LLM, but ~0.15–0.3 obs/s REST ingest → ~15h**, §6.1), Cognee, Zep/Graphiti (+Neo4j), Letta |
| **Un-benchable locally** | hosted-only / paper-only / framework-coupled / score-withheld | n/a | Mem0-platform-v3 (cloud), Mastra (framework), Supermemory (hosted), True Memory (no code), Engram-paper, EverOS, Memmachine, Celiums, Open-Brain, Claude-Mem, CaviraOSS, EngramX, iai-mcp |

This is why Hindsight and Mem0-OSS are **verified-qa-deferred** rather than measured: their adapters are confirmed working against the real clients (Hindsight #220/#184; Mem0 vs mem0ai 2.0.4 on a $0 local ollama stack), but a full on-harness QA run is throughput-bound to many hours of LLM extraction. We record the field-reported number plus the deferral reason — never a fabricated on-harness QA number. The famous "OSS / installable" KG frameworks (Cognee, Graphiti, Letta) are installable but land in the extraction class — they are not cheap, and they produce QA rows, not the R@5 headline.

### 6.1 Scoping-vs-reality correction: agentmemory is throughput-walled despite being LLM-free at write

The #234 scoping classified **agentmemory** as cheap-(a) on the strength of having *no write-time LLM* (embedding/BM25-only, LLM-compress off by default). The real bench overturned that on a second axis the scoping didn't weigh: **ingest throughput.** agentmemory's per-observation iii-engine REST ingest runs at **~0.15–0.3 obs/s and wedges as the index grows** — at ~70 chunked `observe` calls × ~48 distractor sessions per question, a full strat150 run extrapolates to **~15h**. So agentmemory joins the throughput-walled tier *alongside* Mem0/Hindsight despite being LLM-free at write — a flag-don't-thrash call, not a completed bench. The refinement to the taxonomy: **"no write-time LLM" is necessary but not sufficient for cheap benching; per-item ingest throughput is a second, independent wall.** agentmemory's published 95.2% R@5 stays in the published-field column, **unverified-on-harness (throughput)** — distinct from ai-memory's 0.920, which *is* on-harness. The adapter is built and verified on small loads (committed, #234); a fast-follow with a bulk-ingest path could finish the bench. This is exactly the kind of scoping-estimate-vs-measured-reality gap SME exists to surface — applied to its own provisioning plan.

---

## 7. Per-system verdicts

**mempalace (techempower-org fork) — `sme_measured`, the substrate-under-test.**
Strengths: best-in-class retrieval (R@5 0.927, parity-or-better with OMEGA); a *real* knowledge graph that is diverse (entropy 0.645), connected (61.87% giant), and hierarchical (modularity 0.796) once measured correctly; live ontology introspection. Weaknesses: 0 emergent `supersedes` edges (Cat 6 supersession is a genuine gap); the structural value-add over the vector backbone is marginal on the corpora tested (the four NULLs); reader/synthesis is the QA bottleneck, not the substrate.

**OMEGA — `sme_measured`, competitor substrate.**
Strengths: retrieval parity with mempalace (R@5 0.900, QA 0.593 ≈ 0.580); auto-relate emits a fully-connected single-component graph on good-dog (0 isolates). Weaknesses: high declared-vs-effective ontology drift (0.875), no `supersedes` semantics, introspection 0.0; pure-library usage means Cat 9a/9b are N/A (a real finding — ships an MCP server but the adapter uses the library path).

**Hindsight — verified-qa-deferred, extraction competitor.**
Field-strong (91.4% LongMemEval QA, indie-verified by Virginia Tech + Washington Post). Adapter verified + runnable. Verdict deferred on the ~150h extraction throughput wall. Exposes no graph endpoint → structural cats N/A.

**Mem0-OSS — verified-qa-deferred, extraction competitor.**
The most-cited system, but the platform-vs-OSS gap is enormous (cloud v3 92–94% vs OSS 61–68%). Adapter verified vs mem0ai 2.0.4. Two real findings: ~18h extraction wall, and **graph memory was removed from the OSS package** — structural cats score against an empty edge set.

**flat — `sme_measured`, the no-structure control.**
The floor every structural delta is measured against (R@5 0.833, QA 0.384). Exhibits the textbook verbatim signature: no hop-depth scaling, no structural cats by design. Indispensable as the control, not a product.

**postgres_ingest — `sme_measured`, storage-equivalence probe.**
Identical to flat on both retrieval (0.833) and QA (0.392). Its entire value is the §5.3 finding: the storage engine is not the variable.

**ai-memory — `sme_measured`, field retrieval competitor (#234/#247).**
Pure SQLite FTS5 + MiniLM, no graph, no write-time LLM. Benched R@5 0.920 on strat150 (n=150, 0 errors) — level with mempalace's drawer-level 0.920, reinforcing "the backbone carries retrieval." Measured 0.920 vs published 0.978 = a 5.8pp measured-vs-claimed gap (subset + per-question isolation; single-session cat_1 0.853 the soft spot). The cheapest, fastest competitor to bench in the whole field.

**agentmemory — adapter built + verified-on-small-loads; on-harness bench throughput-walled (#234).**
LLM-free at write, yet *not* cheap to bench: per-observation REST ingest ~0.15–0.3 obs/s → ~15h extrapolated for strat150 (§6.1). The scoping-vs-reality correction that added the second axis to the cost-wall taxonomy. Published 95.2% R@5 stays published-field, unverified-on-harness (throughput); a bulk-ingest fast-follow could finish it.

**rlm — `sme_measured`, the Cat 9a orchestrator arm.**
Carries the invocation finding: 46.7% recall at 7–27% invocation, parameter-count-invariant (7B == 70B). The ceiling is willingness-to-invoke, a failure mode no retrieval benchmark captures.

**Karpathy D1/D2 (full_context, karpathy_compiled) — wired, not benched.**
Methodology baselines (whole-vault-in-context; LLM-compiled wiki). Adapters wired in the registry; not yet run through any cat on a shared corpus. Honest coverage gap.

**longhand / ladybugdb — wired, not benched (#164).**
Verbatim-first cohort. Un-provisionable on this harness: `ingest_corpus` raises NotImplementedError (longhand only ingests Claude-Code session JSONL via its own hooks; ladybugdb is a reader needing an existing `.ldb`). Footnoted.

---

## 8. Statistical rigor — bootstrap CIs + FDR correction

The campaign's informal "±1 question = noise" language is now replaced by formal statistics (#21, `baselines/headline_delta_significance_2026-05-31.json`): **paired bootstrap confidence intervals** (10k resamples on per-question deltas, paired by `question_id`) plus a **Benjamini-Hochberg FDR correction** across the whole metric family (α = 0.05). The section's own honesty story is **two-tier** — some nulls are *CI-confirmed*, others are *descriptive-only* because no committed per-question paired baseline exists. Both are reported as what they are.

### 8.1 CI-backed nulls (true paired per-question, FDR-corrected)

These comparisons pair the *same* per-question metric across two conditions, so a bootstrap CI is meaningful. All are non-significant — every CI straddles zero, every adjusted p = 0.84.

| Comparison | metric | Δ (pp) | 95% CI (pp) | n | n_discordant | p_adj | significant? |
|---|---|---:|---|---:|---:|---:|:--:|
| **Storage-equivalence** (postgres vs flat, LoCoMo) | qa_correct | **+0.4** | **[−2.0, +2.8]** | 250 | **9** | 0.84 | **No (null)** |
| Storage-equivalence (same) | sme_recall | 0.0 | identical | 250 | 0 | — | definitional 0 |
| Age-fusion (LongMemEval-S strat150) | drawer R@5 | −0.7 | [−5.3, +4.0] | 150 | 13 | 0.84 | No (null) |
| Age-fusion (LoCoMo daemon) | qa_correct | 0.0 | [−2.0, +2.0] | 250 | 6 | 0.84 | No (null) |
| Age-fusion (LoCoMo daemon) | drawer R@5 | 0.0 | identical | 250 | 0 | — | definitional 0 |

- **Storage-equivalence is now statistically null, not merely "looks equal" (§5.3).** The +0.4pp postgres-vs-flat QA delta has a 95% CI of [−2.0, +2.8]pp and only **9 of 250 questions discordant** — the runs answer 241/250 identically. **Basis note:** this CI is on the strict-correct per-question vector (abstentions excluded), so its means (postgres 0.264 / flat 0.260) sit below §3's 0.392/0.384, which count correct-abstentions. Both bases agree — the §3 correct-or-abstain delta (+0.8pp) and this strict-correct delta (+0.4pp) are *both* null, same direction. `sme_recall` is byte-identical across all 250 questions (a definitional 0).
- **Two of the four NULLs (§5.2) are CI-confirmed.** Age-fusion on LongMemEval-S (ΔR@5 −0.7pp, CI [−5.3, +4.0]) and on LoCoMo (ΔQA 0.0pp, CI [−2.0, +2.0]; drawer R@5 byte-identical) are both formally non-significant. These are *true* paired A/Bs — same metric, same questions, one endpoint toggled — so the CI is legitimate.

### 8.2 Descriptive-only — point estimates, NO CI (the honest tier)

Three comparisons are **deliberately left without a CI** because the per-question metrics are not paired-comparable, or no committed per-question baseline exists. Computing a CI here would *launder a methodology error into a rigorous-looking number* — so we don't.

- **Cross-system mempalace vs OMEGA R@5 (0.920 vs 0.900)** — `status: descriptive_only`. Reported as **point estimates**, no CI. The `not_comparable_reason` is concrete: mempalace's R@5 is `drawer_hit_at_5` (drawer-level), OMEGA's is `omega_hit_at_5` (native unit), from different runners. Same strat150 subset, but **not the same hit semantics** → not paired-comparable. This is §4.2's comparability caveat made operational — the gate *refused* to emit a CI on incomparable data, which is the framework's honesty enforced in code, not just prose. (The two systems still land within ~2pp on their own metrics, consistent with the "backbone carries retrieval" reading, but that is a descriptive observation, not a tested delta.)
- **CE-rerank on/off (#103)** — `status: no_paired_baseline`. The R@10-flat / MRR-drops / 3×-slower finding stands as a **descriptive** result; no per-question baseline was committed.
- **Hybrid-vs-union / graph-leg-inert (#111)** — `status: no_paired_baseline`. The "hybrid ≡ union, graph leg inert" finding is **descriptive** — though it rests on a byte-identical candidate trace, which is stronger evidence of a tie than a statistical near-miss would be, just not a bootstrap CI.
- **The retrieval-breadth ladder (§5.1, +17.3pp limit5→20)** is a within-system QA ladder, not a paired A/B in this artifact; its weight comes from effect size (17.3pp on n=150 = 26 questions, far outside any plausible noise band), reported as a large descriptive effect, not an FDR-corrected one.

**This two-tier split IS the §8 result.** The central equivalence claim and two of the four NULLs are statistically confirmed; the cross-system delta and the remaining nulls are descriptive — and the section names which is which rather than dressing a summary-only or semantically-mismatched finding in CI language. A diagnostic framework that would *decline* to compute a tempting cross-system CI because the units don't match is the strongest possible statement of the measured-vs-claimed discipline this whole report is built on.

---

## 9. What SME uniquely contributes

No competing benchmark tests Cat 3/4/5/6/8 (ingestion integrity, gap detection, ontology coherence, contradiction/supersession) or Cat 9 (the handshake — invocation rate). The campaign produced the first independent structural-quality readings across memory systems *and* surfaced failure modes the QA leaderboards cannot see: the invocation plateau (Cat 9a), the removed-graph regression (Mem0-OSS), the no-graph-endpoint architecture (Hindsight), and — turned inward — its own capped-projection measurement artifact. It also delivered the meta-results a diagnostic framework owes its own readings: which categories are cross-comparable across differently-built systems (Cat 5 topology robust; Cat 4 ontology-sensitive, §5.5), and construct validity demonstrated on public corpora at scale rather than only designed on a seeded corpus (Cat 2c on HotpotQA's 7,405 questions, §2). The constitutional posture held throughout: lightweight, locally runnable, diagnostic-not-leaderboard, measured-never-blurred-with-claimed.

---

## Appendix: provenance index

Every number above traces to a committed artifact:

- **Matrix spine:** `baselines/cross_system_multipass_matrix_2026-05-30.json`
- **Survey (field):** `memorypalace/docs/research/2026-05-24-memory-system-benchmarks.md`
- **Reader-bottleneck ladder:** `docs/benchmarks/2026-05-30-deployed-e2e-ladder.md` + `baselines/longmemeval_deployed_qa_strat150_limit{5,20,50}_2026-05-30.json`
- **True-oracle ceiling (0.868):** `docs/benchmarks/2026-05-29-true-oracle-floor.md`
- **Age-fusion NULL (LongMemEval):** `docs/benchmarks/2026-05-29-longmemeval-s-results.md`
- **Age-fusion NULL (LoCoMo) + storage-equivalence QA:** `docs/benchmarks/2026-05-30-locomo-daemon-results.md`
- **Graph-leg-inert + hybrid weight:** `docs/benchmarks/2026-05-31-hybrid-scorer-weight-tuning.md`
- **CE-rerank NULL:** `docs/benchmarks/2026-05-31-ce-rerank-corpus-seeded.md`
- **Measurement-artifact corrections:** `docs/benchmarks/2026-05-31-cat458-real-kg-crossvalidation.md` + `baselines/mempalace_cat{4,5,8}_realkg_*_2026-05-31.json`
- **Ontology-sensitivity (Cat 4 sensitive / Cat 5 robust):** `docs/benchmarks/2026-05-31-ontology-sensitivity-cat45.md` + `baselines/ontology_sensitivity_good_dog_2026-05-31.json` (#45)
- **HotpotQA Cat 2c construct validity:** `sme/corpora/hotpotqa/` (loader + README) + `scripts/hotpotqa_retrieval_smoke.py` (#43)
- **Storage-equivalence retrieval:** `baselines/jp_realm_v0_1_{flat,postgres}_condA_*.json`
- **Cost-wall taxonomy:** `docs/mem0_adapter.md`, `docs/hindsight_adapter.md`, #234 scoping (`scratch/nebula-234/scoping.md`)
- **Spec / methodology:** `docs/sme_spec_v8.md`
- **Statistical significance (§8):** `baselines/headline_delta_significance_2026-05-31.json` (#21 / #245 / #246) — paired bootstrap CIs + BH-FDR + the descriptive-only comparability gate
- **Field retrieval wave (§3.1/§6):** `baselines/longmemeval_s_strat150_ai_memory_2026-05-31.json` (ai-memory R@5 0.920) + agentmemory throughput-wall finding (#234/#247)
