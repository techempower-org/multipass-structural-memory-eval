# SME Campaign Synthesis — Structural Memory Evaluation of the 2026 Memory-System Field

**Date:** 2026-05-31
**Author:** Nebula (SME dream-team), synthesizing the full campaign
**Status:** DRAFT — two sets of numbers are TODO-FILL pending in-flight runs (the ai-memory / agentmemory R@5 wave under #234/#177, and the bootstrap-CI / FDR significance pass under #21). Marked clearly inline. The synthesis is otherwise complete and every number traces to a committed `baselines/` artifact.
**Posture:** Diagnostic, not a leaderboard. Every cell is a controlled reading of one substrate under stated conditions. We never blur *measured* (our harness) against *claimed* (a vendor's self-report), and we never mix R@K retrieval recall with end-to-end QA accuracy.

---

## 0. Executive summary

Over this campaign SME — the Structural Memory Evaluation framework — was driven across **nine diagnostic categories**, **six-plus benched substrates**, and a **33-system published field** drawn from the MemPalace memory-system survey. Five findings carry the report:

1. **Retrieval is near-ceiling; the reader is the bottleneck.** On LongMemEval-S, oracle retrieval R@5 is 0.974 and the deployed substrate reaches R@5 ≈ 0.927 — yet end-to-end QA tops out far lower because the *reader* (retrieve → synthesize → answer) loses the points, not the *retriever*. Widening the retrieval window from top-5 to top-20 buys **+17.3pp QA** (0.567 → 0.740) and then plateaus; the residual gap to the 0.868 reader ceiling is synthesis, not retrieval.

2. **Four independent structural levers came back NULL — the vector backbone is the lever.** Age-fusion (the AGE knowledge-graph re-ranker) showed no significant retrieval gain on *three* corpora (LongMemEval-S, LoCoMo, jp-realm); the hybrid retriever's **graph leg is inert** on the golden query set (hybrid ≡ union, byte-identical); and **cross-encoder reranking is neutral-to-negative** on a corpus-complete daemon (R@10 flat at 0.60, MRR drops, the big CE is 3× slower). The consistent message across all four: the **dense vector + BM25 backbone already does the work**, and the structural add-ons are targeted tools, not blanket improvements.

3. **Storage substrate is equivalent on retrieval and QA — the engine isn't the variable.** Swapping ChromaDB for postgres+pgvector while holding embedding, corpus, reader, and judge fixed leaves both retrieval (R@5 0.833 == 0.833) and QA (0.392 ≈ 0.384) statistically identical. The substrate carries the answer; the storage engine does not.

4. **Three structural cells were measurement artifacts; the corrected readings overturn them.** A capped `/graph` projection reported Cat 4 as "98.98% one edge type, normalized entropy 0.020" — a combinatorial `tunnel`-scaffold artifact. The real knowledge graph reads **normalized entropy 0.645**, a **61.87% giant component** (Cat 5), and **Louvain modularity 0.796** (Cat 8, hierarchy PASS). The artifact said "monoculture / fragmented / flat"; the truth is "diverse / connected / hierarchical." This is the framework catching itself.

5. **The field splits cleanly on extraction cost.** Verbatim-first / retrieval-only systems are **$0 marginal to bench** (no LLM at write time). Extraction-based systems (Mem0, Hindsight, Cognee, Graphiti, Letta) hit an **LLM-fact-extraction-per-ingest throughput wall** — Mem0 OSS ≈ 18h, Hindsight ≈ 150h for a full strat150 QA run — making them *verified-but-deferred* rather than freely measured.

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

---

## 3. The benched readings — per category

All numbers from `baselines/cross_system_multipass_matrix_2026-05-30.json` and the underlying per-run JSONs. `sme_measured` only.

### 3.1 Retrieval & QA cats (1, 2c, 7, 9a)

| Cat | mempalace | OMEGA | flat | postgres_ingest | rlm | Hindsight / Mem0 |
|---|---|---|---|---|---|---|
| **1** R@5 | **0.927** (daemon /search, strat150) | 0.900 (−2.7pp = 4 q) | 0.833 (jp-realm Cond-A) | 0.833 (==flat) | 0.467 (invocation-capped) | deferred (field 91.4% / 67.8%) |
| **2c** R@5 | 0.960 (cat_2c) | 0.920 (−1 q noise) | 0.833 (hop-1 0.852 / hop-2 0.667, no depth scaling) | 0.833 (==flat) | by-hop avail | deferred |
| **7** QA macro | 0.580 (same-reader) | 0.593 (+1.3pp parity) | 0.384 (LoCoMo E2E n=250) | **0.392** (==flat, Δ noise) | n/a | deferred (cost wall) |
| **9a** invocation | 0.983 @ 100% invoke (Opus, Tau2 99.3) | N/A (library) | N/A (library) | N/A (library) | **0.467 @ 7–27% invoke** | N/A |

**Reading the rows:**
- **Cat 1/2c:** mempalace and OMEGA are within sampling noise of each other (±1–4 questions on n=150). flat and postgres_ingest are *identical* (storage-equivalence, §5.3). flat shows the expected verbatim signature — hop-1 0.852 collapsing to hop-2 0.667, no traversal.
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

| # | Lever tested | Result | Evidence |
|---|---|---|---|
| 1 | **Age-fusion on LongMemEval-S** | ΔR@5 = **−0.0067** (−1 q of 150); no significant gain | `2026-05-29-longmemeval-s-results.md` |
| 2 | **Age-fusion on LoCoMo** | ΔQA = **+1.2pp** (noise, n=250); drawer-R@5 Δ = **exactly 0.0** | `2026-05-30-locomo-daemon-results.md` |
| 3 | **Age-fusion on jp-realm** | no significant retrieval gain (same pattern) | matrix Cat 2c provenance |
| 4a | **Hybrid graph leg (#111)** | **inert** — zero graph candidates on 12 golden queries; `hybrid` byte-identical to `union`; the only real lever is convex vector/BM25 weight | `2026-05-31-hybrid-scorer-weight-tuning.md` |
| 4b | **Cross-encoder rerank (#103)** | **neutral-to-negative** — R@10 flat at 0.60 across all 3 legs, MRR *drops* (0.299 → 0.293 → 0.284), big CE 3× slower (1523ms vs 555ms) | `2026-05-31-ce-rerank-corpus-seeded.md` |

The convergent message: **the dense-vector + BM25 backbone already does the retrieval work.** Age-fusion is a *targeted* re-ranker (directionally plausible on temporal/knowledge-update categories, but unproven at n=25/category — never reported as an effect). The graph leg and the cross-encoder add latency, not recall. The vector backbone — not the graph, not the reranker — is where retrieval quality lives.

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

Both retrieval *and* QA are statistically identical. The substrate (embedding + corpus) carries the answer; the storage engine is not the variable. This validates the chroma→postgres migration and isolates "what the backend swap costs" from every other factor — equal recall confirms the migration, and the QA parity confirms it downstream too.

### 5.4 The measurement-artifact corrections — the framework catching itself

Three mempalace structural cells were **artifacts of a capped `/graph` projection**, since overturned by full-graph computation (`docs/benchmarks/2026-05-31-cat458-real-kg-crossvalidation.md`):

| Cat | Artifact (capped projection) | FINAL (real full graph) | What flipped |
|---|---|---|---|
| **4** | 98.98% one edge type, normalized entropy 0.020 | **entropy 0.645**, dominant `other` 26.83%, 40 types | "severe monoculture" → "diverse vocabulary" |
| **5** | 44.8% isolates (bogus) | **giant component 61.87%**, isolates 22.2% | "fragmented" → "well-connected" |
| **8** | modularity 0.009, hierarchy FAIL | **modularity 0.796**, hierarchy PASS, introspection 1.0 | "flat / no hierarchy" → "strongly modular" |

Root cause (Cat 4): `_graph_mapping.py` generated `tunnel` edges **combinatorially** — O(k²) per room shared across wings — so popular rooms produced 167,645 of 169,372 edges (98.98%), and the entropy was computed over both the structural scaffold *and* the semantic KG layer with no filter. The fix (`kg_only` mode, #147/PR #210) skips the structural projection. Underneath it sat a *genuine* defect — 55% of `:RELATION` edges in the `other` fallback bucket — fixed by an expanded deterministic predicate pre-pass (mempalace#336: `other` 55% → 27%), and a `--drop-code-tokens` DELETE of 48,135 junk edges, yielding the FINAL 0.645 entropy / 40 types.

Two cross-validations make these trustworthy: (a) a direct-cypher full-graph aggregate reproduced the pre-fix `other` 55.05% / entropy 0.3402 **to the digit**; (b) the corrected numbers were verified read-only against the live AGE graph. The honesty note runs both ways — deleting junk edges *raised* isolates (20.4% → 22.2%) because ~20k entities whose only edge was junk became honestly isolated. The framework's own measurement surface was the bug; SME found it, fixed it, and cross-checked the fix from three vantage points.

---

## 6. Cost-wall taxonomy — verbatim-first vs extraction-based

The decisive axis for *whether a system can be benched cheaply* is **write-time extraction cost**, not installability (the #234 scoping finding). Three classes:

| Class | Definition | Marginal cost | Systems |
|---|---|---|---|
| **Verbatim / retrieval-only** | embedding/FTS-only at write; no LLM per document | **$0** | flat, postgres_ingest, mempalace, agentmemory, engram-2, ai-memory, mcp-memory-service |
| **Extraction-based** | LLM fact-extraction per ingest; throughput-bound | **hours** | Mem0-OSS (~9s/ingest warm → **~18h** strat150), Hindsight (~60–96s/session → **~150h**), Cognee, Zep/Graphiti (+Neo4j), Letta |
| **Un-benchable locally** | hosted-only / paper-only / framework-coupled / score-withheld | n/a | Mem0-platform-v3 (cloud), Mastra (framework), Supermemory (hosted), True Memory (no code), Engram-paper, EverOS, Memmachine, Celiums, Open-Brain, Claude-Mem, CaviraOSS, EngramX, iai-mcp |

This is why Hindsight and Mem0-OSS are **verified-qa-deferred** rather than measured: their adapters are confirmed working against the real clients (Hindsight #220/#184; Mem0 vs mem0ai 2.0.4 on a $0 local ollama stack), but a full on-harness QA run is throughput-bound to many hours of LLM extraction. We record the field-reported number plus the deferral reason — never a fabricated on-harness QA number. The famous "OSS / installable" KG frameworks (Cognee, Graphiti, Letta) are installable but land in the extraction class — they are not cheap, and they produce QA rows, not the R@5 headline.

> **TODO-FILL (in-flight, #234/#177, Selene):** the verbatim/retrieval-only wave. ai-memory and agentmemory R@5 on strat150 ($0 local) are benching now. When they land, add two `sme_measured` R@5 rows here and in §3.1 — they are directly comparable to the mempalace 0.927 R@5 headline (both R@5, same subset), no metric-mixing. engram-2 is gated on a cloud-embedding decision (its default embedding is Gemini Embed 2); Cognee/Graphiti/Letta are parked as a deliberate, JP-gated extraction-QA arc.

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

**rlm — `sme_measured`, the Cat 9a orchestrator arm.**
Carries the invocation finding: 46.7% recall at 7–27% invocation, parameter-count-invariant (7B == 70B). The ceiling is willingness-to-invoke, a failure mode no retrieval benchmark captures.

**Karpathy D1/D2 (full_context, karpathy_compiled) — wired, not benched.**
Methodology baselines (whole-vault-in-context; LLM-compiled wiki). Adapters wired in the registry; not yet run through any cat on a shared corpus. Honest coverage gap.

**longhand / ladybugdb — wired, not benched (#164).**
Verbatim-first cohort. Un-provisionable on this harness: `ingest_corpus` raises NotImplementedError (longhand only ingests Claude-Code session JSONL via its own hooks; ladybugdb is a reader needing an existing `.ldb`). Footnoted.

---

## 8. Statistical rigor — bootstrap CIs + FDR correction

> **TODO-FILL (in-flight, #21, Morpheus).** This section will carry the formal statistics for every delta claimed above:
> - **Bootstrap confidence intervals** on each readout delta (R@5, QA, the four NULLs, the storage-equivalence Δ). The campaign has been disciplined about "±1 question = ±0.04 at n=25" informal noise bounds; this replaces them with resampled CIs.
> - **Benjamini-Hochberg FDR correction** across the family of delta tests, so the multiple-comparisons problem across ~dozen A/B deltas is controlled. The expected outcome: the four NULLs survive as non-significant (CIs straddling 0), the retrieval-breadth ladder +17.3pp survives as significant, and the storage-equivalence Δ survives as non-significant — but these will be *stated with corrected p-values*, not asserted.
> - When #21 lands, fold the per-delta CI/q-value table in here and back-reference it from §5.2 (NULLs) and §5.3 (equivalence) so every "noise" / "significant" word in this report is backed by a corrected statistic.

Until then, the significance language in this draft is the campaign's informal sampling-bound convention (deltas within ±1–2 questions at the stated n are called noise), explicitly *not* a formal test.

---

## 9. What SME uniquely contributes

No competing benchmark tests Cat 3/4/5/6/8 (ingestion integrity, gap detection, ontology coherence, contradiction/supersession) or Cat 9 (the handshake — invocation rate). The campaign produced the first independent structural-quality readings across memory systems *and* surfaced failure modes the QA leaderboards cannot see: the invocation plateau (Cat 9a), the removed-graph regression (Mem0-OSS), the no-graph-endpoint architecture (Hindsight), and — turned inward — its own capped-projection measurement artifact. The constitutional posture held throughout: lightweight, locally runnable, diagnostic-not-leaderboard, measured-never-blurred-with-claimed.

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
- **Storage-equivalence retrieval:** `baselines/jp_realm_v0_1_{flat,postgres}_condA_*.json`
- **Cost-wall taxonomy:** `docs/mem0_adapter.md`, `docs/hindsight_adapter.md`, #234 scoping (`scratch/nebula-234/scoping.md`)
- **Spec / methodology:** `docs/sme_spec_v8.md`
- **TODO-FILL pending:** #234/#177 (ai-memory + agentmemory R@5), #21 (bootstrap CIs + BH-FDR)
