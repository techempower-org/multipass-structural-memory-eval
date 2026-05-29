# Comparison Readiness: How Far to a Head-to-Head vs. the Memory-System Field

**Date:** 2026-05-29
**Author:** Selene (SME dream-team)
**Question (JP's ask):** read the mempalace research docs — especially the
benchmarking ones — and assess *how far we have to go to have results that
let us compare against all the other memory systems.*

**Method:** read-then-synthesize across the mempalace fork's `docs/research/`
(21 docs, local clone at `~/Projects/memorypalace`, 0 behind upstream) and
this repo's `docs/related_work/` + `docs/cross_validation_2026.md`. Every
claim cites its source doc. Our own verified numbers are read directly from
`baselines/*.json` (this session's runs).

**Bottom line up front:** the *survey* work is done — we know exactly which
systems and benchmarks define the field, what metric each leaderboard
reports, and where the comparability traps are. The *measurement* work is
not. We have a clean, defensible **retrieval (R@5)** story on LongMemEval-S,
but the leaderboards everyone cites are **end-to-end QA accuracy**, and our
only QA numbers to date are confounded by reader/judge model choice (Pass A
oracle tops out at ~0.52 with gpt-5.3-chat as both reader and judge — not
the canonical GPT-4o judge, not a frontier answer model). To publish a
defensible "mempalace vs the field" table we need: (1) the LongMemEval
**canonical GPT-4o judge** wired and run, (2) a **fixed, frontier answer
model** held constant, and (3) at minimum a **LoCoMo loader** so we're not a
one-benchmark entrant. Items 1–3 are scoped and mostly un-blocked; the
honest estimate is **medium effort, not a research project** — see §4.

---

## 1. Landscape — who and what we're comparing against

The field and its benchmarks are catalogued in
`2026-05-24-memory-system-benchmarks.md` (the 32KB landscape survey), with
the architectural framing in the two `compass_artifact_*.md` syntheses and
`2026-05-24-true-memory-comparison.md`. Cross-referenced against this repo's
`docs/related_work/longmemeval.md` and `locomo-and-memorybench.md`.

### 1.1 The benchmarks (and the metric each reports)

| Benchmark | Canonical metric | What "good" is | Source |
|---|---|---|---|
| **LongMemEval-S** (Wu et al., ICLR 2025) | **E2E QA accuracy** (retrieve → answer → GPT-4o judge, >97% human agreement) | full-context GPT-4o 60.2%; oracle 87.0%; >90% is frontier | `2026-05-24-memory-system-benchmarks.md` §1; `related_work/longmemeval.md` |
| **LoCoMo** (Maharana et al., ACL 2024) | **E2E QA accuracy** (LLM-as-judge); also F1 / R@k | full-context ~52–66%; >90% frontier | `2026-05-24-memory-system-benchmarks.md` §1; `related_work/locomo-and-memorybench.md` Part 1 |
| **BEAM** (mem0ai/memory-benchmarks) | **E2E QA pass rate** at 100K–10M token buckets | 70%+ at 1M competitive; 50% at 10M notable | `2026-05-24-memory-system-benchmarks.md` §1 |
| **ConvoMem** | personalization / preference | (no fixed bar) | `2026-05-24-memory-system-benchmarks.md` §1; via MemoryBench |
| **DMR** (MemGPT's metric) | deep memory retrieval | Zep 94.8%, MemGPT 93.4% | `2026-05-24-memory-system-benchmarks.md` §1 |
| **MemoryBench** (supermemoryai) | *runner*, not a dataset — wraps LoCoMo / LongMemEval / ConvoMem; judge-agnostic | n/a | `related_work/locomo-and-memorybench.md` Part 2 |

The decisive fact, stated in `2026-05-24-memory-system-benchmarks.md` §3
("The Retrieval Recall vs. QA Accuracy Gap"): **R@5 and QA accuracy are not
comparable.** Celiums shows 100% retrieval but 62.3% QA; True Memory's §6.2
diagnostic shows 330/357 wrong answers fixed when handed the correct
context. The leaderboards are QA; our headline numbers so far are R@5.

### 1.2 The systems (best published LongMemEval, sorted)

From `2026-05-24-memory-system-benchmarks.md` Appendix B. Metric type is the
load-bearing column:

| System | LongMemEval | LoCoMo | Metric | Answer model | Verification |
|---|---|---|---|---|---|
| OMEGA | 95.4% | — | QA | GPT-4.1 | self |
| Mem0 (platform v3) | 94.4% | 92.5% | QA | undisclosed | self |
| Mastra | 94.87% | — | QA | GPT-5-mini | self |
| Hindsight | 91.4% | 89.61% | QA | Gemini 3 Pro | **indie** (VT + WaPo) |
| True Memory (Pro) | 87.8% | 93.0% | QA | gpt-4.1-mini | **paper** (arXiv:2605.04897) |
| Supermemory | 81.6–85.2% | 65.4% | QA | GPT-4o / Gemini-3 | indie |
| EverOS/EverMind | 83.0% | 93.05% | QA | undisclosed | self |
| ENGRAM (paper) | 71.4% | 77.55% | QA | GPT-4o-mini | paper |
| Zep/Graphiti | 71.2% | 75.14% | QA | GPT-4o | self/paper |
| Celiums | 62.3% | — | QA | Opus (best of 5) | self |
| **MemPalace (upstream)** | **96.6–98.4% R@5** | 88.9% R@10 | **R@K only** | n/a | indie (ChromaDB baseline) |
| engram-2 | 99.0% R@5 | 74.5% QA | mixed | GPT-5.4 | self |
| agentmemory | 95.2% R@5 | — | R@K only | n/a | self |
| ai-memory | 97.8% R@5 | — | R@K only | n/a | self |
| mcp-memory-service | 80.4–86.0% R@5 | 49.7% R@5 | R@K only | n/a | self |

Architectural framing (for the comparison's narrative, not the numbers) —
the field splits into **compile-upstream** (Pinecone Nexus, Cognee,
GraphRAG), **verbatim-first** (MemPalace, True Memory, traditional RAG), and
**hybrid** (Zep/Graphiti bi-temporal, Letta/MemGPT tiered, Mem0
self-editing): `compass_artifact_wf-28bac4e8…md`. The
**four-layer model** (Storage → Encoder → Retrieval → **Consumption**) and
the claim that the QA gap lives in the Consumption layer:
`compass_artifact_wf-ad108fcc…md` and `2026-05-24-true-memory-comparison.md`.

### 1.3 Two comparability traps the survey already documented

These constrain *how* we publish, not just *what* we run
(`2026-05-24-memory-system-benchmarks.md` §3):

1. **Answer-model variation is ~24pp.** Same benchmark, GPT-4.1 vs
   GPT-4o-mini, is a 24-point swing. "No apples-to-apples comparison exists
   unless the answer model is held constant." → we must publish a fixed
   reference answer model.
2. **Mem0 has two numbers in circulation** (platform-v3 92.5% vs OSS
   61.4–66.88% LoCoMo) — the platform-vs-OSS distinction is "almost never
   made explicit." → any table we publish must label provider version.
3. **LoCoMo cross-comparisons are unreliable** without knowing the exact
   question subset (200 vs 300 vs 1,540) and whether the adversarial
   category was included. → if we run LoCoMo we must pin the subset.

---

## 2. What we have (verified this session)

Mapped onto the benchmarks/metrics above. Numbers read directly from
`baselines/*.json`; corpus is LongMemEval-S / oracle via
`sme/corpora/longmemeval/loader.py`.

### 2.1 Retrieval (R@5) — clean and defensible

| Reading | R@1 | R@5 | R@10 | n | Source file |
|---|---|---|---|---|---|
| `/search` (mempalace-daemon) | 0.853 | **0.927** | 0.927 | 150 (stratified) | `longmemeval_s_strat150_search_2026-05-29.reagg.json` |
| `/search/age-fused` | 0.867 | **0.920** | 0.920 | 150 (stratified) | `longmemeval_s_strat150_age_fused_2026-05-29.reagg.json` |

The age-fused vs /search delta is **neutral** (−0.7pp, within noise at
n=150). This is the first clean R@5 on the techempower-org production
palace, directly comparable to the R@K column in §1.2 (upstream 96.6%,
agentmemory 95.2%, engram-2 99.0%, mcp-memory-service 80.4–86.0%). The
`2026-05-24-memory-system-benchmarks.md` §5 short-term recommendation —
"run LongMemEval-S R@5 through the daemon adapter" — **is now done.** We sit
mid-pack on R@5 (below the ChromaDB-baseline leaders, above
mcp-memory-service), which the survey predicts: upstream's 96.6% is
effectively a ChromaDB baseline, and enabling palace features *lowers* R@5
(rooms −7.2pp, AAAK −12.4pp per §3).

### 2.2 E2E QA — exists, but confounded

| Reading | QA acc | n | Reader | Judge | Source file |
|---|---|---|---|---|---|
| Pass A oracle, /search ctx | **0.522** | 500 | gpt-5.3-chat | gpt-5.3-chat | `reader_sweep_passA_search-default_2026-05-29.json` |
| Pass A oracle, /search ctx | 0.504 | 500 | o4-mini | gpt-5.3-chat | same |
| Pass A oracle, age-fused ctx | 0.466 | 500 | gpt-5.3-chat | gpt-5.3-chat | `reader_sweep_passA_age-fused_2026-05-29.json` |
| Pass A oracle, age-fused ctx | 0.432 | 500 | o4-mini | gpt-5.3-chat | same |
| Pass B, Opus reader (orig judge) | 0.393 | 150 | claude-opus-4-8 | gpt-5.3-chat | `reader_sweep_passB_opus_REJUDGED_2026-05-29.json` |

**This is the crux of the gap.** Pass A is the *oracle* setting — retrieval
is bypassed, the reader is handed the correct sessions — and QA still tops
out at **~0.52**. Published oracle GPT-4o is **87.0%**
(`2026-05-24-memory-system-benchmarks.md` §1). A 35-point oracle gap is not
a retrieval failure and not a capability failure; it is a **reader + judge
confound**: we used gpt-5.3-chat as *both* reader and judge (and o4-mini /
Opus as alternate readers), none of which is the LongMemEval-canonical
`gpt-4o-2024-08-06` judge with type-specific prompts
(`related_work/longmemeval.md` "Scorer"). The Pass B Opus rejudge confirms
the judge is the swing factor — orig-judge labels move wildly by category
(single-session-user 0.84 vs single-session-preference 0.0), which is the
signature of a judge-prompt mismatch, not a reader deficiency. **None of
these QA numbers are publishable against the leaderboards** — they're
internal diagnostics establishing the reader/judge sensitivity.

### 2.3 Daemon internals (no field analogue — these differentiate us)

- **Cross-encoder rerank** infra shipped (`ms-marco-MiniLM-L-6-v2`,
  off-by-default), A/B deferred until daemon capacity frees
  (`2026-05-28-cross-encoder-rerank.md`). Prediction: 1–3pp R@5 lift max,
  per True Memory's 56-config ablation.
- **RRF-vs-hybrid A/B**, **multi-encoder RRF** (`2026-05-15-multi-encoder-rrf.md`),
  **isotonic calibration**, **chunking ablation** (`2026-05-06-chunking-strategy-ablation.md`).
- **Cat 9 / The Handshake** invocation data: RLM 46.67% vs Familiar 78.33%
  on jp-realm — the "works in theory, fails in practice" failure mode no
  leaderboard captures (`2026-05-24-true-memory-comparison.md`; both
  compass artifacts).

These are SME-unique. No competing system publishes ingestion-integrity
(Cat 4), gap-detection (Cat 5), ontology-coherence (Cat 8), or
harness-invocation (Cat 9) scores — confirmed against the five-ability
LongMemEval taxonomy and LoCoMo's five QA categories
(`related_work/longmemeval.md` "What SME measures that LongMemEval doesn't";
`related_work/locomo-and-memorybench.md` mapping table).

---

## 3. The gap — concretely, what's missing for a head-to-head

### 3.1 Metric mismatch (the #1 blocker)

We report R@5; the leaderboards report E2E QA. Our only QA numbers use a
non-canonical judge and produce an implausible 35pp oracle deficit. **We
cannot place a number on the LongMemEval QA leaderboard today.** The
`2026-05-24-memory-system-benchmarks.md` §4 names this exactly: "E2E QA
scoring pipeline … This is the critical gap." The cross-validation harness
(`scripts/cross_validate_longmemeval.py`) and the design
(`cross_validation_2026.md` §2) are 80% specified but the canonical
GPT-4o-judge wrapper (`sme/eval/longmemeval_judge.py`) is **not yet wired**
— it's the "concrete next-PR scope" in `cross_validation_2026.md`, still
pending.

### 3.2 Datasets we haven't run

- **LoCoMo** — *not wired.* Only LongMemEval has a loader
  (`cross_validation_2026.md` Status: "Currently only LongMemEval is
  wired"). Without it we can't touch the EverOS 93.05% / True Memory 93.0% /
  Mem0 92.5% / Hindsight 89.61% LoCoMo column. Loader is "Option B" in
  `related_work/locomo-and-memorybench.md` Part 2; most of the work overlaps
  the MemoryBench provider ingestion path.
- **BEAM** — *not wired.* Production-scale (100K–10M tokens); only 4 systems
  publish results. High differentiation, but the survey rates it long-term
  (`2026-05-24-memory-system-benchmarks.md` §4).
- **ConvoMem** — *not wired.* Available via MemoryBench.
- **LongMemEval-M** (~1.5M tokens) — *not run* and "rarely used due to
  context limits" (`2026-05-24-memory-system-benchmarks.md` §1); skip.

### 3.3 Competitor numbers — do we need to run them?

**Mostly no — the survey already cites published numbers we can compare
against.** `2026-05-24-memory-system-benchmarks.md` is a fully-sourced
competitor table (every cell has a citation + verification level). For a
"vs the field" table we can cite OMEGA 95.4%, Mem0 94.4%, Hindsight 91.4%,
True Memory 87.8%, Zep 71.2%, etc. directly. **What we'd need to *run*
ourselves** is only:

1. **Our own QA number** under the canonical judge + fixed answer model (so
   our row is real, not R@5-masquerading-as-QA).
2. **Optionally, 2–3 competitors on identical infrastructure** (OMEGA,
   Hindsight, Mem0-OSS) if we want the *first independent multi-system
   benchmark* rather than a cite-the-leaderboard table — this is the
   "would differentiate SME" tier (`2026-05-24-memory-system-benchmarks.md`
   §4: adapters for competing systems), and the §1.2 verification-level
   caveat (most competitor numbers are self-reported) is the reason it has
   real value.

### 3.4 Comparability hygiene we must enforce

Per §1.3: publish (a) a **fixed reference answer model**, (b) the **judge
model + version**, (c) **dataset split + question count**, (d) **provider
version** for any competitor, (e) whether **adversarial/abstention** items
were included. The survey's long-term recommendation is literally "build a
unified leaderboard with mandatory disclosure of metric type, answer model,
judge model, dataset split, question count" — that disclosure discipline is
the deliverable, and we already have the template.

---

## 4. How far to go — prioritized checklist

Ordered by effort/impact. "Done", "scoped", "open" flag where the research
docs already have the answer vs where it's open work.

### Tier 0 — already done (this session)
- [x] LongMemEval-S **R@5** on the daemon, `/search` and `/search/age-fused`,
  n=150 stratified. (`baselines/longmemeval_s_strat150_*.reagg.json`) —
  satisfies `2026-05-24-memory-system-benchmarks.md` §5 short-term #1–2.
- [x] Reader/judge **sensitivity sweep** establishing that QA numbers are
  judge-confounded (Pass A/B). Diagnostic, not publishable.
- [x] The **landscape survey** — competitor numbers + metric definitions +
  comparability traps fully sourced. (`2026-05-24-memory-system-benchmarks.md`)

### Tier 1 — unblocks the *first* defensible QA row (highest impact)
- [ ] **Wire the canonical LongMemEval GPT-4o judge** with type-specific
  prompts (`sme/eval/longmemeval_judge.py`, the pending next-PR in
  `cross_validation_2026.md` §2). *Scoped — ~half a day per that doc;
  blocked only on `OPENAI_API_KEY` access to gpt-4o-2024-08-06.*
- [ ] **Fix a reference answer model** (held constant) and re-run Pass A
  oracle + full-haystack QA. Re-running with the canonical judge should
  collapse the 35pp oracle anomaly toward the published 87% oracle ceiling;
  if it doesn't, that itself is the finding. *Scoped.*
- [ ] **Publish R@5 + QA side-by-side** with full disclosure metadata
  (§3.4). This is the "show the R@K→QA gap on the same runs" differentiator
  named in `2026-05-24-memory-system-benchmarks.md` §4 medium-term #1.

### Tier 2 — second benchmark, so we're not a one-dataset entrant
- [ ] **LoCoMo loader** (`sme/corpora/locomo/loader.py`) — pin the question
  subset + adversarial inclusion (§1.3, §3.4). *Open; ~1 day. Overlaps the
  MemoryBench-provider ingestion path (`locomo-and-memorybench.md` Option
  C).* Unlocks the LoCoMo column (EverOS/TM/Mem0/Hindsight).
- [ ] Run R@5 + QA on LoCoMo through the daemon + Familiar adapters.

### Tier 3 — independent multi-system benchmark (research-grade, highest novelty)
- [ ] **Adapters for 2–3 competitors** on identical corpus/model/judge —
  priority OMEGA (local, pip-installable), Hindsight (MCP), Mem0-OSS
  (`2026-05-24-memory-system-benchmarks.md` §4 / §5 medium-term #2). *Open;
  the first independent head-to-head on one harness.*
- [ ] **Register SME as a MemoryBench provider** (TS shim → Python adapters)
  for third-party calibration on LoCoMo/LongMemEval alongside
  Mem0/Zep/Supermemory (`cross_validation_2026.md` §3;
  `locomo-and-memorybench.md` Option A). *Scoped — ~2 days, separate
  sub-repo.*
- [ ] Run SME-unique cats (4/5/8/9) on competitors — the first published
  structural-quality comparison (`2026-05-24-memory-system-benchmarks.md`
  §4 medium-term #3). No other system has these scores.

### Tier 4 — long-term differentiators
- [ ] **BEAM loader** (production-scale; only 4 systems publish).
- [ ] **Cross-encoder rerank A/B** once daemon capacity frees
  (`2026-05-28-cross-encoder-rerank.md` — harness exists, run deferred).
- [ ] **Cat 9 across MCP systems** — quantify invocation-discipline as a
  cross-system axis (`true-memory-comparison.md` "What True Memory is
  missing").

---

## 5. Honest verdict

**How far to go for a defensible "mempalace vs the field" table?**

- The **survey is done** — we know the field, the metrics, and the traps
  cold (`2026-05-24-memory-system-benchmarks.md` is publication-quality).
- A **cite-the-leaderboard comparison table** (our R@5 + our canonical-judge
  QA, alongside competitors' published QA with full disclosure) is **Tier 1
  only — roughly 1–2 days of un-blocked work** (wire the GPT-4o judge, fix a
  reference answer model, re-run, publish with metadata). The harness is
  80% built; the missing piece is the judge wrapper + an API key.
- A **two-benchmark table** (add LoCoMo) is **+~1 day** (Tier 2).
- An **independent multi-system head-to-head** (we run competitors
  ourselves) is the **research-grade contribution and the real moat**
  (Tier 3, ~1–2 weeks) — and per §1.2 it's worth it, because most
  competitor numbers are self-reported and unverified.

We are **not far** from a credible table; we are **one judge-wrapper PR + a
reader-model decision** away from our first honest QA row. The thing that
makes us *unique* — structural cats (4/5/8) and the Cat 9 invocation gap —
is already measured and has **no competitor analogue**, so even the minimal
table ships something no other entrant has.

---

## Sources cited

mempalace fork `docs/research/` (local clone `~/Projects/memorypalace`):
- `2026-05-24-memory-system-benchmarks.md` (landscape survey)
- `2026-05-24-true-memory-comparison.md`
- `compass_artifact_wf-28bac4e8-71d9-4175-837a-d4ad563aec8d_text_markdown.md`
- `compass_artifact_wf-ad108fcc-3960-4eab-ad5d-234bf365b2f4_text_markdown.md`
- `2026-05-28-cross-encoder-rerank.md`
- `2026-05-15-multi-encoder-rrf.md`, `2026-05-06-chunking-strategy-ablation.md`
- `convergent-findings-kostadis-comparison.md`

this repo:
- `docs/related_work/longmemeval.md`
- `docs/related_work/locomo-and-memorybench.md`
- `docs/cross_validation_2026.md`
- `baselines/longmemeval_s_strat150_search_2026-05-29.reagg.json`
- `baselines/longmemeval_s_strat150_age_fused_2026-05-29.reagg.json`
- `baselines/reader_sweep_passA_search-default_2026-05-29.json`
- `baselines/reader_sweep_passA_age-fused_2026-05-29.json`
- `baselines/reader_sweep_passB_opus_REJUDGED_2026-05-29.json`
