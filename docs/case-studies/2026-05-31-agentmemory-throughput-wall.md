# Case Study 3 — The agentmemory throughput wall (cost-wall taxonomy)

**SME applied to its own provisioning plan.** The campaign's scoping
classified `agentmemory` as cheap-to-bench on the strength of having *no
write-time LLM*. The real bench overturned that on a second axis the
scoping hadn't weighed — **ingest throughput** — and the correction
refined the whole cost-wall taxonomy.

- **Category:** Cost-wall taxonomy (benchability, not a Cat 1–9 score)
- **System:** agentmemory (REST iii-engine ingest)
- **Scale target:** LongMemEval-S strat150 (~70 chunked `observe` calls ×
  ~48 distractor sessions per question)
- **Status:** Adapter built + verified on small loads; on-harness bench
  deferred (throughput-walled) — a flag-don't-thrash call.

---

## The finding

The #234 scoping used a single axis — **write-time extraction cost** —
to decide whether a system could be benched cheaply, and put agentmemory
in the cheap tier: embedding/BM25-only, LLM-compress off by default, no
write-time LLM. By that axis it should bench like `flat` or `ai-memory`.

The real bench measured the actual ingest path and found:

> **~0.15–0.3 observations/sec, wedging as the index grows.**
> Extrapolated over strat150's per-question chunked `observe` calls ×
> distractor sessions → **~15 hours** for a full run.

So agentmemory is throughput-walled **despite being LLM-free at write** —
it lands alongside Mem0-OSS (~18h) and Hindsight (~150h), but for a
different reason.

## The fix (the taxonomy correction)

The actionable change was to the *framework's own model* of the field:

> **"No write-time LLM" is necessary but not sufficient for cheap
> benching; per-item ingest throughput is a second, independent wall.**

The cost-wall taxonomy gained a second axis (synthesis §6 / §6.1):

| Class | Definition | Marginal cost | Examples |
|---|---|---|---|
| Verbatim / retrieval-only, cheap | no write LLM **and** fast bulk ingest | **$0** | flat, postgres_ingest, mempalace, **ai-memory (benched 0.920)** |
| Throughput-walled | per-item ingest too slow — *whether or not* there's a write LLM | hours | Mem0-OSS (~18h), Hindsight (~150h), **agentmemory (~15h, no write LLM)** |
| Un-benchable locally | hosted / paper-only / framework-coupled | n/a | Mem0-platform, Mastra, True Memory, … |

The downstream fix for agentmemory itself: a **bulk-ingest path** would
move it back to the cheap tier and let the bench finish. Until then its
published 95.2% R@5 stays in the published-field column,
**unverified-on-harness (throughput)** — distinct from ai-memory's 0.920,
which *is* on-harness.

## The re-run (the contrast that proves the axis)

The cleanest re-verification is the side-by-side with `ai-memory`, run in
the same wave:

| System | Write LLM? | Ingest throughput | Benchable cheaply? | Result |
|---|---|---|---|---|
| ai-memory | no | fast (FTS5 + MiniLM) | **yes** | **benched R@5 0.920**, n=150, 0 errors |
| agentmemory | no | ~0.15–0.3 obs/s | **no (~15h)** | deferred, published 95.2% unverified |

Same "no write LLM" property; opposite benchability — *because* the
second axis differs. That is the axis made visible.

## The lesson

**A scoping estimate is a hypothesis; the measured ingest path is the
test.** SME exists to surface measured-vs-claimed gaps — and here it
applied that discipline to *its own provisioning plan*, catching a
scoping error before it became a fabricated on-harness number. The honest
move when a bench is throughput-walled is to **flag it, record the
field-reported number with its deferral reason, and never invent an
on-harness score** — exactly what the `verified-qa-deferred` /
`unverified-on-harness` honesty levels exist to express.

**Artifacts:** synthesis §6 / §6.1, `baselines/longmemeval_s_strat150_ai_memory_2026-05-31.json`
(the ai-memory contrast), #234/#247 field-bench wave.
