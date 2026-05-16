# multipass-structural-memory-eval

![Multi Pass. Mem Palace. A mockup of Leeloo Dallas's MULTI PASS ID card from The Fifth Element, stamped MEM PALACE where the issuing authority would normally appear, with the caption FREE PASS UNLOCK YOUR CREATIVITY ELIMINATES THE NEED TO REMEMBER EVERYTHING YOU'VE EVER TYPED.](docs/assets/issue_101_multipass_header.png)

**A diagnostic framework for memory systems** — RAG, knowledge graphs,
personal knowledge bases, conversational memory — that tests what the
system knows about its own structure, not just whether it can retrieve
memories.

> *"Multipass!"* — Leeloo, *The Fifth Element*. The name is a nod to a
> joke from an earlier MemPalace issue thread (visual reference above),
> and also to what the framework actually does: **multiple passes**
> over every memory system under test, across multiple corpus shapes
> and multiple retrieval conditions (A / B / C), so brittle default
> behaviours that hide on any single pass become visible when the
> readings are compared side by side.

## Contents

[What this is](#what-this-is) · [Status](#status) · [Install](#install) ·
[Next steps](#next-steps) · [Adapters](#adapters)

## What this is

See the [nine-category menu](docs/ideas.md#who-should-run-which-categories)
for what each test measures and which to run for your setup.

Standard memory benchmarks (LongMemEval, LoCoMo, MINE, GraphRAG-Bench,
BEAM) ask "can you find a memory?" That's necessary but not
sufficient. A filing cabinet can find a memory. The question is what
the **structure** of a memory system gives you beyond retrieval — and
whether your specific build, under your specific harness, with your
specific model, actually uses it.

SME defines a nine-category test menu. Categories 1–8 measure graph
structure and offline retrieval. Category 9 (The Handshake) measures
harness integration — whether the model actually reaches the memory
when it runs in production. Cats 1–8 are where every published
benchmark stops; Cat 9 is the gap every deployment engineer runs into.

Each category has a `Cat N` identifier (for code) and a descriptive
"palace-nod" name (The Lookup, The Stairway, The Blueprint, The
Handshake, and so on) so readable output doesn't require a lookup
table.

## Status

**Beta-level instrumentation, actively evolving.** Eight adapters
(`flat-baseline`, `mempalace`, `mempalace-daemon`, `familiar`, `rlm`,
`ladybugdb`, `full-context`, `karpathy-compiled`), nine CLI commands
(`retrieve`, `analyze`, `cat8`, `cat2c`, `cat4`, `cat5`, `check`,
`cat9`, `compile-wiki`), Cat 4 and Cat 5 partially implemented, and
a specification for the remaining categories. Diagnostic posture,
not benchmark — the defensible findings are before/after deltas
under identical conditions and within-system A/B/C ablations.
Absolute recall numbers inherit a substring-on-filename matcher
with known biases. See the [spec](docs/sme_spec_v8.md) and the
[onboarding guide](docs/ideas.md) for the full honest-limitations
discussion.

## Install

```bash
pip install -e .
# Optional extras:
pip install -e ".[topology]"   # Ripser + python-louvain (for gap detection)
pip install -e ".[ladybugdb]"  # LadybugDB adapter
pip install -e ".[dev]"        # pytest, ruff
```

Installs as the Python package `sme-eval` with CLI entrypoint
`sme-eval`. The GitHub repo is `multipass-structural-memory-eval`;
the acronym **SME** (Structural Memory Evaluation) is used throughout
the documentation and code.

**Quick start:** run your first diagnostic in 5 minutes with the
[onboarding guide](docs/ideas.md#quickstart-your-first-diagnostic-run).
Need the spec? Start at [docs/sme_spec_v8.md](docs/sme_spec_v8.md).

## Next steps

- **[`docs/ideas.md`](docs/ideas.md) — onboarding guide.** Start here
  if you want to run SME against your own memory system. Covers the
  nine-category menu, how to write an adapter for your backend, how
  to write a corpus from your own content, how to run the implemented
  categories, and how to read what comes out the other end. This is
  also where the methodology framing lives — why A/B/C isolation
  matters, why multi-corpus testing is load-bearing, and why "the
  delta is the product, the levels are decoration."

- **[`docs/sme_spec_v8.md`](docs/sme_spec_v8.md) — full specification.**
  Precise category-by-category definitions, metric formulas, adapter
  interface contract, topology layer details, and the Cat 9 (The
  Handshake) harness-integration spec. Reference material — read the
  onboarding guide first if you want to get a test run going.

- **[`docs/cross_validation_2026.md`](docs/cross_validation_2026.md) —
  current work.** Cross-validation of SME categories against
  LongMemEval / MemoryBench, Karpathy-condition D baselines (full-
  corpus-in-context), and first readings from the live benchmark
  harness. Active development; this is where near-term SME findings
  land.

- **[`docs/industry_standards_integration.md`](docs/industry_standards_integration.md)
  — integration audit.** Survey of where SME rolls its own vs. where
  battle-tested standards exist (SHACL, PROV-O, OpenLineage, B-Cubed,
  Ripser). Constitutional principle: SME stays lightweight and locally
  runnable — no server hosting required.

## Adapters

SME ships adapters for several memory systems. Each adapter teaches
the framework to speak the wire protocol of a specific system so the
same eval questions can run across multiple backends. Adapters live in
`sme/adapters/` and implement the `SMEAdapter` ABC.

### `mempalace-daemon` — by [jphein](https://github.com/jphein)

`sme/adapters/mempalace_daemon.py` talks to a running
[`palace-daemon`](https://github.com/jphein/palace-daemon) over HTTP —
by [`jphein`](https://github.com/jphein). No filesystem access, no
ChromaDB import, no shared-process constraint with the daemon. Use
this adapter when MemPalace is fronted by the daemon (the daemon is
the single writer to the palace) — the existing `mempalace` adapter
is still correct for single-process upstream installs without the
daemon.

**Wired endpoints:**

- `query()` → `GET /search?q=…&limit=…` with `X-API-Key`. Daemon-side
  `warnings` (e.g. degraded vector index) are surfaced into
  `QueryResult.error` as `WARN: …` so Cat 9 scoring can distinguish
  flagged retrieval from clean retrieval. *The adapter still passes a
  `kind=` query parameter for backward compat, but as of mempalace fork
  v1.7.1 ([`7ba28dc`](https://github.com/techempower-org/mempalace/commit/7ba28dc))
  the daemon silently ignores it — Stop-hook checkpoints moved to a
  dedicated collection, making the binary `kind=content`/`kind=all`
  filter inert. The vestigial CLI flag will be removed pending the
  scope-filter design call in
  [techempower-org/mempalace#76](https://github.com/techempower-org/mempalace/issues/76).*
- `get_graph_snapshot()` → tries `GET /graph` first (palace-daemon
  ≥1.6.0); on 404, falls back to walking `mempalace_list_wings`,
  `mempalace_list_rooms` per wing, and `mempalace_list_tunnels` via
  `POST /mcp`. The MCP fallback is slower (~30s on a 151K-drawer
  palace) but works against any palace-daemon version. Note: passive
  (room-shared-across-wings) tunnels are under-reported on the MCP
  fallback path until
  [techempower-org/mempalace#75](https://github.com/techempower-org/mempalace/issues/75)
  lands — `/graph` already returns the correct count.

**Auth resolution:** explicit `--api-url` / `--api-key` flags →
`~/.config/palace-daemon/env` (`PALACE_DAEMON_URL`, `PALACE_API_KEY`)
→ process environment.

**Invocation:**

```bash
# With explicit daemon URL
sme-eval retrieve --adapter mempalace-daemon \
    --api-url http://your-daemon:8085 \
    --questions corpus.yaml \
    --json out.json

# Or, if ~/.config/palace-daemon/env is populated, no flags needed
sme-eval retrieve --adapter mempalace-daemon --questions corpus.yaml
```

The same `--api-url` / `--api-key` flags work on the `cat4`, `cat5`,
and `check` subcommands. (The `--kind` flag still parses for backward
compat but is operationally a no-op against current daemons — see the
endpoint note above.)

**Why this matters:** the engram-2 critique ("0.984 R@5 but 17% E2E
QA accuracy") is about the integration-under-production-model slice
that Cat 9 measures. Running SME's `retrieve` through the daemon
surfaces exactly the kind of gap that critique describes — the
adapter's WARN-soft-error treatment means the framework records
"retrieval ran but the daemon flagged it as degraded" as a first-
class signal, not as a hard failure that hides the issue.

#### Why the existing adapter still has a use

For users running upstream MemPalace without palace-daemon (the
default install pattern), the existing `mempalace` adapter is
correct — single process, no daemon, direct ChromaDB access is
fine. The daemon adapter is *additive*, for users who've adopted
palace-daemon's single-writer architecture.

> **Backend note.** The `mempalace-daemon` adapter targets palace-daemon
> regardless of the storage backend underneath. JP's production palace
> migrated from ChromaDB to **postgres + pgvector + Apache AGE** in May
> 2026; the daemon's HTTP surface is unchanged, so the adapter works
> against either era without modification. The existing `mempalace`
> adapter (direct, no daemon) still assumes ChromaDB and would need
> updates to target a postgres-backed palace directly.

### familiar — by [jphein](https://github.com/jphein)

[`familiar.realm.watch`](https://github.com/jphein/familiar.realm.watch)
is a retrieval pipeline that wraps palace-daemon with reranking,
temporal decay, extractive compression, and grounding directives.
`[jphein](https://github.com/jphein)` built it; `sme/adapters/familiar.py`
lets SME measure its full end-to-end contribution on top of the raw
daemon. The sibling `mempalace-daemon` adapter measures palace alone —
running both on the same corpus shows what the pipeline layer adds.

**Wired endpoints:**

- `query()` → `POST /api/familiar/eval` with body
  `{query, limit, kind, mock}`. Familiar's eval endpoint already
  returns SME-shape `{answer, context_string, retrieved_entities,
  retrieved_edges, error, warnings, available_in_scope}` natively
  (it was designed against the SME contract), so the adapter is
  mostly deserialization with the same WARN: error-prefix
  translation as `mempalace-daemon`.
- `get_graph_snapshot()` → `GET /api/familiar/graph`. Familiar
  proxies palace-daemon's `/graph` with a 5-minute server-side cache;
  payload mapping reuses `sme/adapters/_graph_mapping.py` shared with
  `mempalace-daemon`.
- `get_harness_manifest()` → forward-compat for Cat 9. Returns
  `[ToolCall, MCPResource]` once `sme.harness` ships; `[]` until then.

**Determinism:** `--mock` (default) skips LLM inference so Cat 1
substring scoring is reproducible across runs. Use `--no-mock` to
include the model output in the per-question record (intended for
future Cat 9 work).

**Invocation:**

```bash
# Default: --mock for Cat 1 determinism
sme-eval retrieve --adapter familiar     --api-url https://familiar.jphe.in     --questions corpus.yaml     --json familiar.json

# Compare against the same palace via the daemon adapter
sme-eval retrieve --adapter mempalace-daemon     --api-url http://your-daemon:8085     --questions corpus.yaml     --json daemon.json

# The score delta = what familiar's v0.2 pipeline is worth
```

The `--api-url`, `--mock`/`--no-mock`, and `--familiar-timeout` flags
work on `cat4`, `cat5`, `check`, and `retrieve` subcommands.

### Shipped: `rlm` adapter

`sme/adapters/rlm_adapter.py` treats [RLM](https://github.com/jphein/rlm)
(a fork of [alexzhang13/rlm](https://github.com/alexzhang13/rlm)) as
the **read-side orchestrator** rather than a deterministic retrieval
pipeline. The LLM itself decides when to call `mempalace_search`,
with what queries, and how to compose results. `familiar`'s pipeline
is the *baseline* this adapter is benchmarked against, not the thing
it replaces.

**Design:** RLM gets `mempalace_search` registered as a `custom_tools`
callable. The adapter wraps that callable to capture every search
result into a per-query buffer; after `rlm.completion()` returns, the
buffer's contents become `context_string` (in tool-call order) and
`retrieved_entities` (one Entity per drawer). Same scoring contract
as every other adapter.

**Endpoint override:** `RLM_BASE_URL` / `RLM_MODEL` / `RLM_API_KEY`
env vars point the openai backend at any compatible endpoint —
local llama.cpp, hosted Llama 3.3 70B, anything OpenAI-shaped —
without touching the cloud-chat-assistant config-file fallback path.

**Live readings on `jp-realm-v0.1` (30 questions).**

*April 2026 — ChromaDB-era palace backend:*

| Run | Mean recall | Tool-call distribution |
|---|---|---|
| rlm + Qwen 2.5 7B Q5_K_M | 46.67% | 25/30 zero-call, 2/30 used tool |
| rlm + Llama 3.3 70B | 46.67% | 22/30 zero-call, 8/30 used tool |
| familiar v0.3.9 (deterministic) | 78.33% | n/a |

Both RLM runs land at the same aggregate recall despite a 4×
difference in tool-invocation rate — they ceiling at the
orchestrator's willingness to invoke the tool, not at retrieval
quality. This is the data behind the [9a invocation-rate
issue](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/3)
filed upstream. See the [onboarding
guide](docs/ideas.md#rlmadapter--research-scaffold-2026-04-26) for
the full discussion and the per-question deltas.

*May 2026 — postgres+pgvector+AGE palace backend, Step 1 retrieval-breadth probe (complete):*

| Adapter | n=5 | n=20 | Δ (breadth) |
|---|---|---|---|
| `mempalace-daemon` (retrieval-only) | 73.3% | **81.7%** | **+8.4pp** |
| `familiar` (full pipeline, gemma3:4b) | **88.3%** | 86.7% | -1.7pp |
| `rlm` + gemma4:e4b | 41.7% | 41.7% | **0.0pp** |
| `rlm` + qwen3.5:4b | 71.7% | 75.0% | +3.3pp |

**Findings:**

- **Backend migration lift**: daemon n=5 went from 70.0% (April,
  ChromaDB-era) → 73.3% (May, postgres+pgvector+AGE). +3.3pp from
  the hybrid-search substrate alone.
- **Retrieval breadth helps at the substrate layer** (+8.4pp daemon
  n=5 → n=20) but **saturates at every layer above it**: familiar's
  rerank already finds the right drawers from top-5; both RLM
  orchestrators also saturate near n=5.
- **Base model choice dominates orchestrator behavior at fixed
  parameter count**: gemma4:e4b and qwen3.5:4b are both 4B, same
  wrapper code, same backend, same corpus — yet **30pp recall gap**
  between them. Tool-use-training (Qwen 3.5's RL on tool
  trajectories) shows up as both higher recall AND higher hit-rate
  on Cat 9a-shaped tasks. Empirical validation of the published Tau2
  benchmark gap (~37.7 points) on an independent corpus.
- **gemma4-RLM saturates 31.6pp BELOW the daemon retrieval-only
  floor** — the orchestrator is actively dropping signal the
  substrate provides for free. By contrast, **qwen3.5-RLM saturates
  near the daemon floor** (71-75% vs 73-82%) — first orchestrator
  we've measured that doesn't regress retrieval.
- **Hit-rate (any-match) is the cleanest summary**: gemma4 17-18/30
  (~57%), qwen3.5 27-28/30 (~92%), daemon 27/30 (~90%), familiar
  29-30/30 (~97%).

See [upstream comment thread on #3](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/3#issuecomment-4457514474)
for the discriminating-experiment context, and the [Step 1 follow-up
comment](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/3)
(to be posted alongside this commit).

**Invocation:**

```bash
RLM_BASE_URL=https://your-endpoint RLM_MODEL=llama-3.3-70b RLM_API_KEY=...     PALACE_DAEMON_URL=http://your-daemon:8085 PALACE_API_KEY=...     sme-eval retrieve --adapter rlm     --questions sme/corpora/jp_realm_v0_1/questions.yaml     --json baselines/rlm_$(date +%Y%m%d).json
```

## Upstream conversation (2026-05)

Active threads across the SME ↔ mempalace fork boundary that this
fork is currently driving:

**On [M0nkeyFl0wer/multipass-structural-memory-eval](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval) (SME upstream):**

- PR [#7](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/pull/7) — RlmAdapter + Qwen-7B/Llama-70B baselines; cat5 API-arg forwarding fix landed in [`7d081c3`](https://github.com/jphein/multipass-structural-memory-eval/commit/7d081c3); kind-filter integration test xfail'd in [`6184680`](https://github.com/jphein/multipass-structural-memory-eval/commit/6184680).
- Issue [#3](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/3) — Cat 9a invocation rate; discriminating experiment proposed ([RLM-forced / RLM-grounded / RLM-on-familiar](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/3#issuecomment-4457514474)) and n=200 git-derived probe corpus offered ([addendum](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/3#issuecomment-4457757792)) as a domain-relevant alternative to LongMemEval cross-validation.
- Issue [#4](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/4) — Phantom Wall (Cat 8b); proposed [mixed PROV-O / SHACL alignment](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/4#issuecomment-4457514600) with primary-source verification flagged.
- Issue [#8](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/8) — Adapter contract testkit; argued [strict with documented opt-out](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/8#issuecomment-4457514719) based on bugs caught/missed across the three recent adapters.
- Issue [#9](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/9) — MemoryBench cross-validation; recommended [sub-repo bridge with upstream-PR target on the horizon](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/9#issuecomment-4457514825).

**On [techempower-org/mempalace](https://github.com/techempower-org/mempalace) (mempalace fork):**

- Issue [#75](https://github.com/techempower-org/mempalace/issues/75) — `mempalace_list_tunnels` MCP tool surfaces only explicit (agent-created) tunnels; passive (room-shared-across-wings) tunnels are computed in `graph_stats.top_tunnels` but have no direct MCP query path. Proposed `include_passive=True` parameter or sibling tool.
- Issue [#76](https://github.com/techempower-org/mempalace/issues/76) — Design call: bring back a more general scope/collection filter on search now that the palace has multiple stores (drawers, session_recovery, …), given the binary `kind=` filter was retired in [`7ba28dc`](https://github.com/techempower-org/mempalace/commit/7ba28dc) after the checkpoint-collection split made it inert.

## License

MIT. See [`LICENSE`](LICENSE).
