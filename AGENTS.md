# AGENTS.md — multipass-structural-memory-eval (SME)

Project context for AI code-review agents (Gemini Code Assist, Claude PR
Review, Copilot, etc.) reviewing changes to this repository.

## What this is

SME is a diagnostic framework for memory systems — RAG, knowledge graphs,
personal knowledge bases, conversational memory. It tests what a memory
system knows about its own structure, not just whether it can retrieve.
Eight to ten adapters (`flat-baseline`, `mempalace`, `mempalace-daemon`,
`familiar`, `rlm`, `ladybugdb`, `full-context`, `karpathy-compiled`,
`postgres`, `postgres-age`) plug into a nine-category test menu; categories
1–8 measure graph structure and retrieval, Category 9 measures harness
integration. Upstream is `M0nkeyFl0wer/multipass-structural-memory-eval`;
the fork at `techempower-org/multipass-structural-memory-eval` carries
extensive measurement work (Step 1/2/3 RLM benchmarks, LongMemEval
substrate-floor parity, adaptmem FT-300 reproduction, AGE write-through
spike, encoder-conditional chunking ablation).

## Design principles

These guide review priorities and PR decisions.

- **Diagnostic posture, not benchmark.** The defensible findings are
  before/after deltas under identical conditions and within-system
  A/B/C ablations — absolute recall numbers inherit a substring-on-
  filename matcher with known biases. PRs that introduce new
  measurements should preserve a clean baseline-vs-condition shape
  rather than chasing absolute-number leaderboards.
- **Adapter contract is a hard interface.** Every adapter implements
  `SMEAdapter` from `sme/adapters/base.py` with the same `query` /
  `get_graph_snapshot` / `close` shape. New adapters MUST conform —
  the harness assumes the contract holds. See PR #7 (RlmAdapter) for
  the canonical adapter-shape example.
- **Stochastic LLM evals need n ≥ 25.** Partial readings at n<25 swing
  significantly on the same underlying distribution. PRs reporting new
  recall numbers below that threshold should label them as partial-N
  and not draw conclusions. See `feedback_stochastic_partial_n.md` in
  the project memory for the empirical anchor.
- **Cross-repo references must be fully qualified.** Naked `#NNN` in
  comments and docs auto-links to the comment's home repo, which is
  wrong across the upstream/fork pair (`MemPalace/mempalace` vs
  `techempower-org/mempalace`, etc.). Use `owner/repo#NNN`. Escape
  numbered prose bullets: `point (3)`, not `point #3`.
- **Don't kill in-flight runs to "preserve" data.** Incremental JSON
  snapshots survive the kill; the cost of restarting a 26-hour chain
  to recover a half-hour of saved work is asymmetric. If a run looks
  off, `cp` the artifact to a side path and let it finish.

## Style + structure

- **Adapters** live in `sme/adapters/`. Each adapter is one file
  named after the system it adapts. Constructor args are kwargs-
  only past `db_path` so CLI plumbing in `sme/cli.py::_load_adapter_from_args`
  doesn't break when adapters add params.
- **Corpora** live in `sme/corpora/<corpus_name>/questions.yaml`
  with a single `version:` string + `questions:` list of
  `{id, text, expected_sources, min_hops}`. Per-question vault
  directories under `sme/corpora/<name>/vault/<question_id>/`.
- **Bench scripts** live in `scripts/` and write JSON results to
  `baselines/<corpus>_<adapter>_<date>_<config>.json`. Per-question
  incremental snapshots so timeouts mid-run don't lose data.
- **Writeups** live in `docs/benchmarks/<date>-<topic>.md` with a
  setup table, results table, and methodology caveats section.
- **Tests** run via `pytest tests/` from the repo's venv (`venv/`).
  Skip tests that need an external service (the daemon at
  `disks.jphe.in:8085`) under `@pytest.mark.integration`.

## What's special on this fork

The `techempower-org` fork has carried independent measurement work
since 2026-05-13, primarily on the `feat/rlm-adapter` branch. Recently
landed:

- **RlmAdapter** (`sme/adapters/rlm_adapter.py`) — RLM-as-orchestrator
  using palace-daemon HTTP API for retrieval. Forced/grounded
  invocation modes via `invocation_mode` constructor arg.
- **postgres adapter** (`sme/adapters/postgres_ingest.py`) — per-question
  ingest + query against `mempalace.backends.postgres.PostgresCollection`.
  Used in LongMemEval substrate-floor parity bench.
- **postgres-age adapter** (`sme/adapters/postgres_age_ingest.py`) —
  vector + AGE write-through fusion. Used in the +9pp graph spike.
- **n=200 git-derived probe corpus** (`sme/corpora/mempalace_git_probes_v2/`)
  — deterministic from `techempower-org/mempalace` git log;
  file-shaped `expected_sources` (vs jp-realm-v0.1's substring-shaped
  ones).
- **Step 2/3 RLM benchmark suite** — gemma4:e4b + qwen3.5:4b across
  vanilla/forced/grounded × n=5/n=20 on jp-realm-v0.1 and
  mempalace_git_probes_v2.
- **AGE write-through spike + 6-phase integration plan** (2026-05-17;
  see `docs/benchmarks/2026-05-17-age-write-through-spike.md`) —
  graph signal adds +9pp R@5 over vector on n=200 git-probes.

## Review priorities (high → low)

1. Adapter contract violations (signature mismatches, missing methods,
   ignoring `read_only`, `n_results` semantics)
2. Bench-scoring correctness — `recall_at_k` / `mean_recall` math, hop
   counting, source-matching shape consistency
3. Partial-N claims drawn without the `feedback_stochastic_partial_n`
   caveat (n<25 conclusions are a regression)
4. Cross-repo reference disambiguation in comments/docs (`#NNN` →
   `owner/repo#NNN`)
5. Public API changes on CLI subcommands (`retrieve`, `analyze`, `cat*`)
6. Test coverage on new adapters + new scoring code

## Out of scope for review

- Style nits in `docs/benchmarks/*.md` writeups (these are research
  artifacts, not source code)
- Coverage on `baselines/*.json` files (data, not code)
- Performance on bench scripts (they run overnight; per-question
  latency is bounded by the LLM, not by SME's bookkeeping)
- Existing inconsistencies in older corpora's question shape
  (jp-realm-v0.1 uses substring-shaped `expected_sources`;
  mempalace_git_probes_v2 uses file-shaped) — this is by design
