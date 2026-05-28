# Narrow-age paraphrase corpus

Operator note for `sme/corpora/paraphrase_narrow_age/questions.yaml`.

## Why this corpus exists

Aurora's PR #94 (techempower-org/multipass-structural-memory-eval) diagnosed
that the rerank-on vs rerank-off A/B on the original 51-question paraphrase
corpus returns **bit-identical retrieval orderings**. Root cause: that corpus
draws target drawers from a ~2-month window (2026-04-23 .. 2026-05-28). When
familiar's pipeline applies temporal-decay multipliers on top of palace-side
rerank, the per-drawer multipliers span an order of magnitude — the decay sort
overwrites the rerank scores and the two arms produce indistinguishable
orderings.

The fix is to bunch drawer ages tightly so decay multipliers stay roughly
uniform (within ~7% across a 7-day window for the current decay parameters).
Once decay is approximately constant, the rerank score actually decides the
order and the A/B becomes informative again. Same construction trick exposes
modality lifts (vector vs BM25 vs graph) without recency masking them.

This corpus does that: every non-canary target drawer was created between
**2026-05-21** and **2026-05-28**.

## Construction method

1. **Source candidate drawers** via the live palace.
   Used the `mempalace_search` MCP tool with topical queries that surface
   drawers from the last 7 days. Each result includes `created_at`, so
   filtering to the 7-day window is direct. Topical queries used:
   - reranker / FlashRank / MiniLM
   - postgres / pgvector / AGE / knowledge graph
   - familiar / Phi-4 / llama-server / GPU
   - candidate strategy / multi-limit sweep
   - palace-daemon recent PR work

   Search recall for narrow-age drawers is good in May 2026 because activity
   has been concentrated — the AGE backfill (2026-05-25), the FlashRank
   rerank eval (2026-05-27), the slot-picker work (2026-05-28), and the
   multipass candidate-strategy sweep (2026-05-28) all landed within the
   window.

2. **Author paraphrase-stressed questions** that target the surfaced
   drawers without using the drawer's own technical vocabulary. Each
   question carries `expected_drawers` (drawer IDs — any-of match) and/or
   `expected_substrings` (case-insensitive substring match against the
   retrieved `content_snippet`). The dual matcher lets a question still
   score a hit when the specific drawer ID is unstable but the topical
   content surfaces.

3. **Cover all seven shapes** used by the parent corpus
   (`vocab_mismatch`, `topical_mismatch`, `cross_project`, `temporal`,
   `recency`, `contradiction`, `canary`) so per-shape statistics from
   `run_paraphrase_probe.py` are comparable between corpora. The
   distribution skews toward vocab/topical mismatch since those are the
   shapes most sensitive to rerank, but `canary` questions are included to
   detect false-positive retrieval (the corpus should fail closed on
   them).

4. **Verify with the runner.** The YAML must parse via
   `run_paraphrase_probe._load_questions()` and every entry must round-trip
   to the expected shape buckets. Baseline run (`--top-n 5`, mock mode)
   captures the no-HyDE recall starting point.

## How to refresh

The 7-day window slides. Each refresh authors a fresh corpus YAML in a
sibling directory rather than mutating this one — the historical corpus
stays runnable so baselines can be replayed.

1. Pick a new 7-day window (typically the most recent 7 days).
2. Repeat the topical sweep via `mempalace_search` and the `/api/familiar/eval`
   endpoint (`POST` with `{"query": "...", "mock": true}` — the response's
   `retrieved_entities[].id` and the YAML `date:` lines surface drawer
   ages without needing direct palace-daemon access).
3. Replace ~half the questions per refresh; keep the topical
   structure (counts per shape) constant so comparisons remain valid.
4. Re-run the baseline with `run_paraphrase_probe.py` against the new
   YAML and save the JSON to `baselines/probe-results-<date>-narrow-age-baseline.json`.

A practical cadence: weekly refresh keeps the window aligned with current
activity. Refreshing more often than the rerank-eval cadence is wasted
churn; less often risks the window aging out of the active development
slice.

## What a sweep on this corpus tests that the parent corpus cannot

The original `paraphrase_questions.yaml` mixes drawer ages spanning ~2
months. Temporal-decay sort then masks every other downstream scoring
signal. As a result the original corpus is good for measuring:

- end-to-end recall on a wide topic mix
- HyDE on/off lift (HyDE adds a model call, doesn't reorder by decay)
- gross regressions in palace search

…but it is **blind to**:

- rerank lift (decay rescores overwrite rerank scores)
- modality lift (vector vs BM25 vs graph) when the modality difference
  is finer than the decay spread
- score-shape changes in palace-daemon's rerank model swaps

The narrow-age corpus is the diagnostic instrument for those questions.
Concretely:

- Sweep `PALACE_RERANK_ENABLED=true` vs `false` on this corpus and you
  should see Δrecall and ΔMRR move (on the original corpus, both are
  ~zero).
- Sweep `candidate_strategy={vector,bm25,hybrid,graph}` and the
  per-strategy curves should diverge per-shape (vocab_mismatch favors
  vector + rerank, topical_mismatch favors BM25 / graph).
- Swap rerank models (TinyBERT-L-2 ↔ MiniLM-L-6 ↔ MiniLM-L-12) and the
  before/after orderings should change non-trivially. On the parent
  corpus, the rerank model is effectively a no-op because decay
  dominates.

## Limitations and known issues

- **Baseline recall is intentionally low** (~3% in the 2026-05-28 run).
  That's the design: the corpus is paraphrase-stressed by construction so
  there's headroom for rerank and modality to demonstrate lift. A corpus
  with 80% baseline recall couldn't show a 10pp lift — there's no room.
- **Drawer-ID drift.** Drawer IDs in `expected_drawers` are content
  hashes that should be stable, but garbage collection or re-ingest can
  delete them. The fallback to `expected_substrings` mitigates this
  for most questions but not all. Refresh restores tight ID coverage.
- **Canary questions are by design unanswerable** in the 7-day window. A
  good retrieval system should miss them; if it starts hitting canaries,
  that's a false-positive signal worth investigating before drawing
  conclusions from the corpus.
- **Single-window snapshot.** This corpus measures one week. A retrieval
  change that helps narrow-age but regresses wide-age won't be caught
  here — pair this with the parent corpus for any landing decision.

## Files

- `sme/corpora/paraphrase_narrow_age/questions.yaml` — the corpus itself
- `baselines/probe-results-2026-05-28-narrow-age-baseline.json` — first
  baseline run (no-HyDE arm = the canonical baseline; HyDE arm provided
  for completeness)
- Runner: `familiar.realm.watch/tests/eval/run_paraphrase_probe.py` (lives
  in the familiar.realm.watch repo, invoked from here via absolute path)

## Related

- familiar.realm.watch issue #73 — task spec
- techempower-org/multipass-structural-memory-eval PR #94 (Aurora) —
  rerank/decay interaction diagnosis that motivated this corpus
- `tests/eval/paraphrase_questions.yaml` (parent corpus) — the wide-age
  control set
