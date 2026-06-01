# Case Study 2 — The storage-equivalence null (Cat 1 / Cat 7)

**A null result that earns its place.** Swapping the storage engine —
ChromaDB → postgres+pgvector — while holding everything else fixed left
both retrieval and end-to-end QA statistically identical. The finding
isn't "the migration is fine" (though it is); it's that **the storage
engine is not the variable** people think it is.

- **Categories:** Cat 1 (R@5 retrieval) + Cat 7 (E2E QA, A/B condition)
- **Systems:** `flat` (ChromaDB) vs `postgres_ingest` (postgres+pgvector)
- **Scale:** jp-realm-v0.1 Condition-A (retrieval); LoCoMo-10, n=250 (QA)
- **Status:** RESOLVED — CI-confirmed central null of the campaign.

---

## The finding

Operators routinely assume the vector store *engine* is a quality lever:
"switch to pgvector and retrieval will improve." SME ran the controlled
experiment — **hold embedding (all-MiniLM-L6-v2), corpus, reader, and
judge fixed; swap only the storage engine** — to test it.

| Metric | flat (ChromaDB) | postgres_ingest (pg+pgvector) | Δ |
|---|---|---|---|
| Cat 1 R@5 (jp-realm) | 0.833 | 0.833 | **0.000** |
| Cat 2c by-hop | hop-1 0.852 / hop-2 0.667 | hop-1 0.852 / hop-2 0.667 | **identical** |
| Cat 7 LoCoMo E2E QA (n=250) | 0.384 | 0.392 | **+0.008 (noise)** |

Retrieval recall is byte-identical; QA is within noise.

## The fix (what the reading tells you to change — or not)

The actionable guidance here is **negative, and that's the point**: do
*not* spend effort migrating storage engines expecting a retrieval or QA
gain. The reading says the lever is elsewhere — the embedding model and
the corpus carry the answer; the engine is interchangeable.

The complementary positive lever, from the same campaign: **widening the
retrieval window** (top-5 → top-20) buys **+17.3pp QA** and then plateaus
(synthesis §5.1). That is where an operator chasing QA should spend the
effort — not on the storage substrate.

## The re-run (the verification that makes it a null, not a guess)

Eyeballing "0.384 ≈ 0.392" is not enough — SME's #21 statistics pass
turned it into a tested statement (`baselines/headline_delta_significance_2026-05-31.json`):

| Comparison | Δ (pp) | 95% CI (pp) | n | n_discordant | p_adj | significant? |
|---|---:|---|---:|---:|---:|:--:|
| postgres vs flat, QA | **+0.4** | **[−2.0, +2.8]** | 250 | **9** | 0.84 | **No (null)** |
| postgres vs flat, R@5 | 0.0 | identical | 250 | 0 | — | definitional 0 |

- The paired bootstrap CI (10k resamples, paired by `question_id`)
  **straddles zero**; the BH-FDR-adjusted p = 0.84.
- Only **9 of 250 questions** answer differently — the two engines
  produce the same answer 241/250 times.
- Basis note: this CI is on the strict-correct vector (abstentions
  excluded), so its means (postgres 0.264 / flat 0.260) sit below the
  correct-or-abstain 0.392/0.384 — **both bases agree the delta is null**,
  same direction.

## The lesson

A diagnostic framework's most valuable output is sometimes a
**rigorously-confirmed null**: it redirects effort away from a
non-lever. The A/B condition isolation (swap exactly one factor, hold the
rest) plus a paired CI is what lets SME say "the engine is not the
variable" as a *measured* claim rather than folklore. When you read a
near-equal pair of numbers, demand the CI and the discordant count before
calling it equivalence.

**Artifacts:** `docs/benchmarks/2026-05-30-locomo-daemon-results.md`,
`baselines/jp_realm_v0_1_{flat,postgres}_condA_*.json`,
`baselines/headline_delta_significance_2026-05-31.json`, synthesis §5.3 / §8.1.
