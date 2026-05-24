# good-dog-corpus Cat 3 / Cat 6 — coverage audit and flat-baseline reading

**Issue:** [techempower-org/multipass-structural-memory-eval#21](https://github.com/techempower-org/multipass-structural-memory-eval/issues/21)
— *Run good-dog-corpus Cat 3/6 to validate True Memory's consolidation
claims.*

**Date:** 2026-05-24
**Author:** Lucid (dreamweavers team)
**Status:** Initial pass. Flat baseline only. Full pipeline / structural
adapter readings deferred until upstream LongMemEval+MemPalace E2E
plumbing (issues #17–#19) lands.

---

## What this document is

A coverage audit of the good-dog-corpus question set for SME Cat 3
(contradiction surfacing — *The Dissonance*) and Cat 6 (temporal
supersession — *The Archive*), plus the flat-baseline floor reading
against the expanded question set. It exists to answer two related
questions:

1. **Does the corpus's question set adequately probe the consolidation-
   layer claims** that consolidation-aware systems (True Memory,
   MemPalace with temporal facts, Zep, mem0g) make in marketing
   material — i.e., that the system *flags both old and new framings*
   and *tracks the supersession chain across documents*?
2. **What is the flat-baseline ceiling on the substring matcher** for
   these questions? Where any structural system's per-question recall
   tops out at or near the flat number, the matcher is reading filename
   / paragraph overlap, not retrieval quality, and the headline metric
   for that question is `(structural − flat)` — not the absolute
   number.

The answers, in one paragraph: the question set was thin (3+3) before
this pass; six new questions land it at 6+6, above the 5+ aim from #21.
The flat baseline on the expanded set scores 90.3% mean recall on Cat 3
and 100% on Cat 6 with the substring matcher — meaning **the substring
matcher cannot distinguish a consolidation-aware system from a flat
RAG on Cat 6 at all, and can only do so weakly on Cat 3**. The real
consolidation signal lives in the `ContradictionPair[]` field of
`QueryResult` (spec v8 §3) and in the `_superseded_by` edge property
(spec v8 §6 / `_created_by` provenance), neither of which the substring
matcher reads. That is the finding — a corpus-side gap, not a system
defect.

---

## §1 — Coverage audit, before and after

### 1.1 Before (v0.1, 2026-05-01)

| Category | Questions | Probes |
|---|---|---|
| Cat 3 | q08, q09, q10 (3 total) | DCM grain-free pair (1 question); dominance theory pair (1 question); UKC/AKC pit-bull/AmStaff distinction (1 question, unbound to a `contradiction_pair_id`) |
| Cat 6 | q14, q15, q16 (3 total) | Montreal BSL lifecycle; Schenkel→Mech→AVSAB chain; FDA 2018→2019→2022 framing shift |

**Identified gaps:**

- The Hill's vitamin D recall lifecycle is the cleanest 4-document
  `supersedes` chain in the corpus (Jan 31 → Mar 20 → May 20 → Nov 20
  2019, three explicit `supersedes` edges in the vault frontmatter)
  but was probed only as a Cat 2c multi-hop walk (q07), not as a
  Cat 6 supersession question.
- Tufts BEG ("It's Not Just Grain-Free", 2018-11) and Freeman JVIM
  2022 are the strongest academic counterweights to the 2018
  grain-free framing, but the existing Cat 3 question set only probed
  the FDA self-reframing — not the academic-side contradiction.
- The AVSAB-as-current-recommendation framing (positive reinforcement
  as the recommended replacement) was implicit in q09 but not probed
  as a discrete clinical-consensus question.
- Cross-jurisdiction policy contradiction (Ontario DOLA pit-bull
  restriction vs Montreal repeal vs Calgary no-BSL) was present in
  the vault across four notes but not probed as Cat 3 — even though
  the contradiction is "all three of these are simultaneously
  current law in different jurisdictions, and a system that returns
  only one is silently picking a side."
- The June 2019 FDA Third Status Report's *ingredient* reframing
  (grain-free → pulses / peas / lentils) was an intermediate state
  not probed by q16, which lumped 2018 → 2022 together.
- The 9-year lag between the academic correction (Mech 1999) and the
  clinical-practice codification (AVSAB 2008) was a Cat 6 evolution
  not probed at the temporal-lag granularity.

3-questions-per-category was below the 5+ aim from the issue. The
margin between "the corpus happened to have one question that worked"
and "the corpus exercises the consolidation primitive" was too thin
to draw conclusions from.

### 1.2 After (v0.1 expanded, 2026-05-24)

Six new questions added in this pass:

| ID | Category | Probes | `expected_sources` |
|---|---|---|---|
| q19_beg_diet_alternate_framing | cat_3 | Tufts academic counter-framing to grain-free DCM (BEG, pulses, lentils) | `Tufts`, `BEG`, `boutique`, `pulses`, `lentils` |
| q20_pit_bull_policy_jurisdictional_contradiction | cat_3 | Three Canadian jurisdictions with three different current legal frameworks for pit-bull-type dogs | `Ontario`, `DOLA`, `Montreal`, `repeal`, `Calgary`, `behaviour-based` |
| q21_dominance_research_vs_clinical_consensus | cat_3 | AVSAB's recommended replacement for dominance-based training | `AVSAB`, `positive reinforcement`, `reward-based`, `dominance` |
| q22_hills_recall_final_state | cat_6 | Final stage of the Hill's vitamin D lifecycle (Warning Letter, Nov 2019, Topeka) | `Hill's`, `Warning Letter`, `November`, `Topeka`, `vitamin premix` |
| q23_dcm_ingredient_reframing | cat_6 | Intermediate ingredient reframing in FDA Third Status Report (grain-free → pulses) | `grain-free`, `pulses`, `peas`, `lentils`, `June 2019`, `Third Status Report` |
| q24_clinical_consensus_lag_after_mech | cat_6 | Temporal lag between Mech 1999 academic correction and AVSAB 2008 clinical recommendation | `Mech`, `1999`, `AVSAB`, `2008`, `dominance` |

All `expected_sources` substrings verified present in ≥1 vault note via
`tests/test_questions_yaml_smoke.py` (8/8 tests passing post-additions).

**New totals: Cat 3 = 6, Cat 6 = 6. Total corpus questions: 24.**

---

## §2 — Flat baseline reading

### 2.1 Setup

- **Adapter:** `FlatBaselineAdapter` (`sme/adapters/flat_baseline.py`).
  Pure top-K cosine similarity over a ChromaDB collection. No metadata
  filtering, no graph traversal, no reranking. The reference Condition
  A per spec v8 §7.
- **Ingestion:** `scripts/ingest_good_dog_flat.py` (added in this PR).
  Reads every `.md` under `sme/corpora/good-dog-corpus/vault/`, splits
  on blank-line boundaries with a 120-char floor (so single-line
  headings glue onto the preceding paragraph), and writes to a
  ChromaDB persistent collection using the default embedding
  (all-MiniLM-L6-v2). 24 files → 309 chunks.
- **Command:**

  ```
  ./venv/bin/python scripts/ingest_good_dog_flat.py --out /tmp/good_dog_chroma
  ./venv/bin/sme-eval retrieve --adapter flat \
      --db /tmp/good_dog_chroma --collection-name good_dog_flat \
      --questions sme/corpora/good-dog-corpus/questions.yaml \
      --json /tmp/good_dog_flat.json
  ```

- **Scoring:** substring-presence on the retrieved `context_string`.
  This is the SME default for `sme-eval retrieve`; the LongMemEval
  GPT-4o judge upgrade is gated on the cross-validation harness
  (`scripts/cross_validate_longmemeval.py`) per the questions.yaml
  header comments.

### 2.2 Headline numbers

```
cat   |  n  |  full-recall  |  mean_recall  |  avg_tokens
cat_1 |   4 |    4/4        |    1.000      |      826
cat_2c |  3 |    1/3        |    0.750      |      696
cat_3 |   6 |    4/6        |    0.903      |      805
cat_4 |   3 |    3/3        |    1.000      |      938
cat_6 |   6 |    6/6        |    1.000      |      801
cat_7 |   2 |    1/2        |    0.833      |      848
total |  24 |   19/24       |    0.931      |      814
```

Total tokens 19,538; tokens-per-correct-answer 1,028; hit-rate
(any-substring) 100%; full-recall (all-substrings) 19/24.

### 2.3 Per-question Cat 3 / Cat 6 detail

```
[cat_3] q08_grain_free_dcm_causal_status                        recall=1.00 ✓
[cat_3] q09_alpha_wolf_dominance_framing_status                 recall=1.00 ✓
[cat_3] q10_pit_bull_apbt_amstaff_relationship                  recall=1.00 ✓
[cat_3] q19_beg_diet_alternate_framing                          recall=1.00 ✓
[cat_3] q20_pit_bull_policy_jurisdictional_contradiction        recall=0.67 — miss: Calgary, behaviour-based
[cat_3] q21_dominance_research_vs_clinical_consensus            recall=0.75 — miss: positive reinforcement
[cat_6] q14_montreal_bsl_lifecycle                              recall=1.00 ✓
[cat_6] q15_dominance_theory_supersession                       recall=1.00 ✓
[cat_6] q16_fda_dcm_investigation_status_progression            recall=1.00 ✓
[cat_6] q22_hills_recall_final_state                            recall=1.00 ✓
[cat_6] q23_dcm_ingredient_reframing                            recall=1.00 ✓
[cat_6] q24_clinical_consensus_lag_after_mech                   recall=1.00 ✓
```

Two partial-recall Cat 3 questions — both probe cross-document
contradiction that the top-K retrieval doesn't surface because the
relevant terms live in different vault domains:

- **q20** misses `Calgary` and `behaviour-based`. Top-K retrieved
  Ontario DOLA + Montreal notes, but the Calgary RPOB note did not
  rank high enough for the cross-jurisdiction sweep at K=10.
- **q21** misses `positive reinforcement`. AVSAB note ranks but the
  retrieval pulls dominance-framing chunks rather than the
  positive-reinforcement-as-replacement chunk.

These are the questions where structural retrieval / cross-document
traversal *could* plausibly beat flat — they are the questions worth
re-running once a structural adapter is wired up against this corpus.
The other Cat 3 questions and all Cat 6 questions are at the matcher
ceiling.

---

## §3 — What the flat-baseline reading does and doesn't say

### 3.1 What it says

- **The corpus's question set is well-grounded.** 19/24 questions
  achieve full substring recall against a no-structure baseline; the
  remaining 5 are partials with identifiable misses (one substring
  out of several, typically the cross-domain term). No question
  scored zero — the corpus has no "expected_sources nobody can
  retrieve" authoring bugs.
- **Substring recall cannot distinguish flat from consolidation-aware
  on Cat 6 in this corpus.** All 6 Cat 6 questions are at 1.00 on the
  flat baseline. A consolidation-aware system that scored 1.00 on Cat 6
  here is matching the floor, not beating it. The structural signal
  (correct identification of the *current* state vs the *superseded*
  ones, correct ordering of the supersession chain) is not measured by
  substring presence — it's measured by whether the system marks
  edges with `_superseded_by` and returns them ordered by `_created_at`
  per spec v8 §6.
- **Substring recall can distinguish flat from consolidation-aware on
  Cat 3 only weakly.** The two partial-recall Cat 3 questions (q20,
  q21) are the only ones where a structural system has substring
  headroom to demonstrate cross-document traversal. The other four
  Cat 3 questions need the `ContradictionPair[]` structured-response
  channel (spec v8 §3) to measure the actual consolidation claim —
  "did the system flag BOTH framings as conflicting?" — which the
  substring matcher cannot detect.

### 3.2 What it doesn't say

The flat baseline reading **does not** measure:

- **Contradiction surfacing.** Whether the system explicitly returns a
  `ContradictionPair(claim_a, claim_b, source_a, source_b)` for the
  seeded contradiction pairs `dcm_grain_free_2018` and
  `dominance_theory_pre_vs_post_1999`. The flat adapter returns zero
  such pairs by design. Spec v8 §3 metrics —
  *Contradiction Detection Rate* and *Contradiction Precision* —
  measure exactly this, and need a structural adapter to be non-trivial.
- **Provenance chain integrity.** Whether the system can answer "which
  edges were reclassified from RELATED to typed?" or "which document
  in the chain is currently authoritative?" The flat adapter has no
  edges; this is the spec v8 §6 sub-test 6b.
- **Temporal persistence stability.** Persistence diagrams at different
  time slices showing structural evolution (spec v8 §6 Topology
  integration). Out of scope for the substring-recall harness.

These are the readings a consolidation-aware system must take to
*validate the consolidation claim*. The flat baseline establishes only
that the corpus content is *retrievable* — a prerequisite for the
consolidation reading, not a substitute for it.

---

## §4 — What a consolidation-aware system needs to score well

For a system to demonstrate consolidation-layer value on good-dog-corpus
Cat 3 / Cat 6, it would need to:

**Cat 3 — Contradiction surfacing.**

1. **Return both framings.** For q08, q19, q23: surface both the 2018
   FDA grain-free framing AND the 2019/2022 multifactorial framing /
   Tufts BEG framing, not just whichever embedding ranks highest.
2. **Populate the `ContradictionPair[]` field.** Per spec v8 §3, the
   `QueryResult.contradictions` field is the structured channel for
   contradiction detection. A system that returns the right context
   substrings but leaves this field empty scores 0 on Cat 3 even with
   1.00 substring recall.
3. **Cross-document traversal.** For q20 (pit-bull policy): pull from
   Ontario, Montreal, Calgary notes simultaneously, recognise they
   constitute three current-but-incompatible frames, and flag the
   *jurisdictional* dimension as the discriminator.

**Cat 6 — Temporal supersession.**

1. **Identify the current state.** For q22 (Hill's final state): not
   the Jan 31 announcement (superseded), not the Mar 20 expansion
   (superseded), not the May 20 expansion (superseded by Warning
   Letter on Nov 20). The *current* document is the Warning Letter.
   A system that returns all four with equal weight has not consolidated.
2. **Order the chain.** For q07, q15, q16, q23: return the chain in
   temporal order, with explicit supersession markers between adjacent
   stages. Spec v8 §6 sub-test 6b: "which edges were reclassified" —
   the `_superseded_by` edge property is the channel for this.
3. **Recognise intermediate states.** For q23, q24: return not just
   the endpoints but the load-bearing intermediate (Third Status Report
   for ingredient reframing; the 9-year Mech→AVSAB gap for clinical
   codification lag).

A True Memory implementation, a MemPalace deployment with the temporal
KG enabled, a Zep instance, or a mem0g instance should — per their own
marketing claims — be capable of returning these structured signals.
The substring matcher reading will mostly come back at the flat ceiling;
the meaningful signal will live in the structured-response channels
(`contradictions[]`, `retrieved_edges[]` with `_superseded_by` /
`_created_at`) that this initial reading does not yet exercise.

The next step for issue #21 is wiring `sme-eval cat3` and `sme-eval
cat6` against the good-dog-corpus to read those structured channels —
which is the structural-adapter task currently blocked on the upstream
adapter plumbing (issues #17–#19, #18-Mem0g/OMEGA/Hindsight).

---

## §5 — Reproducibility

Everything needed to reproduce this reading is in-tree:

```
sme/corpora/good-dog-corpus/vault/        — 24 vault notes
sme/corpora/good-dog-corpus/questions.yaml — 24 questions (6 new in this pass)
sme/corpora/good-dog-corpus/ontology.yaml  — schema and contradiction-pair registry
tests/test_questions_yaml_smoke.py         — 8 smoke checks, all passing
scripts/ingest_good_dog_flat.py            — flat-baseline ingestion helper
```

Reproduce with:

```
./venv/bin/python -m pytest tests/test_questions_yaml_smoke.py -v
./venv/bin/python scripts/ingest_good_dog_flat.py --out /tmp/good_dog_chroma
./venv/bin/sme-eval retrieve --adapter flat \
    --db /tmp/good_dog_chroma --collection-name good_dog_flat \
    --questions sme/corpora/good-dog-corpus/questions.yaml \
    --json /tmp/good_dog_flat.json
```

JSON output schema: `{adapter, db_path, collection_name, corpus_version,
n_results, questions: [{id, text, min_hops, expected_sources,
matched_sources, recall, hit, tokens, elapsed_ms, retrieval_path,
error}], summary}`.
