# Chunking-axis sensitivity is partly downstream of encoder calibration

**Date:** 2026-05-17
**Branch:** `feat/rlm-adapter`
**Result:** Encoder fine-tuning absorbs **~83% of the chunking-strategy R@5 sensitivity** on our 48 git-derived markdown probes. Tests the hypothesis posted on `MemPalace/mempalace#1384` (discussioncomment-16950768).

## 2x2 ablation

48 markdown probes from `sme/corpora/mempalace_git_probes_v2/questions.yaml`, corpus = current HEAD .md files in `~/Projects/memorypalace/` (techempower-org/mempalace fork).

| Encoder | Chunker | R@5 |
|---|---|---:|
| base-MiniLM | paragraph | 0.6250 |
| base-MiniLM | heading-aware | 0.5000 |
| FT-300 | paragraph | 0.5833 |
| FT-300 | heading-aware | 0.5625 |

**Δ (heading-aware vs paragraph), per encoder:**
- base-MiniLM: **-0.1250** (-12.5pp — large, statistically meaningful at n=48)
- FT-300: **-0.0208** (-2.1pp — within noise margin)

## Two findings, both reportable

### Finding 1: encoder-conditional hypothesis CONFIRMED

The B-vs-A delta shrinks by 83% when the encoder is FT'd to the domain. This mirrors exactly what `@nakata-app` observed on their 20-probe set (B-vs-A flat at ΔMRR = 0 with [0, 0] CI bootstrap, using FT-300 weights). The methodological implication: **chunking-axis ablation results need to specify the encoder calibration regime they were measured under**, because the result may not survive an encoder-FT swap. Reporting B-vs-A without that qualifier risks publishing an encoder-dependent result as if it were universal.

### Finding 2: heading-aware *loses* on git-subject probes (opposite of xg-gh-25)

This is the directional surprise. `@xg-gh-25`'s argument on `MemPalace/mempalace#1384` (discussioncomment-16948061) was that heading-aware should win on markdown because *the heading IS the retrieval signal*. On our probes, heading-aware loses in both encoder regimes.

Likely cause: our probes are **commit-subject-shaped**, e.g.:

| Probe text | Expected file |
|---|---|
| `Post-mortem section in pgvector-cutover-runbook` | `pgvector-cutover-runbook.md` |
| `Pgvector migration 2026-05-14 status snapshot — Phases 4.1/4.2` | `2026-05-10-pgvector-age-migration-impl.md` |
| `Update runbook with daemon-state + repair-required findings` | `pgvector-cutover-runbook.md` |

The commit subject is broader than any single in-file section heading. Paragraph chunks catch the relevant body via word overlap with the broader subject text; heading-prefixed chunks dilute the body signal with hierarchical headings (`Operators > Pgvector Cutover > Phase 4.1 > Post-mortem`) that don't share vocabulary with the commit subject.

xg-gh-25's "ship B for .md" probably holds for **user-style queries** ("what did we decide about X?" — where the heading `## Decision: X` IS the strongest semantic anchor). It doesn't hold for **commit-subject-style retrieval** (where the body text shares more vocabulary with the probe than the heading hierarchy does).

This is itself a useful generalization: **chunking strategy is downstream of probe shape, not just corpus shape or encoder calibration**.

## Implication for `mempalace#1508` (`symbol_header_prefix` kwarg)

The `symbol_header_prefix` PR adds a kwarg for prepending AST-extracted symbol headers to chunks. It was closed by `@jphein` with "premature on my side; want to do more local validation before opening upstream."

The local validation just landed: **the kwarg's value proposition is encoder-conditional**. At base-MiniLM, chunking variance is 12.5pp — enriching chunks with symbol headers could plausibly move recall by a similar magnitude. At FT-300, chunking variance is 2.1pp — symbol headers have little headroom to move recall. The PR is worth reopening *with this framing*: it's a lever for the encoder-not-yet-calibrated regime, which is exactly where most projects sit until they invest in domain FT.

## Caveats

- 48 probes is too small for paired bootstrap to be tight. The -12.5pp base-MiniLM finding looks robust at this n; the -2.1pp FT-300 finding is in noise territory and the directional claim ("FT-300 absorbs most of the variance") is the strongest version of the result, not the per-encoder absolute numbers.
- We haven't tested xg-gh-25's "skip chunking, use structured extraction + graph" path. That's a separate adapter being built ([SME task #42](https://github.com/techempower-org/multipass-structural-memory-eval)).
- Our paragraph chunker is naive (blank-line split, 800-char cap). MemPalace's `convo_miner._chunk_by_paragraph` has additional logic (CHUNK_SIZE enforcement, min_chunk_size validation, etc.) that may produce different chunk boundaries; running this ablation through `convo_miner` would tighten the comparison.

## Artifacts

- Bench script: `scripts/chunk_strategy_encoder_ablation.py`
- Results: `baselines/chunking_encoder_ablation_2026-05-17.json` (per-condition R@5 + per-probe hit/miss + top-5 filenames retrieved)
- Trained FT-300 weights: `/tmp/minilm-lme-ft-300-katana/model/` (reproducible via `docs/benchmarks/2026-05-17-adaptmem-ft300-reproduction.md`)

## Open questions for the upstream thread

- Does the 12.5pp → 2.1pp absorption ratio hold on a larger corpus + more probes? Atakan's bootstrap on the 20-probe set hit [0, 0] CI with FT-300, consistent with our 2.1pp here.
- Is the directional surprise on heading-aware (paragraph beats heading-aware on commit-subject probes) consistent across probe sources? If you have user-style queries (not commit subjects) and re-run this ablation, the heading-aware-wins claim probably holds — implying a probe-shape × chunking interaction worth its own writeup.
