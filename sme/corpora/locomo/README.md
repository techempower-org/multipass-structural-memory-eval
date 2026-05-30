# LoCoMo — SME corpus loader

This directory holds the loader and SME-shape conversion for the
**LoCoMo** benchmark (Maharana et al., ACL 2024; arXiv 2402.17753).
The dataset itself is **not committed** to this repo; see the download
section below.

The loader is the Tier-2 deliverable of the comparison-readiness plan
(`docs/research/2026-05-29-comparison-readiness.md` §3.2) — a second
benchmark so SME is not a one-dataset entrant. It unlocks the LoCoMo QA
column (EverOS 93.05% / True Memory 93.0% / Mem0 92.5% / Hindsight
89.61%). It mirrors the LongMemEval loader's interface exactly.

## Pinned subset (the comparability contract)

LoCoMo cross-comparisons are unreliable unless the exact question
subset and adversarial inclusion are pinned
(`docs/research/2026-05-29-comparison-readiness.md` §1.3). This loader pins:

| Constant | Value | Meaning |
|---|---|---|
| `SUBSET` | `"locomo10"` | the canonical released benchmark file |
| `SUBSET_SAMPLE_COUNT` | `10` | 10 conversations |
| `SUBSET_QA_COUNT` | `1986` | all QA across the 10 conversations |
| `ADVERSARIAL_INCLUDED` | `True` | category-5 items are loaded and flagged |

**Any reading published from this loader must state all four.** The
1986 figure is the sum over the 10 conversations
(199+105+193+260+242+158+190+239+196+204) of `data/locomo10.json`. It
is **not** the paper's 7,512-QA number — that counts the larger
50-conversation *construction* set, which was never publicly released.
`locomo10.json` is what the field actually benchmarks on.

## Dataset download

```bash
mkdir -p sme/corpora/locomo/data
cd sme/corpora/locomo/data
wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

The `data/` directory is gitignored — keep the upstream JSON local.
(`locomo10.json` is ~2.8 MB; downloaded + verified 2026-05-29.)

## Category numbering caveat (READ THIS)

LoCoMo has a notorious mismatch between its **paper prose** and its
**released JSON**:

- The paper (arXiv 2402.17753) lists categories in prose as
  (1) single-hop, (2) multi-hop, (3) temporal, (4) open-domain,
  (5) adversarial.
- The official scorer `task_eval/evaluation.py`, which consumes
  `locomo10.json`, uses a **different** integer mapping: category
  **1 = multi-hop**, **2/3/4 = single-hop / temporal / open-domain**,
  **5 = adversarial**.

The released JSON labels its data with the *scorer's* numbering. We
verified empirically on `locomo10.json`: category 1 has the highest
mean evidence-reference count (3.13 refs/question vs 1.0–2.1 for cats
2/3/4) — the signature of multi-session synthesis (multi-hop). This
loader therefore pins to the **scorer numbering** (`LOCOMO_CATEGORY_NAMES`).
A system that used the prose numbering for cats 1–4 must reconcile
this before any head-to-head.

Observed category distribution in `locomo10.json` (1986 QA):

| category | name (scorer) | n | mean evidence refs |
|---|---|---|---|
| 1 | multi-hop | 282 | 3.13 |
| 2 | single-hop | 321 | 1.17 |
| 3 | temporal | 96 | 2.08 |
| 4 | open-domain | 841 | 1.07 |
| 5 | adversarial | 446 | 1.03 |

## Format mapping (LoCoMo → SME)

| LoCoMo field | SME mapping |
|---|---|
| `sample_id` (`conv-26`…) | preserved under `locomo.sample_id`; the per-sample vault dir |
| `qa[i]` | one `LoCoMoQuestion`, `id = "<sample_id>::q<i>"` |
| `category` (1–5 int) | preserved under `locomo.category`; named via `LOCOMO_CATEGORY_NAMES`; mapped to SME category via `LOCOMO_CATEGORY_TO_SME` |
| `question` | `text` |
| `answer` | `gold_answer` (LoCoMo QA judge target; outside SME's substring matcher) |
| `adversarial_answer` | preserved under `locomo.adversarial_answer` — the *wrong* answer the question baits; correct behavior is abstention |
| `category == 5` | `locomo.is_adversarial: true`, `sme_category: cat_1_negative` |
| `evidence` (`["D1:3"]`) | preserved under `locomo.evidence`; collapsed to `D<N>` session ids for `expected_sources` |
| `conversation.session_N` | one markdown file per session under `vault/<sample_id>/D<N>.md` |
| turn `img_url` / `blip_caption` | BLIP caption folded into the turn body as `_[shared image: …]_`; raw url preserved in an HTML comment |

## SME ↔ LoCoMo category mapping (`LOCOMO_CATEGORY_TO_SME`)

| LoCoMo (scorer #) | SME category | Mapping confidence |
|---|---|---|
| single-hop (2) | Cat 1 | **Exact primitive match.** Only divergence is scorer (substring vs LLM judge). |
| multi-hop (1) | Cat 2c | **Partial.** LoCoMo does not break out hop depth (1/2/3) the way SME Cat 2c does. |
| temporal (3) | Cat 6 | Strong match on time-point queries; LoCoMo does not test Cat 6b provenance. |
| open-domain (4) | `unmapped` | Fusing dialogue with external world knowledge — no SME analogue. |
| adversarial (5) | Cat 1 (negative class) | System must **abstain**, not retrieve — mirrors LongMemEval `_abs`. |

See `docs/related_work/locomo-and-memorybench.md` for the full
primary-source mapping and the per-SME-category divergence analysis.

## Architectural note: per-sample vaults (not per-question)

LongMemEval gives each question its own haystack, so its loader writes
a vault per *question*. LoCoMo instead shares one conversation across
all of a sample's questions, so this loader writes a vault per
*sample* (`vault/<sample_id>/`) and every question in that sample
queries the same vault. A cross-validation run loops per sample:

```python
for sample_id, sample_questions in group_by_sample(questions):
    adapter.reset()
    adapter.ingest_corpus_from_dir(vault_dir / sample_id)
    for q in sample_questions:
        result = adapter.query(q.text, n_results=5)
        sme_score = sme_substring_match(result, q.expected_sources_session_level())
        judge_score = locomo_judge(result.context_string, q.gold_answer, q.is_adversarial)
        record(q.question_id, sme_score, judge_score)
```

## Status

- `loader.py` — `LoCoMoQuestion` / `LoCoMoSession` / `LoCoMoTurn`
  dataclasses, `load_questions(path)` iterator,
  `materialize_sme_corpus(questions, output_dir)` for per-sample vault
  rendering. Pinned-subset constants exported.
- `tests/test_locomo_loader.py` — schema-fidelity tests against an
  inline 2-sample fixture (no download needed).
- **Pending (team-lead, daemon lane):** the actual daemon
  retrieval + QA run. See `scratch/zephyr-locomo/findings.md` for the
  exact command.

## Citation

Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., &
Fang, Y. (2024). *Evaluating Very Long-Term Conversational Memory of
LLM Agents.* ACL 2024 (Long Papers). arXiv:2402.17753.

Upstream repo: https://github.com/snap-research/locomo
Project page: https://snap-research.github.io/locomo/
