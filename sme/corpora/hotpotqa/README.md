# HotpotQA — SME corpus loader

This directory holds the loader and SME-shape conversion for the
**HotpotQA** benchmark (Yang et al., EMNLP 2018; arXiv 1809.09600).
The dataset itself is **not committed** to this repo; see the download
section below.

HotpotQA is the Phase-1 multi-hop calibration surface from the
standard-corpora integration plan (upstream
`M0nkeyFl0wer/multipass-structural-memory-eval#43`). It is the public,
1000s-scale corpus with **sentence-level annotated supporting facts**
that lets SME demonstrate — not just design — construct validity for
**Cat 2c** (multi-hop retrieval recall by depth): *"Cat 2c ran against
HotpotQA's known 2-hop evidence and recovered N of M gold paragraphs."*
It mirrors the LoCoMo / LongMemEval loaders' interface exactly.

## Pinned subset (the comparability contract)

HotpotQA cross-comparisons are unreliable unless the split and
retrieval setting are pinned. This loader pins:

| Constant | Value | Meaning |
|---|---|---|
| `SUBSET` | `"dev_distractor"` | the `hotpot_dev_distractor_v1.json` split |
| `SETTING` | `"distractor"` | 10-paragraph haystack (2 gold + 8 distractor) |
| `SUBSET_QUESTION_COUNT` | `7405` | questions in the dev distractor split |
| `HOTPOT_MIN_HOPS` | `2` | every question is 2-hop by construction |

**Any reading published from this loader must state the split and
setting.** The **fullwiki** setting (retrieve over all of Wikipedia) is
a different, IR-heavy task and is out of scope for a loader. The train
split (`hotpot_train_v1.1.json`, 90,447 questions) is also loadable —
pass its path explicitly — but the *pinned* comparability subset is the
dev distractor split.

## Dataset download

```bash
mkdir -p sme/corpora/hotpotqa/data
cd sme/corpora/hotpotqa/data
# dev distractor split (~44 MB) — the pinned subset
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
# optional: train split (~535 MB)
# wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
```

The `data/` directory is gitignored — keep the upstream JSON local.
HotpotQA is released under **CC BY-SA 4.0**; redistribution requires
attribution and share-alike, so the corpus is downloaded per-machine
rather than vendored here.

## Hop depth (the Cat 2c join)

SME Cat 2c groups questions by `min_hops`. HotpotQA does not annotate an
explicit integer hop depth, but **every released question is 2-hop by
construction** (two gold supporting paragraphs), so the loader assigns
`min_hops = 2` to every record. The `type` field is the qualitative
multi-hop *shape*:

| `type` | shape | retrieval behavior |
|---|---|---|
| `bridge` | sequential 2-hop | resolve a bridge entity in paragraph A, then use it to answer from paragraph B (true chaining) |
| `comparison` | parallel 2-hop | retrieve a fact from each of two paragraphs and compare them (both must be found, no chaining) |

Both require ≥2 distinct gold paragraphs, hence `min_hops = 2`. A
deeper-hop corpus (e.g. MuSiQue) would extend the depth axis; HotpotQA
pins the 2-hop calibration point that jp-realm-v0.1 cannot reach at
scale.

## Format mapping (HotpotQA → SME)

| HotpotQA field | SME mapping |
|---|---|
| `_id` | `question_id` (preserved as the SME question `id` and the per-question vault dir) |
| `question` | `text` |
| `answer` | `gold_answer` (QA judge target; `"yes"`/`"no"` for comparison questions) |
| `type` (`comparison`/`bridge`) | preserved under `hotpotqa.type`; named via `HOTPOT_TYPE_NAMES`; all map to SME `cat_2c` |
| `level` (`easy`/`medium`/`hard`) | preserved under `hotpotqa.level` |
| `supporting_facts` (`[[title, sent_id], …]`) | preserved under `hotpotqa.supporting_facts`; gold titles → `expected_sources`; sentence texts via `expected_sources_sentence_level()` |
| `context` (`[[title, [sentence, …]], …]`) | one markdown file per paragraph under `vault/<question_id>/<title>.md`; gold paragraphs flagged `is_gold: true` |
| — (assigned) | `min_hops: 2`, `sme_category: cat_2c` |

## Architectural note: per-question vaults (not per-sample)

LoCoMo shares one conversation across all of a sample's questions, so
its loader writes a vault per *sample*. HotpotQA instead gives each
question its own ~10-paragraph haystack, so this loader writes a vault
per *question* (`vault/<question_id>/`) — the same per-question scoping
as LongMemEval. A cross-validation run loops per question:

```python
for q in load_questions(dev_distractor_path):
    adapter.reset()
    adapter.ingest_corpus_from_dir(vault_dir / q.question_id)
    result = adapter.query(q.text, n_results=5)
    sme_score = sme_substring_match(result, q.expected_sources_paragraph_level())
    # multi-hop recall: did retrieval surface BOTH gold paragraphs?
    record(q.question_id, sme_score, min_hops=q.min_hops)
```

`materialize_sme_corpus(..., gold_only=True)` drops the distractors for
an oracle-retrieval upper bound; the default writes the full distractor
haystack (the standard setting).

## Status

- `loader.py` — `HotpotQuestion` / `HotpotParagraph` dataclasses,
  `load_questions(path)` iterator,
  `materialize_sme_corpus(questions, output_dir)` for per-question vault
  rendering. Pinned-subset constants exported.
- `tests/test_hotpotqa_loader.py` — schema-fidelity tests against an
  inline fixture (no download needed).
- **Pending (downstream):** the Cat 2c cross-validation run against the
  daemon. The loader is the prerequisite; the run is a separate task.

## Citation

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov,
R., & Manning, C. D. (2018). *HotpotQA: A Dataset for Diverse,
Explainable Multi-hop Question Answering.* EMNLP 2018. arXiv:1809.09600.

Upstream repo: https://github.com/hotpotqa/hotpot
Project page: https://hotpotqa.github.io/
License: CC BY-SA 4.0
