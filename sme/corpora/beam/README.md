# BEAM — SME corpus loader

This directory holds the loader and SME-shape conversion for the
**BEAM** benchmark — *"Beyond a Million Tokens: Benchmarking and
Enhancing Long-Term Memory in LLMs"* (ICLR 2026; arXiv 2510.27246;
[mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks)).
The full dataset is **not committed** to this repo; see the download
section. A small **pinned sample** (`sample/beam_100K_sample.json`)
*is* committed so the loader, its test, and a retrieval smoke run
without any network access.

BEAM is the third cross-validation benchmark (after LongMemEval and
LoCoMo). It is structurally different from the other two: an
**end-to-end QA pass-rate** benchmark scored with rubric *nuggets*
(0 / 0.5 / 1.0 per nugget), graded at **token buckets** (100K, 500K,
1M, 10M) on conversations far longer than LoCoMo's. It mirrors the
LongMemEval / LoCoMo loader interface (`load_questions`,
`materialize_sme_corpus`, dataclasses, an SME category map).

## Pinned contract (the comparability claim)

BEAM grades the *same* conversation at multiple token buckets. The
100K-token version and the 10M-token version are different retrieval
problems, so a BEAM number is meaningless without its bucket. This
loader records `bucket` on **every** record and in the materialized
`questions.yaml`.

| Constant | Value | Meaning |
|---|---|---|
| `VALID_BUCKETS` | `("100K", "500K", "1M", "10M")` | the four graded scales |
| `QUESTIONS_PER_CONVERSATION` | `20` | 10 ability types × 2 questions |
| `ABILITY_TYPES` | 10 keys | the memory abilities BEAM probes |

Released split sizes (HuggingFace dataset card, downloaded 2026-05-29):

| bucket | examples | questions (20/conv) | HF source |
|---|---|---|---|
| 100K | 20 | 400 | `Mohammadta/BEAM` split `100K` |
| 500K | 35 | 700 | `Mohammadta/BEAM` split `500K` |
| 1M | 35 | 700 | `Mohammadta/BEAM` split `1M` |
| 10M | — | — | `Mohammadta/BEAM-10M` split `10M` |

**Any reading published from this loader must state the bucket and the
conversation count.** Mem0 reports BEAM QA accuracy of **64.1 / 48.6**
at the **1M / 10M** buckets (mem0.ai research), averaging <7K tokens
per retrieval call.

## Dataset download

The full dataset lives on HuggingFace (`Mohammadta/BEAM` for
100K/500K/1M, `Mohammadta/BEAM-10M` for 10M), license CC-BY-SA-4.0. The
upstream runner (`benchmarks/beam/run.py`) downloads each split and
caches it as a per-bucket JSON file (`beam_<bucket>.json`, a top-level
array of conversation dicts). This loader consumes those cached files.

To produce them locally (the `data/` dir is gitignored):

```bash
pip install datasets    # HuggingFace datasets library
mkdir -p sme/corpora/beam/data
```

```python
import ast, json
from pathlib import Path
from datasets import load_dataset

OUT = Path("sme/corpora/beam/data")
SPLITS = {"100K": ("Mohammadta/BEAM", "100K"),
          "500K": ("Mohammadta/BEAM", "500K"),
          "1M":   ("Mohammadta/BEAM", "1M"),
          "10M":  ("Mohammadta/BEAM-10M", "10M")}

for bucket, (name, split) in SPLITS.items():
    ds = load_dataset(name, split=split)
    convs = []
    for idx, item in enumerate(ds):
        conv = {
            "conversation_id": item.get("conversation_id", f"{bucket}_{idx}"),
            "conversation_seed": item.get("conversation_seed", {}),
            "user_profile": item.get("user_profile", {}),
            "chat": item.get("chat", []),
            # probing_questions is a repr/JSON string upstream; the
            # loader parses it, so passing it through verbatim is fine.
            "probing_questions": item.get("probing_questions", "{}"),
        }
        convs.append(conv)
    (OUT / f"beam_{bucket}.json").write_text(
        json.dumps(convs, ensure_ascii=False))
    print(bucket, len(convs), "conversations")
```

Then:

```python
from sme.corpora.beam import load_questions, materialize_sme_corpus
qs = list(load_questions("sme/corpora/beam/data/beam_100K.json", bucket="100K"))
materialize_sme_corpus(qs, "sme/corpora/beam/_corpus_100K")
```

(`beam_100K.json` ≈ 14 MB, `beam_500K.json` ≈ 86 MB, `beam_1M.json`
≈ 172 MB — hence gitignored. The 10M bucket is a separate, larger HF
dataset.)

## Pinned sample (committed, no download)

`sample/beam_100K_sample.json` is a 1-conversation, 12-turn, 10-question
slice of the real 100K split (`conversation_id` "1", downloaded
2026-05-29). Turn contents, questions, answers, and rubrics are
**verbatim** from the release; only the chat was truncated to two
6-turn sessions and the non-abstention `source_chat_ids` were remapped
onto the kept turns so the fixture exercises evidence resolution
deterministically. It is what the loader test and the retrieval smoke
run against.

## Format mapping (BEAM → SME)

| BEAM field | SME mapping |
|---|---|
| `conversation_id` | preserved under `beam.conversation_id`; the per-conversation vault dir |
| `bucket` (loader arg) | preserved under `beam.bucket`; part of the comparability contract |
| `probing_questions[ability][i]` | one `BEAMQuestion`, `id = "<bucket>::<conv_id>::q<index>"` |
| ability key (`information_extraction`…) | preserved under `beam.ability_type`; mapped to SME via `BEAM_ABILITY_TO_SME` |
| item `question` | `text` |
| item `answer` / `ideal_response` | `gold_answer` (abstention items use `ideal_response`) |
| item `rubric` (`[str]` nuggets) | preserved under `beam.rubric`; joined `' | '` as `beam.ground_truth_nuggets` (the upstream judge target) |
| item `source_chat_ids` (`[int]`) | preserved under `beam.source_chat_ids`; resolved to `S<N>` session ids for `expected_sources` |
| `ability == "abstention"` | `beam.is_abstention: true`, `sme_category: cat_1_negative` |
| `conversation.chat[session][turn]` | one markdown file per session under `vault/<conv_id>/S<N>.md`; turn `id` preserved in an HTML comment, `time_anchor` folded into the body |

## SME ↔ BEAM ability mapping (`BEAM_ABILITY_TO_SME`)

| BEAM ability | SME category | Mapping confidence |
|---|---|---|
| information_extraction | Cat 1 | **Exact primitive match** (recall a specific entity/date/number). |
| multi_session_reasoning | Cat 2c | **Partial** — BEAM doesn't break out hop depth (1/2/3) the way SME Cat 2c does. |
| contradiction_resolution | Cat 3 | **Partial** — closest match to Cat 3; BEAM grades the reconciled answer, not the surfacing of both sides. |
| knowledge_update | Cat 3 | **Partial, with KU divergence** — BEAM rewards returning the *revised* value; SME Cat 3 rewards *flagging* old vs new. A silent-overwriter scores better on KU. See `docs/related_work/longmemeval.md`. |
| temporal_reasoning | Cat 6 | Strong on time-point/duration; BEAM does not test Cat 6b provenance. |
| event_ordering | Cat 6 | Chronological reconstruction (Kendall tau-b upstream, not a substring match). |
| abstention | Cat 1 (negative class) | System must **abstain**, not retrieve — mirrors LongMemEval `_abs` and LoCoMo adversarial. |
| preference_following | `unmapped` | Grades generation adherence, not structural retrieval. |
| instruction_following | `unmapped` | Grades sustained constraint adherence — no SME analogue. |
| summarization | `unmapped` | Grades compression — no SME analogue. |

## Architectural note: per-conversation vaults (not per-question)

Like LoCoMo (and unlike LongMemEval's per-question haystacks), BEAM
shares one conversation across all of that conversation's 20 questions.
This loader therefore writes a vault per *conversation*
(`vault/<conversation_id>/`) and every question in that conversation
queries the same vault. A cross-validation run loops per conversation:
materialize + ingest the conversation's vault once, then query all 20
questions against it.

## Status

- `loader.py` — `BEAMQuestion` / `BEAMSession` / `BEAMTurn` dataclasses,
  `load_questions(path, bucket=...)` iterator,
  `materialize_sme_corpus(questions, output_dir)` for per-conversation
  vault rendering. Pinned-contract constants exported.
- `sample/beam_100K_sample.json` — committed 10-question sample of the
  real 100K split (no download needed for tests / smoke).
- `tests/test_beam_loader.py` — schema-fidelity tests against the
  pinned sample + an inline fixture.
- **Follow-ups (this is a long track):** wire BEAM into
  `scripts/cross_validate_longmemeval.py` (a per-conversation runner
  analogous to `run_locomo_questions`), nugget-based judge scoring, and
  the 500K / 1M / 10M buckets. The first PR delivers the loader + a
  100K-bucket retrieval smoke + tests; the larger-bucket QA runs are
  separate work.

## Citation

Taghibakhshi, M., et al. (2026). *Beyond a Million Tokens: Benchmarking
and Enhancing Long-Term Memory in LLMs.* ICLR 2026. arXiv:2510.27246.

Upstream repo: https://github.com/mem0ai/memory-benchmarks
Dataset: https://huggingface.co/datasets/Mohammadta/BEAM
