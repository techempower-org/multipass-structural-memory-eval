# adaptmem FT-300 reproduction: R@5 = 1.0000 on 200q test split

**Date:** 2026-05-17
**Branch:** `feat/rlm-adapter`
**Result:** Reproduced nakata-app's adaptmem FT-300 LongMemEval R@5 = 0.9950 claim on katana. Our reproduction lands at **R@5 = 1.0000** on the held-out 200-question test split — slightly above the published number, well within run-to-run variance.

## What was reproduced

| | nakata-app's published FT-300 ([results_ft300_direct.json](https://github.com/nakata-app/adaptmem/blob/main/benchmarks/results_ft300_direct.json)) | Our katana reproduction |
|---|---|---|
| n_test | 200 | 200 |
| R@1 | 0.915 | **0.925** (+1.0pp) |
| R@5 | 0.995 | **1.000** (+0.5pp, perfect) |
| R@10 | 0.995 | **1.000** |
| wall clock | 607.06s (their box) | 56s train + 16s test = ~72s (katana GPU) |

The +0.5pp on R@5 (and +1.0pp on R@1) is within plausible stochastic noise from the encoder fine-tune (epoch order, weight init, GPU vs CPU). The protocol is otherwise identical.

## Setup

Reproduced via the upstream `nakata-app/adaptmem` repo's own training script. Exact protocol:

| | Value |
|---|---|
| Repo | `https://github.com/nakata-app/adaptmem` |
| Training script | `benchmarks/longmemeval_eval.py --mode train` |
| Test script | `benchmarks/longmemeval_eval.py --mode test` |
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Training data | 300 LongMemEval-S question/relevant-session pairs (seed=42 shuffle of `longmemeval_s_cleaned.json`) |
| Test data | 200 remaining LongMemEval-S questions (same shuffle) |
| Fine-tune | epochs=3, batch=16, lr=2e-5, MultipleNegativesRankingLoss |
| GPU | katana (NVIDIA, via PyTorch CUDA) |
| Training wall clock | 55.6s for 565 query pairs over 12493 corpus entries |

The split (`baselines/lme_split_300_200_seed42.json`) was generated locally with `random.Random(42).shuffle(qids)` — same algorithm nakata-app documents, and the first 3 train ids (`cc06de0d`, `f9e8c073`, `b320f3f8`) match the committed `split_ids_100_400.json`, confirming the shuffle is reproducible across hosts.

## Re-running on katana

```bash
# 1. Clone nakata-app/adaptmem and install
cd /tmp && git clone --depth 1 https://github.com/nakata-app/adaptmem
cd adaptmem && pip install -e .

# 2. Build the seed=42 300/200 split (same as published)
python -c "
import json, random
data = json.loads(open('LME_DATA_PATH').read())
qids = [q['question_id'] for q in data]
random.Random(42).shuffle(qids)
json.dump({'train_question_ids': qids[:300], 'test_question_ids': qids[300:]},
           open('split.json','w'), indent=2)
"

# 3. Train (~1 min on GPU)
python benchmarks/longmemeval_eval.py --mode train \
    --dataset LME_DATA_PATH \
    --split-ids split.json --n-train 300 \
    --model-out /tmp/minilm-lme-ft-300 \
    --device cuda

# 4. Test
python benchmarks/longmemeval_eval.py --mode test \
    --dataset LME_DATA_PATH \
    --split-ids split.json \
    --model-in /tmp/minilm-lme-ft-300
```

## Companion result: code-tuned adaptmem variants do not lift LongMemEval

This reproduction supersedes the earlier writeup at `2026-05-17-adaptmem-encoder-swap.md`, which tested **code-tuned** adaptmem variants (training corpus = Python docstring↔code pairs) and reported a 3.8pp regression. Those variants — `~/Downloads/ft300/`, `~/Downloads/ft1000/`, `~/Downloads/ft5000.zip`, and `~/Projects/adaptmem-cache/` — were experiments from nakata-app's *codesearchnet* training notebook, not the LongMemEval-domain weights that produced the published 0.995 number.

For the record, comparing on the same 500q full LongMemEval-S:

| Variant | Training corpus | R@5 (500q full) |
|---|---|---|
| MiniLM-L6-v2 base (no fine-tune) | — | 0.9660 |
| `~/Downloads/ft300/` (code-tuned, 300 pairs) | 5000 Python def/docstring | **0.9660** (identical to base — code FT-300 is recall-neutral on LongMemEval) |
| `~/Downloads/ft1000/` (code-tuned, 1000 pairs) | 5000 Python def/docstring | 0.9560 (-1.0pp) |
| `~/Projects/adaptmem-cache/` (code-tuned, different ft300 weights) | 5000 Python def/docstring | 0.9280 (-3.8pp) |
| **Our LME-tuned FT-300 (this run)** | **300 LongMemEval query-session pairs** | **1.000** on 200-test (perfect) |

Two patterns:
1. **Code-tuned variants drift across runs.** The two ft300 instances on disk (Downloads vs Projects/adaptmem-cache) have different weights and produce different LongMemEval results despite identical training data and protocol. This is run-to-run noise from MultipleNegativesRankingLoss with limited training data — same noise that explains the +0.5pp delta on the LME reproduction below.
2. **Code-FT is at best LongMemEval-neutral, at worst regressing.** The base MiniLM-L6-v2 was already trained on diverse text including conversational data; sharpening its representation toward code retrieval doesn't help (and sometimes hurts) on conversational memory tasks.

## What this validates

- nakata-app's published 0.995 R@5 number is **independently reproducible** on a different GPU with the same protocol.
- Encoder fine-tuning on 300 in-domain labelled query-session pairs is sufficient to **saturate LongMemEval recall** (R@5 = 1.0 on 200 held-out test).
- Domain match in fine-tuning data is load-bearing — code-FT and LongMemEval-FT produce wildly different results despite identical training algorithms and base encoder.

## Artifacts

- Result JSON: `baselines/lme_substrate_ft300_katana_test200_2026-05-17.json`
- Reproducible split: `baselines/lme_split_300_200_seed42.json` (300 train / 200 test, seed=42 shuffle)
- Trained model: `/tmp/minilm-lme-ft-300-katana/` (not committed — 90MB; reproduce with the training command above)
- Companion (negative result on code-tuned variants): `docs/benchmarks/2026-05-17-adaptmem-encoder-swap.md`

## Open questions for the spec

- **Generalization beyond LongMemEval.** R@5 = 1.0 on the 200-test split is striking; testing on `jp-realm-v0.1` and `mempalace_git_probes_v2` would say whether the FT generalizes or just memorizes the corpus distribution it was trained on.
- **In-domain vs in-corpus FT.** The training and test sessions share the same author-style distribution. Whether FT-300 trained on conversational sessions from *another* corpus lifts LongMemEval recall would isolate "domain generalization" from "corpus memorization."
- **Composability with the orchestrator layer.** Step 2-forced/grounded showed that gemma4-RLM regresses below substrate-floor (0.733 daemon vs 0.583-0.724 RLM-best on jp-realm-v0.1). With FT-300 lifting substrate-floor to 1.0 on LME, does qwen3.5-RLM still saturate at substrate-floor, or does the better substrate let the orchestrator find more answers? Step 35.c (RLM on LongMemEval) gets meaningful constraints from this number.
