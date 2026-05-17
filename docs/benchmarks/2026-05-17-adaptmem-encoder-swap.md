# adaptmem encoder swap on LongMemEval: -3.8pp regression vs MiniLM baseline

**Date:** 2026-05-17
**Branch:** `feat/rlm-adapter`
**Result:** Adaptmem-tuned MiniLM-L6-v2 scores **R@5 = 0.9280** on LongMemEval-S 500q — a **3.8 percentage point regression** vs the same architecture's base weights (R@5 = 0.9660).

## Setup

Single-variable swap from the just-validated substrate-floor parity (see `2026-05-17-longmemeval-substrate-parity.md`):

| | Substrate-parity bench | This run |
|---|---|---|
| Encoder | `all-MiniLM-L6-v2` base | `adaptmem-cache/model/` (same arch, fine-tuned weights) |
| Embedding dim | 384 | 384 |
| Dataset | `longmemeval_s_cleaned.json` (500q) | same |
| Ingest content | user turns only, no metadata | same |
| Scoring | recall_any@5 on `answer_session_ids` | same |
| Retrieval | brute-force cosine (or postgres pgvector — both gave 0.9660) | brute-force cosine via numpy |

Encoder swap is the only variable. Same input text → same indexing → same scoring → different embedding model.

## Per-category results

| Question type | n | baseline (R@5) | adaptmem (R@5) | Δ |
|---|---:|---:|---:|---:|
| knowledge-update | 78 | 1.0000 | 0.9744 | **-2.6pp** |
| multi-session | 133 | 0.9925 | 0.9624 | **-3.0pp** |
| single-session-assistant | 56 | 0.9643 | 0.9107 | **-5.4pp** |
| single-session-preference | 30 | 0.9667 | 0.9000 | **-6.7pp** |
| single-session-user | 70 | 0.9143 | 0.8143 | **-10.0pp** |
| temporal-reasoning | 133 | 0.9474 | 0.9398 | -0.8pp |
| **OVERALL** | **500** | **0.9660** | **0.9280** | **-3.8pp** |

**Adaptmem regresses across every category.** Largest drops on single-session-user (-10.0pp), single-session-preference (-6.7pp), and single-session-assistant (-5.4pp) — the three categories closest to "user describes themselves and asks a question about it later," which is the exact retrieval shape conversational memory systems exist to handle.

## Why: domain shift in fine-tuning data

The model card at `~/Projects/adaptmem-cache/model/README.md` shows adaptmem was fine-tuned with `MultipleNegativesRankingLoss` on **Python docstring ↔ code pairs** (5000 training pairs). Sample source sentences from the training data include things like:

- "카카오톡 전송내역 팝업 URL" → Python function returning a KakaoTalk send-history URL
- "Creates a Chapter object from a url" → Python function that scrapes a webpage into a Chapter
- "ignore a set of tokens with specific names" → Python token-ignoring decorator

The fine-tuning task was **align natural-language descriptions to Python code**. The LongMemEval task is **align a user's question to a past conversational session** — different domain, different prose register, different evidence shape.

The result is a textbook case of **negative transfer** in encoder fine-tuning: by sharpening the model's representation of code-vs-description distinctions, it lost calibration on user-conversation-vs-other-conversation distinctions. The base MiniLM-L6-v2 was trained on a large, diverse corpus including conversational text; adaptmem traded that breadth for code-retrieval specialization.

## Implication for SME

The "encoder swap" lever in memory-system benchmarks is **not a one-way upgrade.** It is a *domain-matching* decision. A model labeled "domain-tuned" or "fine-tuned" can systematically underperform the base model when:

1. The test domain ≠ the fine-tuning domain
2. The fine-tuning task narrows the representation in a way that the test task needed

This is worth a paragraph in the SME methodology spec under "encoder selection" — the cross-validation protocol should always include the base model as a control, not just the candidate encoder.

## Cost-structure observation

The brute-force numpy cosine bench ran in **41.6s** (12.0 q/s) — vs **22 minutes** (0.38 q/s) for the postgres+pgvector parity bench on identical input. Roughly 32× speedup. For LongMemEval-shape per-question vaults (50–200 sessions each), 95%+ of postgres bench time is TRUNCATE+UPSERT+index-maintenance overhead, not actual KNN search. Indexed vector DBs only earn their keep when corpus reuse amortizes the index cost across many queries — exactly the production-palace setup, not per-question microbenches.

## Important caveat — this is NOT nakata-app's published FT-300

The model tested here is the adaptmem variant cached locally at `~/Projects/adaptmem-cache/`. Inspecting `corpus.tsv` shows 5000 Python-code training pairs. The published `nakata-app/adaptmem FT-300` (the 0.9950 R@5 number this task originally intended to validate) is presumably a different variant fine-tuned on conversational data.

This run therefore measures:
- ✅ "The code-domain adaptmem variant present on katana regresses on LongMemEval" — confirmed.
- ❌ "nakata-app's published FT-300 doesn't reproduce" — **not** what was tested. That validation still requires the actual FT-300 weights.

The methodology lesson (negative transfer is real; encoder-domain matching matters) holds regardless. But the headline shouldn't be "adaptmem regresses" — it should be "this specific code-tuned variant regresses, and we haven't yet pulled the published conversational variant."

## Artifacts

- Bench script: `scripts/lme_substrate_adaptmem_bench.py`
- Results: `baselines/lme_substrate_adaptmem_2026-05-17.json`
- Adaptmem variant used: `~/Projects/adaptmem-cache/model/` (90MB, fine-tuned MiniLM-L6-v2 on `corpus.tsv` = 5000 Python code pairs)
- Variant NOT yet tested: `nakata-app/adaptmem FT-300` (the conversational variant claimed to reach R@5 ≈ 0.9950)
