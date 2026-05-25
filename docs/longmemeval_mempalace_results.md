# LongMemEval — MemPalace Results

This is the results template for [SME #19 — Run MemPalace fork through
LongMemEval E2E QA](https://github.com/techempower-org/multipass-structural-memory-eval/issues/19).
The first published score for the MemPalace fork on LongMemEval lands
here once the run completes.

**Status:** scaffolded — awaiting first live run. Tables below contain
placeholders. The exact CLI commands that produce these numbers are
documented at the bottom.

---

## Methodology

### Dataset

- **LongMemEval oracle** (`longmemeval_oracle.json`) — 500 questions,
  ~15 MB; the smallest of the three official splits. Each question
  ships with only its evidence sessions, so the haystack per question
  is small (~3-6 sessions) and the per-question ingest cost is
  negligible. Source: [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned),
  MIT-licensed.
- Oracle (not S or M) is the conservative starting point: it lets us
  publish a number first, then climb the haystack-size ladder. Running
  S (~115K tokens per question) and M (~1.5M tokens per question) is
  follow-up scope.

### Pipeline

For each LongMemEval question:

1. **Ingest** — POST each haystack session as one drawer to the daemon's
   `/memory` endpoint under wing `lme_<question_id>`, room `longmemeval`.
   The per-question wing isolates each question's haystack so prior
   questions can't leak into the current question's retrieval.
2. **Retrieve** — query the daemon's `/search?wing=lme_<question_id>`
   with the question text, top-K=5. Returns up to 5 drawers' text.
3. **Read** — feed the retrieved context plus the question to the
   reader model (default `gpt-4.1-mini`), receive a natural-language
   answer.
4. **Judge** — feed `(question, gold_answer, system_answer,
   question_type)` to the LongMemEval GPT-4o judge
   (`gpt-4o-2024-08-06`), receive `CORRECT | PARTIAL | INCORRECT |
   ABSTAIN | ERROR`. The judge prompt is question-type-specific (see
   `sme/eval/longmemeval_judge.py`).
5. **Score**:
   - **R@5** — substring match: does the gold session id appear in the
     top-5 retrieved chunks? Cheap and deterministic; SME's existing
     metric.
   - **QA accuracy** — fraction of judged questions where the label is
     `CORRECT` or `ABSTAIN`. This is the LongMemEval-canonical number,
     directly comparable to OMEGA / Hindsight / True Memory.
   - **Retrieval-QA gap** — `R@5 - QA accuracy`. Isolates retrieval
     quality from answer-generation quality; a large positive gap
     means the right session was retrieved but the reader couldn't
     piece the answer together.

### Models

| Role | Model | Why |
|---|---|---|
| Reader | `gpt-4.1-mini` | Cheap, good enough for multi-session synthesis. |
| Judge  | `gpt-4o-2024-08-06` | The LongMemEval canonical judge (paper §4). |

### KU caveat

Per-category numbers are reported separately by design. LongMemEval's
**Knowledge Update (KU)** task rewards returning the *new* (post-overwrite)
value; SME's **Cat 3** rewards *flagging both old and new*. A
silent-overwriter scores better on KU than a contradiction-surfacing
system. Reporting a single overall correlation would mislead — see
`docs/related_work/longmemeval.md` for the divergence analysis.

The dual-metric aggregator emits an `overall` row anyway because the
LongMemEval published numbers report a single overall accuracy; that
overall row is the number directly comparable to OMEGA / Hindsight /
True Memory. Per-category rows are the SME-internal diagnostic.

---

## R@5 Retrieval Recall — placeholder

Substring-match retrieval recall at top-5. R@5 = 1.0 means the gold
session id appeared in the top-5 retrieved chunks.

| SME Category | LME Question Type | n | R@5 (mempalace-daemon) | R@5 (familiar) |
|---|---|---:|---:|---:|
| cat_1          | single-session-* (IE)         | TBD | TBD | TBD |
| cat_2c         | multi-session (MR)            | TBD | TBD | TBD |
| cat_3_partial  | knowledge-update (KU)         | TBD | TBD | TBD |
| cat_6          | temporal-reasoning (TR)       | TBD | TBD | TBD |
| cat_1_negative | abstention (ABS)              | TBD | TBD | TBD |
| **Overall**    | —                             | 500 | **TBD** | **TBD** |

---

## QA Accuracy — placeholder

Judge-scored end-to-end QA accuracy. This is the number directly
comparable to published LongMemEval scores.

| SME Category | LME Question Type | n | QA-acc (mempalace-daemon) | QA-acc (familiar) |
|---|---|---:|---:|---:|
| cat_1          | single-session-* (IE)         | TBD | TBD | TBD |
| cat_2c         | multi-session (MR)            | TBD | TBD | TBD |
| cat_3_partial  | knowledge-update (KU)         | TBD | TBD | TBD |
| cat_6          | temporal-reasoning (TR)       | TBD | TBD | TBD |
| cat_1_negative | abstention (ABS)              | TBD | TBD | TBD |
| **Overall**    | —                             | 500 | **TBD** | **TBD** |

---

## Retrieval-QA Gap — placeholder

`R@5 - QA accuracy`. Positive gap means the right session was retrieved
but the reader couldn't produce the answer; negative gap means the
reader got the answer right despite imperfect retrieval (typically by
having world knowledge or by being lucky with paraphrase).

| SME Category | Gap (mempalace-daemon) | Gap (familiar) |
|---|---:|---:|
| cat_1          | TBD | TBD |
| cat_2c         | TBD | TBD |
| cat_3_partial  | TBD | TBD |
| cat_6          | TBD | TBD |
| cat_1_negative | TBD | TBD |
| **Overall**    | **TBD** | **TBD** |

---

## Comparison vs Published Numbers

LongMemEval scores reported elsewhere (sourced from product pages /
papers — verify dates before citing for publication):

| System | LongMemEval split | Overall QA acc | Reader / Judge | Source |
|---|---|---:|---|---|
| OMEGA          | oracle | 95.4% | (paper) | TBD link |
| Hindsight      | oracle | 91.4% | (paper) | TBD link |
| True Memory    | oracle | 87.8% | (paper) | TBD link |
| MemPalace (this run) | oracle | **TBD** | gpt-4.1-mini / gpt-4o-2024-08-06 | this doc |
| Familiar (this run)  | oracle | **TBD** | gpt-4.1-mini / gpt-4o-2024-08-06 | this doc |

Notes:
- All published numbers above are on the **oracle** split. M and S
  splits are larger haystacks; numbers shift downward with corpus size.
  Comparisons must hold the split constant.
- Reader/judge choice affects QA accuracy. The published numbers above
  use a mix of reader/judge models; comparing systems under different
  reader/judge stacks is the standard practice in this benchmark, but
  flag it explicitly when reporting.

---

## Disagreement Set

Questions where SME's R@5 and the judge's verdict imply opposite
conclusions (R@5 ≥ 0.5 + INCORRECT, OR R@5 < 0.5 + CORRECT) are
diagnostic gold — they characterize the substring matcher's limits
against a stronger judge.

The full disagreement list lands in the per-question JSON output; a
summary count goes here:

- Total disagreements: **TBD**
- Mostly Cat 3 (KU divergence): **TBD**
- Mostly Cat 2c (synthesis paraphrase): **TBD**

---

## How to reproduce

### Prerequisites

- A running palace-daemon. JP's homelab daemon is at
  `http://disks.jphe.in:8085`; `PALACE_API_KEY` lives in
  `~/.config/palace-daemon/env`.
- `OPENAI_API_KEY` exported in the environment (the reader and judge
  call OpenAI).
- The LongMemEval oracle JSON downloaded to
  `sme/corpora/longmemeval/data/longmemeval_oracle.json` (see that
  directory's README for the wget command).

### Dry-run first (no API calls — just cost estimate)

```bash
./venv/bin/python scripts/run_longmemeval_mempalace.py \
    --adapter mempalace-daemon \
    --api-url http://disks.jphe.in:8085 \
    --questions sme/corpora/longmemeval/data/longmemeval_oracle.json \
    --dry-run
```

This prints an estimated total USD and a per-model token breakdown so
you can sanity-check before launching the live run.

### Live run — MemPalace daemon

```bash
./venv/bin/python scripts/run_longmemeval_mempalace.py \
    --adapter mempalace-daemon \
    --api-url http://disks.jphe.in:8085 \
    --questions sme/corpora/longmemeval/data/longmemeval_oracle.json \
    --json baselines/longmemeval_mempalace_daemon_$(date +%Y%m%d).json
```

The PALACE_API_KEY is read from the environment automatically. Use
`--api-key KEY` to override.

### Live run — Familiar

```bash
./venv/bin/python scripts/run_longmemeval_mempalace.py \
    --adapter familiar \
    --familiar-url http://familiar.realm.watch:8080 \
    --questions sme/corpora/longmemeval/data/longmemeval_oracle.json \
    --json baselines/longmemeval_familiar_$(date +%Y%m%d).json
```

### Smoke test — first 5 questions

Use this to verify the pipeline end-to-end without burning the full
run's API budget:

```bash
./venv/bin/python scripts/run_longmemeval_mempalace.py \
    --adapter mempalace-daemon \
    --api-url http://disks.jphe.in:8085 \
    --questions sme/corpora/longmemeval/data/longmemeval_oracle.json \
    --max-questions 5 \
    --json /tmp/lme_smoke.json
```

### R@5-only (no reader, no judge — no OpenAI required)

```bash
./venv/bin/python scripts/run_longmemeval_mempalace.py \
    --adapter mempalace-daemon \
    --api-url http://disks.jphe.in:8085 \
    --questions sme/corpora/longmemeval/data/longmemeval_oracle.json \
    --skip-judge \
    --json baselines/longmemeval_mempalace_daemon_r5only_$(date +%Y%m%d).json
```

### Equivalent CLI subcommand

The same machinery is available via `sme-eval longmemeval` for the
non-daemon adapters (full-context, flat, karpathy-compiled, mempalace,
mempalace-daemon). The run script in this doc adds the per-question
daemon ingest and the dry-run estimator; the CLI subcommand assumes the
corpus is already in the palace.

---

## Estimated cost

The dry-run path emits the canonical estimate at runtime. As a planning
floor for the full 500-question oracle run with the default reader
(`gpt-4.1-mini`) and judge (`gpt-4o-2024-08-06`), measured 2026-05-24
against the script's char→token approximation and OpenAI's list prices:

- **Reader (gpt-4.1-mini):** ~$0.34 (761K input tokens, 25K output)
- **Judge  (gpt-4o):**       ~$0.47 (150K input tokens, 10K output)
- **Total per adapter:** ~**$0.82** for 500 questions

A full run across both MemPalace + Familiar lands in the **$2 range**.
Re-run the `--dry-run` command before launching to get the current
estimate against today's pricing — the script's pricing table is in
`_MODEL_PRICING_USD_PER_M_TOKENS` and should be updated when OpenAI
moves rates.

---

## Files produced

Each live run writes one JSON file:

- `baselines/longmemeval_<adapter>_<YYYYMMDD>.json` — full per-question
  records (R@5, judge label, judge rationale, retrieval context length,
  reader output, judge token usage) plus the aggregated summary.

The disagreement set lives under `summary.disagreements` in the same
file.

---

## Open follow-ups

- **Familiar wing scoping.** Familiar's `/api/familiar/eval` endpoint
  doesn't yet accept a wing scope, so the familiar run currently shares
  one palace across all 500 questions. This means familiar's numbers
  here are best-case (no cross-question contamination is guaranteed,
  but is unlikely with the per-question wing prefix because familiar's
  reranker scores against the question itself). Track and tighten this
  in familiar.realm.watch once it adds wing scoping to the eval
  endpoint.
- **S and M splits.** Once the oracle number is published, climb the
  haystack-size ladder — S (~115K tokens per Q), then M (~1.5M tokens
  per Q). The scale curve is the structural reading SME exists to
  surface.
- **Cross-validation against published numbers.** OMEGA, Hindsight,
  and True Memory used different reader/judge stacks than the defaults
  here. Add `--answer-model` / `--judge` swaps for an apples-to-apples
  re-run if the comparison demands it.

---

## Entities-only baseline reading — 2026-05-25 (n=50, cat_6 only)

This reading was captured as the "before" leg of an upcoming A/B that
isolates the value of typed RELATION triples in the AGE knowledge
graph. At capture time the daemon's substrate was:

- **Entity nodes:** ~264,800 (post entity-extraction backfill, 99.6% of
  drawers checkpointed).
- **MENTIONS edges:** drawer → entity (full mesh from the regex
  extractor).
- **RELATION triples:** 1 (sentinel only — typed subject-predicate-object
  facts not yet extracted).
- **Search primitive used:** `GET /search` (vector + BM25). The
  age-fused fusion (`POST /search/age-fused`) was not exercised because
  the RELATION layer it relies on is empty today.

### Result

| n  | Category                | R@5    | QA-acc | Gap |
|---:|-------------------------|-------:|-------:|----:|
| 50 | cat_6 (temporal-reason) | **47.0%** | n/a    | n/a |

`--skip-judge` mode — R@5 is substring-match against gold session id;
no reader, no judge, no OpenAI cost. Output:
`baselines/longmemeval_entities_only_2026-05-25.json`.

### Methodology note — why this corrects the prior smoke readings

Earlier smoke runs reported R@5 = 0.733 (n=5, 2026-04-26 ChromaDB) and
R@5 = 0.817 (n=20, 2026-05-15 postgres+AGE) on the same first-N
questions of `longmemeval_oracle.json`. The n=50 reading at 47.0%
**drops 26-35pp** from those smoke numbers despite drawing from a
superset of the same questions.

The first 50 questions in the oracle file are all `cat_6`
(temporal-reasoning), so this isn't a category-mix shift. The most
parsimonious explanation is that the smoke samples were
upward-biased — the easier `cat_6` questions cluster early in the
file. This is exactly the failure mode that the `n≥25` threshold rule
exists to prevent.

**Implication for the prior "+3.3pp ChromaDB → postgres+AGE" claim:**
both readings used in that delta (0.700 and 0.733) were at n=5 — below
threshold and bias-prone. Treat that claim as withdrawn pending a
matched n≥50 reading on a comparable substrate; ChromaDB has been
retired so the comparison cannot be re-run, but the claim should not
be cited as evidence of substrate-level retrieval lift.

### How to reproduce

```bash
./venv/bin/python scripts/run_longmemeval_mempalace.py \
    --adapter mempalace-daemon \
    --questions sme/corpora/longmemeval/data/longmemeval_oracle.json \
    --api-url http://familiar.jphe.in:8085 \
    --api-key "$PALACE_API_KEY" \
    --max-questions 50 \
    --skip-judge \
    --json baselines/longmemeval_entities_only_<date>.json
```

### Next leg of the A/B

Once `mempalace_kg_stats.triples` climbs out of the sentinel `1` (i.e.
typed RELATION triples are extracted onto the AGE graph), re-run the
same command but swap the adapter's retrieval primitive from `/search`
to `POST /search/age-fused` (RRF fusion of vector + AGE-graph walk).
Same corpus, same questions, same scoring — only the search primitive
changes. The delta will be the cleanest available reading of
"does the RELATION layer help retrieval recall?".
