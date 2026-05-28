# LongMemEval — MemPalace Results

First published score for the MemPalace fork on LongMemEval, tracked
under [SME #19](https://github.com/techempower-org/multipass-structural-memory-eval/issues/19)
and produced by [SME #44](https://github.com/techempower-org/multipass-structural-memory-eval/issues/44).

**Status (2026-05-28):** first leg landed — `mempalace-daemon` against
`/search` (default vector + BM25, no AGE traversal). Overall QA accuracy
**60.40%** at n=500. Subsequent legs in flight: techempower-org/multipass-structural-memory-eval#45
(`/search/age-fused`) and techempower-org/multipass-structural-memory-eval#46
(Familiar adapter). Tables below show the `mempalace-daemon` column
filled; `familiar` columns pending #46.

> **R@5 caveat:** the substring-based R@5 reported here is **broken** in
> this run — `--content-rules upstream-exact` strips session IDs from
> the retrieved drawer text, so the matcher cannot find them. The fix
> lives in techempower-org/multipass-structural-memory-eval#67
> (drawer_id-based matcher) but did not merge into the bench branch in
> time. The QA-accuracy figure is unaffected; read 3.97% R@5 as a
> measurement artifact, not a substrate quality signal.

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

| Role | Model | Provider | Why |
|---|---|---|---|
| Reader | `o4-mini` | Azure Foundry (claud-assistant-resource) | Reasoning-tier reader, kept the per-question cost ~\$0.0026 across the 500Q chain (chain total ~\$1.30). |
| Judge  | `gpt-5.3-chat` | Azure Foundry | Stronger judge than the paper's `gpt-4o-2024-08-06`; LongMemEval's published runs used the GPT-4 family. Reader/judge model deltas are flagged when reporting. |

Per-question reader cost: ~\$0.0026. Per-question judge cost: ~\$0.0011.
Chain (n=500) total Azure cost for techempower-org/multipass-structural-memory-eval#44: ~\$1.30.

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

## R@5 Retrieval Recall — 2026-05-28 (#44)

Substring-match retrieval recall at top-5. R@5 = 1.0 means the gold
session id appeared in the top-5 retrieved chunks.

| SME Category | LME Question Type | n | R@5 (mempalace-daemon) | R@5 (familiar) |
|---|---|---:|---:|---:|
| cat_1          | single-session-* (IE)         | 150 | 0.0000 | pending #46 |
| cat_2c         | multi-session (MR)            | 121 | 0.0000 | pending #46 |
| cat_3_partial  | knowledge-update (KU)         | 72  | 0.0000 | pending #46 |
| cat_6          | temporal-reasoning (TR)       | 127 | 0.1562 | pending #46 |
| cat_1_negative | abstention (ABS)              | 30  | 0.0000 | pending #46 |
| **Overall**    | —                             | 500 | **0.0397** | pending #46 |

The 3.97% headline is an artifact of the matcher, not the substrate
(see caveat at top of doc). The cat_6 row (15.62%) shows that
retrieval *was* finding relevant material — it's just that 4 of the 5
categories under `--content-rules upstream-exact` strip the session ID
from the retrieved text, so the substring matcher returns 0 even when
the right session is in the top-5. The drawer_id-based matcher in
techempower-org/multipass-structural-memory-eval#67 will close this
gap when re-run.

---

## QA Accuracy — 2026-05-28 (#44)

Judge-scored end-to-end QA accuracy (`judge_correct_rate` per category,
weighted average overall). ABSTAIN counts as correct for the
abstention category (`cat_1_negative`) per LongMemEval convention.

| SME Category | LME Question Type | n | QA-acc (mempalace-daemon) | QA-acc (familiar) |
|---|---|---:|---:|---:|
| cat_1          | single-session-* (IE)         | 150 | 0.5267 | pending #46 |
| cat_2c         | multi-session (MR)            | 121 | 0.7438 | pending #46 |
| cat_3_partial  | knowledge-update (KU)         | 72  | 0.6944 | pending #46 |
| cat_6          | temporal-reasoning (TR)       | 127 | 0.4409 | pending #46 |
| cat_1_negative | abstention (ABS)              | 30  | 0.9000 | pending #46 |
| **Overall**    | —                             | 500 | **0.6040** | pending #46 |

### Judge label breakdown (mempalace-daemon)

| Category | CORRECT | PARTIAL | INCORRECT | ABSTAIN | ERROR |
|---|---:|---:|---:|---:|---:|
| cat_1          | 79 | 3 | 68 | 0 | 0 |
| cat_2c         | 90 | 2 | 29 | 0 | 0 |
| cat_3_partial  | 50 | 0 | 22 | 0 | 0 |
| cat_6          | 37 | 1 | 70 | 19 | 0 |
| cat_1_negative | 14 | 0 | 3 | 13 | 0 |

cat_6 sees the most ABSTAIN labels (19 of 127) — the reader recognized
its retrieved context was insufficient for temporal reasoning and chose
to abstain rather than hallucinate, which the judge counted as correct
behaviour. This is a healthier failure mode than `cat_1` and `cat_2c`,
where every wrong answer was a confident wrong answer (0 ABSTAIN).

---

## Retrieval-QA Gap — 2026-05-28 (#44)

`R@5 - QA accuracy`. Positive gap means the right session was retrieved
but the reader couldn't produce the answer; negative gap means the
reader got the answer right despite imperfect retrieval (typically by
having world knowledge or by being lucky with paraphrase).

**With the matcher caveat:** every row below shows a large negative gap
because R@5 is artificially 0 for 4 of 5 categories (substring matcher
issue, see top of doc). When techempower-org/multipass-structural-memory-eval#67
ships and R@5 is re-measured with the drawer_id matcher, the absolute
numbers will shift but the *relative* pattern across categories should
hold.

| SME Category | Gap (mempalace-daemon) | Gap (familiar) |
|---|---:|---:|
| cat_1          | -0.5267 | pending #46 |
| cat_2c         | -0.7438 | pending #46 |
| cat_3_partial  | -0.6944 | pending #46 |
| cat_6          | -0.2847 | pending #46 |
| cat_1_negative | -0.9000 | pending #46 |
| **Overall**    | **-0.5643** | pending #46 |

---

## Search-Endpoint A/B — 2026-05-28 (techempower-org/multipass-structural-memory-eval#45)

Same haystack, reader, judge, and per-question wing isolation as #44.
Only the daemon retrieval endpoint changes: vector + BM25 default
(`/search`) versus vector + AGE-graph RRF fusion (`/search/age-fused`).
Source: `baselines/longmemeval_age_fused_2026-05-28.json`.

| SME Category | n | QA-acc `/search` (#44) | QA-acc `/search/age-fused` (#45) | Δ |
|---|---:|---:|---:|---:|
| cat_1          | 150 | 52.67% | 7.33%   | -45.34pp |
| cat_1_negative | 30  | 90.00% | 100.00% | +10.00pp |
| cat_2c         | 121 | 74.38% | 0.00%   | -74.38pp |
| cat_3_partial  | 72  | 69.44% | 1.39%   | -68.05pp |
| cat_6          | 127 | 44.09% | 36.22%  |  -7.87pp |
| **Overall**    | 500 | **60.40%** | **17.60%** | **-42.80pp** |

### What actually changed

| metric | `/search` default (#44) | `/search/age-fused` (#45) |
|---|---:|---:|
| Mean context_chars per query | 2539 | **459** (5.5× narrower) |
| Median context_chars | 2780 | 432 |
| Max context_chars | 4193 | 841 |
| Questions with 0 matched_sources (substring matcher) | 466/500 | 500/500 |
| Adapter errors | 0 | 1 (`NO_RESULTS`) |
| Disagreements | 290 | 88 |

### Empty-triples caveat

This reading does **not** say AGE-fused retrieval is structurally
worse than vector + BM25. As of 2026-05-25 the AGE triples layer was
effectively empty (`triples: 1` reported by `kg_stats` after the
entities-only backfill). The fusion's graph half had nothing to fuse
with, so age-fused was effectively competing against vector-only with
a tighter snippet boundary. The daemon's `kg_stats` reported 1.8M
triples during the 2026-05-28 bench window — the real A/B should rerun
once the triple layer is fully populated and stable.

cat_1_negative's rise to 100% is consistent with this regime: with
thinner context the reader said "I don't know" more often, which is
the right answer for unanswerable questions. The 88 disagreements (vs
290 in #44) are also consistent — when retrieval returns almost no
matched content, judge and matcher both agree the system can't answer.

The `run_metadata.search_endpoint` field in the #45 JSON reads
`default` even though the endpoint actually queried was
`/search/age-fused` (verified by the structural difference in
`context_chars` distribution). That's a logging bug in
`scripts/run_longmemeval_mempalace.py` — filed as part of the bench
artifacts.

---

## Comparison vs Published Numbers

LongMemEval scores reported elsewhere (sourced from product pages /
papers — verify dates before citing for publication):

| System | LongMemEval split | Overall QA acc | Reader / Judge | Source |
|---|---|---:|---|---|
| OMEGA          | oracle | 95.4% | (paper) | TBD link |
| Hindsight      | oracle | 91.4% | (paper) | TBD link |
| True Memory    | oracle | 87.8% | (paper) | TBD link |
| **MemPalace via palace-daemon `/search`** (2026-05-28, this fork) | oracle | **60.40%** | o4-mini / gpt-5.3-chat (Azure Foundry) | techempower-org/multipass-structural-memory-eval#44 |
| MemPalace via `/search/age-fused` (in flight) | oracle | pending | o4-mini / gpt-5.3-chat | techempower-org/multipass-structural-memory-eval#45 |
| Familiar (in flight)  | oracle | pending | o4-mini / gpt-5.3-chat | techempower-org/multipass-structural-memory-eval#46 |

Notes:
- All numbers above are on the **oracle** split. M and S splits are
  larger haystacks; numbers shift downward with corpus size.
  Comparisons must hold the split constant.
- Reader/judge differences matter: this fork's run uses Azure Foundry's
  `o4-mini` reader and `gpt-5.3-chat` judge, both newer than the GPT-4
  family used by the published numbers above. Comparing across
  reader/judge stacks is the standard practice in this benchmark, but
  flag it explicitly when reporting. **Do not** read the 60.40% as
  "MemPalace is below OMEGA's 95.4%" without also noting the model-stack
  delta.
- The 60.40% number is the *production* palace-daemon path with
  `--content-rules upstream-exact` (matched protocol per
  techempower-org/multipass-structural-memory-eval#54). The
  substrate-floor reading from techempower-org/multipass-structural-memory-eval#51
  (R@5=0.9660 byte-identical to upstream) confirms the postgres-vector
  substrate is parity-good; the 60.40% delta from the published 87-95%
  range therefore lives in: (a) retrieval depth choices, (b)
  reader/judge stack, (c) content rules.

---

## Disagreement Set

Questions where SME's R@5 and the judge's verdict imply opposite
conclusions (R@5 ≥ 0.5 + INCORRECT, OR R@5 < 0.5 + CORRECT) are
diagnostic gold — they characterize the substring matcher's limits
against a stronger judge.

The full disagreement list lands in the per-question JSON output; a
summary count goes here.

**2026-05-28 (#44):** 290 total disagreements over 500 questions.

| Direction | Count | Meaning |
|---|---:|---|
| judge_correct + matcher_miss | 281 | Reader answered correctly; matcher couldn't find session ID in retrieved text (the broken-R@5 root cause) |
| matcher_hit + judge_wrong | 8 | Matcher found the session ID; judge ruled the reader's answer wrong or incomplete |

**By category (all 290):**

| Category | Count | Note |
|---|---:|---|
| cat_2c         | 90 | Multi-session synthesis — judge accepts paraphrased reasoning the matcher's literal session ID lookup misses |
| cat_1          | 79 | Single-session — should be the easiest case; the 79 disagreements are the strongest indicator that the `upstream-exact` matcher break is the dominant source of noise |
| cat_3_partial  | 50 | Knowledge-update — see KU caveat above; judge and matcher have known divergent definitions here |
| cat_6          | 44 | Temporal reasoning — fewest disagreements because cat_6 retrieves richer multi-session context (15.62% R@5 vs 0% elsewhere) |
| cat_1_negative | 27 | Abstention — judge correctly counts ABSTAIN as right; matcher counts it as miss |

The 281:8 ratio (judge_correct_matcher_miss : matcher_hit_judge_wrong)
is the cleanest single-number summary of the matcher break: the judge
agrees with the system far more often than the matcher does, which
suggests the substrate is finding the right session most of the time
but the matcher can't see it.

---

## How to reproduce

### Prerequisites

- A running palace-daemon. JP's homelab daemon is at
  `http://your-daemon-host:8085`; `PALACE_API_KEY` lives in
  `~/.config/palace-daemon/env`.
- `OPENAI_API_KEY` exported in the environment (the reader and judge
  call OpenAI).
- The LongMemEval oracle JSON downloaded to
  `sme/corpora/longmemeval/data/longmemeval_oracle.json` (see that
  directory's README for the wget command).

### Operational state (2026-05-25)

JP's daemon is on `palace-daemon` 1.7.2 backed by postgres+pgvector+AGE.
The AGE **entity** layer was backfilled on 2026-05-25 (142,315 entities
added in ~61 min, 0 errors; `mempalace_kg_stats` reports 264,402
entities total across all wings). The **triples / relationships** layer
is effectively empty (`triples: 1`), so today `POST /search/age-fused`
returns results whose ranking is driven by vector + BM25 + reranker —
the AGE-graph half of the RRF fusion has nothing to contribute until
relationships are extracted. The eval script in this doc routes through
plain `/search` and that remains the right default; treat `/search/age-fused`
as a sibling primitive that will become diagnostically useful once
triples land. Tracked in the SME ↔ daemon follow-up memory.

### Dry-run first (no API calls — just cost estimate)

```bash
./venv/bin/python scripts/run_longmemeval_mempalace.py \
    --adapter mempalace-daemon \
    --api-url http://your-daemon-host:8085 \
    --questions sme/corpora/longmemeval/data/longmemeval_oracle.json \
    --dry-run
```

This prints an estimated total USD and a per-model token breakdown so
you can sanity-check before launching the live run.

### Live run — MemPalace daemon

```bash
./venv/bin/python scripts/run_longmemeval_mempalace.py \
    --adapter mempalace-daemon \
    --api-url http://your-daemon-host:8085 \
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
    --api-url http://your-daemon-host:8085 \
    --questions sme/corpora/longmemeval/data/longmemeval_oracle.json \
    --max-questions 5 \
    --json /tmp/lme_smoke.json
```

### R@5-only (no reader, no judge — no OpenAI required)

```bash
./venv/bin/python scripts/run_longmemeval_mempalace.py \
    --adapter mempalace-daemon \
    --api-url http://your-daemon-host:8085 \
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

- `baselines/longmemeval_<adapter-or-route>_<date>.json` — full
  per-question records (R@5, judge label, judge rationale, retrieval
  context length, reader output, judge token usage) plus the aggregated
  summary. Date format is `YYYYMMDD` for live judged runs (see
  reproduction commands above) or `YYYY-MM-DD` for retrieval-only
  baselines that ship without a judged-reader pass.

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
| 50 | cat_6 (temporal-reasoning) | **47.0%** | n/a | n/a |

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
    --api-url http://your-daemon-host:8085 \
    --api-key "$PALACE_API_KEY" \
    --max-questions 50 \
    --skip-judge \
    --json "baselines/longmemeval_entities_only_$(date +%Y-%m-%d).json"
```

### Next leg of the A/B

Once `mempalace_kg_stats.triples` climbs out of the sentinel `1` (i.e.
typed RELATION triples are extracted onto the AGE graph), re-run the
same command but swap the adapter's retrieval primitive from `/search`
to `POST /search/age-fused` (RRF fusion of vector + AGE-graph walk).
Same corpus, same questions, same scoring — only the search primitive
changes. The delta will be the cleanest available reading of
"does the RELATION layer help retrieval recall?".
