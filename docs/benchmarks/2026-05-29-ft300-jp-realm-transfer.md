# FT-300 encoder transfer to jp-realm — robust null

**Date:** 2026-05-29
**Branch:** `feat/84-ft300-jp-realm`
**Issue:** [techempower-org/multipass-structural-memory-eval#84](https://github.com/techempower-org/multipass-structural-memory-eval/issues/84)
**Result:** **Tau2's predicted +30–33pp recall gap on jp-realm is not borne out.** Two
independently-trained fine-tuned encoders (the published FT-300, and a from-recipe
reproduction) show **no meaningful transfer** to JP's personal knowledge base — best
covered-only delta **+1.73pp at R@1**, with **R@10 actually −1.73pp**. At n=30 this is a
**null / falsification** of the cross-domain prediction, not a partial lift.

## Why this run exists

A memory note recorded a Tau2-derived prediction: a fine-tuned retrieval encoder would
open a **+30–33pp recall gap** on `jp-realm-v0.1` (see
`reference_tau2_predicts_cat9a`). #84 tests that directly with the AdaptMem **FT-300**
encoder — a `MultipleNegativesRankingLoss` fine-tune of `all-MiniLM-L6-v2`.

The standard jp-realm path (`sme-eval retrieve`) runs retrieval **server-side inside the
palace-daemon** (familiar:8085), so the encoder cannot be swapped locally. To keep this
**daemon-independent** (the #84 requirement), the bench reconstructs a fully local haystack
from a ChromaDB **palace backup** (`~/.mempalace/palace-backup-20260416-110359`, 135,399
drawers, frozen 2026-04-16, no network), embeds it with each encoder, and scores the 30
jp-realm questions with the **same `expected_sources` substring recall as `cmd_retrieve`**,
evaluated at K ∈ {1, 5, 10}.

## Provenance correction — FT-300 is a code/science fine-tune, not LongMemEval

The genuine published artifact lives at `/home/jp/Downloads/ft300/model` (`dataset_size:300`,
MNR loss, base `all-MiniLM-L6-v2`, 384d). Its model-card widget shows **code + scientific
computing** training content (e.g. *"Trial function to solve 2 eqns … Correa et al. (2015c)"*,
`def _minimize_c(...)`) — **not** the LongMemEval conversational data that the repo's
`ft-300-base/` skeleton (a separate, weightless `dataset_size:565` artifact) had implied.
Either way it carries **zero jp-realm training signal**, so the cross-corpus generalization
test stands; the honest label is "code/science-FT", not "LongMemEval-FT".

The published `nakata-app/minilm-lme-ft-300` HF repo is **404** (confirmed authenticated as
JP) — the artifact is not externally downloadable; it was sourced from JP's local Downloads.

## Results — jp-realm-v0.1, n=30 questions, 135,399-drawer local backup, top-K=10

### Headline (covered-only, n=29 — excludes 1 snapshot-uncovered question)

| Leg | Encoder | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| **A** | `all-MiniLM-L6-v2` (base / baseline) | 0.3448 | 0.5172 | **0.6207** |
| **B** | FT-300 (published, code/science-FT) | 0.3621 | 0.5172 | 0.6034 |
| **C** | FT-300-approx (LongMemEval recipe, 467 pairs, from-recipe repro) | 0.3621 | 0.4828 | 0.6207 |

**A→B delta (the headline test):** R@1 **+1.73pp**, R@5 **0.00pp**, R@10 **−1.73pp**.
Best delta across all K is **+1.73pp** — against a predicted **+30–33pp**.

Overall (all 30, including the snapshot-uncovered question): A R@10 = 0.600, B R@10 = 0.583.

### Per-question R@10 movement (covered, n=29)

- **24 / 29 questions: exactly 0.0 delta** — the FT encoders rank the same drawers as base.
- **Helped:** q07_palace_daemon_role (+0.5).
- **Hurt:** q26_caddy_dns_resolvers (−0.5), q28_postgres_migration (−0.5).

Net: a wash. No category shows the systematic lift the prediction requires.

### Leg C — recipe sensitivity

The from-recipe reproduction (`scripts/train_ft300_approx.py`, 467 pairs = 203 base + 264
synthetic-preference, 16 optimizer steps, 3.5s on an RTX 2080 Ti) is a **faithful
approximation, not bit-exact**: the original `dataset_size:565` card includes ~98 pairs from
`s2_syn_all.jsonl`, which is absent on this workstation. Leg C lands within ±2pp of base at
every K (and *drops* R@5 by ~3.4pp), agreeing with the published artifact's null. Two
independently-trained FT encoders failing to transfer is a **robust** result, not an artifact
of one checkpoint.

## Snapshot caveat

The backup is frozen at 2026-04-16, predating some jp-realm topics. The bench auto-flags
questions whose `expected_sources` appear in **no** drawer as `snapshot_uncovered` (here:
`q13_graphpalace`) and reports a `covered_only` aggregate that excludes them, so the encoder
is never blamed for content the snapshot cannot contain. q07/q23 reference palace-daemon
(newer than the snapshot) but still had partial coverage via related drawers.

## Interpretation

The Tau2 cross-domain prediction does not hold on jp-realm. A retrieval encoder fine-tuned on
an unrelated domain (code/science, or LongMemEval conversational recall) does **not** generalize
to JP's technical personal KB — base `all-MiniLM-L6-v2` is already at or above both FT encoders.
The lever the null points to is **our-corpus fine-tuning**: an encoder trained on jp-realm-shaped
data, not a transplanted FT. That is filed as the #84 follow-up.

## Reproduce

```bash
# A: baseline   B: published FT-300   C: from-recipe approx (optional)
./venv/bin/python scripts/jp_realm_encoder_swap.py --model all-MiniLM-L6-v2 \
    --json baselines/jp_realm_encoder_swap_default_2026-05-29.json
./venv/bin/python scripts/jp_realm_encoder_swap.py --model /home/jp/Downloads/ft300/model \
    --json baselines/jp_realm_encoder_swap_ft300_2026-05-29.json
./venv/bin/python scripts/train_ft300_approx.py --out baselines/ft300_approx_model   # ~4s, 16 steps
./venv/bin/python scripts/jp_realm_encoder_swap.py --model baselines/ft300_approx_model \
    --json baselines/jp_realm_encoder_swap_ft300approx_2026-05-29.json
./venv/bin/python scripts/jp_realm_encoder_delta.py \
    --a baselines/jp_realm_encoder_swap_default_2026-05-29.json \
    --b baselines/jp_realm_encoder_swap_ft300_2026-05-29.json \
    --json baselines/jp_realm_encoder_delta_2026-05-29.json
```

Trained weights (`baselines/ft300_approx_model/`, ~90MB) are gitignored; the bench JSON
results are committed. The published FT-300 lives outside the repo at
`/home/jp/Downloads/ft300/model`.
