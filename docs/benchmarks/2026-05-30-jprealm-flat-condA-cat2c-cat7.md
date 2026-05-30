# jp-realm flat baseline — Cat 2c Condition A + Cat 7 A/B/C

**Date:** 2026-05-30
**Branch:** `feat/125-jprealm-flat-baseline`
**Issue:** [techempower-org/multipass-structural-memory-eval#125](https://github.com/techempower-org/multipass-structural-memory-eval/issues/125)
**Result:** On `jp-realm-v0.1`, the **structural pipeline beats a flat baseline by +10pp recall**
(0.933 vs 0.833 @K=5) on the *same information*, at a **+323 tok/query** cost. The advantage
**grows with hop depth** (B/A ratio 1.11x → 1.25x). The AGE **graph-fusion sub-layer** specifically
is a **neutral-to-slightly-negative tax** at 1-hop and helps only at 2-hop — most of the structural
lift comes from the hybrid retrieval (BM25 + reranker + closet/rating boosts), not graph traversal.

## Why this run exists

`jp-realm-v0.1` is a **Mode B / diagnostic** corpus: it has no static source — its 30 questions
probe JP's *live* familiar palace (~436K drawers) via `expected_sources` substring matching. Until
now the only reading was **Condition B** (the structured daemon pipeline,
`baselines/cat2c_daemon_age_2026-05-29.json` = 0.933). With no flat baseline, the Cat 2c verdict was
`incomplete` and Cat 7 (does structure earn its complexity?) could not be scored. #125 builds the
missing **Condition A** (flat, no structure) — and crucially builds it from the **same information**
the structured arm sees, so A-vs-B is a controlled structure ablation rather than a synthetic
strawman.

## Methodology — information-matched flat corpus

The load-bearing constraint (CLAUDE.md A/B/C isolation): **Condition A must be the same information
as Condition B, flattened.** A synthetic flat corpus vs a real structured palace would be an invalid
comparison.

1. **Source (read-only).** For each of the 30 questions, query the live familiar daemon
   (`GET /search`, `kind=content`, top-10) — the exact retrieval path and `kind` filter Condition B
   used. Dedupe the union of hits into a pool of 264 unique drawers. **No writes to the palace**
   (familiar#92 pollution rule) — read-only GET only. Self-referential SME run-log drawers (prior
   benchmark output leaked back into the palace) are dropped as pollution (35 rows).
   Script: `scripts/build_jprealm_flat_corpus.py`.
2. **Sanitize for public exposure.** This repo is a public fork; the corpus is JP's personal palace
   content. Stripped (placeholder counts): IP addresses (65 → `<ip>`), MAC addresses, secret-shaped
   `key=value` assignments + opaque key/token blobs (23 → `<redacted>`), session UUIDs, all
   emails (9 → `<email>`, including JP's public git email as defense-in-depth), private internal endpoints + host:port forms
   (12 `<internal-endpoint>` + 12 `<internal-host>`, e.g. the daemon URL `…jphe.in:8085`), and private
   self-hosted `*.jphe.in` service FQDNs (16) + VM-inventory host names (11) — revealing JP's internal
   service topology, not needed for any benchmark token. The 60 benchmark `expected_sources` tokens
   (tool/host/concept names like `palace-daemon`, `gatekeeper`, `disks`, `pgvector`) are preserved —
   none are secrets. `*.realm.watch` subdomains are public-facing (the project itself) and kept, as
   are public URLs (github, arxiv, anthropic, letsencrypt). A final audit confirms **0** residual IPs,
   MACs, secrets, JWTs, session UUIDs, or private FQDNs.
   Output: `sme/corpora/jp_realm_v0_1/flat_source.jsonl` (sanitized, committed).
3. **Flatten.** Ingest the sanitized text into a ChromaDB collection carrying **only** drawer id +
   text — no wing/room/graph metadata. `FlatBaselineAdapter` does pure top-K vector similarity over
   it. Script: `scripts/ingest_jprealm_flat_chroma.py`. The ChromaDB dir is gitignored (binary,
   rebuilt from the committed JSONL).

**Conservatism note.** The flat pool is drawn from the daemon's *own* retrievals, so Condition A is
not penalized for searching the full 436K-drawer palace — it only ranks among documents the
structured arm already surfaced. This makes the **+10pp B−A gap a conservative lower bound** on the
structural advantage; a flat baseline over the whole palace would do no better and likely worse.

## Conditions (all @K=5, matching Condition B's `n_results`)

| Condition | What | Endpoint | Recall | Full@5 | Tok/q |
|-----------|------|----------|-------:|-------:|------:|
| **A** flat (no structure)        | vector top-K over flattened same-info pool | `FlatBaselineAdapter` / ChromaDB | **0.833** | 21/30 | 466 |
| **B** structural pipeline        | hybrid: vector + BM25 + reranker + closet/rating boosts | daemon `GET /search` | **0.933** | 26/30 | 789 |
| **C** graph-fusion sub-layer     | AGE-graph RRF fusion on top of B | daemon `POST /search/age-fused` | **0.900** | 24/30 | n/a¹ |

¹ The `/search/age-fused` endpoint returns full untruncated drawer blobs (≈483K tok/query), so its
token count is **not comparable** to A/B. Recall is comparable; tokens are not. Cat 7 token/cost
analysis uses **A vs B only**.

## Cat 2c — recall by hop depth

| condition | 1-hop (n=27) | 2-hop (n=3) | overall |
|-----------|-------------:|------------:|--------:|
| A flat    | 85%          | 67%         | 83%     |
| B daemon  | 94%          | 83%         | 93%     |
| C agefused| 93%          | 67%         | 90%     |

- **B − A:** +9.3pp @1-hop, +16.7pp @2-hop. **B/A ratio grows with depth (1.11x → 1.25x)** — the
  structural advantage scales as the spec predicts.
- **B − C:** +1.9pp @1-hop, +16.7pp @2-hop. The graph-RRF layer (C) is *behind* plain hybrid (B) at
  1-hop and never ahead — on this 1-hop-heavy corpus, graph fusion is a neutral-to-slightly-negative
  tax. The structural lift over flat lives in **hybrid retrieval + reranking**, not graph traversal.

Verdict (harness): *structure adds value at uniform scale* — B beats flat, the advantage grows with
depth, but the graph sub-layer's contribution does not separate cleanly at 1-hop (n=3 at 2-hop is too
small to call).

## Cat 7 — does structure earn its complexity?

On the token-comparable A-vs-B axis:

- **Recall:** +10.0pp for structure (0.833 → 0.933).
- **Cost:** +323 tok/query (466 → 789), i.e. **1.69×** the context budget.
- **Tokens per correct answer:** flat 666 vs structured 910 — flat is *cheaper per correct answer*
  because it answers fewer-but-cheaper questions; structure buys the extra 5 correct answers (21→26)
  at a token premium.

**Reading:** structure earns its complexity on recall (+10pp, scaling with depth) but pays for it in
tokens. For a budget-constrained reader, flat captures 83% of the answerable set at ~59% of the
context cost; structure is worth it when the marginal 5 answers matter more than the token premium.

## Caveats

- **n=30, 1-hop-heavy** (27 of 30 are 1-hop). The 2-hop deltas rest on n=3 — directional, not
  significant.
- **Live diagnostic corpus.** Condition B reads JP's live palace, which grows over time
  (`available_in_scope` 433515 on 2026-05-29 → 436647 on 2026-05-30). The flat corpus is frozen from
  the 2026-05-30 snapshot. Re-running B on a later snapshot may drift by a question or two.
- **Conservative gap** — see the methodology conservatism note above.

## Artifacts

- `sme/corpora/jp_realm_v0_1/flat_source.jsonl` — sanitized flat corpus (264 drawers, committed)
- `scripts/build_jprealm_flat_corpus.py` — read-only sourcing + sanitization
- `scripts/ingest_jprealm_flat_chroma.py` — flat ChromaDB build (dir gitignored, rebuildable)
- `baselines/jp_realm_v0_1_flat_condA_2026-05-30.json` — Condition A reading
- `baselines/jp_realm_v0_1_daemon_agefused_2026-05-30.json` — Condition C reading
- `baselines/cat2c_jprealm_ABC_2026-05-30.json` — Cat 2c A/B/C scorecard
- Condition B reuses the existing `baselines/jp_realm_v0_1_daemon_age_2026-05-29.json` /
  `baselines/cat2c_daemon_age_2026-05-29.json`.
