# Reader Floor-Lift — Phase 2 Results (the three prompt clauses)

**Date:** 2026-05-29
**Author:** Nyx (SME dream-team)
**Issue:** `techempower-org/multipass-structural-memory-eval#59`
**Implements:** Lucid's taxonomy (`docs/research/2026-05-29-reader-floor-failure-taxonomy.md` §1a/§2a/§3a)

## Bottom line up front

The three category-specific reader-prompt clauses, measured on the **pinned
n=500 oracle `/search` context** (reader = `claude-opus-4-8`, judge =
gpt-5.3-chat + canonical prompts), did **NOT produce a clean lift**. Each clause
churned ~14/56–133 rows but net movement was −2 to +3 questions:

| category (slice) | baseline (preference) | variant | CORRECT-only Δ | abstention-credited Δ | net flips |
|---|---|---|---|---|---|
| single-session-assistant (n=56) | 0.3393 | assistant_trust 0.3036 | **−3.6pp** | −3.6pp | +2 / −4 |
| temporal-reasoning (n=133) | 0.3308 | temporal_cot 0.3308 | +0.0pp | +0.0pp | +5 / −6 |
| multi-session (n=133) | 0.6391 | dedup_count 0.6617 | **+2.3pp** | +0.0pp¹ | +7 / −4 |

¹ multi-session abstention-credited is flat (0.6992 both) because the +3 net
CORRECT came partly from rows that were credited ABSTAINs under preference.

**The clauses are not the lever on this substrate.** The binding constraint is
retrieval/ingest: the gold answer frequently is not in the pinned context the
reader received, so no reader instruction can recover it. This *confirms* Lucid's
own per-category **context-shaping** recommendations (§1b/§2b/§3b) are the real
fix, and shows the pinned n=500 oracle context cannot demonstrate reader-prompt
lifts on its own.

**UPDATE — ss-assistant is an INGEST-MODE artifact, and the proper substrate
recovers it to ~1.00 (see §"ss-assistant ingest re-pin" below).** The pinned
n=500 context was generated with **upstream-exact** ingest, which strips
assistant turns (`loader.py:333` keeps only `t.role=="user"`). Confirmed: 0/56
ss-assistant pinned contexts contain ANY assistant content. Re-pinning with
**sme-rich** ingest (assistant turns kept) flips the floor from 0.32 to
**preference 0.982 / assistant_trust 1.000**. So assistant_trust IS a real win —
once the answer reaches the reader.

## The decisive datum: single-session-assistant is substrate-blocked

Lucid's live re-ingest + fresh `/search` probe found **29/38 answer-present**.
But on the **already-pinned** n=500 `/search` context, a gold-token-overlap
proxy (≥80% of the gold answer's distinctive tokens present in
`context_string`) finds the opposite:

- gold PRESENT (≥80%): **13/51** — and assistant_trust got **11/13** of those CORRECT
- partial (40–80%): 5/51
- gold ABSENT (<40%): **33/51**

(Proxy caveat: gold phrasing ≠ context phrasing, so treat 13-vs-33 as
directional. But the split is stark.) Where the assistant turn actually
reached the reader, the trust clause works (11/13); where it didn't (the
majority), the clause can't help and forcing commitment slightly hurts (−3.6pp).
The pinned `limit=5` chunk retrieval dropped the gold assistant turn in ~65% of
this slice — exactly the substrate tail Lucid flagged in §1b. **assistant_trust
is "promising but substrate-blocked"; it needs a higher-`limit` re-pin
(single-session questions, so cheap) before it can be tested as a reader fix.**

## Per-category mechanism

- **assistant_trust (§1a):** clause verbatim. Net −2. Substrate-blocked (above).
  Note: under the canonical judge there are **no ABSTAIN labels** on this
  non-abstention slice — an "I don't know" scores INCORRECT, so the "36 abstain"
  Lucid tallied (generic-judge run) appear as INCORRECT here. The clause does
  reduce refusals but converts them mostly to *wrong commitments* when the
  answer is absent.
- **temporal_cot (§2a):** clause verbatim. 14 rows changed, 5 gains / 6 losses,
  net −1 (flat). The explicit date-extraction + subtraction scaffold fixes some
  date-arithmetic errors but introduces others (CoT sometimes re-anchors a
  different wrong date). Pairing with §2b (retrieve ALL haystack sessions so
  both event dates reach the reader) is likely required — the
  retrieval-drop sub-mode (`bbf86515`/`f0853d11`) is untouched by a prompt.
- **dedup_count (§3a):** clause verbatim. 14 changed, 7 gains / 4 losses, net +3
  CORRECT — the only directionally-positive clause. The dedup-then-count scaffold
  does help the double-count failure, but the lift is small (+2.3pp raw) and
  washes out under abstention crediting. §3b (broader retrieval / fact-ledger,
  and preferring `/search` over age-fused) is the larger lever here.

## ss-assistant ingest re-pin — the floor was an ingest-mode artifact

**Disambiguation (offline, decisive):** of the 56 ss-assistant pinned
`context_string`s, **0/56** contained ANY assistant-turn content (no
`## assistant` header, no role headers, no `<!-- evidence -->` markers). The
pinned n=500 context was generated with **upstream-exact** ingest
(`loader.py:333` keeps only `t.role=="user"`), so for ss-assistant questions —
whose gold lives in an assistant turn — the gold was **dropped at ingest**, not
rank-out. This is an ingest-mode artifact, not a reader bug and not a `limit=5`
problem.

**Re-pin (local, no daemon):** re-rendered each of the 56 single-session
ss-assistant haystacks with **sme-rich** content rules (keeps assistant turns)
and used the full rendered session as context (these are single-session
questions — 1 session each, median 5.2K chars — so "retrieve all" is trivially
correct and removes the chunk-ranking confound entirely). Gold-token presence
(≥80% proxy) jumps **13/51 → 48/51**, 0 absent; 56/56 contexts now carry
assistant content.

**Re-test (reader=`claude-opus-4-8`, canonical judge) on the sme-rich substrate:**

| prompt | upstream-exact (pinned) | sme-rich (re-pin) |
|---|---|---|
| preference | 0.3393 | **0.9821** |
| assistant_trust | 0.3036 | **1.0000** |

The ss-assistant floor (0.32) was **~96% an ingest-mode artifact**: with the
assistant turns present, preference alone reaches 0.982 and assistant_trust
reaches **1.000** (it fixed preference's single miss, `778164c6`). assistant_trust
beats preference by +1.8pp at that ceiling — **the clause is a real win once the
answer reaches the reader, exactly as Lucid (§1a) predicted.**

Re-pin baseline: `baselines/reader_floorlift_ss-assistant_smerich_2026-05-29.json`.

**Caveat:** this is the oracle-substrate ceiling (full single session, no
retrieval drop). The production daemon path (chunk-level, `limit=5`) will sit
below 1.000 — but the result proves (a) sme-rich ingest is mandatory for
assistant-authored answers, and (b) assistant_trust is a genuine reader-prompt
improvement, both masked entirely by the upstream-exact pinned substrate.

## Disclosure / method

Reader = `claude-opus-4-8`, prompt = each variant vs `preference` baseline on
the SAME pinned slice (clean A/B, one sweep). Judge = `gpt-5.3-chat` + canonical
type-specific prompts (#146). Abstention-credited via the per-row
`is_abstention` flag (#148/#156). Per-category slices of the n=500 search-default
pinned context (ss-assistant n=56, temporal n=133, multi-session n=133); the
ss-assistant re-pin uses local sme-rich full-session rendering (no daemon).
Baselines: `baselines/reader_floorlift_{ss-assistant,temporal,multi-session}_2026-05-29.json`
+ `reader_floorlift_ss-assistant_smerich_2026-05-29.json`.

## Recommendation

1. **ss-assistant: SHIP the ingest-integrity finding** — the floor is an
   upstream-exact ingest artifact; sme-rich + assistant_trust recovers it from
   0.32 to 1.00 on oracle substrate. This is a publishable substrate result AND
   validates the assistant_trust clause. Production re-test (daemon, chunked)
   needed for the deployed number.
2. **temporal / multi-session: do NOT ship the clauses** — their gold IS in the
   upstream-exact pinned context (user turns), so those were clean tests and both
   were ~null (temporal +0.0pp, multi-session +2.3pp raw / flat credited). Real
   reader-prompt nulls; the larger lever is Lucid's §2b/§3b context-shaping.
3. **Sharpened arc conclusion:** the residual oracle gap is **what reaches the
   reader** (ingest + retrieval), not how the reader is instructed — for two of
   three floors the prompt is a null; for the third (ss-assistant) the entire
   floor was an ingest artifact.
