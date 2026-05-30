# True-Oracle Floor Test — the published "oracle" was retrieval-limited

**Date:** 2026-05-29
**Author:** Nyx (SME dream-team)
**Issue:** `techempower-org/multipass-structural-memory-eval#59`
**Touches:** the published comparison card's headline oracle-QA number.

## Verdict (card correction needed)

**Our published "oracle 0.61" was NOT true oracle — it was retrieval-limited.**
The Pass A "oracle" context was built by running `/search` at `limit=5` over the
oracle haystack and concatenating the top-5 chunks, so chunking + the top-5 cut
dropped or fragmented the gold for the floor categories. When the gold is made
**definitionally present** (full evidence sessions, verbatim, sme-rich, no
retrieval), **all three floor categories lift substantially:**

| category | /search-pinned floor | true-oracle (preference) | true-oracle (+clause) | lift (best) |
|---|---|---|---|---|
| single-session-assistant | 0.339 | 0.982 | **1.000** (assistant_trust) | **+66.1pp** |
| temporal-reasoning | 0.331 | 0.752 | **0.790** (temporal_cot) | **+45.9pp** |
| multi-session | 0.699¹ | 0.865 | **0.880** (dedup_count) | **+18.1pp** |

¹ multi-session /search-pinned shown abstention-credited (0.699); CORRECT-only
was 0.639. Either way the true-oracle lift is large.

**The card's framing — "residual ~26pp = the reader leaves it on the table even
when handed the gold" — is wrong for the floor categories.** The gold often
never reached the reader. The dominant residual is **what reaches the reader**
(ingest mode + retrieval breadth), not reader reasoning. A true-oracle number
belongs on the card alongside the retrieval-limited one.

## Method (local, no daemon)

For each floor-category question, the context_string is the **evidence
session(s)** (`answer_session_ids`) rendered **sme-rich** (keeps user AND
assistant turns), concatenated verbatim — no `/search`, no `limit`, no
chunk-drop. Retrieval is fully bypassed; the gold is definitionally present.
Confirmed via token-overlap presence proxy (≥80%): ss-assistant 48/51 (was
13/51 on /search-pinned), temporal 77/120 strong + 33 partial, multi-session
29/49 strong + 17 partial. (The proxy understates temporal/multi-session because
their golds are dates/numbers with weak token overlap, but the full evidence
sessions are present by construction.)

Reader = `claude-opus-4-8`; judge = `gpt-5.3-chat` + canonical type-specific
prompts (#146); abstention-credited via `is_abstention` (#148/#156). n = 56 / 133
/ 133. Baselines: `baselines/reader_trueoracle_{ss-assistant,temporal,multi-session}_2026-05-29.json`.

## Per-category reading

- **ss-assistant → 1.000.** This floor was an ingest-mode artifact:
  `/search`-pinned context used **upstream-exact** ingest (`loader.py:333`
  strips assistant turns), so the gold — in an assistant turn — was dropped at
  ingest (0/56 pinned contexts had any assistant content). sme-rich + full
  session recovers it completely; assistant_trust takes preference's lone miss
  to 1.000. (Cross-ref: the dedicated ss-assistant ingest write-up.)
- **temporal → 0.79** (+46pp). The dates ARE present in the evidence sessions;
  the /search-pinned floor was largely evidence-session drop (limit=5 cut one of
  the two date-bearing sessions). With both present, temporal_cot adds +3.8pp
  over preference (the explicit date-extraction + subtraction scaffold), and the
  residual ~21pp is genuine date-arithmetic/anchoring reader error — that part
  of the original framing survives, but it's half the apparent floor.
- **multi-session → 0.88** (+18pp). Smaller lift because its /search-pinned floor
  was already the highest (0.64–0.70) — chunk retrieval dropped *some* of the
  scattered mentions but not all. With all evidence co-located, dedup_count adds
  +1.5pp (the dedup-then-count scaffold), residual ~12pp is genuine
  aggregation/dedup reader error.

## What this means for the arc

The decomposition's final lever ("the reader") splits cleanly:
- **A large share of the floor was substrate** (ingest mode for ss-assistant;
  retrieval breadth / `limit=5` evidence-drop for temporal + multi-session) —
  recoverable by ingest + retrieval fixes, NOT reader prompting.
- **A genuine reader residual remains** (temporal arithmetic ~21pp, multi-session
  aggregation ~12pp) — and the per-category clauses help there (temporal_cot
  +3.8pp, dedup_count +1.5pp on true-oracle), now testable because the answer is
  present.

## Recommendation

1. **Correct the card:** publish a TRUE-oracle floor number distinct from the
   retrieval-limited "0.61". The honest headline is "with the gold actually in
   context, the floors recover to 0.79–1.00; the published 0.61 was bounded by
   `/search` limit=5 + chunking, not by the reader."
2. The reader-prompt clauses (assistant_trust / temporal_cot / dedup_count) ARE
   real improvements on true-oracle substrate (+3.8 / +1.5 / +18→clause).
3. Production daemon re-test (chunked, real retrieval) still needed for the
   DEPLOYED number — this is the oracle ceiling, not the production number.

## Caveat

True oracle = evidence sessions only (the cleanest "gold is present" substrate).
A real system must still *retrieve* those sessions; this measures the reader
ceiling once retrieval is perfect, which is exactly what "oracle" should mean —
and what the published number failed to be.
