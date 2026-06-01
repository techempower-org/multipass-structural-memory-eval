# SME Case Studies — fix → re-run

This catalog is the operator-side evidence that SME does what it claims:
it runs against a real memory system, finds a structural defect, and the
finding comes with **"fix this and re-run"** guidance that — when
followed — moves the reading. Each entry is shaped the same way, so a
reader can answer "is this useful for *my* shape of graph?" without
reading the category spec:

1. **The system** — what was measured, and at what scale.
2. **The finding** — the reading SME produced.
3. **The fix** — the concrete change it pointed at.
4. **The re-run** — the reading *after* the fix, as a before→after delta.
5. **The lesson** — what generalizes.

These mirror the `remediation` field SME now attaches to its category
reports (upstream M0nkeyFl0wer#44): the report tells you what to fix and
how to re-verify; these case studies are completed instances of exactly
that loop. Every number traces to a committed `baselines/` artifact and
the campaign synthesis (`docs/research/2026-05-31-sme-campaign-synthesis.md`).

| # | Case | Category | Headline delta |
|---|---|---|---|
| 1 | [Tunnel-projection measurement artifact](2026-05-31-cat4-tunnel-projection-artifact.md) | Cat 4 (Ingestigation) | entropy 0.020 → 0.645 (the framework catching *itself*) |
| 2 | [Storage-equivalence null](2026-05-31-storage-equivalence-null.md) | Cat 1 / Cat 7 | postgres == flat; QA Δ +0.4pp, CI [−2.0, +2.8] — the engine is not the lever |
| 3 | [agentmemory throughput wall](2026-05-31-agentmemory-throughput-wall.md) | Cost-wall taxonomy | LLM-free at write, still ~15h to bench — a second cost axis |

> **A note on what these are NOT.** SME is diagnostic, not a leaderboard.
> These are deltas under controlled conditions, not absolute product
> rankings — and case 1 is the framework finding a bug in *its own*
> measurement surface, which is the strongest possible honesty signal a
> diagnostic can publish.
