# Reader Floor-Category Failure Taxonomy — diagnosing the residual oracle gap

**Date:** 2026-05-29
**Author:** Lucid (SME dream-team)
**Issue:** `techempower-org/multipass-structural-memory-eval#59` (reader/substrate arc)

**Question (team-lead's ask):** the session's decomposition closed the encoder
(#84 null), retrieval (#91/#109 near-ceiling), and judge (#146 ~+2pp) levers.
The residual ~26pp oracle gap is the **reader** failing on three floor
categories even when handed good context and scored by the canonical judge.
Find the *specific* failure mode per category — the way the "preference" prompt
fix diagnosed a specific over-abstention mode and lifted
single-session-preference 0.04 → 0.76 — and propose concrete interventions.

**Method:** join every non-CORRECT, non-ABSTAIN case from the n=500 Pass A
baselines (`reader_sweep_passA_canonical-judge_opus-preference_{search-default,
age-fused}_2026-05-29.json`, reader = `claude-opus-4-8|preference`, judge =
canonical) back to `longmemeval_oracle.json` to recover the question, the gold
answer, and the gold-bearing turns (the `has_answer` turns, with their
`role`). Then re-ingest a sample of failing haystacks into the **live
palace-daemon** (`sme-rich` content rules, the sweep default) and replay the
exact `/search` retrieval to measure, per failure, **whether the answer was
actually in the context the reader received**. That last step is what
separates a *reader* fault from a *substrate* fault. All probe scripts are in
`scratch/lucid-diagnosis/`.

**Bottom line up front:** the three floors have three *different* root causes,
and only one of them is a retrieval problem.

| Category | n=500 acc (search / age) | Dominant root cause | Lever |
|---|---|---|---|
| single-session-assistant | 0.32 / 0.27 | **Reader distrusts assistant-authored turns** — answer is present, reader abstains | **Prompt** (highest-confidence fix) |
| temporal-reasoning | 0.33 / 0.37 | **Date mis-anchoring + arithmetic** (dates present) + some evidence-session drops | Prompt (CoT) + Context (per-event date surfacing) |
| multi-session | 0.65 / 0.41 | **Aggregation over scattered + duplicated facts** — under-count / double-count | Prompt (dedup-then-count) + Context (cross-session co-location) |

The single-session-assistant floor is **not** a substrate bug: in **29 of 38**
search-endpoint failures the gold answer was verbatim in the retrieved context
and Opus still refused it. That is a clean analogue of the "preference" win and
is the single highest-leverage fix in this report.

---

## 0. The structural fact that frames everything

The reader's context is **not** the raw oracle session. The harness
(`scripts/run_longmemeval_mempalace.py::ingest_question_haystack`) posts each
session to the daemon's `/memory`, and the daemon **re-chunks** every drawer
into `<parent>_chunk_NNNNNN` sub-drawers. `/search` then returns the **top-N
chunks** (the sweep pinned at `limit=5`), which the adapter concatenates into
`context_string`. Two consequences:

1. With the default `sme-rich` rules, assistant turns **are** ingested (role
   headers `## user` / `## assistant`, evidence turns marked
   `<!-- evidence -->`). Assistant content is *not* dropped at ingest. (The
   alternate `upstream-exact` rule *would* drop them — `loader.py:333` keeps
   only `t.role == "user"` — but that was not the sweep default. I verified
   live that `sme-rich` round-trips the assistant turns into the retrieved
   context.)
2. Because retrieval is chunk-level and capped at 5, a long session can have
   its answer **split across chunks** or **ranked out**, and a session with
   many *superseded drafts* floods the top-5 with stale near-duplicates.

This is the substrate surface the reader is working against. The taxonomy
below says, per category, whether the failure lives **above** this surface
(reader) or **at** it (chunking/ranking).

---

## 1. single-session-assistant — 0.32 (the worst floor)

**Question shape:** "remind me what *you* told me / the list *you* gave / the
sheet *you* made about X." The gold answer lives in an **assistant** turn
(38/38 search, 40/41 age — the role composition is essentially 100% assistant).

**Failure-mode tally (search, 38 cases):** 36 abstain ("not in context"), 2
committed-but-wrong.

### Root cause: the reader treats only USER turns as "the conversation history"

Live retrieval probe (`scratch/lucid-diagnosis/probe_present.py`,
re-ingest + replay `/search` at `limit=5`):

- **ANSWER-PRESENT: 29 / 38** — the gold answer's distinctive tokens were in
  the retrieved context and the reader **still failed** (almost all "I don't
  know").
- ANSWER-PARTIAL: 4 / 38.
- ANSWER-ABSENT (retrieval/chunk genuinely dropped it): **5 / 38**.

So ~76% of this floor is a **pure reader fault**, not retrieval. The reader's
own words give the mechanism away. Representative PRESENT cases (answer
verbatim in context, reader abstained):

- **`1903aded`** — Q: "what was the 7th job in the list you provided?" Gold:
  *Transcriptionist.* The retrieved chunk is the assistant's numbered list
  with `7. Transcriptionist` right there. Reader: *"the conversation history
  shows that you asked me to brainstorm work-from-home jobs … but it doesn't
  contain the actual list of jobs I provided."*
- **`7e00a6cb`** — Q: "name of that hostel near the Red Light District you
  recommended?" Gold: *International Budget Hostel.* Reader: *"the conversation
  history doesn't contain the actual names … only the questions you asked."*
- **`e9327a54`** — Q: "that unique dessert shop with the giant milkshakes?"
  Gold: *The Sugar Factory at Icon Park.* Present in context. Reader: *"the
  specific names … aren't included in the text provided."*

The pattern is identical across the 29: the reader cleanly recognizes the
**user's request** turn, then asserts that the assistant's **response** ("the
content *I* provided") is absent — when it is sitting in the same retrieved
context. The "preference"/"committed" prompts told the reader not to
over-abstain on *inference* questions, but said nothing about **trusting the
assistant's own prior outputs as evidence**. This is the next over-abstention
mode in the same family.

### Secondary cause (~5/38): superseded-draft flooding + chunk split

A minority of cases *are* substrate. Two sub-flavours:

- **Superseded drafts (`7161e7e2`, the shift-rotation sheet):** the session is
  an iterative refinement where the assistant regenerated a table 6 times. At
  `limit=5`, the top chunks include **placeholder drafts** ("Agent 1, Agent
  2…", Week-based, Day-based) alongside the one final table with real names
  (`| Sunday | Admon | …`). Even when the final table is retrieved, it sits
  among 5 stale near-duplicates the reader must disambiguate.
- **Answer split mid-enumeration (`6ae235be`, CITGO refineries):** gold is a
  4-item list (atmospheric distillation, FCC, alkylation, hydrotreating).
  Chunking split the list; even at `limit=50` only the head ("Atmospheric
  distillation") is retrieved — "alkylation" never reaches the reader.

### Interventions

**(a) Prompt — the primary fix (preference-style).** Add an
assistant-turn-trust clause. Exact wording to drop into the prompt variant:

> *The conversation history includes BOTH the user's messages and the
> assistant's previous replies. When the question asks what "you" (the
> assistant) said, recommended, listed, or produced earlier, the answer is in
> the assistant's own turns — treat the assistant's previous replies as
> authoritative evidence, exactly like the user's. Do NOT say the history
> lacks the content just because the user only asked for it; find the
> assistant turn that answered and quote it. If the assistant produced
> several drafts and later revised them, answer from the MOST RECENT / final
> version.*

The final-version clause also neutralizes the superseded-draft sub-mode.
Expected lift is large: 29/38 of this floor's failures are answer-present, so
flipping even most of them moves the category from ~0.32 toward the
single-session-user ceiling (~0.86).

**(b) Context-shaping (handles the ~5/38 substrate tail).** (i) Raise the
single-session-assistant retrieval `limit` (these are *single*-session
questions — there is only one haystack session; retrieving more chunks of it
is cheap and recovers split answers like CITGO). (ii) For ingest, prefer a
chunker that does not split a single assistant enumeration/table across chunk
boundaries (chunk on turn boundaries, or keep `## assistant … <!-- evidence -->`
blocks atomic). (iii) Optionally de-duplicate near-identical superseded drafts
at ingest so stale tables don't crowd the top-k.

---

## 2. temporal-reasoning — 0.33 (worst in absolute count: 85/133 wrong)

**Question shape:** "how many days between event A and event B?", "which came
first, X or Y?", "how long before Z?". Gold turns are **always user** turns
(0/85 assistant), and usually **multiple** (`user+user`, `user+user+user` …):
the two/three event dates live in *different* sessions, stated inline in prose
("on January 10th", "since February 20th", "I got back from … on March 19th").

**Failure-mode tally (search, 85):** 56 committed-but-wrong, 29 abstain.

### Root cause: three reasoning sub-modes over scattered inline dates

1. **Date mis-anchoring (most common).** Both dates are present, but the reader
   binds the wrong date to one event — frequently substituting the
   **question_date** (the "now") for the second event's actual date.
   - `0bb5a684`: workshop Jan 10 (sess 0), meeting **Jan 17** (sess 1). Gold 7
     days. Reader read the meeting as **Jan 13** (the question_date) → "3
     days." It never anchored to the Jan 17 stated in session 1.
2. **Arithmetic / ordering errors with both dates present.**
   - `gpt4_385a5000`: marigolds started **March 3**, tomatoes **Feb 20**. Q:
     which first? Gold *Tomatoes.* Reader: *"marigolds were started first"* —
     ignored that Feb 20 < March 3.
   - `a3838d2b`: count events **before** "Run for the Cure" (Oct 15); 4 such
     events exist across 6 sessions. Reader: *"zero — it was your first."*
3. **Missing evidence session (genuine retrieval drop).**
   - `bbf86515`: reader says *"no event called 'Rack Fest'"* but session 1
     states "the 'Rack Fest' … on June 18th." One of the two evidence sessions
     did not reach the reader.
   - `f0853d11`: reader says *"no reference to a 'Coastal Cleanup' event"* but
     session 0 states "Coastal Cleanup event on March 7th." Same drop.

**Evidence-session coverage (live-daemon replay, `probe_sessions.py`, `limit=5`):**
the dominant temporal failures are sub-modes (1) and (2) — the dates *are* in the
retrieved context and the reader mis-reasons — with a genuine retrieval-drop
minority (sub-mode 3). Token-presence is the wrong instrument for this category
(the gold is a *derived* number — "7 days" — that never appears verbatim), so the
diagnosis rests on the anchor-token coverage probe + manual read of the cases
above: mis-anchoring and arithmetic errors over present dates account for the
majority (56/85 are committed-but-wrong, i.e. the reader *had enough to attempt*
and got it wrong), while the abstain cases (29/85) are split between "couldn't do
the subtraction" and the retrieval-drop minority.

### Interventions

**(a) Prompt — temporal CoT with explicit date extraction.** A targeted
variant that forces the reader to (1) list every dated event it can find with
its date, (2) identify the two the question asks about, (3) compute the delta
explicitly, and (4) never use "today"/the question date as an event date unless
the question says so. Draft clause:

> *This is a temporal question. First, extract EVERY event mentioned in the
> history together with its explicit calendar date (quote the date as written,
> e.g. "January 10th", "3/1", "since February 20th"). Do NOT use the current
> date as an event's date. Then identify the two events the question compares,
> and compute the number of days between their dates by explicit date
> subtraction. State the two dates and the subtraction, then give the final
> number.*

**(b) Context-shaping — surface per-event dates, not just session dates.** The
`sme-rich` rendering carries a `_Date:` **session** header, but the dates that
matter are the **event** dates buried in prose. Two substrate moves: (i) at
ingest, prepend a per-session "events on record" line that pulls each inline
date + its event noun to the top of the drawer (a cheap date-NER pass), so the
embedding *and* the reader see "[2023-03-07] Coastal Cleanup" explicitly; (ii)
for temporal questions, retrieve **all** sessions of the haystack (they are
short) rather than top-5 chunks, so neither evidence session can be dropped —
this directly fixes the `bbf86515`/`f0853d11` retrieval-drop sub-mode.

---

## 3. multi-session — 0.65 (search) / 0.41 (age)

**Question shape:** "how many X have I …?", "how much total did I spend on …?",
"how many different Y did I …?". The answer is a **count or sum** derived by
aggregating facts scattered across many sessions. Gold turns are **all user**,
spread over 2–6 sessions.

**Failure-mode tally (search, 38):** 36 committed-but-wrong, 2 abstain — the
reader almost always *commits to a wrong number*.

### Root cause: aggregation over scattered AND duplicated facts

1. **Under-count (missed a scattered instance).**
   - `0a995998`: 3 clothing items (dry cleaning + Zara boots + …). Reader: 2.
   - `3a704032`: 3 plants (peace lily + succulent + snake plant from sister).
     Reader: 2 — missed the snake-plant turn (a different session).
   - `gpt4_f2262a51`: 3 doctors (PCP + ENT + dermatologist). Reader: 2.
2. **Double-count / dedup failure.**
   - `gpt4_d84a3211`: bike spend gold **$185**; reader **$225** — the **$40
     bike lights are mentioned in THREE sessions** (sess 0, 2, 3) and the
     reader summed duplicates. The cross-session restatement is the trap.
   - `6d550036`: "projects led" gold 2; reader 3 — counted a non-leadership
     item as leadership.
   - `gpt4_59c863d7`: model kits gold 5; reader enumerated then miscounted
     "That's 4" (counting bug on its own list).

The structural enabler is **duplication across sessions**: LongMemEval
deliberately restates the same fact in multiple sessions, so a naive reader
either misses an instance (under-count) or counts a restatement twice
(double-count). At `limit=5` chunk retrieval the reader also frequently sees
only a **subset** of the mentions.

**Why this is mostly a reader fault (with a retrieval-breadth assist):** the
n=500 tally is decisive on its own — **36/38 search-endpoint failures are
committed-but-wrong, only 2 abstain.** The reader almost always *finds enough
to commit to a number* and gets the count/sum wrong, rather than failing to
retrieve. That points the primary fix at the reader's aggregation procedure
(dedup-then-count), with retrieval breadth as the secondary lever for the
under-count cases where a scattered instance was genuinely outside the top-5.
(Token-presence is again the wrong instrument here — the gold is a derived
count like "3" — so the split rests on the committed-vs-abstain ratio plus the
manual case reads above.)

### Interventions

**(a) Prompt — dedup-then-count scaffold.** Force an explicit
enumerate → dedup → count procedure:

> *This question asks for a COUNT or TOTAL aggregated across the whole history.
> First, list every distinct item/event/expense relevant to the question, each
> with the detail that makes it unique (name, date, amount). Treat the SAME
> item mentioned in different sessions as ONE entry — do not count a restated
> fact twice. Then count (or sum) the deduplicated list and give the final
> number. Show the list before the number.*

The "treat the same item mentioned in different sessions as ONE entry" clause
directly targets the `$40-lights-3×` double-count, the dominant sum-error.

**(b) Context-shaping — co-locate cross-session mentions.** Multi-session is
the category most hurt by chunk-level top-5 retrieval, because the relevant
mentions are *spread* and each is individually low-salience. Two moves: (i)
retrieve more broadly for these questions (higher `limit`, or union the top
chunks of *every* session so no instance is dropped); (ii) at ingest or
retrieval time, emit a per-haystack "fact ledger" — a deduplicated bullet list
of the salient user-stated facts across sessions — so the reader counts over a
clean ledger instead of re-deriving it from scattered prose. Note the
**age-fused endpoint is markedly worse here (0.41 vs 0.65)**: graph-fusion is
*hurting* multi-session aggregation (likely pulling entity-centric chunks that
fragment the count) — Nyx should A/B the count-questions on `/search` vs
`/search/age-fused` and prefer plain `/search` for this category until
understood.

---

## 4. Cross-cutting reading

- **The reader is the last lever, but it is not one lever.** ss-assistant is a
  *trust/abstention* prompt bug; temporal is a *reasoning + date-surfacing*
  problem; multi-session is a *dedup/aggregation + retrieval-breadth* problem.
  A single prompt won't move all three — they need three targeted clauses, and
  two of them want context-shaping help.
- **Highest-confidence, lowest-cost win first:** the ss-assistant
  assistant-turn-trust clause. 29/38 of that floor's failures already have the
  answer in context; this is the cleanest "preference"-style flip available.
- **`limit=5` is doing real damage** on temporal and multi-session (evidence
  drops, partial mention sets). For *single*-session-assistant and the short
  LongMemEval haystacks, raising the retrieval breadth is nearly free and
  recovers the substrate tail.
- **age-fused is not uniformly better** as a reader substrate: it helps
  knowledge-update slightly but *hurts* multi-session (−24pp) and is roughly
  flat on the other floors. Treat endpoint choice as per-category.

## 5. Hand-off to implementation (Nyx)

1. **ss-assistant prompt variant** with the assistant-turn-trust + final-draft
   clause (§1a). Re-run the n=500 (or stratified n=150) Pass A on
   single-session-assistant only; expect the largest single lift.
2. **temporal CoT variant** with explicit date-extraction + no-question-date
   anchoring (§2a); pair with all-session retrieval for the haystack (§2b).
3. **multi-session dedup-then-count variant** (§3a); pair with higher retrieval
   breadth and prefer `/search` over `/search/age-fused` for count questions
   (§3b).
4. Optional substrate: turn-boundary / atomic-evidence-block chunking and a
   per-haystack date/fact ledger injected into context (§1b, §2b, §3b).

All claims here are reproducible from `scratch/lucid-diagnosis/{join,classify,
probe_present,probe_sessions}.py` against the n=500 baselines + the live
daemon.
