# LongMemEval-S rerun: prep + ingest-cost sizing

**Date:** 2026-05-29
**Branch:** `feat/91-longmemeval-s-prep`
**Issue:** prep for `techempower-org/multipass-structural-memory-eval#91` (LongMemEval-S `/search` vs `/search/age-fused` A/B)
**Scope:** loader verification + per-question ingest-cost sizing + a tiny n=3 daemon smoke. **This is prep only — the full bench is run by Sandman on the daemon.**

## TL;DR for sizing the full run

| Quantity | Value | Source |
|---|---:|---|
| Sessions / question (S-split) | min 38, **p50 48**, p90 52, max 62 (mean 48.7) | static probe, all 500q |
| Turns / question | min 396, p50 491, p90 532, max 616 | static probe |
| Tokens / question (chars/4) | **~122 K** (p50 122,448) | static probe |
| Per-session POST body | p50 **10.3 KiB**, mean 10.2 KiB, max 31.9 KiB | sme-rich render, 50q sample |
| POST bytes / question | ~486 KiB | derived |
| **Total POSTs, n=500** | **23,867** sessions → 23,867 `/memory` POSTs | static probe, all 500q |
| Total POSTs, n=100 | ~4,773 sessions | derived |
| Drawers ingested / question | ~48 (one drawer per session, then daemon chunks each) | ingest path `ingest_question_haystack` |

Note: each `/memory` POST is chunked by the daemon into `<parent>_chunk_NNNNNN` sub-drawers, so the *drawer* count in ChromaDB/pg is a multiple of the session count. n=500 means ~24K parent drawers plus their chunks — a large, sustained write load.

## Deliverable 1 — loader handles the S-split: **confirmed, no fix needed**

`sme/corpora/longmemeval/loader.py` is schema-agnostic by design (docstring §3, lines 4–5 and `load_questions` docstring lines 197–199: "All three files share the same record schema; only the haystack length differs").

Verified empirically against `longmemeval_s_cleaned.json` (277 MB, already present in the main checkout under `sme/corpora/longmemeval/data/`):

- **500 records**, top-level JSON array.
- Record keys exactly match the loader's expected set — `{answer, answer_session_ids, haystack_dates, haystack_session_ids, haystack_sessions, question, question_date, question_id, question_type}`. **Zero missing, zero extra keys.**
- `load_questions(S)` parsed all 500 records with **no errors** — including the `zip(..., strict=True)` length check across `haystack_session_ids` / `haystack_dates` / `haystack_sessions`, which would have raised on any per-record length skew. It did not.
- question_type histogram (matches the upstream S distribution exactly): `multi-session` 133, `temporal-reasoning` 133, `knowledge-update` 78, `single-session-user` 70, `single-session-assistant` 56, `single-session-preference` 30.

The only oracle-specific text in the codebase is a cosmetic `"version": "longmemeval-oracle-v1"` string written into `questions.yaml` by `materialize_sme_corpus` (loader.py:294) — that is a label, not a parse assumption, and the daemon ingest path (`scripts/run_longmemeval_mempalace.py`) does not use `materialize_sme_corpus` at all. No code change required for S.

The run script also already accepts S directly: `--questions` doc says "longmemeval_oracle.json (or _s / _m)" (run script argparse line 814), and `--max-questions` / `--skip-judge` both exist.

## Deliverable 2 — n=3 smoke: **ran, but the daemon died mid-run**

Command (daemon-light, retrieval-only):

```
scripts/run_longmemeval_mempalace.py \
  --adapter mempalace-daemon --api-url http://familiar:8085 \
  --questions .../longmemeval_s_cleaned.json \
  --max-questions 3 --skip-judge --json <scratch>/smoke_s_n3.json
```

Result: **R@5 = 0.00% on all 3 questions — this is a daemon-death artifact, NOT a retrieval finding.**

| q | posted | ingest errors | recall |
|---|---:|---:|---:|
| e47becba | 36 / 53 | 17 (`Connection refused`) | 0.0 |
| 118b2229 | 0 / 45 | 45 (`Connection refused`) | 0.0 |
| 51a45a95 | 0 / 50 | 50 (`Connection refused`) | 0.0 |

All 112 ingest failures were `HTTP -1 URLError: [Errno 111] Connection refused` — the daemon stopped accepting connections partway through Q1's haystack. With nothing ingested for Q2/Q3 and only a partial Q1, every query returned `NO_RESULTS`, hence R@5 = 0.

## Deliverable 4 — daemon instability: **HIT IT. Stopped, did not retry, notified Sandman.**

The daemon was **healthy at smoke start** (`status:ok`, `crash_loop:false`, `restart_count:0`, uptime 658 s, postgres memcg 0.37 %). It died at **07:42:21 PDT**, ~midway through Q1's ingest.

Root cause, from `journalctl -u palace-daemon.service`:

- `palace-daemon.service` is a **systemd unit** (not a container — only `mempalace-db` is containerized on familiar). It is now `inactive (dead)`, `enabled` but **did not auto-restart**.
- Main PID `code=killed, signal=TERM`.
- The traceback at kill time is inside an **auto-mine subprocess**: `main.py:2867 mine → _run_mine_subprocess → proc.communicate()`, ending in `asyncio.CancelledError: Task cancelled, timeout graceful shutdown exceeded`.
- It flushed and tore down ChromaDB cleanly, then systemd `Deactivated successfully`.

**The `.bench-active.lock` did not prevent the auto-mine.** I touched `/srv/mempalace-data/palace/.bench-active.lock` *before* the smoke (per palace-daemon#104), yet the daemon was killed while running a mine. Either #104's lock check isn't honored on the running build (daemon version **1.9.1**), or the mine was already in-flight when I touched the lock and the lock only gates *new* mines. This is the most important thing to resolve before the full n=500 run — the lock is the load-protection mechanism and it appears not to have protected this smoke.

Per task instructions, I stopped immediately on the instability, did **not** retry, removed my bench lock as cleanup, did **not** touch the systemd unit (daemon recovery is the orchestrator's call), and notified Sandman.

## Recommended full-run sizing

1. **Resolve the lock/auto-mine interaction first.** A 23,867-POST ingest at n=500 sustained will keep the daemon under heavy write load for a long time; if auto-mine can still fire (or if a graceful-shutdown timeout can be tripped by mine + bench contention), the full run will die the same way this smoke did. Confirm palace-daemon#104 is actually live on 1.9.1 and that the lock gates in-flight mines, not just new ones. Consider hard-disabling the mine timer for the duration of the run rather than relying on the advisory lock.
2. **Start at n=100, not n=500.** ~4,773 POSTs is a meaningful load test of the fixed lock behavior without committing to the full ~24K-POST run. If n=100 completes clean with stable `restart_count`/`crash_loop`, scale to n=500.
3. **Watch `restart_count` / `crash_loop` / `:8085` reachability throughout** — the daemon does not auto-restart, so a mid-run death silently zeroes recall for every subsequent question (exactly what happened here).
4. The retrieval A/B itself (`/search` vs `/search/age-fused`) is wired and ready in the run script (`--search-endpoint`); no code blockers remain once the daemon is stable.

## Artifacts

- Static-shape probe: `~/.claude/.../scratch/somnia-91/probe_s_shape.py`
- Smoke report JSON + log: `~/.claude/.../scratch/somnia-91/smoke_s_n3.{json,log}`
