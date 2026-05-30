# LoCoMo on the mempalace daemon (pgvector + Apache AGE) — E2E QA + R@5

**Date:** 2026-05-30
**Issue:** [techempower-org/multipass-structural-memory-eval#176](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/176)
**Adapter:** `mempalace-daemon` · postgres backend (pgvector) + Apache AGE knowledge graph · `/search/age-fused`
**Reader + judge:** `gpt-5.3-chat` (canonical LongMemEval type-specific prompts; temporal → off-by-one, adversarial → abstention)
**Comparison anchor:** the flat-adapter LoCoMo run (Cassia, #82) on the **identical** stratified subset.

## TL;DR

This is the **"daemon next"** half of JP's flat-now/daemon-next decision for the
LoCoMo cell of the comparison matrix. We ran LoCoMo through the production
retrieval substrate — a postgres-backed palace-daemon (pgvector vectors + an
Apache AGE graph), queried via wing-scoped `/search/age-fused` — on a
**throwaway scratch palace** so the ~thousands-of-turns LoCoMo ingest never
touched the production familiar palace's live ~1.9M-triple KG.

Subset: stratified per LoCoMo `question_type`, **50 per type, seed=1729, n=250**
— the same subset the flat run used, so all legs are a like-for-like A/B.

| Metric | Flat (Chroma) | Daemon vector-only† | Daemon age-fused (fixed) | age-fused − vector |
|---|---:|---:|---:|---:|
| Overall QA accuracy | 0.384 | 0.376 | **0.388** | **+0.012** |
| drawer R@5 | — (not exposed) | 0.5556 | **0.5556** | **0.0000** |
| substring recall (mean) | — | 0.2005 | 0.2025 | +0.0020 |

Per-LoCoMo-type QA accuracy:

| question_type | n | Flat | Daemon vector-only† | Daemon age-fused (fixed) |
|---|---:|---:|---:|---:|
| adversarial | 50 | 0.62 | 0.78 | 0.84 |
| multi-hop | 50 | 0.22 | 0.10 | 0.08 |
| open-domain | 50 | 0.42 | 0.42 | 0.40 |
| single-hop | 50 | 0.40 | 0.28 | 0.30 |
| temporal | 50 | 0.26 | 0.30 | 0.32 |

**The finding (the A/B that matters):** restoring the KG graph-only hydration
(the palace-daemon#202 fix) moved overall QA **+1.2pp** (0.376 → 0.388) and
drawer-R@5 by **exactly 0.0** (0.5556 either way) on LoCoMo. So on this corpus
the age-fused graph half contributes **~nothing to top-5 retrieval** — the
graph-only candidate hits don't displace the vector top-5 — and the QA bump is a
~3-question swing well inside noise on n=250. The #202 fix is correct and live
(graph held 2,425 vertices / 13,768 edges at query time), but its *LoCoMo
retrieval impact is marginal*. This is consistent with the LongMemEval-S age-fusion
A/B (#91), where age-fusion also showed no significant R@5 gain over plain `/search`.

> **† CAVEAT — the first daemon run was vector-only, not age-fused.** The scratch
> daemon process behind the "vector-only" column started at 23:55 on 2026-05-29,
> ~20 minutes *before* the palace-daemon#202 fix
> (`age-fused graph-only hydration uses real columns`) was written to
> `search_routes.py` at 00:15 on 2026-05-30. A long-running Python process holds
> the code it imported at start, so that run executed the buggy
> `SELECT id, content` hydration query — `mempalace_drawers` has `document`, not
> `content` — which failed and **silently fell back to vector-only retrieval**.
> Confirmed via `git show 2eb9e3d^` (buggy) vs HEAD (fixed) and via
> process-start-time vs file-mtime. The "age-fused (fixed)" column is a re-run
> against a daemon restarted *after* the fix (verified: a smoke age-fused query
> returned real drawer text, no empty fallback). The vector-only numbers are
> kept as a legitimate data point, not discarded.

> Diagnostic posture (per CLAUDE.md): these are deltas under controlled
> conditions on a stratified subset, not absolute leaderboard scores. The only
> variable that changed vs the flat run is the retrieval substrate + ingest
> topology; reader, judge, prompts, subset, and seed are held fixed.

## Method

### Ingest topology (per-sample wing isolation)

LoCoMo shares one conversation across all of a sample's questions. Each sample's
sessions were POSTed to the daemon's `/memory` endpoint under a **per-sample
wing** `locomo_<sample_id>` (room `sessions`), one drawer per session,
**sme-rich** rendering (the same drawer shape the flat run materialises to
disk — so we compare substrate, not text rendering). The adapter scopes every
`/search/age-fused` query to that sample's wing, making cross-sample
contamination impossible even though the daemon owns a single palace. This
mirrors the per-question wing scoping in `scripts/run_longmemeval_mempalace.py`.

The AGE graph is populated two ways: the daemon ran with
`MEMPALACE_KG_WRITETHROUGH=1`, so each `/memory` write extracts entities inline,
and `/backfill-age` runs once after ingest as a backstop. By query time the
scratch graph held **2,425 vertices / 13,768 edges** across 1,136 chunked
drawers — so the age-fused queries fused against a real, populated graph, not an
empty one. (The explicit `/backfill-age` reported `entities_added: 0` because
the inline write-through had already covered every drawer; the graph was live
regardless.)

### Retrieval + QA scoring

- **drawer R@5** uses a session_id → drawer_id map captured at ingest, with the
  daemon's `<parent>_chunk_NNNNNN` chunk suffix stripped before comparison (#98).
- **substring recall** matches expected session evidence as substrings of the
  retrieved context (SME's native matcher) — comparable across substrates.
- **QA** feeds the wing-scoped retrieval to the `gpt-5.3-chat` reader, then the
  canonical LongMemEval type-specific judge. Adversarial items are judged
  abstention-aware (a correct refusal scores as success).

## Isolation — never touched prod

The whole stack was throwaway and local to katana, with zero proximity to the
production palace (which lives behind `familiar:8085`, postgres on
`familiar:5433`):

- **Scratch Postgres** (`apache/age` PG16 + pgvector + pg_trgm) in Docker:
  container `mempalace-db-locomo-scratch`, bound to `127.0.0.1:5434`, DB
  `locomo_scratch`, a Docker **named volume** (no host bind-mount → cannot
  collide with prod's `/var/lib/mempalace-db`).
- **Scratch daemon** (`main.py --manual --host 127.0.0.1 --port 8086`) with
  `MEMPALACE_BACKEND=postgres`, `MEMPALACE_KG_BACKEND=age`,
  `MEMPALACE_KG_WRITETHROUGH=1`, and `MEMPALACE_POSTGRES_DSN` pointed at the
  scratch DB.
- **Isolation guard** in the runner refuses to start unless the daemon URL is a
  localhost instance AND the palace reports 0 drawers — so the run physically
  cannot point at a populated production palace.
- **Teardown:** scratch daemon stopped, `docker compose down -v` removed the
  container + named volume. Production state untouched throughout.

## Reproduce

```bash
# 1. scratch DB (named volume, port 5434, DB locomo_scratch)
MEMPALACE_DB_PASSWORD=$(openssl rand -hex 16) \
  docker compose -f <scratch-compose> up -d

# 2. scratch daemon (postgres + AGE) on localhost:8086
MEMPALACE_BACKEND=postgres MEMPALACE_KG_BACKEND=age MEMPALACE_KG_WRITETHROUGH=1 \
MEMPALACE_POSTGRES_DSN=postgresql://palace:***@localhost:5434/locomo_scratch \
  python <palace-daemon>/main.py --manual --host 127.0.0.1 --port 8086 \
    --palace <scratch-palace> --api-key <scratch-key>

# 3. the run (isolation guard enforces localhost + empty)
PALACE_DAEMON_URL=http://localhost:8086 PALACE_API_KEY=<scratch-key> \
AZURE_API_KEY=... AZURE_API_BASE=... \
  python scripts/run_locomo_mempalace_daemon.py \
    --per-type 50 --seed 1729 --search-endpoint /search/age-fused \
    --out baselines/locomo_daemon_age_fused_2026-05-30.json \
    --status /tmp/locomo_daemon.STATUS
```

Baseline JSON (the A/B pair):
- `baselines/locomo_daemon_age_fused_2026-05-30.json` — fixed age-fused (headline).
- `baselines/locomo_daemon_vector_only_2026-05-30.json` — the #202-buggy run, a
  legitimate vector-only data point (kept, not discarded).

Runner: `scripts/run_locomo_mempalace_daemon.py` (tested in
`tests/test_run_locomo_mempalace_daemon.py`).
