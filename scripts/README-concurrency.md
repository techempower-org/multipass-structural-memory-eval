# Concurrent benches against the palace-daemon

The palace-daemon serves requests concurrently (`run_in_executor` threadpool
offload on async FastAPI + Postgres MVCC), so running multiple SME benches
against one daemon is **data-safe**. The old single-file bench-active lock was
a 1-bench *mutex* — it serialized benches and left familiar's 24-thread Ryzen
9 3900X mostly idle. The lock is now **refcounted** (a directory of
`<pid>.marker` files; palace-daemon#196), so benches coexist and auto-mine
only resumes when the **last** bench deregisters.

## How to run a bench concurrently

Wrap the bench in `scripts/bench_runner.sh` (the shared acquire-refcount → run
→ release helper). Two forms:

```bash
# Wrapper form (recommended):
scripts/bench_runner.sh run --kind ingest -- \
    ./venv/bin/python scripts/run_longmemeval_mempalace.py --adapter mempalace-daemon ...

# Or use the worked example supervisor:
scripts/run_longmemeval_bench.sh --adapter mempalace-daemon \
    --questions sme/corpora/longmemeval/longmemeval_s.json --max-questions 500
```

`bench_runner.sh` registers this bench's marker on the daemon host, **never
aborts** because another bench is registered, releases on `EXIT`/`INT`/`TERM`,
and heartbeats the marker so the daemon's 6h stale-reaper never drops a live
long-running bench.

## `--kind`: the only knob that matters

| kind | meaning | concurrency |
|---|---|---|
| `retrieval` (default) | retrieval-only (no `/memory` POSTs) | **any N** the RAM allows — ungated, always SAFE |
| `ingest` | ingest-heavy (POSTs `/memory`, e.g. LongMemEval per-question topology) | **single bench only** by default; 2nd concurrent ingest REFUSED (exit 4). Thread-safe as of #331, but still gated on **prod contamination** — never run concurrent ingest against the prod palace. RAM-floored + capped at `SME_BENCH_MAX_INGEST` |

### ⚠️ Concurrent ingest: thread-safe now, but DON'T run it against prod

Two separate barriers gate concurrent ingest. The first is resolved; the
second is not, and the second is why the default stays OFF.

1. **Thread-safety — RESOLVED (mempalace#331).** Live familiar runs with
   **`MEMPALACE_KG_WRITETHROUGH=1`** (verified in `~/.config/palace-daemon/env`),
   which takes the daemon's *inline* KG write-through path. That path had a
   data race under concurrent writers (autocommit=False shared connection →
   transaction-span interleaving → silently dropped/mis-committed KG triples).
   The audit originally rated this "latent" on the belief prod doesn't set
   WRITETHROUGH — it does — so the race was live. **#331's RLock fix is
   deployed to familiar (2026-05-29) and verified in the loaded module**, so
   concurrent KG writes are now data-safe (no dropped triples).

2. **Prod contamination — NOT solved by the RLock.** `familiar:8085` fronts
   the **production** palace (the live ~1.9M-triple AGE graph + JP's companion
   data). An ingest-heavy bench POSTs *test data* into whatever palace the
   daemon serves — so running it against familiar pollutes prod regardless of
   thread-safety. The RLock prevents dropped triples; it does **not** prevent
   contamination.

Therefore `bench_runner.sh` **still refuses a 2nd concurrent ingest bench by
default** (exit 4). A *single* ingest bench is safe (no concurrent writer);
retrieval-only is always safe (no writes). The flag's meaning has shifted from
"thread-safety unknown" to **"requires a scratch palace + explicit opt-in"**:
flip `SME_BENCH_ALLOW_CONCURRENT_INGEST=1` **only** when pointed at a
throwaway/scratch palace (e.g. a separate daemon or `PALACE_BENCH_LOCK_PATH` +
a non-prod `--api-url`), **never** against the prod familiar daemon. With the
flag on, `SME_BENCH_MAX_INGEST=3` (RAM-bounded, matching the audit's 2-3) is
the operative limit.

## Tunables (env)

| var | default | meaning |
|---|---|---|
| `PALACE_DAEMON_HOST` | `familiar` | ssh target hosting the lock dir |
| `PALACE_BENCH_LOCK_PATH` | `/srv/mempalace-data/palace/.bench-active.lock` | lock dir on the daemon host |
| `SME_BENCH_MAX_INGEST` | `3` | max concurrent ingest-heavy benches when the flag is on (matches the audit's 2-3) |
| `SME_BENCH_ALLOW_CONCURRENT_INGEST` | `0` | `1` permits ≥2 concurrent ingest benches — set ONLY against a scratch palace, NEVER prod familiar (thread-safe per #331, but ingest pollutes whatever palace it hits) |
| `SME_BENCH_MIN_AVAIL_MIB` | `2048` | RAM floor — refuse a new ingest bench below this |
| `SME_BENCH_HEARTBEAT_SECONDS` | `300` | marker-refresh interval (≪ the 6h stale-age guard) |
| `PALACE_BENCH_PID` | `$$` | PID recorded in the marker name |

Refusals are explicit: count-cap → exit 2, RAM-floor → exit 3.

## Deploy requirement

The refcount semantics require the daemon to be on palace-daemon#196 code
(refcounted `bench_lock.py`) **deployed and restarted** on the daemon host.
On a pre-#196 daemon, the marker directory still pauses auto-mine, but an
*empty* dir (all benches released) is read as still-active until it ages out
6h — so the "resume on last release" semantic won't work until the daemon is
updated. Deploy: sync the new `bench_lock.py` to the daemon's WorkingDirectory
and `systemctl restart palace-daemon`.
