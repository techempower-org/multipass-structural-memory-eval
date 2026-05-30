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
| `retrieval` (default) | retrieval-only (no `/memory` POSTs) | **any N** the RAM allows — ungated |
| `ingest` | ingest-heavy (POSTs `/memory`, e.g. LongMemEval per-question topology) | capped at `SME_BENCH_MAX_INGEST` (default **3**) AND a RAM floor |

Concurrency is **SAFE** per the mempalace ingest thread-safety audit
(mempalace#331): the cap on ingest-heavy benches is bounded by **RAM**
(~6 GB free on familiar), **not** thread-safety. The daemon's `_write_sem=2`
shared-connection write path is data-safe today. (A latent race in the inline
KG write-through — `MEMPALACE_KG_WRITETHROUGH`, not set in prod — was fixed
with an RLock in mempalace#331 regardless.)

## Tunables (env)

| var | default | meaning |
|---|---|---|
| `PALACE_DAEMON_HOST` | `familiar` | ssh target hosting the lock dir |
| `PALACE_BENCH_LOCK_PATH` | `/srv/mempalace-data/palace/.bench-active.lock` | lock dir on the daemon host |
| `SME_BENCH_MAX_INGEST` | `3` | max concurrent ingest-heavy benches (matches the audit's 2-3) |
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
