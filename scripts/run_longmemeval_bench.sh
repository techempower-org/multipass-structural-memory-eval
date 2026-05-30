#!/usr/bin/env bash
# run_longmemeval_bench.sh — example supervisor that runs a LongMemEval bench
# UNDER the refcounted bench-runner (scripts/bench_runner.sh), so multiple
# benches can share the daemon concurrently instead of serializing on a
# single-bench mutex.
#
# This is the reference pattern for wiring any bench supervisor to the
# refcount: pick the --kind, hand the real command to `bench_runner.sh run`,
# and let it acquire → run → release (with the RAM-aware ingest cap + the
# trap-on-exit release + the marker heartbeat handled for you).
#
# run_longmemeval_mempalace.py POSTs each question's sessions into /memory
# (per-question ingest topology), so it is INGEST-HEAVY → --kind ingest. A
# retrieval-only sweep (no /memory POSTs) would pass --kind retrieval and run
# ungated.
#
# Concurrency is SAFE (mempalace ingest thread-safety audit, mempalace#331):
#   - retrieval-only: any N the RAM allows
#   - ingest-heavy:   cap ~2-3, bounded by RAM (~6 GB free on familiar), NOT
#                     thread-safety. bench_runner's SME_BENCH_MAX_INGEST=3 +
#                     RAM floor enforce this; the daemon's _write_sem=2
#                     shared-conn write path is data-safe today.
# REQUIRES the daemon on palace-daemon#196 code (refcounted lock) deployed +
# restarted on the daemon host — otherwise a finished bench leaves an empty
# lock dir that the pre-#196 daemon still reads as "active" for 6h.
#
# Usage:
#   scripts/run_longmemeval_bench.sh --adapter mempalace-daemon \
#       --questions sme/corpora/longmemeval/longmemeval_s.json \
#       --max-questions 500 --json baselines/lme_s_500.json
#
# All args are forwarded verbatim to run_longmemeval_mempalace.py.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
PYBIN="${PYBIN:-${REPO_ROOT}/venv/bin/python}"

# Preflight (daemon health, no in-flight mine, KG sanity). Non-fatal here —
# the bench command itself surfaces a hard failure; preflight is a courtesy
# early-out. Comment out if running purely retrieval-only against a quiet box.
if [ -x "${HERE}/preflight_bench.sh" ]; then
    "${HERE}/preflight_bench.sh" || {
        echo "run_longmemeval_bench: preflight failed — fix the above or set SKIP_MINE_CHECK=1" >&2
        exit 1
    }
fi

# Hand the real bench to the refcounted runner as an INGEST-heavy bench.
exec "${HERE}/bench_runner.sh" run --kind ingest -- \
    "${PYBIN}" "${REPO_ROOT}/scripts/run_longmemeval_mempalace.py" "$@"
