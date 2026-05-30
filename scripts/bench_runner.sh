#!/usr/bin/env bash
# bench_runner.sh — shared acquire-refcount → run → release wrapper for bench
# supervisors (techempower-org/multipass-structural-memory-eval#NNN, pairs
# with techempower-org/palace-daemon#196).
#
# WHY: the bench-active lock used to be a single-bench mutex — a supervisor
# would abort/wait if it was held, serializing benches and leaving familiar's
# 24-thread Ryzen 9 3900X mostly idle. The daemon now refcounts the lock
# (palace-daemon#196: a directory of <pid>.marker files), and the Postgres
# backend + run_in_executor offload make concurrent benches data-safe. This
# wrapper is the SME-fork half: register THIS bench in the refcount, run it
# concurrently with others, and deregister on exit — without ever aborting
# just because another bench is registered.
#
# The lock lives on the DAEMON host (familiar); SME runs on katana and reaches
# it over SSH. We record a katana-side PID in the marker name; the daemon
# reaps purely by age (it can't see katana PIDs), so a long bench must let the
# wrapper heartbeat-touch its marker (handled automatically while the bench
# runs).
#
# USAGE — two forms:
#
#   # 1. Wrapper form (recommended): run a command under the refcount.
#   scripts/bench_runner.sh run --kind ingest -- \
#       ./venv/bin/python scripts/run_longmemeval_mempalace.py --n 500 ...
#
#   # 2. Source form (for supervisors that need to interleave logic):
#   source scripts/bench_runner.sh
#   bench_acquire ingest    # registers + sets a trap to release on EXIT
#   ... run the bench ...
#   # release happens automatically on EXIT; or call bench_release explicitly
#
# --kind:
#   ingest      — ingest-heavy (POSTs /memory, drives mining-adjacent RAM).
#                 Gated by the RAM-aware concurrency cap.
#   retrieval   — retrieval-only (no ingest). Cheap on RAM; allowed to go wide.
#                 (default)
#
# ENV / tunables:
#   PALACE_DAEMON_HOST            ssh target for the daemon host (default: familiar)
#   PALACE_BENCH_LOCK_PATH        lock dir path ON THE DAEMON HOST
#                                 (default: /srv/mempalace-data/palace/.bench-active.lock)
#   SME_BENCH_MAX_INGEST          max concurrent ingest-heavy benches (default: 3)
#   SME_BENCH_MIN_AVAIL_MIB       RAM floor: refuse to register an ingest bench
#                                 if available RAM on the daemon host is below
#                                 this (default: 2048 MiB ≈ headroom for ~1 more)
#   SME_BENCH_HEARTBEAT_SECONDS   marker-touch interval (default: 300; must be
#                                 well under the daemon's 6h stale-age guard)
#   PALACE_BENCH_PID              override the marker PID (default: $$)
#   SME_BENCH_ALLOW_CONCURRENT_INGEST
#                                 set 1 to permit ≥2 CONCURRENT ingest-heavy
#                                 benches. Default 0 → a 2nd concurrent ingest
#                                 bench is REFUSED (exit 4). See the gate below.
#
# ⚠️  CONCURRENT-INGEST SAFETY GATE (mempalace#331)
# Live familiar runs with MEMPALACE_KG_WRITETHROUGH=1, which takes the daemon's
# *inline* KG write-through path — and that path has a data race under
# concurrent writers (autocommit=False shared connection → transaction-span
# interleaving → silently dropped/mis-committed KG triples). The mempalace
# ingest thread-safety audit's RLock fix (mempalace#331) is NOT yet merged or
# deployed to familiar. Until it is, running ≥2 ingest-heavy benches
# concurrently risks KG CORRUPTION. A SINGLE ingest bench is safe (no
# concurrent writer); retrieval-only is always safe (no writes). So this
# wrapper REFUSES a 2nd concurrent ingest bench by default; flip
# SME_BENCH_ALLOW_CONCURRENT_INGEST=1 only after #331 is deployed to familiar.

set -euo pipefail

PALACE_DAEMON_HOST="${PALACE_DAEMON_HOST:-familiar}"
PALACE_BENCH_LOCK_PATH="${PALACE_BENCH_LOCK_PATH:-/srv/mempalace-data/palace/.bench-active.lock}"
SME_BENCH_MAX_INGEST="${SME_BENCH_MAX_INGEST:-3}"
SME_BENCH_ALLOW_CONCURRENT_INGEST="${SME_BENCH_ALLOW_CONCURRENT_INGEST:-0}"
SME_BENCH_MIN_AVAIL_MIB="${SME_BENCH_MIN_AVAIL_MIB:-2048}"
SME_BENCH_HEARTBEAT_SECONDS="${SME_BENCH_HEARTBEAT_SECONDS:-300}"
_BENCH_PID="${PALACE_BENCH_PID:-$$}"
_BENCH_MARKER="${PALACE_BENCH_LOCK_PATH}/katana-${_BENCH_PID}.marker"
_BENCH_HEARTBEAT_PID=""
_BENCH_REGISTERED=0

_ssh() { ssh -o BatchMode=yes "${PALACE_DAEMON_HOST}" "$@"; }

# Count current non-stale markers on the daemon host. The daemon reaps by age
# on its own checks; here we just count *.marker entries (a lightweight read).
_bench_refcount() {
    _ssh "ls -1 '${PALACE_BENCH_LOCK_PATH}'/*.marker 2>/dev/null | wc -l" 2>/dev/null || echo 0
}

# Available RAM (MiB) on the daemon host — the real concurrency cap.
_bench_avail_mib() {
    _ssh "free -m | awk '/^Mem:/{print \$7}'" 2>/dev/null || echo 0
}

# bench_acquire <kind>: register this bench in the refcount. For ingest-heavy
# benches, enforce the RAM-aware cap (count + available-RAM floor). NEVER
# aborts just because other benches are registered — only backs off when the
# host genuinely lacks headroom. Sets a trap so the marker is always released.
bench_acquire() {
    local kind="${1:-retrieval}"
    if ! _ssh "mkdir -p '${PALACE_BENCH_LOCK_PATH}'" 2>/dev/null; then
        echo "bench_runner: FATAL cannot create lock dir on ${PALACE_DAEMON_HOST}" >&2
        return 1
    fi

    if [ "$kind" = "ingest" ]; then
        local count avail
        count=$(_bench_refcount)
        avail=$(_bench_avail_mib)
        echo "bench_runner: ingest gate — ${count} bench(es) registered, ${avail} MiB avail on ${PALACE_DAEMON_HOST}"
        # mempalace#331 safety gate: refuse a 2nd CONCURRENT ingest bench while
        # the inline KG-write-through race is live on familiar (WRITETHROUGH=1,
        # RLock fix not yet deployed). A single ingest bench (count 0) is safe.
        if [ "${count:-0}" -ge 1 ] && [ "${SME_BENCH_ALLOW_CONCURRENT_INGEST}" != "1" ]; then
            echo "bench_runner: REFUSE — ${count} ingest bench(es) already registered and concurrent ingest is GATED (mempalace#331: live KG-write-through race under concurrent writers). A single ingest bench is safe; flip SME_BENCH_ALLOW_CONCURRENT_INGEST=1 only after #331's RLock is deployed to familiar." >&2
            return 4
        fi
        if [ "${count:-0}" -ge "${SME_BENCH_MAX_INGEST}" ]; then
            echo "bench_runner: REFUSE — ${count} ingest benches already registered (cap ${SME_BENCH_MAX_INGEST}). Re-run when one finishes." >&2
            return 2
        fi
        if [ "${avail:-0}" -lt "${SME_BENCH_MIN_AVAIL_MIB}" ]; then
            echo "bench_runner: REFUSE — only ${avail} MiB available (floor ${SME_BENCH_MIN_AVAIL_MIB}). Not enough headroom for another ingest bench." >&2
            return 3
        fi
    fi

    if ! _ssh "touch '${_BENCH_MARKER}'" 2>/dev/null; then
        echo "bench_runner: FATAL cannot write marker ${_BENCH_MARKER}" >&2
        return 1
    fi
    _BENCH_REGISTERED=1
    # shellcheck disable=SC2064  # expand marker/host now, not at trap time
    trap "bench_release" EXIT INT TERM
    echo "bench_runner: registered ${_BENCH_MARKER} (kind=${kind})"

    # Heartbeat: keep the marker fresh so the daemon's age guard never reaps a
    # still-running bench. Background loop; killed in bench_release.
    (
        while sleep "${SME_BENCH_HEARTBEAT_SECONDS}"; do
            _ssh "touch '${_BENCH_MARKER}' 2>/dev/null" 2>/dev/null || true
        done
    ) &
    _BENCH_HEARTBEAT_PID=$!
}

# bench_release: deregister this bench's marker and stop the heartbeat. Safe to
# call multiple times (idempotent) and from the EXIT trap.
bench_release() {
    trap - EXIT INT TERM
    if [ -n "${_BENCH_HEARTBEAT_PID}" ]; then
        kill "${_BENCH_HEARTBEAT_PID}" 2>/dev/null || true
        _BENCH_HEARTBEAT_PID=""
    fi
    if [ "${_BENCH_REGISTERED}" = "1" ]; then
        _ssh "rm -f '${_BENCH_MARKER}' 2>/dev/null; rmdir '${PALACE_BENCH_LOCK_PATH}' 2>/dev/null || true" 2>/dev/null || true
        _BENCH_REGISTERED=0
        echo "bench_runner: released ${_BENCH_MARKER}"
    fi
}

# Wrapper form: `bench_runner.sh run [--kind K] -- <cmd...>`
_run_wrapped() {
    local kind="retrieval"
    while [ $# -gt 0 ]; do
        case "$1" in
            --kind) kind="$2"; shift 2 ;;
            --) shift; break ;;
            *) echo "bench_runner: unknown arg '$1'" >&2; return 2 ;;
        esac
    done
    if [ $# -eq 0 ]; then
        echo "bench_runner: no command after '--'" >&2
        return 2
    fi
    bench_acquire "$kind" || return $?
    local rc=0
    "$@" || rc=$?
    bench_release
    return "$rc"
}

# Only act as a CLI when executed directly (not when sourced).
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    case "${1:-}" in
        run) shift; _run_wrapped "$@" ;;
        *)
            echo "usage: $0 run [--kind ingest|retrieval] -- <command...>" >&2
            echo "   or: source $0 ; bench_acquire <kind> ; ... ; bench_release" >&2
            exit 2
            ;;
    esac
fi
