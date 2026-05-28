#!/usr/bin/env bash
# preflight_bench.sh — verify the bench-host state before launching a 500Q
# LongMemEval run via scripts/run_longmemeval_mempalace.py.
#
# Closes the workaround half of #60 (auto-mine vs bench resource conflict).
# The full fix is #61's option C (daemon-side bench-active.lock) — until
# that lands, this script catches the contention pattern at launch time
# instead of mid-run.
#
# Exit codes:
#   0 — all checks passed, bench is safe to launch
#   1 — at least one check failed (output names what)
#
# Defaults assume palace-daemon is on familiar:8085. Override via env vars:
#   PALACE_HOST          — daemon host (default: familiar)
#   PALACE_PORT          — daemon port (default: 8085)
#   PALACE_DAEMON_HOST   — ssh target for daemon-side checks (default: $PALACE_HOST)
#   PALACE_API_KEY       — auth (default: from ~/.config/palace-daemon/env)

set -euo pipefail

PALACE_HOST="${PALACE_HOST:-familiar}"
PALACE_PORT="${PALACE_PORT:-8085}"
PALACE_DAEMON_HOST="${PALACE_DAEMON_HOST:-${PALACE_HOST}}"

# Resolve API key from env or config file.
if [[ -z "${PALACE_API_KEY:-}" ]]; then
  env_file="${HOME}/.config/palace-daemon/env"
  if [[ -f "${env_file}" ]]; then
    PALACE_API_KEY=$(grep '^PALACE_API_KEY=' "${env_file}" | cut -d= -f2 | tr -d '"' | tr -d "'")
  fi
fi
if [[ -z "${PALACE_API_KEY:-}" ]]; then
  echo "FAIL: PALACE_API_KEY not in env and ~/.config/palace-daemon/env unreadable" >&2
  exit 1
fi

failures=0
PREFLIGHT_HEALTH_JSON=$(mktemp -t preflight_health.XXXXXX.json)
PREFLIGHT_KG_JSON=$(mktemp -t preflight_kg.XXXXXX.json)
trap 'rm -f "${PREFLIGHT_HEALTH_JSON}" "${PREFLIGHT_KG_JSON}"' EXIT

warn() { echo "  WARN: $*" >&2; }
fail() { echo "  FAIL: $*" >&2; failures=$((failures + 1)); }
ok() { echo "  OK:   $*"; }

echo "=== preflight_bench: ${PALACE_HOST}:${PALACE_PORT} ==="

# 1. Daemon health — must return HTTP 200 with status=ok.
echo "→ check: daemon /health"
health=$(curl -s -m 5 -o "${PREFLIGHT_HEALTH_JSON}" -w "%{http_code}" \
  -H "X-API-Key: ${PALACE_API_KEY}" \
  "http://${PALACE_HOST}:${PALACE_PORT}/health" 2>&1 || echo "000")
if [[ "${health}" != "200" ]]; then
  fail "daemon /health returned HTTP ${health} — daemon is down or unreachable"
else
  status=$(python3 -c "import json; print(json.load(open('${PREFLIGHT_HEALTH_JSON}')).get('status', '?'))" 2>/dev/null)
  if [[ "${status}" == "ok" ]]; then
    ok "daemon /health = ok"
  else
    fail "daemon /health returned 200 but status=${status} (expected ok)"
  fi
fi

# 2. Daemon hasn't restarted recently — restart_count should be 0.
echo "→ check: daemon restart_count"
restart_count=$(python3 -c "
import json
try:
  d = json.load(open('${PREFLIGHT_HEALTH_JSON}'))
  print(d.get('restart_count', '?'))
except Exception as e:
  print(f'parse-err: {e}')
" 2>&1)
if [[ "${restart_count}" =~ ^[0-9]+$ ]]; then
  if [[ "${restart_count}" -eq 0 ]]; then
    ok "daemon restart_count = 0 (stable)"
  elif [[ "${restart_count}" -le 2 ]]; then
    warn "daemon restart_count = ${restart_count} (recently restarted; may stabilize)"
  else
    fail "daemon restart_count = ${restart_count} (active crash cycle, see #61)"
  fi
else
  warn "daemon restart_count unreadable: ${restart_count}"
fi

# 3. No active mempalace mine subprocesses on the daemon host (per #60).
# Single SSH call (was two — Gemini PR #70 review) returning the PID list.
# `[m]empalace` regex prevents pgrep from self-matching the ssh argv.
# SKIP_MINE_CHECK=1 downgrades this to a warning — the daemon's own
# auto-mine is independent of the workstation Stop hook (techempower-org/
# multipass-structural-memory-eval#76 / techempower-org/palace-daemon#104),
# so a permanent SKIP is sometimes the only way to launch under the
# techempower-org/multipass-structural-memory-eval#61 stability stack.
echo "→ check: no active mempalace mine"
if ! mine_pids_raw=$(ssh -o BatchMode=yes "${PALACE_DAEMON_HOST}" \
    'pgrep -f "[m]empalace mine" 2>/dev/null; exit 0'); then
  fail "SSH to ${PALACE_DAEMON_HOST} failed — check connectivity / auth"
else
  mine_pids=$(echo "${mine_pids_raw}" | tr '\n' ',' | sed 's/,$//')
  if [[ -z "${mine_pids}" ]]; then
    ok "no mempalace mine running on ${PALACE_DAEMON_HOST}"
  elif [[ "${SKIP_MINE_CHECK:-0}" == "1" ]]; then
    warn "mempalace mine running on ${PALACE_DAEMON_HOST} (pids: ${mine_pids}) — SKIP_MINE_CHECK=1, proceeding"
  else
    fail "mempalace mine running on ${PALACE_DAEMON_HOST} (pids: ${mine_pids}) — pause Stop hook before launching (or set SKIP_MINE_CHECK=1 if stability stack applied)"
  fi
fi

# 4. KG stats sanity — daemon can answer MCP calls (last-line defence
# against the cycle where /health passes but /mcp tools fail because the
# psycopg connection is stale).
echo "→ check: daemon MCP kg_stats responds"
curl -s -m 15 \
  -H "X-API-Key: ${PALACE_API_KEY}" \
  -H "Content-Type: application/json" \
  -X POST "http://${PALACE_HOST}:${PALACE_PORT}/mcp" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"mempalace_kg_stats","arguments":{}}}' \
  > "${PREFLIGHT_KG_JSON}" 2>&1 || true
if grep -q '"result"' "${PREFLIGHT_KG_JSON}"; then
  triples=$(python3 -c "
import json, sys
try:
    d = json.load(open('${PREFLIGHT_KG_JSON}'))
    text = d['result']['content'][0]['text']
    print(json.loads(text).get('triples', '?'))
except Exception as e:
    print(f'parse-err:{e}', file=sys.stderr)
    print('?')
" 2>/dev/null)
  ok "MCP kg_stats responded — triples: ${triples}"
else
  fail "MCP kg_stats failed — daemon's psycopg connection may be stale: $(head -c 200 "${PREFLIGHT_KG_JSON}")"
fi

echo
if [[ "${failures}" -eq 0 ]]; then
  echo "PREFLIGHT PASSED — bench is safe to launch"
  exit 0
else
  echo "PREFLIGHT FAILED — ${failures} check(s) failed (see above)"
  echo
  echo "Workaround recap (from #61):"
  echo "  1. Pause Stop hook:  edit ~/.claude/plugins/cache/mempalace/.../hooks.json"
  echo "  2. Kill in-flight mines:  ssh ${PALACE_DAEMON_HOST} 'sudo pkill -f \"mempalace mine\"'"
  echo "  3. Restart daemon:  ssh ${PALACE_DAEMON_HOST} 'sudo systemctl restart palace-daemon'"
  echo "  4. Re-run this preflight"
  exit 1
fi
