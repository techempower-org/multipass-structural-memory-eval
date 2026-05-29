"""Live-daemon smoke tests for MemPalaceDaemonAdapter.

Skipped automatically when PALACE_DAEMON_URL is not set in the
environment, so CI without a daemon stays green. Run locally with:

    PALACE_DAEMON_URL=http://your-daemon-host:8085 \
    PALACE_API_KEY=$(grep ^PALACE_API_KEY ~/.config/palace-daemon/env | cut -d= -f2) \
    pytest tests/test_mempalace_daemon_integration.py -v

The tests are read-only: query() and get_graph_snapshot() only.

#115: env vars being *set* is necessary but not sufficient — a shared daemon
that's down or mid-redeploy returns HTTP 500 on every call, which used to make
these smoke tests hard-FAIL and masquerade as a code regression. The
session-scoped `_daemon_health` fixture probes `/health` once and skips the
whole module when the daemon isn't reachable or reports a non-``ok`` status,
so "the shared daemon is down" reads as SKIP, not FAIL.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from sme.adapters.base import QueryResult
from sme.adapters.mempalace_daemon import MemPalaceDaemonAdapter


pytestmark = pytest.mark.skipif(
    not (os.environ.get("PALACE_DAEMON_URL") and os.environ.get("PALACE_API_KEY")),
    reason="needs a running palace-daemon; set PALACE_DAEMON_URL and PALACE_API_KEY to enable",
)


def _probe_health(api_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Probe the daemon's /health once. Returns (reachable_and_ok, detail).

    The daemon returns status='ok' when the palace collection opens and
    'degraded' (HTTP 503) when it can't (postgres down, AGE init failure,
    mid-redeploy). Either a transport error or a non-ok status means the
    live-daemon tests can't run meaningfully → caller should skip, not fail.
    """
    url = f"{api_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return False, f"/health HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 — connection refused, timeout, bad JSON
        return False, f"/health unreachable: {type(e).__name__}: {e}"
    status = body.get("status")
    if status != "ok":
        return False, f"/health status={status!r}"
    return True, "ok"


def _probe_graph_backend(api_url: str, api_key: str, timeout: float = 20.0):
    """Probe whether the /graph AGE backend is actually serving (#115).

    Returns True if the graph backend looks healthy, False if it reports
    ``kg_stats.error == 'backend_unavailable'`` (or equivalent down signal),
    and None if the probe itself couldn't determine state (transport error,
    no kg_stats) — in which case the caller should not skip on this basis.
    """
    url = f"{api_url.rstrip('/')}/graph"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — can't tell; let the test proceed
        return None
    kg_stats = body.get("kg_stats")
    if isinstance(kg_stats, dict) and kg_stats.get("error") == "backend_unavailable":
        return False
    return True


@pytest.fixture(scope="session")
def _daemon_health() -> None:
    """Skip the live-daemon tests when the daemon is down/degraded (#115).

    Probed once per session. PALACE_DAEMON_URL is guaranteed set here by the
    module-level skipif, so a missing URL never reaches this fixture.
    """
    api_url = os.environ.get("PALACE_DAEMON_URL", "")
    ok, detail = _probe_health(api_url)
    if not ok:
        pytest.skip(f"palace-daemon not serving ({detail}); skipping live-daemon tests")


@pytest.fixture
def adapter(_daemon_health):
    a = MemPalaceDaemonAdapter()
    yield a
    a.close()


def test_query_returns_query_result(adapter):
    r = adapter.query("hello", n_results=2)
    assert isinstance(r, QueryResult)
    # Either we got results, or we got a soft-warn / NO_RESULTS — never an
    # uncaught exception.


def test_snapshot_returns_at_least_one_wing(adapter):
    # /health can be 'ok' (the search collection opens) while the /graph
    # AGE backend is still down/recovering — e.g. post-redeploy it returns
    # wings:{} with kg_stats.error='backend_unavailable' (#115). That's a
    # daemon-state condition, not a code regression, so skip rather than
    # fail when the graph backend isn't serving.
    raw = _probe_graph_backend(os.environ.get("PALACE_DAEMON_URL", ""),
                               os.environ.get("PALACE_API_KEY", ""))
    if raw is not None and not raw:
        pytest.skip("graph backend unavailable (kg_stats backend_unavailable); "
                    "search is up but /graph is recovering")

    entities, _ = adapter.get_graph_snapshot()
    wing_names = {e.name for e in entities if e.entity_type == "wing"}
    # Live palace has 30+ wings on JP's install; even a fresh palace has >=1.
    assert len(wing_names) >= 1


@pytest.mark.xfail(
    reason="palace-daemon#194: kind=content filter is a no-op post-redeploy "
           "(returns identical results to kind=all). strict=False so this "
           "XPASSes once #194 is fixed — the XPASS is the signal to remove "
           "this xfail.",
    strict=False,
)
def test_kind_content_excludes_stop_hook_checkpoints(adapter):
    """Cross-check the README's behavioural claim: kind='content'
    excludes Stop-hook auto-save checkpoints (which start with
    'CHECKPOINT:' in the live palace) while kind='all' includes them.

    The earlier-shape assertion on total_before_filter conflated
    metadata math with filter behaviour — `total_before_filter` is
    not "scope size before kind filter". The reliable signal is in
    the returned context_string itself: do CHECKPOINT: strings
    appear or not?
    """
    r_all = adapter.query("CHECKPOINT", n_results=5, kind="all")
    r_content = adapter.query("CHECKPOINT", n_results=5, kind="content")

    # If the live palace has zero checkpoints, both will be empty —
    # skip rather than fail.
    if "CHECKPOINT:" not in (r_all.context_string or ""):
        pytest.skip(
            "live palace has no Stop-hook checkpoints to test against"
        )

    # The behavioural invariant: kind='content' must have strictly
    # fewer (or zero) CHECKPOINT: strings than kind='all'.
    n_all = (r_all.context_string or "").count("CHECKPOINT:")
    n_content = (r_content.context_string or "").count("CHECKPOINT:")
    assert n_content < n_all, (
        f"kind='content' should filter checkpoints, got "
        f"{n_content} vs {n_all} for kind='all'"
    )
