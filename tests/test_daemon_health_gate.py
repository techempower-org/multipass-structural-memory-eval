"""Unit tests for the #115 daemon-health gate.

These cover `_probe_health` in isolation (mocked HTTP) so the skip-vs-fail
logic is verified without a live daemon and regardless of whether
PALACE_DAEMON_URL is set — unlike test_mempalace_daemon_integration.py,
whose module-level skipif gates it on the env vars.
"""
from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

from tests.test_mempalace_daemon_integration import (
    _probe_graph_backend,
    _probe_health,
)


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _resp(body: dict) -> _FakeResp:
    return _FakeResp(json.dumps(body).encode("utf-8"))


def test_probe_health_ok_status_is_reachable():
    with mock.patch("urllib.request.urlopen", return_value=_resp({"status": "ok"})):
        ok, detail = _probe_health("http://daemon")
    assert ok is True
    assert detail == "ok"


def test_probe_health_degraded_status_not_ok():
    """Daemon up but DB down (the #84 scenario) → degraded → not ok → skip."""
    with mock.patch("urllib.request.urlopen", return_value=_resp({"status": "degraded"})):
        ok, detail = _probe_health("http://daemon")
    assert ok is False
    assert "degraded" in detail


def test_probe_health_http_500_not_ok():
    err = urllib.error.HTTPError("http://daemon/health", 500, "ISE", {}, None)
    with mock.patch("urllib.request.urlopen", side_effect=err):
        ok, detail = _probe_health("http://daemon")
    assert ok is False
    assert "500" in detail


def test_probe_health_connection_refused_not_ok():
    with mock.patch("urllib.request.urlopen",
                    side_effect=urllib.error.URLError("connection refused")):
        ok, detail = _probe_health("http://daemon")
    assert ok is False
    assert "unreachable" in detail


def test_probe_health_strips_trailing_slash():
    """api_url with a trailing slash must not produce //health."""
    seen = {}

    def _capture(url, timeout=5.0):
        seen["url"] = url
        return _resp({"status": "ok"})

    with mock.patch("urllib.request.urlopen", side_effect=_capture):
        _probe_health("http://daemon/")
    assert seen["url"] == "http://daemon/health"


# --- _probe_graph_backend (#115, the /health=ok-but-/graph-down case) ----


def test_probe_graph_backend_healthy():
    with mock.patch("urllib.request.urlopen",
                    return_value=_resp({"wings": {"a": 1}, "kg_stats": {"entities": 1}})):
        assert _probe_graph_backend("http://daemon", "k") is True


def test_probe_graph_backend_unavailable_returns_false():
    """The exact post-redeploy shape: wings empty, kg_stats backend_unavailable."""
    body = {"wings": {}, "kg_stats": {"error": "backend_unavailable",
                                      "detail": "the connection is closed",
                                      "retryable": True}}
    with mock.patch("urllib.request.urlopen", return_value=_resp(body)):
        assert _probe_graph_backend("http://daemon", "k") is False


def test_probe_graph_backend_transport_error_returns_none():
    """Can't determine state → None, so the test proceeds (doesn't over-skip)."""
    with mock.patch("urllib.request.urlopen",
                    side_effect=urllib.error.URLError("refused")):
        assert _probe_graph_backend("http://daemon", "k") is None
