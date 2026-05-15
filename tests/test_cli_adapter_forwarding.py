"""Regression tests for CLI → adapter argument forwarding.

These exist because of an upstream code-review finding on PR #7:
``cmd_cat5`` accepted ``--api-url``/``--api-key``/``--kind``/``--mock``/
``--familiar-timeout`` at the argparse layer but quietly dropped them
before constructing the adapter — so ``sme-eval cat5 --adapter
mempalace-daemon --api-url ...`` raised
``ValueError: MemPalaceDaemonAdapter needs api_url`` even though the URL
was on the command line. ``cmd_check`` had already been migrated to the
shared ``_load_adapter_from_args`` helper; ``cmd_cat5`` had not.

Each test below names the command path it covers; if a future contributor
adds a new subcommand that goes through ``_add_db_or_api_args`` they
should add a row here too.
"""

from __future__ import annotations

import argparse

import pytest

from sme import cli


def _cat5_namespace(**overrides) -> argparse.Namespace:
    """Build an argparse.Namespace matching what the cat5 parser produces.

    Every attribute that ``_load_adapter_from_args`` or ``cmd_cat5`` reads
    via ``getattr`` is included so the namespace is fully drop-in.
    """
    defaults: dict = dict(
        adapter="mempalace-daemon",
        db=None,
        api_url=None,
        api_key=None,
        kind=None,
        mock_inference=None,
        familiar_timeout=None,
        auto_discover=False,
        node_tables=None,
        edge_tables=None,
        kg_path=None,
        collection_name=None,
        seeded_gaps=None,
        no_homology=True,
        betti_max_nodes=2000,
        min_component_size=3,
        max_type_prevalence=0.5,
        top_k=20,
        json=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_load_adapter_from_args_forwards_api_url_and_key(monkeypatch, tmp_path):
    """Helper used by cat4/cat5/check/cat9 must forward --api-url/--api-key.

    This is the unit-level proof that the shared plumbing is correct; the
    cmd_cat5 test below is the integration-level proof that cat5 uses it.
    """
    monkeypatch.delenv("PALACE_API_KEY", raising=False)
    monkeypatch.delenv("PALACE_DAEMON_URL", raising=False)

    # Empty env file so the adapter has no fallback resolution path —
    # if --api-url/--api-key aren't forwarded, construction raises.
    env_file = tmp_path / "env"
    env_file.write_text("")
    monkeypatch.setattr(
        "sme.adapters.mempalace_daemon.DEFAULT_ENV_FILE", str(env_file)
    )

    ns = _cat5_namespace(
        api_url="http://example.test:8085",
        api_key="test-key",
        kind="all",
    )
    adapter = cli._load_adapter_from_args(ns)
    assert adapter.api_url == "http://example.test:8085"
    assert adapter.api_key == "test-key"
    assert adapter.kind == "all"


def test_cmd_cat5_forwards_api_url_to_adapter(monkeypatch):
    """Regression for M0nkeyFl0wer's exact PR #7 repro.

    Before the fix, ``cmd_cat5`` built its own ``adapter_kwargs`` dict and
    omitted ``api_url``/``api_key``/``kind`` even though the argparse
    layer accepted them. Calling cat5 with ``--api-url`` raised
    ``ValueError: MemPalaceDaemonAdapter needs api_url``. After the fix
    cat5 funnels through ``_load_adapter_from_args``, the same helper
    cat4/check/cat9 use, and the URL reaches the adapter.
    """
    captured_kwargs: dict = {}

    class _FakeAdapter:
        def __init__(self, **kw):
            captured_kwargs.update(kw)

        def get_graph_snapshot(self):
            return ([], [])

        def close(self):
            pass

    def _fake_load(name, **kwargs):
        captured_kwargs["_name"] = name
        captured_kwargs.update(kwargs)
        return _FakeAdapter()

    monkeypatch.setattr(cli, "_load_adapter", _fake_load)

    ns = _cat5_namespace(
        adapter="mempalace-daemon",
        api_url="http://example.test:8085",
        api_key="test-key",
    )
    rc = cli.cmd_cat5(ns)
    assert rc == 0
    assert captured_kwargs["_name"] == "mempalace-daemon"
    assert captured_kwargs.get("api_url") == "http://example.test:8085"
    assert captured_kwargs.get("api_key") == "test-key"


def test_cmd_cat5_forwards_familiar_args_to_adapter(monkeypatch):
    """Same regression surface, familiar adapter path.

    cat5 with ``--adapter familiar --no-mock --familiar-timeout 10`` was
    equally broken by the dropped-kwargs bug. Verify both are now plumbed.
    """
    captured_kwargs: dict = {}

    class _FakeAdapter:
        def __init__(self, **kw):
            captured_kwargs.update(kw)

        def get_graph_snapshot(self):
            return ([], [])

        def close(self):
            pass

    monkeypatch.setattr(
        cli,
        "_load_adapter",
        lambda name, **kw: (
            captured_kwargs.update({"_name": name, **kw}) or _FakeAdapter()
        ),
    )

    ns = _cat5_namespace(
        adapter="familiar",
        api_url="http://familiar.test",
        mock_inference=False,
        familiar_timeout=10.0,
    )
    rc = cli.cmd_cat5(ns)
    assert rc == 0
    assert captured_kwargs["_name"] == "familiar"
    assert captured_kwargs.get("api_url") == "http://familiar.test"
    assert captured_kwargs.get("mock_inference") is False
    assert captured_kwargs.get("timeout_s") == 10.0
