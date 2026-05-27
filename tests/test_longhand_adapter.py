"""Tests for sme.adapters.longhand — subprocess-mocked, no live Longhand."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from sme.adapters.base import Edge, Entity, HarnessDescriptor, QueryResult
from sme.adapters.longhand import LonghandAdapter


@pytest.fixture
def fake_bin(tmp_path):
    """A resolvable stub binary so construction succeeds without a real CLI."""
    b = tmp_path / "longhand"
    b.write_text("#!/bin/sh\necho '[]'\n", encoding="utf-8")
    b.chmod(0o755)
    return b


def _completed(stdout="", returncode=0, stderr=""):
    return SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)


# --- Construction / bin resolution -----------------------------------


def test_construct_resolves_abs_bin(fake_bin, tmp_path):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    assert a.bin_path == str(fake_bin)


def test_construct_raises_when_bin_missing(tmp_path):
    with pytest.raises(ValueError, match="not found on PATH"):
        LonghandAdapter(bin_path=str(tmp_path / "nope"), home_dir=str(tmp_path))


def test_construct_warns_when_home_missing(fake_bin, tmp_path, caplog):
    missing = tmp_path / "no-longhand-here"
    with caplog.at_level("WARNING"):
        LonghandAdapter(bin_path=str(fake_bin), home_dir=str(missing))
    assert any("uninitialised" in r.message for r in caplog.records)


# --- query() result mapping ------------------------------------------


def test_query_maps_list_payload(fake_bin, tmp_path, monkeypatch):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    payload = [
        {
            "text": "fixed the stripe webhook race",
            "session_id": "sess-1",
            "project": "billing",
            "file": "/repo/src/webhook.ts",
            "score": 0.91,
            "event_type": "edit",
        }
    ]

    def fake_run(argv, **kw):
        assert argv[0] == str(fake_bin)
        assert argv[1] == "search"
        assert "--json" in argv
        return _completed(stdout=json.dumps(payload))

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = a.query("stripe webhook")
    assert isinstance(result, QueryResult)
    assert result.error is None
    assert "stripe webhook race" in result.context_string
    assert len(result.retrieved_entities) == 1
    ent = result.retrieved_entities[0]
    assert isinstance(ent, Entity)
    assert ent.name == "webhook.ts"
    assert ent.properties["project"] == "billing"
    assert ent.properties["session_id"] == "sess-1"
    assert ent.entity_type == "event:edit"


def test_query_maps_dict_wrapped_payload(fake_bin, tmp_path, monkeypatch):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    payload = {"results": [{"content": "hello", "session": "s", "title": "t"}]}
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: _completed(stdout=json.dumps(payload))
    )
    result = a.query("anything")
    assert result.error is None
    assert "hello" in result.context_string
    assert result.retrieved_entities[0].name == "t"


def test_query_no_results(fake_bin, tmp_path, monkeypatch):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _completed(stdout="[]"))
    result = a.query("nothing matches")
    assert result.error == "NO_RESULTS"
    assert result.context_string == ""
    assert result.retrieved_entities == []


def test_query_cli_error_returns_error_not_raise(fake_bin, tmp_path, monkeypatch):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _completed(returncode=2, stderr="boom"),
    )
    result = a.query("q")
    assert result.error is not None
    assert result.error.startswith("CLI_ERROR rc=2")
    assert "boom" in result.error


def test_query_timeout_returns_error(fake_bin, tmp_path, monkeypatch):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))

    def fake_run(argv, **kw):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kw.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = a.query("q")
    assert result.error is not None
    assert result.error.startswith("TIMEOUT")


def test_query_bad_json_returns_error(fake_bin, tmp_path, monkeypatch):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: _completed(stdout="not json")
    )
    result = a.query("q")
    assert result.error is not None
    assert result.error.startswith("BAD_JSON")


def test_query_passes_project_and_limit(fake_bin, tmp_path, monkeypatch):
    a = LonghandAdapter(
        bin_path=str(fake_bin), home_dir=str(tmp_path), project="billing"
    )
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return _completed(stdout="[]")

    monkeypatch.setattr(subprocess, "run", fake_run)
    a.query("q", n_results=12)
    assert "--limit" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--limit") + 1] == "12"
    assert "--project" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--project") + 1] == "billing"


# --- graph / ingest contract -----------------------------------------


def test_graph_snapshot_empty(fake_bin, tmp_path):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    entities, edges = a.get_graph_snapshot()
    assert entities == []
    assert edges == []
    assert all(isinstance(e, Entity) for e in entities)
    assert all(isinstance(e, Edge) for e in edges)


def test_ingest_corpus_not_implemented(fake_bin, tmp_path):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    with pytest.raises(NotImplementedError, match="diagnostic-only"):
        a.ingest_corpus([{"id": "x", "text": "y"}])


# --- Cat 9 harness manifest ------------------------------------------


def test_harness_manifest_declares_search(fake_bin, tmp_path):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    manifest = a.get_harness_manifest()
    assert len(manifest) == 1
    d = manifest[0]
    assert isinstance(d, HarnessDescriptor)
    assert d.name == "longhand_search"
    assert d.kind == "mcp_resource"


def test_probe_search_success_on_results(fake_bin, tmp_path, monkeypatch):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    payload = [{"text": "x", "session_id": "s"}]
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: _completed(stdout=json.dumps(payload))
    )
    probe = a.get_harness_manifest()[0].probe_fn()
    assert probe.success is True


def test_probe_search_success_on_empty_store(fake_bin, tmp_path, monkeypatch):
    """NO_RESULTS is a successful call-through (Cat 1 signal), not 9b failure."""
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _completed(stdout="[]"))
    probe = a.get_harness_manifest()[0].probe_fn()
    assert probe.success is True


def test_probe_search_failure_on_cli_error(fake_bin, tmp_path, monkeypatch):
    a = LonghandAdapter(bin_path=str(fake_bin), home_dir=str(tmp_path))
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: _completed(returncode=1, stderr="bad")
    )
    probe = a.get_harness_manifest()[0].probe_fn()
    assert probe.success is False
    assert probe.error is not None
