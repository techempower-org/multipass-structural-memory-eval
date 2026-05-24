"""Tests for sme.adapters.mem0 — mock Memory instance, no live mem0ai."""

from __future__ import annotations

import sys
import types

import pytest


class FakeMemory:
    """Stands in for mem0.Memory. Records calls; configurable returns."""

    def __init__(self):
        self.added: list[tuple] = []
        self.deleted: list[dict] = []
        self.search_calls: list[dict] = []
        self.get_all_calls: list[dict] = []
        self.search_return = {
            "results": [
                {
                    "id": "m1",
                    "memory": "Alice loves hiking",
                    "user_id": "alice",
                    "categories": ["hobbies"],
                    "score": 0.92,
                },
                {
                    "id": "m2",
                    "memory": "Alice works at Google",
                    "user_id": "alice",
                    "categories": ["work", "personal_info"],
                    "score": 0.85,
                },
            ]
        }
        self.get_all_return = {"results": [
            {"id": "m1", "memory": "Alice loves hiking", "categories": ["hobbies"]},
            {"id": "m2", "memory": "Alice works at Google", "categories": ["work"]},
        ]}

    def add(self, messages, user_id):
        self.added.append((messages, user_id))

    def search(self, query, filters=None, top_k=10, **kwargs):
        # Post-graph-removal API requires filters dict
        if filters is None and "user_id" in kwargs:
            raise ValueError("user_id must be inside filters dict now")
        self.search_calls.append({"query": query, "filters": filters, "top_k": top_k})
        return self.search_return

    def get_all(self, filters=None, **kwargs):
        if filters is None and "user_id" in kwargs:
            raise ValueError("user_id must be inside filters dict now")
        self.get_all_calls.append({"filters": filters})
        return self.get_all_return

    def delete_all(self, user_id=None, filters=None):
        self.deleted.append({"user_id": user_id, "filters": filters})


@pytest.fixture
def fake_mem(monkeypatch):
    # Install a fake `mem0` module so `from mem0 import Memory` resolves.
    fake = types.ModuleType("mem0")
    fake.Memory = FakeMemory  # Class import inside _require_mem0
    monkeypatch.setitem(sys.modules, "mem0", fake)
    return FakeMemory()


# --- construction --------------------------------------------------


def test_construction_requires_mem0(monkeypatch):
    monkeypatch.delitem(sys.modules, "mem0", raising=False)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def faux_import(name, *a, **kw):
        if name == "mem0":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", faux_import)

    from sme.adapters.mem0 import Mem0Adapter
    with pytest.raises(ImportError, match="mem0ai"):
        Mem0Adapter()


def test_construction_with_injected_memory(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem)
    assert a._memory is fake_mem
    assert a.user_id == "sme"


def test_construction_with_module_import(fake_mem):
    # Tests the from-module path, no injected memory
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter()
    assert isinstance(a._memory, FakeMemory)


# --- ingest_corpus -------------------------------------------------


def test_ingest_calls_add_with_messages_list(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem, user_id="alice")
    report = a.ingest_corpus([
        {"content": "fact one"},
        {"content": "fact two", "role": "assistant"},
    ])
    assert report["entities_created"] == 2
    # Each add call should be (messages_list, user_id)
    assert len(fake_mem.added) == 2
    msgs0, uid0 = fake_mem.added[0]
    assert msgs0 == [{"role": "user", "content": "fact one"}]
    assert uid0 == "alice"
    msgs1, _ = fake_mem.added[1]
    assert msgs1[0]["role"] == "assistant"


def test_ingest_per_row_user_id_override(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem, user_id="default")
    a.ingest_corpus([{"content": "x", "user_id": "override"}])
    _, uid = fake_mem.added[0]
    assert uid == "override"


def test_ingest_skips_empty_content(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem)
    report = a.ingest_corpus([{"content": ""}, {"text": "ok"}])
    assert report["entities_created"] == 1


def test_ingest_falls_back_to_legacy_string_signature(fake_mem, monkeypatch):
    from sme.adapters.mem0 import Mem0Adapter

    legacy_calls = []

    def add_legacy(self_, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], list):
            raise TypeError("legacy add wants a string, not a list")
        legacy_calls.append((args, kwargs))

    monkeypatch.setattr(FakeMemory, "add", add_legacy, raising=False)
    a = Mem0Adapter(memory=fake_mem)
    report = a.ingest_corpus([{"content": "hello"}])
    assert report["entities_created"] == 1
    # Should have retried with the legacy string-positional form
    assert legacy_calls
    args, kwargs = legacy_calls[0]
    assert args[0] == "hello"


def test_ingest_captures_failures(fake_mem):
    def boom(messages, user_id):
        raise RuntimeError("add crashed")
    fake_mem.add = boom
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem)
    report = a.ingest_corpus([{"content": "x"}])
    assert report["entities_created"] == 0
    assert len(report["errors"]) == 1


# --- query ---------------------------------------------------------


def test_query_builds_context_string(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem, user_id="alice")
    result = a.query("what do I love")
    assert result.error is None
    assert "[1] [hobbies] Alice loves hiking" in result.context_string
    assert "[2] [work,personal_info] Alice works at Google" in result.context_string


def test_query_uses_filters_dict(fake_mem):
    """Post-graph-removal API requires filters={'user_id': ...}, not top-level kwarg."""
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem, user_id="alice")
    a.query("hi")
    assert fake_mem.search_calls[0]["filters"] == {"user_id": "alice"}


def test_query_per_call_user_id_override(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem, user_id="default")
    a.query("x", user_id="bob")
    assert fake_mem.search_calls[0]["filters"] == {"user_id": "bob"}


def test_query_top_k_threaded(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem, n_results=4)
    a.query("x")
    assert fake_mem.search_calls[0]["top_k"] == 4


def test_query_per_call_n_results_override(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem)
    a.query("x", n_results=2)
    assert fake_mem.search_calls[0]["top_k"] == 2


def test_query_retrieved_entities_have_score_and_categories(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem)
    result = a.query("x")
    e0 = result.retrieved_entities[0]
    assert e0.id == "mem0:m1"
    assert e0.properties["score"] == 0.92
    assert e0.properties["categories"] == ["hobbies"]


def test_query_no_results(fake_mem):
    fake_mem.search_return = {"results": []}
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem)
    result = a.query("nothing")
    assert result.error == "NO_RESULTS"


def test_query_legacy_user_id_kwarg_fallback(fake_mem):
    """Older Mem0 versions accepted user_id directly. The adapter retries
    with the legacy shape if the new one raises TypeError/ValueError."""
    from sme.adapters.mem0 import Mem0Adapter

    seen = []

    def legacy_search(query, user_id=None, top_k=10, **kwargs):
        if "filters" in kwargs:
            raise TypeError("legacy doesn't accept filters")
        seen.append({"query": query, "user_id": user_id, "top_k": top_k})
        return {"results": [{"id": "m1", "memory": "hi"}]}

    fake_mem.search = legacy_search
    a = Mem0Adapter(memory=fake_mem)
    result = a.query("q")
    assert result.error is None
    assert seen[0]["user_id"] == "sme"


def test_query_internal_error_captured(fake_mem):
    def boom(**kwargs):
        raise RuntimeError("search crashed")
    fake_mem.search = boom
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem)
    result = a.query("x")
    assert result.error.startswith("INTERNAL:")


# --- get_graph_snapshot -------------------------------------------


def test_snapshot_returns_entities_no_edges(fake_mem):
    """Mem0 OSS dropped graph memory. Snapshot has nodes but zero edges."""
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem, user_id="sme")
    entities, edges = a.get_graph_snapshot()
    assert {e.id for e in entities} == {"mem0:m1", "mem0:m2"}
    assert edges == []  # Graph memory removed in OSS


def test_snapshot_handles_legacy_user_id_kwarg(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    seen = []

    def legacy(user_id=None, **kw):
        if "filters" in kw:
            raise TypeError("legacy doesn't accept filters")
        seen.append(user_id)
        return [
            {"id": "x", "memory": "x_text", "categories": []},
        ]

    fake_mem.get_all = legacy
    a = Mem0Adapter(memory=fake_mem, user_id="alice")
    entities, _ = a.get_graph_snapshot()
    assert seen == ["alice"]
    assert len(entities) == 1


def test_snapshot_get_all_failure_returns_empty(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter

    def boom(**kw):
        raise RuntimeError("backend down")

    fake_mem.get_all = boom
    a = Mem0Adapter(memory=fake_mem)
    entities, edges = a.get_graph_snapshot()
    assert entities == []
    assert edges == []


# --- reset -----------------------------------------------------------


def test_reset_calls_delete_all_with_user_id(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem, user_id="alice")
    a.reset()
    assert fake_mem.deleted[0]["user_id"] == "alice"


# --- ontology ------------------------------------------------------


def test_ontology_notes_graph_memory_removed(fake_mem):
    from sme.adapters.mem0 import Mem0Adapter
    a = Mem0Adapter(memory=fake_mem)
    ont = a.get_ontology_source()
    assert ont["type"] == "readme"
    assert "Graph memory" in ont["documentation"] or "graph" in ont["documentation"].lower()
    scopes = next(s for s in ont["schema"] if s["kind"] == "scopes")
    assert set(scopes["values"]) == {"user", "session", "agent"}
