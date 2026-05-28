"""Unit tests for sme/adapters/adaptmem_adapter.py — issue #81.

The adapter wraps `adaptmem.AdaptMem`, which is an optional install.
Tests stub it via monkeypatch so they run without sentence_transformers
or the actual AdaptMem package present.
"""
from __future__ import annotations

import sys
import types

import pytest

from sme.adapters.base import QueryResult


class _FakeHit:
    def __init__(self, chunk_id: str, text: str, score: float):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score


class _FakeAdaptMem:
    """Stand-in for adaptmem.AdaptMem — captures call args for assertions."""

    rerank_enabled = False

    def __init__(self, hits: list[_FakeHit] | None = None, raise_on_search: bool = False):
        self._hits = hits or []
        self._raise = raise_on_search
        self.last_search_kwargs: dict | None = None

    @classmethod
    def load(cls, path: str):
        # store the path in the class so tests can assert
        inst = cls(hits=cls._test_hits)
        inst._test_path = path
        return inst

    def search(self, question, top_k):
        self.last_search_kwargs = {"question": question, "top_k": top_k}
        if self._raise:
            raise RuntimeError("search blew up")
        return self._hits


def _install_fake_adaptmem(monkeypatch, hits: list[_FakeHit], raise_on_search: bool = False):
    """Inject a fake `adaptmem` module so `from adaptmem import AdaptMem`
    resolves to our stub. Returns the FakeAdaptMem class so tests can
    set hits/raise state per-test."""
    fake_mod = types.ModuleType("adaptmem")
    _FakeAdaptMem._test_hits = hits  # type: ignore[attr-defined]

    class _FakeAdaptMemForCase(_FakeAdaptMem):
        rerank_enabled = False
        _test_hits = hits

    if raise_on_search:
        def _raising(self, question, top_k):
            self.last_search_kwargs = {"question": question, "top_k": top_k}
            raise RuntimeError("search blew up")

        _FakeAdaptMemForCase.search = _raising  # type: ignore[method-assign]

    fake_mod.AdaptMem = _FakeAdaptMemForCase
    monkeypatch.setitem(sys.modules, "adaptmem", fake_mod)
    return _FakeAdaptMemForCase


def test_query_returns_entities_for_each_hit(monkeypatch, tmp_path):
    hits = [
        _FakeHit("c1", "PostgreSQL JSONB index basics.", 0.92),
        _FakeHit("c2", "MongoDB aggregation pipelines.", 0.81),
        _FakeHit("c3", "Redis pub/sub patterns.", 0.74),
    ]
    _install_fake_adaptmem(monkeypatch, hits)

    from sme.adapters.adaptmem_adapter import AdaptMemAdapter
    adapter = AdaptMemAdapter(tmp_path / "model", n_results=5)
    result = adapter.query("postgres json")

    assert isinstance(result, QueryResult)
    assert result.error is None
    assert len(result.retrieved_entities) == 3
    assert result.retrieved_entities[0].id == "c1"
    assert result.retrieved_entities[0].name == "c1"
    assert result.retrieved_entities[0].entity_type == "adaptmem_hit"
    assert result.retrieved_entities[0].properties["score"] == 0.92
    assert result.retrieved_entities[0].properties["rank"] == 1
    # Context string contains all hits, rank-numbered
    assert "[1] c1" in result.context_string
    assert "[3] c3" in result.context_string


def test_query_respects_n_results_override(monkeypatch, tmp_path):
    hits = [_FakeHit(f"c{i}", f"text{i}", 0.5) for i in range(10)]
    _install_fake_adaptmem(monkeypatch, hits)

    from sme.adapters.adaptmem_adapter import AdaptMemAdapter
    adapter = AdaptMemAdapter(tmp_path / "model", n_results=5)
    adapter.query("question", n_results=3)

    # The adapter forwarded top_k=3 (not its default 5) to AdaptMem.search.
    # adapter._am is the FakeAdaptMem instance produced by .load().
    assert adapter._am.last_search_kwargs == {"question": "question", "top_k": 3}


def test_query_default_n_results(monkeypatch, tmp_path):
    hits = [_FakeHit("c1", "text", 0.9)]
    _install_fake_adaptmem(monkeypatch, hits)

    from sme.adapters.adaptmem_adapter import AdaptMemAdapter
    adapter = AdaptMemAdapter(tmp_path / "model", n_results=7)
    adapter.query("question")
    assert adapter._am.last_search_kwargs == {"question": "question", "top_k": 7}


def test_query_search_failure_returns_error_in_result(monkeypatch, tmp_path):
    _install_fake_adaptmem(monkeypatch, [], raise_on_search=True)

    from sme.adapters.adaptmem_adapter import AdaptMemAdapter
    adapter = AdaptMemAdapter(tmp_path / "model")
    result = adapter.query("question")

    assert result.error is not None
    assert "RuntimeError" in result.error
    assert "search blew up" in result.error
    assert result.retrieved_entities == []
    assert result.context_string == ""


def test_ingest_corpus_raises_not_implemented(monkeypatch, tmp_path):
    _install_fake_adaptmem(monkeypatch, [])

    from sme.adapters.adaptmem_adapter import AdaptMemAdapter
    adapter = AdaptMemAdapter(tmp_path / "model")
    with pytest.raises(NotImplementedError, match="pre-trained"):
        adapter.ingest_corpus([{"id": "p1", "text": "hello"}])


def test_get_graph_snapshot_returns_empty(monkeypatch, tmp_path):
    _install_fake_adaptmem(monkeypatch, [])

    from sme.adapters.adaptmem_adapter import AdaptMemAdapter
    adapter = AdaptMemAdapter(tmp_path / "model")
    ents, edges = adapter.get_graph_snapshot()
    assert ents == []
    assert edges == []


def test_missing_adaptmem_import_gives_clear_error(monkeypatch, tmp_path):
    # Force the import to fail by removing adaptmem from sys.modules
    # and patching the import machinery
    monkeypatch.setitem(sys.modules, "adaptmem", None)

    # Need to also pop the adapter module if already imported so __init__
    # tries the import fresh
    sys.modules.pop("sme.adapters.adaptmem_adapter", None)

    from sme.adapters.adaptmem_adapter import AdaptMemAdapter
    with pytest.raises(RuntimeError, match="adaptmem.* package"):
        AdaptMemAdapter(tmp_path / "model")
