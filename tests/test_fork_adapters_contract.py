"""SMEAdapter conformance testkit for fork-only adapters.

Upstream's ``tests/test_adapter_contract.py`` ships with three reference
adapters (``mock``, ``flat_baseline``, ``full_context``). This fork adds
seven more — ``rlm``, ``longhand``, ``omega``, ``hindsight``, ``mem0``,
``random_retrieval``, ``oracle_retrieval`` — and they deserve the same
contract coverage.

The earlier approach added those factories to the upstream test file
directly. That worked but caused merge drift every time upstream touched
its testkit. The new approach (per Gemini review on PR #38):

  - Keep ``tests/test_adapter_contract.py`` bit-identical to upstream.
  - Import its nine contract test functions into this file.
  - Define an ``adapter`` fixture parametrized over the fork-only
    factories below.

When pytest collects the imported test functions in *this* module, it
resolves the ``adapter`` fixture from this module's namespace — so the
same nine assertions run once over upstream's adapters (via the upstream
file) and once over the fork's adapters (via this file). No duplicated
assertions, no upstream drift.

Fork adapters that need missing libraries / external services skip
themselves with ``pytest.skip`` from inside the factory — the skip
reason shows in pytest output and CI runs clean even on minimal envs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from sme.adapters.base import SMEAdapter
from sme.cli import _load_adapter, _registry_by_alias

# Re-bind the upstream contract assertions in this module's namespace so
# pytest re-collects them here under this module's ``adapter`` fixture.
# The ``as`` rename keeps the public test name identical to upstream's
# (pytest output groups them per-file).
from tests.test_adapter_contract import (  # noqa: F401
    test_close_is_idempotent as test_close_is_idempotent,
    test_get_harness_manifest_returns_list as test_get_harness_manifest_returns_list,
    test_get_ontology_source_returns_typed_dict as test_get_ontology_source_returns_typed_dict,
    test_graph_snapshot_internally_consistent as test_graph_snapshot_internally_consistent,
    test_graph_snapshot_returns_typed_pair as test_graph_snapshot_returns_typed_pair,
    test_ingest_corpus_accepts_list_of_dicts as test_ingest_corpus_accepts_list_of_dicts,
    test_is_sme_adapter_subclass as test_is_sme_adapter_subclass,
    test_query_returns_QueryResult as test_query_returns_QueryResult,
    test_query_without_n_results_kwarg as test_query_without_n_results_kwarg,
)


# --- Adapter factories ------------------------------------------------


AdapterFactory = Callable[[Path], SMEAdapter]


def _random_retrieval_factory(tmp_path: Path) -> SMEAdapter:
    """RandomRetrievalAdapter (TREC lower bound) — pure in-memory, no env."""
    from sme.adapters.random_retrieval import RandomRetrievalAdapter

    return RandomRetrievalAdapter()


def _oracle_retrieval_factory(tmp_path: Path) -> SMEAdapter:
    """OracleRetrievalAdapter (TREC upper bound) — pure in-memory, no env."""
    from sme.adapters.oracle_retrieval import OracleRetrievalAdapter

    return OracleRetrievalAdapter()


def _rlm_factory(tmp_path: Path) -> SMEAdapter:
    """RlmAdapter — requires a compatible ``rlm`` library. The import and
    ``RLM(**kwargs)`` construction happen inside ``__init__``. ``rlm`` is
    not a project dependency, so CI skips via ImportError. Locally an
    incompatible ``rlm`` (e.g. an editable checkout with a different
    constructor) raises TypeError — also an environmental skip, not a
    contract failure in RlmAdapter."""
    from sme.adapters.rlm_adapter import RlmAdapter

    try:
        return RlmAdapter(api_url="http://127.0.0.1:0", environment="local")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rlm library not installed")
    except TypeError as e:
        pytest.skip(f"installed rlm has an incompatible constructor: {e}")


def _longhand_factory(tmp_path: Path) -> SMEAdapter:
    """LonghandAdapter — requires the ``longhand`` CLI on PATH. Skips
    cleanly when absent (the adapter raises ValueError with a known
    message)."""
    from sme.adapters.longhand import LonghandAdapter

    try:
        return LonghandAdapter(home_dir=str(tmp_path))
    except ValueError as e:
        if "not found on PATH" in str(e):
            pytest.skip("longhand CLI not on PATH")
        raise


def _omega_factory(tmp_path: Path) -> SMEAdapter:
    """OmegaAdapter — requires ``omega-memory``; skips when unavailable."""
    from sme.adapters.omega import OmegaAdapter

    try:
        return OmegaAdapter(db_path=str(tmp_path / "omega.sqlite"))
    except ImportError:
        pytest.skip("omega-memory not installed")


def _hindsight_factory(tmp_path: Path) -> SMEAdapter:
    """HindsightAdapter — constructable without backend; ``query`` and
    ``get_graph_snapshot`` return graceful empty/error results when the
    URL is unreachable. ``127.0.0.1:0`` fails fast on connect."""
    from sme.adapters.hindsight import HindsightAdapter

    return HindsightAdapter(base_url="http://127.0.0.1:0", api_timeout=0.5)


def _mempalace_server_factory(tmp_path: Path) -> SMEAdapter:
    """MemPalaceServerAdapter (Go server) against an unreachable URL.

    Constructs without connecting; every method degrades gracefully when
    the server is absent (query/ingest return error-bearing results,
    get_graph_snapshot returns ([], [])), so the contract runs clean with
    no live server. ``127.0.0.1:0`` fails fast on connect."""
    from sme.adapters.mempalace_server_adapter import MemPalaceServerAdapter

    return MemPalaceServerAdapter(
        api_url="http://127.0.0.1:0",
        api_key="contract-test",
        api_timeout=0.5,
        reset_before_ingest=False,
    )


def _engram_factory(tmp_path: Path) -> SMEAdapter:
    """EngramAdapter (TypeScript MCP server) with no runtime available.

    engram_path unset and no injected transport → ingest/query degrade to
    error-bearing results, and get_graph_snapshot reads a nonexistent
    SQLite DB → ([], []). Constructs and runs the whole contract clean with
    no Node runtime."""
    from sme.adapters.engram_adapter import EngramAdapter

    return EngramAdapter(
        engram_path=None,
        db_path=str(tmp_path / "engramdb"),
        reset_before_ingest=False,
    )


def _mem0_factory(tmp_path: Path) -> SMEAdapter:
    """Mem0Adapter — requires the ``mem0`` OSS library; skips when absent.

    A bare ``Mem0Adapter()`` builds a default ``mem0.Memory`` which tries to
    stand up the OpenAI LLM/embedder and raises ``OpenAIError`` without a key
    (and a local-ollama backend needs a live ollama). The contract test only
    exercises the SMEAdapter *interface*, not mem0's backend, so inject a
    minimal stub ``Memory`` via the adapter's ``memory=`` seam — same approach
    as ``tests/test_mem0_adapter.py``. Skip only when the library is absent."""
    try:
        import mem0  # noqa: F401
    except ImportError:
        pytest.skip("mem0 not installed")

    from sme.adapters.mem0 import Mem0Adapter

    class _StubMemory:
        def add(self, messages, user_id=None, **kw):
            return {"results": []}

        def search(self, query, filters=None, top_k=10, **kw):
            return {"results": []}

        def get_all(self, filters=None, **kw):
            return {"results": []}

        def delete_all(self, user_id=None, filters=None):
            return None

    return Mem0Adapter(memory=_StubMemory())


# Register fork-only adapters here. Keep IDs stable — they show in
# pytest output (e.g. ``test_query_returns_QueryResult[rlm]``).
ADAPTER_FACTORIES: dict[str, AdapterFactory] = {
    "random_retrieval": _random_retrieval_factory,
    "oracle_retrieval": _oracle_retrieval_factory,
    "rlm": _rlm_factory,
    "longhand": _longhand_factory,
    "omega": _omega_factory,
    "hindsight": _hindsight_factory,
    "mem0": _mem0_factory,
    "mempalace_server": _mempalace_server_factory,
    "engram": _engram_factory,
}


@pytest.fixture(params=sorted(ADAPTER_FACTORIES.keys()))
def adapter(request: pytest.FixtureRequest, tmp_path: Path) -> SMEAdapter:
    """Fork-adapter fixture. Shadows the upstream module's fixture for
    every contract test re-collected in this file."""
    factory = ADAPTER_FACTORIES[request.param]
    return factory(tmp_path)


# --- CLI kwarg-forwarding (fork adapters) -----------------------------
#
# The registry-wide invariants (rename targets in accepts, sources not in
# accepts) are checked structurally in tests/test_cli_adapter_forwarding.py.
# These tests assert the *runtime* behavior of `_load_adapter` for the
# fork adapters that have non-trivial forwarding: hindsight renames
# `api_url` -> `base_url`, rlm accepts `api_url` as-is, and both must drop
# unknown kwargs at the boundary (the PR #7 regression class).


class _StubAdapter:
    """Throw-away adapter that captures whatever kwargs reach it."""

    def __init__(self, **kwargs: Any) -> None:
        self.captured_kwargs = kwargs


@pytest.fixture
def stub_loader():
    """Swap an adapter's loader for ``_StubAdapter``, restoring on teardown.

    ``_AdapterSpec`` is a frozen dataclass — ``object.__setattr__`` bypasses
    that. The restore step keeps mutations from bleeding between tests.
    """
    restores: list[tuple[Any, Any]] = []

    def _patch(alias: str) -> None:
        spec = _registry_by_alias()[alias]
        # Record the true original only on first patch of this spec, so a
        # repeat _patch() in one test can't capture the stub as "original".
        if not any(item[0] is spec for item in restores):
            restores.append((spec, spec.loader))
        object.__setattr__(spec, "loader", lambda: _StubAdapter)

    yield _patch

    for spec, original in restores:
        object.__setattr__(spec, "loader", original)


def test_hindsight_renames_api_url_to_base_url(stub_loader):
    """hindsight's registry spec renames ``api_url`` -> ``base_url``; the
    rename must fire at runtime, and ``api_url`` must not leak through."""
    stub_loader("hindsight")
    out = _load_adapter("hindsight", api_url="http://nowhere:1", api_timeout=0.5)
    assert isinstance(out, _StubAdapter)
    assert out.captured_kwargs["base_url"] == "http://nowhere:1"
    assert out.captured_kwargs["api_timeout"] == 0.5
    assert "api_url" not in out.captured_kwargs


def test_rlm_passes_api_url_through(stub_loader):
    """rlm has no rename — it accepts ``api_url`` directly, so the kwarg
    must reach the constructor unchanged (not renamed to ``base_url``)."""
    stub_loader("rlm")
    out = _load_adapter("rlm", api_url="http://nowhere:1", environment="local")
    assert out.captured_kwargs["api_url"] == "http://nowhere:1"
    assert out.captured_kwargs["environment"] == "local"
    assert "base_url" not in out.captured_kwargs


def test_hindsight_drops_unknown_kwargs(stub_loader):
    """The PR #7 regression class: a new CLI flag (or another adapter's
    kwarg) must not blow up hindsight just by being present in the bag."""
    stub_loader("hindsight")
    out = _load_adapter(
        "hindsight",
        api_url="http://nowhere:1",
        # All foreign to hindsight's accepts — must be dropped.
        db_path="/tmp/db",
        collection_name="drawers",
        environment="local",
        backend="x",
        some_future_flag_that_doesnt_exist=42,
    )
    assert out.captured_kwargs == {"base_url": "http://nowhere:1"}


def test_fork_adapter_none_valued_kwargs_stripped(stub_loader):
    """``None`` means 'use the adapter default' — never forward as-is."""
    stub_loader("hindsight")
    out = _load_adapter(
        "hindsight",
        api_url="http://nowhere:1",
        n_results=None,
        bank_id=None,
    )
    assert out.captured_kwargs == {"base_url": "http://nowhere:1"}
    assert "n_results" not in out.captured_kwargs
    assert "bank_id" not in out.captured_kwargs
