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
from typing import Callable

import pytest

from sme.adapters.base import SMEAdapter

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


def _mem0_factory(tmp_path: Path) -> SMEAdapter:
    """Mem0Adapter — requires ``mem0`` OSS library; skips when absent."""
    from sme.adapters.mem0 import Mem0Adapter

    try:
        return Mem0Adapter()
    except ImportError:
        pytest.skip("mem0 not installed")


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
}


@pytest.fixture(params=sorted(ADAPTER_FACTORIES.keys()))
def adapter(request: pytest.FixtureRequest, tmp_path: Path) -> SMEAdapter:
    """Fork-adapter fixture. Shadows the upstream module's fixture for
    every contract test re-collected in this file."""
    factory = ADAPTER_FACTORIES[request.param]
    return factory(tmp_path)
