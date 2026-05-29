"""Unit tests for the chunk-suffix-stripping helper (closes #98).

Lives in run_longmemeval_mempalace.py rather than a shared util so the
bench script stays a self-contained entry point.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "_run_lme",
    Path(__file__).resolve().parents[1] / "scripts" / "run_longmemeval_mempalace.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)
_drawer_parent_id = _mod._drawer_parent_id


def test_no_suffix_returns_unchanged():
    assert _drawer_parent_id("drawer_lme_X_references_AAA") == "drawer_lme_X_references_AAA"


def test_strip_chunk_suffix():
    assert (
        _drawer_parent_id("drawer_lme_X_references_AAA_chunk_000001")
        == "drawer_lme_X_references_AAA"
    )


def test_strip_short_suffix():
    # The suffix matches any number of digits, not just 6
    assert (
        _drawer_parent_id("drawer_lme_X_references_AAA_chunk_3")
        == "drawer_lme_X_references_AAA"
    )


def test_idempotent():
    once = _drawer_parent_id("drawer_X_chunk_000001")
    twice = _drawer_parent_id(once)
    assert once == twice == "drawer_X"


def test_empty_string():
    assert _drawer_parent_id("") == ""


def test_none():
    assert _drawer_parent_id(None) is None


def test_integer_id_casts_to_str():
    # Defensive: if a daemon returns an int PK, the strip is a no-op cast.
    assert _drawer_parent_id(123) == "123"


def test_suffix_only_at_end():
    # _chunk_NNNNNN in the middle should NOT be stripped — daemon's chunk
    # tagging is always trailing.
    assert (
        _drawer_parent_id("drawer_X_chunk_5_real_continues_here")
        == "drawer_X_chunk_5_real_continues_here"
    )


def test_chunk_zero():
    # Common edge — first chunk is _chunk_000000
    assert (
        _drawer_parent_id("drawer_lme_X_references_AAA_chunk_000000")
        == "drawer_lme_X_references_AAA"
    )


def test_realistic_lme_id():
    # Exact shape from #98's evidence
    assert (
        _drawer_parent_id(
            "drawer_lme_gpt4_2655b836_references_ac68e2f1ca0d9e52df52cf2c_chunk_000001"
        )
        == "drawer_lme_gpt4_2655b836_references_ac68e2f1ca0d9e52df52cf2c"
    )
