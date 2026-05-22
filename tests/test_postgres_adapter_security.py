"""Security regression tests for PostgresIngestAdapter and PostgresAgeIngestAdapter.

Covers:
  * issue #1 — DEFAULT_DSN no longer contains hard-coded credentials and the
    adapter refuses to construct when the env var isn't set.
  * issue #2 — Cypher string-literal escaping handles the ``$$`` dollar-quote
    terminator, backslashes, newlines, and single quotes; payloads that would
    break out of the dollar-tagged block are refused.

The Cypher tests exercise the module-level helper ``_cypher_str_lit`` and the
dollar-tag pattern only. Constructing PostgresAgeIngestAdapter requires a live
Postgres + AGE instance, which is intentionally out of scope for unit tests.
"""

from __future__ import annotations

import importlib

import pytest


def _reload_postgres_ingest():
    """Reimport postgres_ingest so DEFAULT_DSN picks up the current env state."""
    from sme.adapters import postgres_ingest

    return importlib.reload(postgres_ingest)


def test_default_dsn_is_empty_when_env_unset(monkeypatch):
    monkeypatch.delenv("SME_POSTGRES_DSN", raising=False)
    module = _reload_postgres_ingest()
    assert module.DEFAULT_DSN == ""


def test_default_dsn_reads_env(monkeypatch):
    monkeypatch.setenv("SME_POSTGRES_DSN", "postgresql://x:y@h:5432/d")
    module = _reload_postgres_ingest()
    assert module.DEFAULT_DSN == "postgresql://x:y@h:5432/d"


def test_no_hardcoded_credentials_in_source():
    """Defensive check by shape, not value: no ``postgresql://<creds>@host``
    URI may appear as a literal in the postgres adapter source. The original
    issue baked a real password + internal IP into ``DEFAULT_DSN``; this guard
    catches anyone who tries to put one back."""
    import pathlib
    import re

    adapter_dir = pathlib.Path(__file__).resolve().parent.parent / "sme" / "adapters"
    # Matches postgresql://user:password@host — the shape that always leaks
    # creds. ``postgresql://`` without an embedded user (e.g. inside an error
    # message or as a Unix-socket DSN) is fine and intentionally allowed.
    creds_in_uri = re.compile(r"postgresql://[^\s'\"@/]+:[^\s'\"@/]+@")
    for path in adapter_dir.glob("postgres*.py"):
        text = path.read_text(encoding="utf-8")
        assert not creds_in_uri.search(text), (
            f"hard-coded postgresql://user:password@... URI found in {path}"
        )


def test_adapter_refuses_to_construct_without_dsn(monkeypatch):
    monkeypatch.delenv("SME_POSTGRES_DSN", raising=False)
    module = _reload_postgres_ingest()
    with pytest.raises(RuntimeError, match="SME_POSTGRES_DSN"):
        module.PostgresIngestAdapter()


def test_cypher_str_lit_escapes_single_quote():
    from sme.adapters.postgres_age_ingest import _cypher_str_lit

    assert _cypher_str_lit("Al's") == "'Al\\'s'"


def test_cypher_str_lit_escapes_backslash_before_quote():
    """Backslash must be escaped first so a value like ``foo\\'bar`` becomes
    ``'foo\\\\\\'bar'`` (escaped-backslash then escaped-quote), not
    ``'foo\\\\'bar'`` (which closes the literal early)."""
    from sme.adapters.postgres_age_ingest import _cypher_str_lit

    result = _cypher_str_lit("foo\\'bar")
    assert result == "'foo\\\\\\'bar'"


def test_cypher_str_lit_escapes_newlines():
    from sme.adapters.postgres_age_ingest import _cypher_str_lit

    assert _cypher_str_lit("line1\nline2") == "'line1\\nline2'"
    assert _cypher_str_lit("line1\r\nline2") == "'line1\\r\\nline2'"


def test_cypher_str_lit_rejects_dollar_tag():
    """A payload containing the Cypher dollar-quote tag must be rejected so
    it can't terminate the outer Postgres dollar-quoted block (issue #2)."""
    from sme.adapters.postgres_age_ingest import _cypher_str_lit

    with pytest.raises(ValueError, match="dollar-quote tag"):
        _cypher_str_lit("safe $sme_cypher$ injected")


def test_cypher_str_lit_allows_bare_double_dollar():
    """The original ``$$`` was the unsafe delimiter; with the new tagged
    delimiter, a bare ``$$`` in a value is harmless and must round-trip."""
    from sme.adapters.postgres_age_ingest import _cypher_str_lit

    assert _cypher_str_lit("price$$10") == "'price$$10'"


def test_cypher_str_lit_coerces_non_strings():
    from sme.adapters.postgres_age_ingest import _cypher_str_lit

    assert _cypher_str_lit(42) == "'42'"
