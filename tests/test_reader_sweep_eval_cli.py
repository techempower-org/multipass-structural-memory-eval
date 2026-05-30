"""CLI-level tests for scripts/reader_sweep_eval.py — the phi4-default /
``--headline`` mode resolution.

These exercise argument parsing + ``_resolve_models`` only; they never make an
LLM call, so they're CI-safe (no ollama / Azure / Bedrock needed).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# The harness lives in scripts/, not the importable package — load it by path.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "reader_sweep_eval.py"
_spec = importlib.util.spec_from_file_location("reader_sweep_eval", _SCRIPT)
rse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rse)


def _parse(argv):
    """Parse argv through the REAL production parser, so the test can't drift
    from the actual CLI surface."""
    return rse.build_parser().parse_args(argv)


# --- exploratory (default) is local phi4 -----------------------------------


def test_default_reader_sweep_is_local_phi4():
    args = _parse(["reader-sweep", "--pinned", "p.json"])
    rse._resolve_models(args)
    assert args.reader_models == ["phi4"]
    assert args.judge == "phi4"
    assert args.headline is False


def test_default_dry_run_is_local_phi4_and_has_no_judge():
    args = _parse(["dry-run", "--pinned", "p.json"])
    rse._resolve_models(args)
    assert args.reader_models == ["phi4"]
    # dry-run has no --judge attribute; _resolve_models must not invent one.
    assert not hasattr(args, "judge")


# --- --headline opts into Azure/Bedrock + canonical judge ------------------


def test_headline_uses_azure_bedrock_readers_and_canonical_judge():
    args = _parse(["reader-sweep", "--headline", "--pinned", "p.json"])
    rse._resolve_models(args)
    assert args.reader_models == rse.HEADLINE_READERS
    assert args.judge == rse.HEADLINE_JUDGE == "gpt-4o"


# --- explicit values always win (back-compat) ------------------------------


def test_explicit_reader_models_win_over_exploratory_default():
    args = _parse(["reader-sweep", "--pinned", "p.json",
                   "--reader-models", "gpt-4o", "gpt-4.1-mini"])
    rse._resolve_models(args)
    assert args.reader_models == ["gpt-4o", "gpt-4.1-mini"]
    # judge still falls back to the exploratory local default
    assert args.judge == "phi4"


def test_explicit_judge_wins_over_exploratory_default():
    args = _parse(["reader-sweep", "--pinned", "p.json", "--judge", "gpt-4o"])
    rse._resolve_models(args)
    assert args.judge == "gpt-4o"
    assert args.reader_models == ["phi4"]  # reader still local


def test_explicit_values_win_even_with_headline_flag():
    args = _parse(["reader-sweep", "--headline", "--pinned", "p.json",
                   "--reader-models", "qwen2.5:14b-instruct-q4_K_M",
                   "--judge", "phi4"])
    rse._resolve_models(args)
    # explicit local models override even when --headline is set
    assert args.reader_models == ["qwen2.5:14b-instruct-q4_K_M"]
    assert args.judge == "phi4"


def test_resolve_is_idempotent():
    """Calling _resolve_models twice must not change an already-resolved value
    (guards against a future double-call clobbering explicit input)."""
    args = _parse(["reader-sweep", "--pinned", "p.json", "--judge", "gpt-4o"])
    rse._resolve_models(args)
    rse._resolve_models(args)
    assert args.judge == "gpt-4o"
    assert args.reader_models == ["phi4"]
