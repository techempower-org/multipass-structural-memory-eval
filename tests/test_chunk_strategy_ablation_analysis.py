"""Unit tests for scripts/chunk_strategy_ablation_analysis.py — issue #85."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "chunk_strategy_ablation_analysis",
    Path(__file__).resolve().parents[1] / "scripts"
    / "chunk_strategy_ablation_analysis.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_mrr():
    probes = [{"rank": 1}, {"rank": 2}, {"rank": None}]
    assert mod.mrr(probes) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_mrr_empty():
    assert mod.mrr([]) == 0.0


def test_recall_at_k():
    probes = [{"rank": 1}, {"rank": 5}, {"rank": 6}, {"rank": None}]
    assert mod.recall_at_k(probes, 5) == pytest.approx(0.5)
    assert mod.recall_at_k(probes, 10) == pytest.approx(0.75)


def test_filter_probes_by_suffix():
    probes = [
        {"expected": "README.md", "rank": 1},
        {"expected": "main.py", "rank": 2},
        {"expected": "guide.md", "rank": None},
    ]
    md = mod.filter_probes(probes, ".md")
    assert {p["expected"] for p in md} == {"README.md", "guide.md"}
    assert mod.filter_probes(probes, None) == probes


def test_strategy_metrics():
    src = {"strategies": {"A": {"probes": [{"rank": 1, "expected": "a.md"},
                                           {"rank": None, "expected": "b.py"}]}}}
    m = mod.strategy_metrics(src, "A")
    assert m["n_probes"] == 2
    assert m["mrr"] == pytest.approx(0.5)
    assert m["recall_at_5"] == pytest.approx(0.5)


def test_strategy_metrics_markdown_only():
    src = {"strategies": {"A": {"probes": [{"rank": 1, "expected": "a.md"},
                                           {"rank": None, "expected": "b.py"}]}}}
    m = mod.strategy_metrics(src, "A", suffix=".md")
    assert m["n_probes"] == 1
    assert m["mrr"] == pytest.approx(1.0)


def test_ablation_deltas():
    src = {"strategies": {
        "A": {"probes": [{"rank": 2, "expected": "x.md"}]},   # mrr 0.5
        "B": {"probes": [{"rank": 1, "expected": "x.md"}]},   # mrr 1.0
    }}
    d = mod.ablation_deltas(src, ["A", "B"])
    assert d["metrics"]["A"]["mrr"] == pytest.approx(0.5)
    assert d["metrics"]["B"]["mrr"] == pytest.approx(1.0)
    assert d["deltas"]["B_minus_A_mrr"] == pytest.approx(0.5)


def test_encoder_swap_compression():
    # baseline: B beats A by 0.5; ft: B beats A by 0.1 -> compression 0.4
    baseline = {"strategies": {
        "A": {"probes": [{"rank": 2, "expected": "x.md"}]},   # 0.5
        "B": {"probes": [{"rank": 1, "expected": "x.md"}]},   # 1.0
    }}
    ft = {"strategies": {
        "A": {"probes": [{"rank": 2, "expected": "x.md"}]},   # 0.5
        "B": {"probes": [{"rank": 2, "expected": "x.md"}]},   # 0.5
    }}
    comp = mod.encoder_swap_compression(baseline, ft, ["A", "B"])
    c = comp["B_minus_A_mrr"]
    assert c["baseline_delta"] == pytest.approx(0.5)
    assert c["ft_delta"] == pytest.approx(0.0)
    assert c["compression"] == pytest.approx(0.5)


def test_build_report_baseline_only_no_compression():
    baseline = {"strategies": {"A": {"probes": [{"rank": 1, "expected": "x.md"}]}}}
    report = mod.build_report(baseline, None)
    assert report["strategies"] == ["A"]
    assert "baseline" in report["by_encoder"]
    assert "FT-300" not in report["by_encoder"]
    assert "encoder_swap_compression" not in report


def test_main_end_to_end(tmp_path):
    base_f = tmp_path / "base.json"
    ft_f = tmp_path / "ft.json"
    base_f.write_text(json.dumps({"strategies": {
        "A_paragraph__cs800": {"probes": [{"rank": 2, "expected": "x.md"}]},
        "B_heading_md__cs800": {"probes": [{"rank": 1, "expected": "x.md"}]},
    }}))
    ft_f.write_text(json.dumps({"strategies": {
        "A_paragraph__cs800": {"probes": [{"rank": 2, "expected": "x.md"}]},
        "B_heading_md__cs800": {"probes": [{"rank": 2, "expected": "x.md"}]},
    }}))
    out = tmp_path / "out.json"
    rc = mod.main(["--baseline", str(base_f), "--ft", str(ft_f),
                   "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert "encoder_swap_compression" in report
    comp = report["encoder_swap_compression"]["all"]
    key = "B_heading_md__cs800_minus_A_paragraph__cs800_mrr"
    assert comp[key]["compression"] == pytest.approx(0.5)
