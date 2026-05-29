#!/usr/bin/env python3
"""Re-aggregate drawer_hit_at_K with the #98 fix on existing run JSONs.

The 2026-05-28 chain rerun JSONs were produced before #98 shipped, so
their ``drawer_hit_at_K`` fields are uniformly low — the daemon returns
chunked drawer IDs (``<parent>_chunk_NNNNNN``) but the matcher compared
exact strings against the parent IDs we stored at ingest. This script
walks each per-question record, strips the suffix on the retrieved
side, and recomputes hit_at_K — no new bench compute, just rescoring
data we've already paid for.

The script is non-destructive: it writes a sidecar
``<input>.reagg.json`` with the corrected records and a summary block.

Usage:
    scripts/reaggregate_drawer_hits.py \\
        baselines/longmemeval_mempalace_daemon_2026-05-28-rerun.json \\
        baselines/longmemeval_age_fused_2026-05-28-rerun.json \\
        baselines/longmemeval_familiar_2026-05-28-rerun.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Same regex shape as scripts/run_longmemeval_mempalace.py's
# _drawer_parent_id — copied here so this script is standalone.
_CHUNK_SUFFIX_RE = re.compile(r"_chunk_\d+$")


def parent_id(drawer_id):
    if not drawer_id:
        return drawer_id
    return _CHUNK_SUFFIX_RE.sub("", str(drawer_id))


def recompute(record):
    """Return a copy of ``record`` with updated drawer_hit_at_K fields."""
    rec = dict(record)
    expected = set(rec.get("expected_drawer_ids") or [])
    raw = list(rec.get("retrieved_drawer_ids") or [])
    parents = [parent_id(d) for d in raw]
    rec["retrieved_parent_ids"] = parents
    rec["drawer_hit_at_1"] = bool(
        expected and parents and parents[0] in expected
    )
    rec["drawer_hit_at_5"] = bool(
        expected and any(p in expected for p in parents[:5])
    )
    rec["drawer_hit_at_10"] = bool(
        expected and any(p in expected for p in parents[:10])
    )
    return rec


def reaggregate(path: Path) -> dict:
    data = json.loads(path.read_text())
    per_q = data.get("per_question") or []
    new_pq = [recompute(r) for r in per_q]
    data["per_question"] = new_pq

    by_cat = defaultdict(lambda: {"n": 0, "h1": 0, "h5": 0, "h10": 0})
    overall = {"n": 0, "h1": 0, "h5": 0, "h10": 0}
    for r in new_pq:
        c = r.get("sme_category", "?")
        for bucket in (by_cat[c], overall):
            bucket["n"] += 1
            bucket["h1"] += int(r.get("drawer_hit_at_1") or False)
            bucket["h5"] += int(r.get("drawer_hit_at_5") or False)
            bucket["h10"] += int(r.get("drawer_hit_at_10") or False)
    summary_drawer = {
        "overall": {
            "n": overall["n"],
            "R@1": round(overall["h1"] / overall["n"], 4) if overall["n"] else 0.0,
            "R@5": round(overall["h5"] / overall["n"], 4) if overall["n"] else 0.0,
            "R@10": round(overall["h10"] / overall["n"], 4) if overall["n"] else 0.0,
        },
        "per_category": {
            c: {
                "n": b["n"],
                "R@1": round(b["h1"] / b["n"], 4) if b["n"] else 0.0,
                "R@5": round(b["h5"] / b["n"], 4) if b["n"] else 0.0,
                "R@10": round(b["h10"] / b["n"], 4) if b["n"] else 0.0,
            }
            for c, b in by_cat.items()
        },
    }
    data.setdefault("summary", {})["drawer_hit_post_98"] = summary_drawer
    return data


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path, help="rerun JSON files to re-aggregate")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite the input file instead of writing a .reagg.json sidecar")
    args = p.parse_args(argv)

    for ip in args.inputs:
        if not ip.exists():
            print(f"skip {ip} (not found)", file=sys.stderr)
            continue
        data = reaggregate(ip)
        op = ip if args.inplace else ip.with_name(ip.stem + ".reagg.json")
        op.write_text(json.dumps(data, indent=2))
        sd = data["summary"]["drawer_hit_post_98"]["overall"]
        print(f"  {ip.name}  ->  {op.name}")
        print(f"    R@1={sd['R@1']:.4f}  R@5={sd['R@5']:.4f}  R@10={sd['R@10']:.4f}  (n={sd['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
