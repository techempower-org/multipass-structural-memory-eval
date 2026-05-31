#!/usr/bin/env python3
"""corpus-doctor CLI — inject known defects, verify the categories catch them.

Runs the inject → detect → assert loop (sme.corpus_doctor) over a clean
graph snapshot and reports, per defect type, whether the relevant SME
category recalled the KNOWN injected defects. Exit code is non-zero if
any injected defect went undetected — so this doubles as a CI guard that
the categories still detect what they claim to.

Defect types (first slice of upstream M0nkeyFl0wer#27):
  duplicate_entity → Cat 4a canonical-collision dedup
  orphan_node      → Cat 5 isolated-node detection
  broken_ref       → referential-integrity check

The default corpus is the committed good-dog graph (no download, no
daemon, no API key — constitutional: lightweight and locally runnable).

Usage:
    python scripts/corpus_doctor.py
    python scripts/corpus_doctor.py --count 10 --seed 3
    python scripts/corpus_doctor.py --defect orphan_node --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sme.corpus_doctor import DEFECT_TYPES, run_all_defects  # noqa: E402


def _load_clean_graph():
    """Load the default clean corpus (good-dog graph)."""
    from sme.corpora import good_dog_graph

    return good_dog_graph.load_graph()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inject known structural defects and verify the SME "
        "categories detect them (upstream #27, first slice).",
    )
    p.add_argument(
        "--defect",
        action="append",
        choices=list(DEFECT_TYPES),
        help="Restrict to specific defect type(s); repeatable. "
        "Default: all three.",
    )
    p.add_argument(
        "--count", type=int, default=5,
        help="Number of defects to inject per type (default 5).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for deterministic injection (default 0).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit the detection results as JSON instead of a table.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    entities, edges = _load_clean_graph()
    defect_types = tuple(args.defect) if args.defect else None

    results = run_all_defects(
        entities, edges, count=args.count, seed=args.seed,
        defect_types=defect_types,
    )

    all_detected = all(r.detected_all for r in results.values())

    if args.json:
        payload = {
            "clean_graph": {"entities": len(entities), "edges": len(edges)},
            "count_per_type": args.count,
            "seed": args.seed,
            "all_detected": all_detected,
            "results": {
                dt: {
                    "injected": r.injected,
                    "recalled": r.recalled,
                    "recall": r.recall,
                    "delta_precision": r.delta_precision,
                    "detected_all": r.detected_all,
                    "missed_ids": r.missed_ids,
                }
                for dt, r in results.items()
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"corpus-doctor — clean graph: {len(entities)} entities, "
            f"{len(edges)} edges; {args.count} defects/type, seed {args.seed}"
        )
        print("-" * 72)
        for dt, r in results.items():
            mark = "PASS" if r.detected_all else "FAIL"
            print(f"  [{mark}] {r.summary()}")
        print("-" * 72)
        print(
            "ALL INJECTED DEFECTS DETECTED"
            if all_detected
            else "SOME INJECTED DEFECTS WENT UNDETECTED — categories regressed"
        )

    return 0 if all_detected else 1


if __name__ == "__main__":
    sys.exit(main())
