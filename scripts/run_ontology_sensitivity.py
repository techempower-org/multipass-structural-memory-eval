#!/usr/bin/env python3
"""Cat 4 / Cat 5 ontology-sensitivity sweep (upstream #45).

Loads one corpus graph and re-reads Cat 4 (ingestion integrity) + Cat 5
(structural gap detection) under three deliberately-different ontology
granularities — flat / moderate / fine-grained — then reports how much
the headline structural numbers move. The graph topology is identical
across conditions; only the entity_type / edge_type projection changes,
so any movement is attributable to ontology choice alone.

Outcome is publishable either way:
  - movement large  → methodological caveat (report Cat 4/5 with the
    ontology; cross-system comparison needs matched ontologies)
  - movement small  → robustness claim (Cat 4/5 are ontology-robust
    within the observed range; comparison is valid)

CLI:

    run_ontology_sensitivity.py
        --corpus good-dog            # only good-dog is wired (graph-bearing)
        --out PATH                   # baseline JSON destination (optional)

Example:

    ./venv/bin/python scripts/run_ontology_sensitivity.py \
        --corpus good-dog \
        --out baselines/ontology_sensitivity_good_dog_2026-05-31.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sme.corpora.good_dog_graph import load_graph
from sme.corpora.good_dog_ontologies import GOOD_DOG_CONDITIONS
from sme.eval.ontology_sensitivity import format_sweep, run_sensitivity_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default="good-dog",
        choices=["good-dog"],
        help="corpus to sweep (only good-dog carries a typed graph today)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the sweep result JSON here (optional)",
    )
    args = parser.parse_args(argv)

    if args.corpus == "good-dog":
        entities, edges = load_graph()
        conditions = GOOD_DOG_CONDITIONS
        corpus_name = "good-dog-corpus"
    else:  # pragma: no cover — choices guards this
        parser.error(f"unknown corpus {args.corpus}")
        return 2

    result = run_sensitivity_sweep(entities, edges, conditions, corpus=corpus_name)

    print(format_sweep(result))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
