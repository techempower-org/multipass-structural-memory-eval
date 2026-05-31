#!/usr/bin/env python3
"""Run OMEGA's *emergent* structure through SME's structural cats.

For the cross-system multipass matrix (#178 Tier 3 / #115). Unlike the
good-dog-graph adapter — which loads the corpus's hand-DECLARED contradicts /
supersedes edges (a ceiling case) — OMEGA does not read corpus edges. It stores
note TEXT and a background auto-relate pass GENERATES typed edges. This script
ingests the good-dog vault text into an isolated OMEGA store, lets auto-relate
run, and reports the emergent-graph snapshot stats that feed Cat 3/4/5/6/8.

The category SCORING is done by the SME CLIs against the same store, e.g.:

    DB=<omega-home>/omega.db
    sme-eval cat3 --adapter omega --db "$DB"   # self-consistency (tautological for OMEGA)
    sme-eval cat4 --adapter omega --db "$DB"   # ingestion integrity
    sme-eval cat5 --adapter omega --db "$DB"   # topology (needs the node_id fix)
    sme-eval cat6 --adapter omega --db "$DB"   # supersession completeness
    sme-eval cat8 --adapter omega --db "$DB" \
        --implied-ontology sme/corpora/implied_ontology_omega.yaml

The COMPARABLE Cat 3 reading (emergent edges vs good-dog GROUND TRUTH, not vs
the system's own edges) is computed by this script's --cat3-ground-truth pass.

Usage:
    ./venv/bin/python scripts/run_omega_good_dog_structural.py \
        --omega-home /tmp/omega_gooddog --out baselines/omega_good_dog_emergent.json

Requires: pip install 'sme-eval[omega]'  (+ ONNX model: omega setup --download-model)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sme.adapters.base import (
    is_contradicts_edge,
    is_supersedes_edge,
)
from sme.adapters.omega import OmegaAdapter

REPO = Path(__file__).resolve().parent.parent
VAULT = REPO / "sme" / "corpora" / "good-dog-corpus" / "vault"

# Ground-truth contradiction themes (good-dog ontology.yaml
# cat_3_contradiction.seeded_pairs), keyed to the vault files that make up the
# two contradictory sides. A surfaced OMEGA edge "hits" a theme when its two
# endpoints' source files straddle the two sides.
GROUND_TRUTH_THEMES = {
    "grain_free_dcm": {
        "side_a": ["2018-07-fda-dcm-investigation", "2018-11-tufts-petfoodology-beg-dcm"],
        "side_b": ["2022-11-fda-qa-non-hereditary-dcm", "2022-03-freeman-jvim-prospective-dcm-diet"],
    },
    "dominance_theory": {
        "side_a": ["1947-schenkel-expression-studies-wolves", "1970"],
        "side_b": ["2008-avsab-dominance-position-statement", "1999-mech-alpha-status-self-correction"],
    },
}


def load_good_dog_rows() -> list[dict]:
    rows = []
    for md in sorted(VAULT.rglob("*.md")):
        if not md.is_file():
            continue
        rel = md.relative_to(VAULT)
        rows.append({
            "content": md.read_text(),
            "type": "summary",
            "metadata": {"source_file": str(rel), "domain": rel.parts[0]},
        })
    return rows


def _file_side(sf: str):
    base = Path(sf).name if sf else sf
    for theme, sides in GROUND_TRUTH_THEMES.items():
        for side, needles in sides.items():
            if any(n in base for n in needles):
                return (theme, side)
    return None


def cat3_vs_ground_truth(adapter: OmegaAdapter) -> dict:
    """OMEGA's emergent contradicts edges scored against good-dog ground truth.

    Resolves each edge endpoint (mem-<hash>) back to its source file via the
    Entity's stored metadata, then asks whether the pair straddles a
    ground-truth contradiction theme."""
    entities, edges = adapter.get_graph_snapshot()
    # Build node_id -> source_file from the snapshot's entity properties.
    # (content_preview holds the note head; we re-read metadata from the db
    #  via the adapter is overkill — instead match on the note text head.)
    # Simpler + robust: pull source_file by re-reading the omega db.
    import sqlite3
    conn = sqlite3.connect(adapter.db_path)
    memid2file = {}
    for node_id, meta in conn.execute("SELECT node_id, metadata FROM memories"):
        sf = ""
        if meta:
            try:
                sf = json.loads(meta).get("source_file", "") or ""
            except Exception:
                pass
        memid2file[f"omega:{node_id}"] = sf
    conn.close()

    contradicts = [e for e in edges if is_contradicts_edge(e.edge_type)]
    surfaced, hits_by_theme = [], {}
    for e in contradicts:
        f_src, f_dst = memid2file.get(e.source_id, ""), memid2file.get(e.target_id, "")
        s_side, d_side = _file_side(f_src), _file_side(f_dst)
        straddles = (
            s_side and d_side and s_side[0] == d_side[0] and s_side[1] != d_side[1]
        )
        theme = s_side[0] if (s_side and d_side and s_side[0] == d_side[0]) else None
        if straddles:
            hits_by_theme[theme] = hits_by_theme.get(theme, 0) + 1
        surfaced.append({
            "src_file": Path(f_src).name if f_src else f_src,
            "dst_file": Path(f_dst).name if f_dst else f_dst,
            "weight": e.properties.get("weight"),
            "straddles_theme": theme if straddles else None,
        })
    n_true = sum(1 for s in surfaced if s["straddles_theme"])
    return {
        "ground_truth_themes": sorted(GROUND_TRUTH_THEMES),
        "omega_emergent_contradicts_edges": len(contradicts),
        "themes_detected": sorted(hits_by_theme),
        "theme_recall": len(hits_by_theme) / len(GROUND_TRUTH_THEMES),
        "edge_precision_vs_ground_truth": (n_true / len(surfaced)) if surfaced else 0.0,
        "true_positive_edges": n_true,
        "surfaced_edges_detail": surfaced,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega-home", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_good_dog_rows()
    adapter = OmegaAdapter(omega_home=args.omega_home, default_memory_type="summary")
    adapter.reset()
    ing = adapter.ingest_corpus(rows)
    entities, edges = adapter.get_graph_snapshot()

    ent_eids = {e.id for e in entities}
    resolved = sum(
        1 for e in edges if e.source_id in ent_eids and e.target_id in ent_eids
    )
    report = {
        "system": "omega",
        "corpus": "good-dog-corpus (vault text, OMEGA auto-relate — EMERGENT)",
        "n_notes_ingested": len(rows),
        "entities": len(entities),
        "edges": len(edges),
        "edges_with_both_endpoints_resolved": f"{resolved}/{len(edges)}",
        "entity_type_distribution": dict(Counter(e.entity_type for e in entities)),
        "edge_type_distribution": dict(Counter(e.edge_type for e in edges)),
        "contradicts_edges": sum(1 for e in edges if is_contradicts_edge(e.edge_type)),
        "supersedes_edges": sum(1 for e in edges if is_supersedes_edge(e.edge_type)),
        "cat3_vs_ground_truth": cat3_vs_ground_truth(adapter),
        "ingest_errors": ing["errors"][:5],
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
