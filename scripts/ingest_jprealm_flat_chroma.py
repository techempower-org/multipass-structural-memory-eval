#!/usr/bin/env python3
"""Ingest the sanitized jp-realm flat corpus into a ChromaDB collection so
``FlatBaselineAdapter`` (Cat 7 Condition A) can retrieve over it.

The collection deliberately carries NO wing/room/graph metadata — only the
drawer id and sanitized text. That is the whole point of Condition A: the
*same information* as the structured palace (Condition B), with the
structure removed. Flat vector similarity does its own top-K ranking.

Uses ChromaDB's default embedding function (all-MiniLM-L6-v2) so the corpus
is self-contained and reproducible from the committed JSONL without external
model state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default=str(REPO / "sme/corpora/jp_realm_v0_1/flat_source.jsonl"),
    )
    ap.add_argument(
        "--db",
        default=str(REPO / "sme/corpora/jp_realm_v0_1/flat_chroma"),
        help="ChromaDB persistent dir (gitignored — rebuilt from JSONL)",
    )
    ap.add_argument("--collection", default="jp_realm_flat")
    args = ap.parse_args()

    import chromadb

    records = [json.loads(line) for line in open(args.source) if line.strip()]
    if not records:
        raise SystemExit(f"no records in {args.source}")

    client = chromadb.PersistentClient(path=args.db)
    # Fresh build each time — drop any prior collection.
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass
    coll = client.create_collection(args.collection)

    ids = [r["id"] for r in records]
    docs = [r["text"] for r in records]
    # No structural metadata — flat baseline must not see wing/room.
    coll.add(ids=ids, documents=docs)

    print(f"ingested {len(records)} drawers")
    print(f"collection: {args.collection}")
    print(f"db:         {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
