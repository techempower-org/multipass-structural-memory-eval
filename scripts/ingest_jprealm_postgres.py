#!/usr/bin/env python3
"""Ingest the sanitized jp-realm flat corpus into a postgres+pgvector table so
``PostgresIngestAdapter`` (Cat 1 / 2c Condition A) can retrieve over it.

This is the postgres+pgvector twin of ``ingest_jprealm_flat_chroma.py``. Same
sanitized JSONL, same all-MiniLM-L6-v2 embedding (PostgresCollection reuses
Chroma's DefaultEmbeddingFunction), no wing/room/graph metadata — only the
drawer id and text. The ONLY variable swapped vs the flat ChromaDB baseline is
the storage/retrieval backend (chroma -> postgres+pgvector). That makes this row
the "upstream MemPalace raw" ablation: mempalace's own verbatim postgres storage
WITHOUT the palace graph structure on top.

Requires the SME_POSTGRES_DSN env var (no hardcoded DSN by design). Point it at
an isolated throwaway instance — do NOT use the prod substrate.

Unlike the per-question LongMemEval substrate bench, this ingests ALL drawers
ONCE into a persistent table; the retrieve step then queries every question
against that single populated table (mirroring the flat Condition-A protocol).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default=str(REPO / "sme/corpora/jp_realm_v0_1/flat_source.jsonl"),
    )
    ap.add_argument("--table", default="jp_realm_postgres_condA")
    args = ap.parse_args()

    if not os.environ.get("SME_POSTGRES_DSN"):
        raise SystemExit(
            "SME_POSTGRES_DSN not set. Point it at an isolated throwaway "
            "postgres+pgvector instance (NOT prod)."
        )

    from sme.adapters.postgres_ingest import PostgresIngestAdapter

    records = [json.loads(line) for line in open(args.source) if line.strip()]
    if not records:
        raise SystemExit(f"no records in {args.source}")

    # flat_source.jsonl uses {"id","text"}; ingest_corpus wants {"id","document"}.
    # No structural metadata — Condition A must not see wing/room.
    corpus = [{"id": r["id"], "document": r["text"]} for r in records]

    adapter = PostgresIngestAdapter(table_name=args.table)
    res = adapter.ingest_corpus(corpus)  # TRUNCATE + upsert: fresh build each run
    adapter.shutdown()

    print(f"ingested {res.get('ingested', 0)} drawers")
    print(f"table:    {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
