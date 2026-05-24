#!/usr/bin/env python3
"""Ingest good-dog-corpus vault into a flat ChromaDB collection.

Reads every .md file under sme/corpora/good-dog-corpus/vault/, splits on
blank-line boundaries, and writes the resulting chunks into a ChromaDB
PersistentClient at the destination path. The collection is named
``good_dog_flat`` and uses ChromaDB's default embedding (all-MiniLM-L6-v2).

This is a deliberately minimal Condition A (flat baseline) setup for
issue #21. The corpus README marks "first end-to-end run: ingest corpus
→ build graph → SME categories report" as unchecked; this script
addresses the flat-baseline half of that for the Cat 3 / Cat 6
audit-and-baseline pass.

Usage::

    ./venv/bin/python scripts/ingest_good_dog_flat.py \\
        --out /tmp/good_dog_chroma
    ./venv/bin/sme-eval retrieve --adapter flat \\
        --db /tmp/good_dog_chroma \\
        --collection-name good_dog_flat \\
        --questions sme/corpora/good-dog-corpus/questions.yaml \\
        --json /tmp/good_dog_flat.json
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "sme" / "corpora" / "good-dog-corpus"
DEFAULT_VAULT = CORPUS_ROOT / "vault"
COLLECTION_NAME = "good_dog_flat"


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_markdown(text: str, min_chars: int = 120) -> list[str]:
    """Split on blank-line boundaries, drop near-empty chunks.

    The 120-char floor avoids one-line headings becoming standalone
    chunks; they get glued onto the preceding paragraph instead.
    Frontmatter (between leading ``---`` markers) is kept — it contains
    canonical names, aliases, and evidence strings that are part of the
    corpus's load-bearing content.
    """
    parts = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    merged: list[str] = []
    buf = ""
    for p in parts:
        if len(p) < min_chars and buf:
            buf = buf + "\n\n" + p
            continue
        if buf:
            merged.append(buf)
        buf = p
    if buf:
        merged.append(buf)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--vault",
        default=str(DEFAULT_VAULT),
        help=f"Vault directory to ingest (default: {DEFAULT_VAULT})",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output ChromaDB PersistentClient directory",
    )
    ap.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help=f"Collection name (default: {COLLECTION_NAME})",
    )
    args = ap.parse_args()

    vault = Path(args.vault)
    if not vault.is_dir():
        raise SystemExit(f"vault does not exist: {vault}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import chromadb

    client = chromadb.PersistentClient(path=str(out))
    try:
        client.delete_collection(args.collection_name)
    except Exception:
        pass
    coll = client.create_collection(args.collection_name)

    md_files = sorted(p for p in vault.rglob("*.md") if p.is_file())
    if not md_files:
        raise SystemExit(f"no .md files under {vault}")

    docs: list[str] = []
    metas: list[dict] = []
    ids: list[str] = []

    for md in md_files:
        rel = md.relative_to(vault)
        text = md.read_text()
        chunks = chunk_markdown(text)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{rel.as_posix()}#{idx}")
            docs.append(chunk)
            metas.append(
                {
                    "source_file": str(rel),
                    "chunk_index": idx,
                    "domain": rel.parts[0] if rel.parts else "",
                }
            )

    print(f"ingesting {len(md_files)} files → {len(docs)} chunks")
    BATCH = 200
    for i in range(0, len(docs), BATCH):
        coll.add(
            ids=ids[i : i + BATCH],
            documents=docs[i : i + BATCH],
            metadatas=metas[i : i + BATCH],
        )
    print(
        f"done. collection={args.collection_name} at {out} "
        f"({coll.count()} chunks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
