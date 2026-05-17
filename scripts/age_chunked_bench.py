#!/usr/bin/env python3
"""Bench A: AGE entity-overlap fused with paragraph-chunked vector retrieval.

Follow-up to the file-level spike (age_writethrough_bench.py). That run hit:
  vector_only (file-level):    R@5 = 0.1850
  graph_only:                  R@5 = 0.2350  (+5.0pp)
  fusion (RRF):                R@5 = 0.2750  (+9.0pp)

The vector_only floor was abnormally low (0.185) because each file became
one vector — paragraph-level vectors are the production-tier setup. This
bench paragraph-chunks the same 238 files, re-runs the three modes, and
answers: does graph still add lift when vector is at production tier?

Chunking: blank-line-separated paragraphs, glued until ~800 chars (the
default chunk_size in mempalace.convo_miner). Source_file metadata
preserved so file-level expected_sources matching still works on
aggregate.

AGE graph is the same one ingested in age_writethrough_bench (file-level
Drawer nodes + Entity nodes from regex extraction). The asymmetry is
intentional and realistic: the graph layer naturally aggregates to file
identity; the vector layer naturally operates on chunk granularity.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import psycopg2
import yaml

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path("/home/jp/Projects/memorypalace")
PROBE_YAML = Path("/home/jp/Projects/multipass-structural-memory-eval/sme/corpora/mempalace_git_probes_v2/questions.yaml")
OUT_PATH = Path("/home/jp/Projects/multipass-structural-memory-eval/baselines/age_chunked_bench_2026-05-17.json")
SKIP_PATTERNS = (".git", ".claude", "node_modules", "__pycache__", "venv", ".venv")
CHUNK_SIZE = 800  # match mempalace.convo_miner default


def paragraph_chunks(text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) > max_chars and buf:
            chunks.append(buf)
            buf = p
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def load_chunked_corpus() -> list[dict]:
    out: list[dict] = []
    for ext in ("*.md", "*.py"):
        for f in REPO_ROOT.rglob(ext):
            if not f.is_file():
                continue
            parts = f.relative_to(REPO_ROOT).parts
            if any(p in SKIP_PATTERNS for p in parts):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue
            for i, chunk in enumerate(paragraph_chunks(text)):
                out.append({
                    "id": f"{f.name}__c{i}",
                    "document": chunk,
                    "metadata": {"source_file": f.name},
                })
    # Dedupe — multiple files with same filename: last wins
    by_id: dict[str, dict] = {}
    for entry in out:
        by_id[entry["id"]] = entry
    return list(by_id.values())


def load_probes() -> list[dict]:
    d = yaml.safe_load(PROBE_YAML.read_text())
    return d["questions"]


def eval_one_mode(adapter, probes, mode: str, n_results: int = 5, k_chunk: int = 50) -> dict:
    """For each probe: query (possibly oversized k_chunk for chunked vector),
    aggregate to top-5 unique source_files."""
    import traceback
    adapter.retrieval_mode = mode
    print(f"\n>>> {mode}", flush=True)
    t0 = time.time()
    hits = 0
    errors = 0
    per_q: list[dict] = []
    for i, q in enumerate(probes):
        expected = set(q.get("expected_sources") or [])
        if not expected:
            continue
        try:
            # Request enough chunks to find 5 unique source_files in worst case
            result = adapter.query(q["text"], n_results=k_chunk)
            # Aggregate to top-5 unique source files (preserve rank)
            seen_files: list[str] = []
            for e in (result.retrieved_entities or []):
                meta = e.properties or {}
                src = meta.get("source_file") or e.name
                # For graph-only path, ids look like 'chunk:<filename>' (no __cN suffix)
                src = src.split("__c")[0]  # strip chunk suffix if present
                if src not in seen_files:
                    seen_files.append(src)
                if len(seen_files) >= n_results:
                    break
            retrieved = seen_files
            hit = 1 if any(f in retrieved[:n_results] for f in expected) else 0
        except Exception as e:  # noqa: BLE001
            errors += 1
            retrieved = []
            hit = 0
            if errors <= 3:
                print(f"  ERROR on probe {q['id']}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        hits += hit
        per_q.append({
            "id": q["id"],
            "expected": list(expected),
            "retrieved": retrieved[:n_results] if retrieved else [],
            "hit": hit,
        })
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            cum = hits / (i + 1)
            print(f"  [{i+1:3d}/{len(probes)}]  cum_R@5={cum:.4f}  rate={(i+1)/elapsed:.2f}q/s  errors={errors}", flush=True)
    n = len(per_q)
    return {
        "errors": errors,
        "mode": mode,
        "n_evaluated": n,
        "hits": hits,
        "r5": hits / n if n else 0,
        "wall_clock_s": round(time.time() - t0, 1),
        "per_q": per_q,
    }


def main() -> int:
    from sme.adapters.postgres_age_ingest import PostgresAgeIngestAdapter

    print(f"[{time.time():.0f}] chunked-corpus build from {REPO_ROOT}")
    corpus = load_chunked_corpus()
    n_files = len({c["metadata"]["source_file"] for c in corpus})
    print(f"  {len(corpus)} chunks from {n_files} files (≈{len(corpus)/n_files:.1f} chunks/file)")

    print(f"[{time.time():.0f}] loading probes")
    probes = load_probes()
    print(f"  {len(probes)} probes")

    print(f"[{time.time():.0f}] connecting to PostgresAgeIngestAdapter")
    # Note: graph_name reuses the same AGE graph from age_writethrough_bench —
    # we keep the file-level entity→drawer edges and don't re-populate AGE per
    # chunk. The asymmetry (chunk-level vectors + file-level graph) is the
    # production-tier shape: graph layer indexes file identity; vector layer
    # indexes passage similarity.
    adapter = PostgresAgeIngestAdapter(n_results=5, retrieval_mode="fusion")
    print(f"  adapter ready — re-ingesting chunked corpus to drawer table")
    print(f"[{time.time():.0f}] ingesting (vector pass writes ~{len(corpus)} chunks; AGE re-populates entity edges)")
    t0 = time.time()
    res = adapter.ingest_corpus(corpus)
    print(f"  ingested {res['ingested']} chunks in {time.time()-t0:.1f}s")

    # Reset graph_top_k to match — we want enough graph candidates that
    # aggregating to top-5 files is feasible
    adapter.graph_top_k = 100

    results = {}
    for mode in ("vector_only", "graph_only", "fusion"):
        results[mode] = eval_one_mode(adapter, probes, mode)

    summary = {
        "corpus": {"n_chunks": len(corpus), "n_files": n_files, "chunks_per_file": round(len(corpus)/n_files, 2)},
        "n_probes": len(probes),
        "modes": {mode: {"r5": r["r5"], "hits": r["hits"], "n": r["n_evaluated"], "wall_clock_s": r["wall_clock_s"], "errors": r["errors"]} for mode, r in results.items()},
        "deltas": {
            "graph_only - vector_only": round(results["graph_only"]["r5"] - results["vector_only"]["r5"], 4),
            "fusion - vector_only": round(results["fusion"]["r5"] - results["vector_only"]["r5"], 4),
        },
    }
    OUT_PATH.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n" + "=" * 60)
    print(f"  AGE+CHUNKED BENCH — n=200 git-probes-v2 (R@5)")
    print("=" * 60)
    for mode, r in results.items():
        print(f"  {mode:<14s}  R@5 = {r['r5']:.4f}  ({r['hits']}/{r['n_evaluated']})  {r['wall_clock_s']}s")
    print()
    print(f"  Δ graph_only − vector_only:  {summary['deltas']['graph_only - vector_only']:+.4f}")
    print(f"  Δ fusion − vector_only:      {summary['deltas']['fusion - vector_only']:+.4f}")
    print(f"  Written: {OUT_PATH}")

    adapter.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
