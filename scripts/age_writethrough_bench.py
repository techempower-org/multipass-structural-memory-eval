#!/usr/bin/env python3
"""AGE write-through spike: vector-only vs graph-only vs fusion on n=200 git-probes.

Pipeline:
  1. Build a shared corpus from techempower-org/mempalace fork's .md + .py files.
  2. Ingest once into PostgresAgeIngestAdapter — postgres+pgvector stores vectors,
     AGE write-through extracts entities into the sme_spike_kg graph.
  3. For each of 200 probes, run three retrieval modes against the same substrate:
       - vector_only:  pgvector cosine (this is just the substrate-floor baseline)
       - graph_only:   entity-overlap from AGE Cypher MATCH (no vector signal)
       - fusion:       RRF combine of vector + graph rankings
  4. Compare R@5 across modes.

Compared against:
  - daemon n=5 on jp-realm-v0.1 = 0.733  (different corpus, reference scale)
  - familiar n=5 on jp-realm-v0.1 = 0.883 (different corpus, reference scale)
  - daemon n=5 on n=200 git-probes-v2 = 0.280 (the closest apples-to-apples baseline)
  - familiar n=5 on n=200 git-probes-v2 = 0.310

The spike asks: does ANY graph signal lift recall above the postgres+pgvector
substrate floor on this corpus? If yes, the AGE write-through architecture
warrants further investment. If no, the encoder + chunking layers are the
right place to invest.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path("/home/jp/Projects/memorypalace")
PROBE_YAML = Path("/home/jp/Projects/multipass-structural-memory-eval/sme/corpora/mempalace_git_probes_v2/questions.yaml")
OUT_PATH = Path("/home/jp/Projects/multipass-structural-memory-eval/baselines/age_writethrough_spike_2026-05-17.json")

# Skip these path patterns — worktree noise, dotdirs, build artifacts.
SKIP_PATTERNS = (".git", ".claude", "node_modules", "__pycache__", "venv", ".venv")


def load_corpus() -> list[dict]:
    """Collect all .md + .py files from the repo into a single corpus."""
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
            # Cap individual files at ~64KB to keep one-doc-per-file practical
            if len(text) > 65000:
                text = text[:65000]
            out.append({
                "id": f.name,  # filename only — matches expected_sources shape
                "document": text,
                "metadata": {"source_file": f.name, "path_rel": str(f.relative_to(REPO_ROOT))},
            })
    # Dedupe by filename (last one wins)
    by_name: dict[str, dict] = {}
    for entry in out:
        by_name[entry["id"]] = entry
    return list(by_name.values())


def load_probes() -> list[dict]:
    d = yaml.safe_load(PROBE_YAML.read_text())
    return d["questions"]


def eval_one_mode(adapter, probes, mode: str, n_results: int = 5) -> dict:
    """Run all probes through adapter in given mode; return aggregate + per-q."""
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
            result = adapter.query(q["text"], n_results=n_results)
            retrieved = [
                (e.id.removeprefix("chunk:") if e.id.startswith("chunk:") else e.id)
                for e in (result.retrieved_entities or [])
            ]
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

    print(f"[{time.time():.0f}] loading corpus from {REPO_ROOT}")
    corpus = load_corpus()
    print(f"  {len(corpus)} files (.md + .py, deduped by filename)")
    md_count = sum(1 for e in corpus if e["id"].endswith(".md"))
    py_count = sum(1 for e in corpus if e["id"].endswith(".py"))
    print(f"  → {md_count} markdown, {py_count} python")

    print(f"[{time.time():.0f}] loading probes from {PROBE_YAML.name}")
    probes = load_probes()
    print(f"  {len(probes)} probes")

    print(f"[{time.time():.0f}] connecting to PostgresAgeIngestAdapter (FT-300-loaded session)")
    adapter = PostgresAgeIngestAdapter(n_results=5, retrieval_mode="fusion")
    print(f"  adapter ready")

    skip_ingest = "--skip-ingest" in sys.argv
    if not skip_ingest:
        print(f"[{time.time():.0f}] ingesting corpus (vector + AGE write-through)")
        t0 = time.time()
        res = adapter.ingest_corpus(corpus)
        print(f"  ingested {res['ingested']} drawers in {time.time()-t0:.1f}s")
    else:
        print(f"[{time.time():.0f}] --skip-ingest: reusing existing AGE graph + drawer table")

    # Quick KG stats
    cur = adapter._age_conn.cursor()
    cur.execute("LOAD 'age'")
    cur.execute("SET search_path = ag_catalog, public")
    cur.execute(f"SELECT * FROM cypher('{adapter.graph_name}', $$ MATCH (n) RETURN labels(n), count(*) $$) AS (lbl agtype, c agtype)")
    print("  AGE graph state:")
    for row in cur.fetchall():
        print(f"    {row[0]} = {row[1]}")

    # Run the three modes
    results = {}
    for mode in ("vector_only", "graph_only", "fusion"):
        results[mode] = eval_one_mode(adapter, probes, mode)

    # Compose summary
    summary = {
        "corpus": {
            "n_files": len(corpus),
            "n_markdown": md_count,
            "n_python": py_count,
            "source_root": str(REPO_ROOT),
        },
        "n_probes": len(probes),
        "modes": {mode: {"r5": r["r5"], "hits": r["hits"], "n": r["n_evaluated"], "wall_clock_s": r["wall_clock_s"]} for mode, r in results.items()},
        "deltas": {
            "graph_only - vector_only": round(results["graph_only"]["r5"] - results["vector_only"]["r5"], 4),
            "fusion - vector_only": round(results["fusion"]["r5"] - results["vector_only"]["r5"], 4),
        },
    }
    OUT_PATH.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n" + "=" * 60)
    print(f"  AGE WRITE-THROUGH SPIKE — n=200 git-probes-v2 (R@5)")
    print("=" * 60)
    for mode, r in results.items():
        print(f"  {mode:<14s}  R@5 = {r['r5']:.4f}  ({r['hits']}/{r['n_evaluated']})  {r['wall_clock_s']}s")
    print()
    print(f"  Δ graph_only − vector_only:  {summary['deltas']['graph_only - vector_only']:+.4f}")
    print(f"  Δ fusion − vector_only:      {summary['deltas']['fusion - vector_only']:+.4f}")
    print()
    print(f"  Written: {OUT_PATH}")

    adapter.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
