#!/usr/bin/env python3
"""2x2 encoder-conditional chunking ablation on git-derived markdown probes.

Tests the hypothesis posted on MemPalace/mempalace#1384 (discussioncomment-16950768):
chunking-axis sensitivity is partly downstream of encoder calibration. If true,
the B-vs-A (heading-aware vs paragraph) delta should shrink or vanish when the
encoder is FT'd to the domain, because the encoder is already finding the right
session regardless of chunking.

2x2 grid:
  encoders: {base-MiniLM, FT-300}
  chunkers: {paragraph, heading-aware}

48 markdown probes from `sme/corpora/mempalace_git_probes_v2/questions.yaml`,
each with `expected_sources` = ['<filename>.md']. Corpus is the current HEAD
.md files in techempower-org/mempalace (~/Projects/memorypalace/).

Metric: R@5 — was a chunk whose source_file matches the expected filename
in the top-5 retrieved chunks?
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path("/home/jp/Projects/memorypalace")
PROBE_YAML = Path("/home/jp/Projects/multipass-structural-memory-eval/sme/corpora/mempalace_git_probes_v2/questions.yaml")
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FT300_MODEL = "/tmp/minilm-lme-ft-300-katana/model"
OUT_PATH = Path("/home/jp/Projects/multipass-structural-memory-eval/baselines/chunking_encoder_ablation_2026-05-17.json")


def paragraph_chunks(text: str, max_chars: int = 800) -> list[str]:
    """Naive paragraph-aware split: blank-line-separated blocks, glued until max_chars."""
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


def heading_aware_chunks(text: str, max_chars: int = 800) -> list[str]:
    """Split on markdown headings (lines starting with `#`), then size-cap each section.

    Each chunk is prefixed by its heading context (last seen `#` heading + immediate
    parents) so the heading IS in the embedded text — the framing from xg-gh-25's
    note on #1384.
    """
    lines = text.split("\n")
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    sections: list[tuple[list[str], str]] = []  # (heading_stack, body)
    current_headings: list[str] = []
    current_body: list[str] = []
    for line in lines:
        m = heading_re.match(line)
        if m:
            if current_body:
                sections.append((list(current_headings), "\n".join(current_body).strip()))
                current_body = []
            level = len(m.group(1))
            text_h = m.group(2)
            current_headings = current_headings[: level - 1] + [text_h]
        else:
            current_body.append(line)
    if current_body:
        sections.append((list(current_headings), "\n".join(current_body).strip()))

    chunks: list[str] = []
    for headings, body in sections:
        if not body:
            continue
        prefix = " > ".join(headings) + "\n\n" if headings else ""
        full = prefix + body
        if len(full) <= max_chars:
            chunks.append(full)
            continue
        # Section too long — fall back to paragraph chunking within the section
        sub = paragraph_chunks(body, max_chars=max_chars - len(prefix))
        for s in sub:
            chunks.append(prefix + s)
    return chunks


def load_markdown_corpus(repo_root: Path) -> list[dict]:
    """Collect every .md file under repo_root, return [{path_rel, filename, text}]."""
    out: list[dict] = []
    for md in repo_root.rglob("*.md"):
        # Skip worktrees, dotdirs (.claude, .git), node_modules, etc.
        rel = md.relative_to(repo_root)
        parts = rel.parts
        if any(p.startswith(".") for p in parts):
            continue
        if any(p == "node_modules" for p in parts):
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out.append({"path_rel": str(rel), "filename": md.name, "text": text})
    return out


def load_md_probes() -> list[dict]:
    """Return the 48 probes whose expected_sources is a single .md file."""
    d = yaml.safe_load(PROBE_YAML.read_text())
    out = []
    for q in d["questions"]:
        exp = q.get("expected_sources") or []
        md_only = [s for s in exp if s.endswith(".md")]
        if not md_only or md_only != exp:
            continue
        out.append({"id": q["id"], "text": q["text"], "expected_filenames": set(md_only)})
    return out


def build_chunks(corpus: list[dict], chunker) -> tuple[list[str], list[str]]:
    """Return (chunk_texts, chunk_filenames) aligned."""
    chunk_texts: list[str] = []
    chunk_files: list[str] = []
    for entry in corpus:
        for chunk in chunker(entry["text"]):
            chunk_texts.append(chunk)
            chunk_files.append(entry["filename"])
    return chunk_texts, chunk_files


def evaluate_condition(
    model,
    chunk_texts: list[str],
    chunk_files: list[str],
    probes: list[dict],
    top_k: int = 5,
) -> dict:
    """Encode chunks once, evaluate every probe."""
    print(f"  encoding {len(chunk_texts)} chunks...", flush=True)
    chunk_embs = model.encode(
        chunk_texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  scoring {len(probes)} probes...", flush=True)
    hits = 0
    per_q: list[dict] = []
    for q in probes:
        q_emb = model.encode(
            [q["text"]],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        sims = chunk_embs @ q_emb
        order = np.argsort(-sims)
        top_files = [chunk_files[i] for i in order[:top_k]]
        hit = 1 if any(f in q["expected_filenames"] for f in top_files) else 0
        hits += hit
        per_q.append({
            "id": q["id"],
            "expected": list(q["expected_filenames"]),
            "top5_files": top_files,
            "hit": hit,
        })
    return {"hits": hits, "n": len(probes), "r5": hits / len(probes), "per_q": per_q}


def main() -> int:
    from sentence_transformers import SentenceTransformer

    print(f"[{time.time():.0f}] loading {len(load_markdown_corpus(REPO_ROOT))}-file corpus")
    corpus = load_markdown_corpus(REPO_ROOT)
    print(f"  {len(corpus)} markdown files from {REPO_ROOT}")

    probes = load_md_probes()
    print(f"  {len(probes)} .md probes")

    para_chunks, para_files = build_chunks(corpus, paragraph_chunks)
    head_chunks, head_files = build_chunks(corpus, heading_aware_chunks)
    print(f"  paragraph chunks: {len(para_chunks)}")
    print(f"  heading-aware chunks: {len(head_chunks)}")

    results: dict = {"conditions": {}}

    for enc_name, enc_path in [("base-MiniLM", BASE_MODEL), ("FT-300", FT300_MODEL)]:
        print(f"\n[{time.time():.0f}] loading encoder: {enc_name}")
        model = SentenceTransformer(enc_path)
        for chunker_name, ct, cf in [("paragraph", para_chunks, para_files), ("heading-aware", head_chunks, head_files)]:
            cond_name = f"{enc_name}+{chunker_name}"
            print(f"\n>>> condition: {cond_name}")
            t0 = time.time()
            r = evaluate_condition(model, ct, cf, probes)
            r["wall_clock_s"] = round(time.time() - t0, 1)
            r["n_chunks"] = len(ct)
            results["conditions"][cond_name] = r
            print(f"    R@5 = {r['r5']:.4f}  ({r['hits']}/{r['n']})  in {r['wall_clock_s']}s")

    # Delta table
    deltas = {}
    for enc_name in ["base-MiniLM", "FT-300"]:
        p = results["conditions"][f"{enc_name}+paragraph"]["r5"]
        h = results["conditions"][f"{enc_name}+heading-aware"]["r5"]
        deltas[f"{enc_name}: heading-aware - paragraph"] = round(h - p, 4)
    results["deltas"] = deltas

    OUT_PATH.write_text(json.dumps({**results, "n_probes": len(probes), "corpus_root": str(REPO_ROOT)}, indent=2))

    print("\n" + "=" * 60)
    print("  ENCODER × CHUNKER ABLATION (48 .md probes, R@5)")
    print("=" * 60)
    print(f"{'condition':<40s}  {'R@5':>8s}  {'hits':>10s}")
    for cond, r in results["conditions"].items():
        print(f"  {cond:<38s}  {r['r5']:>8.4f}  {r['hits']:>5d}/{r['n']:<3d}")
    print()
    print("  Δ (heading-aware vs paragraph), per encoder:")
    for k, v in deltas.items():
        marker = "  ←  larger" if abs(v) > 0.05 else ""
        print(f"    {k:<45s}  {v:+.4f}{marker}")
    print()
    print(f"  Written: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
