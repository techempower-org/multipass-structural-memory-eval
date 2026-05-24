# CLAUDE.md — multipass-structural-memory-eval

## What this is

**SME (Structural Memory Evaluation)** — a diagnostic framework for memory systems (RAG, knowledge graphs, personal KBs, conversational memory) that tests what the system knows about its own structure, not just whether it can retrieve.

Python package `sme-eval` with CLI entrypoint `sme-eval`. Nine categories (Cat 1–9), nine CLI commands, nine adapters.

## Repository relationships

- **Upstream**: `M0nkeyFl0wer/multipass-structural-memory-eval` — the canonical repo. Issues and PRs reference this org.
- **This fork**: `techempower-org/multipass-structural-memory-eval` — JP's working fork. `origin` remote.
- Issue refs in docs use `M0nkeyFl0wer/multipass-structural-memory-eval#N` (upstream). The one exception is `techempower-org/...#6` which was filed on this fork.
- When drafting upstream comments for JP, use the donkey emoji, not butterfly (butterfly is M0nkeyFl0wer's sig).

## Development

```bash
# Install (use project venv)
pip install -e ".[dev,topology]"

# Run tests
./venv/bin/python -m pytest tests/ -x -q

# Lint
./venv/bin/ruff check .
```

- Python 3.10+, venv at `./venv/`
- Core deps: numpy, networkx, pyyaml (lightweight by design)
- No `pytest-timeout` installed — don't pass `--timeout`

## Key directories

```
sme/adapters/       — backend adapters (flat, mempalace, daemon, familiar, rlm, ladybugdb)
sme/categories/     — category implementations (bcubed, gap, ingestion, multi_hop, ontology, harness)
sme/conditions/     — Karpathy baseline adapters (full_context D1, karpathy_compiled D2, wiki_compiler)
sme/corpora/        — evaluation corpora (jp_realm_v0_1, good-dog-corpus, longmemeval, standard_v0_1)
baselines/          — saved baseline JSON readings
docs/               — spec (sme_spec_v8.md), onboarding guide (ideas.md), research docs
scripts/            — cross-validation harness
```

## Conventions

- Diagnostic posture, not benchmark — findings are deltas under controlled conditions, not absolute scores
- A/B/C/D condition isolation pattern is load-bearing methodology
- Multi-corpus testing is required — single corpus gives misleading results
- Constitutional principle: stays lightweight and locally runnable (no server hosting required)
- Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`)
