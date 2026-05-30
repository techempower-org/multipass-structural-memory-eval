# Cat 3 / Cat 6 structural-field plumbing — closing the (structural − flat) gap

**Issue:** [techempower-org/multipass-structural-memory-eval#200](https://github.com/techempower-org/multipass-structural-memory-eval/issues/200)
— *Cat 3 / Cat 6 structured-field plumbing through the daemon adapter.*

**Date:** 2026-05-30
**Author:** Onyx (sme-dreamteam)
**Status:** Closed. Adapter plumbing landed; structural readings produced
on the good-dog corpus.

---

## What was uncomputable before

The multipass Cat 1–9 matrix
([`docs/benchmarks/2026-05-30-multipass-cat-matrix.md`](2026-05-30-multipass-cat-matrix.md))
marked Cat 3 (The Dissonance) and Cat 6 (The Archive) as **gap —
uncomputable on mempalace**. The flat good-dog baselines existed
([`docs/good_dog_cat3_cat6_findings.md`](../good_dog_cat3_cat6_findings.md):
Cat 3 90.3 %, Cat 6 100 % on the substring matcher), but those are
**corpus-side floor measurements, not system readings** — the substring
matcher cannot distinguish a consolidation-aware system from a flat RAG.
The real signal lives in two structured channels the spec defines but the
adapters never populated:

* **Cat 3** — `QueryResult.contradictions` / the adapter's
  `get_contradiction_pairs()` (spec v8 §3 `ContradictionPair[]`).
* **Cat 6** — the reserved `_superseded_by` edge property (spec v8 §6 /
  §6b provenance).

Because neither was wired through `mempalace.py` or `mempalace_daemon.py`,
the honest headline for these cats — `(structural − flat)` — could not be
computed.

## The fix is entirely SME-side — no daemon schema change

The original issue framed this as a 2-repo task (palace-daemon edge-types
+ route, then the adapter). Reading `palace-daemon/kg_reader.py` (read-only
probe) shows that is **not** needed:

* The `/graph` route projects every KG edge with
  `predicate = r.properties->>'relation_type'` (kg_reader.py:317) and
  accepts **arbitrary predicate strings**. So `supersedes` and
  `contradicts` ride through as predicates already — no new schema, no new
  route, no daemon deploy.

The gap was purely that the SME side never *derived the structured fields*
from those predicates. The fix:

1. **`sme/adapters/base.py`** — three predicate-normalizing helpers
   (`normalize_predicate`, `is_supersedes_edge`, `is_contradicts_edge`)
   plus `annotate_superseded_edges()` (stamps the reserved
   `_superseded_by` / `_superseded_target_by` on superseded edges) and
   `contradiction_pairs_from_edges()` (extracts `ContradictionPair[]`
   from `contradicts` edges, de-duplicated on the unordered endpoint set).
   A new optional `SMEAdapter.get_contradiction_pairs()` contract method
   (default derives from the graph snapshot).
2. **`sme/adapters/_graph_mapping.py`** (daemon + familiar `/graph`) and
   **`sme/adapters/mempalace.py`** (direct ChromaDB + SQLite KG) — call
   `annotate_superseded_edges` after building KG edges; `mempalace.py`
   adds a `get_contradiction_pairs()` override that reads only the KG
   layer (cheaper than the full per-drawer snapshot scan).
3. **`sme/categories/contradiction.py`** (Cat 3) +
   **`sme/categories/supersession.py`** (Cat 6) — scorers that consume
   the structured fields and report the `(structural − flat)` delta. The
   Cat 3 result dict mirrors the `contradiction_pairs` field
   `ontology_coherence.py` already reads for the "conflict detection"
   claim cross-reference.
4. **`sme/corpora/good_dog_graph.py`** + **`sme/adapters/good_dog_graph.py`**
   — a vault → graph loader and adapter. The good-dog vault frontmatter
   already declares typed `contradicts` (publication→publication) and
   `supersedes` (publication→publication) edges; this projects them into
   the SME `(Entity, Edge)` shape. It is a deterministic, in-tree,
   service-free structural corpus — **no production-palace contamination
   risk** (cf. familiar#92).
5. **`sme/cli.py`** — `sme-eval cat3` / `sme-eval cat6` commands +
   the `good-dog-graph` adapter registration.

## Headline numbers (good-dog-corpus, 2026-05-30)

Run:

```
./venv/bin/sme-eval cat3 --adapter good-dog-graph \
    --json baselines/good_dog_cat3_structural_2026-05-30.json
./venv/bin/sme-eval cat6 --adapter good-dog-graph \
    --json baselines/good_dog_cat6_structural_2026-05-30.json
```

| Category | metric | flat floor | structural | **(structural − flat)** |
|---|---|---|---|---|
| **Cat 3 — The Dissonance** | contradiction detection rate | 0.00 (no structured pairs) | **1.00** (6/6 seeded pairs surfaced, precision 1.00) | **+1.00** |
| **Cat 6 — The Archive** | supersession completeness | 0.00 (no edges / no `_superseded_by`) | **1.00** (8/8 supersedes edges resolved, 5 chains) | **+1.00** |

The substring-matcher floor for these cats was 90.3 % / 100 % — but on the
*structured* metric the flat baseline is **0** by construction (a flat
retriever surfaces no `ContradictionPair[]` and stamps no `_superseded_by`).
The structural reading is therefore the entire signal, and the
`(structural − flat)` delta is the honest headline.

Cat 6 reconstructs the seeded supersession chains correctly, including the
clean 4-document Hill's vitamin-D recall chain:

```
pub_hills_vitd_announcement_2019_01 → pub_hills_vitd_expansion_2019_03
  → pub_hills_vitd_expansion_2019_05 → pub_hills_warning_letter_2019_11
  (current state: last)
```

## What the delta means (and what it does not)

This reading verifies the **plumbing**: that the structured fields the
spec defines are now derived end-to-end from the typed edges a backend
stores. The +1.00 delta on good-dog is the ceiling case — the corpus
declares its `contradicts` / `supersedes` edges by hand, so a system that
ingests them faithfully surfaces them all.

What it does **not** claim: that MemPalace's *extraction* pipeline would
*generate* these edges from raw text. The good-dog corpus carries them as
ground truth; whether a live palace's enrichment produces `contradicts` /
`supersedes` triples at all is a separate ingestion-quality question (Cat
4). On a live palace whose KG holds no such triples, both scorers return
the honest 0 — "the system retrieves but does not model contradictions /
supersession," which is itself the finding for that system.

The same adapters now surface these fields through the daemon `/graph`
path too (verified structurally in
`tests/test_cat3_cat6_plumbing.py::test_project_graph_surfaces_superseded_and_contradicts`
and the direct-adapter SQLite-KG path), so the matrix's mempalace column
becomes computable the moment a palace KG carries the edges.

## Reproducibility

```
./venv/bin/python -m pytest tests/test_cat3_cat6_plumbing.py -q   # 19 tests
./venv/bin/sme-eval cat3 --adapter good-dog-graph
./venv/bin/sme-eval cat6 --adapter good-dog-graph
```

Baselines: `baselines/good_dog_cat3_structural_2026-05-30.json`,
`baselines/good_dog_cat6_structural_2026-05-30.json`.
