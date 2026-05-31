# Ontology-Sensitivity Sweep — how much do Cat 4 / Cat 5 move under different ontologies?

**Date:** 2026-05-31
**Author:** Lucid (sme-dreamteam)
**Issue:** [M0nkeyFl0wer/multipass-structural-memory-eval#45](https://github.com/M0nkeyFl0wer/multipass-structural-memory-eval/issues/45) — *Ontology-sensitivity reading: how much do Cat 4 / Cat 5 readings move under different ontology choices?*
**Status:** RESOLVED with a split finding (see Verdict). Read-only over the in-tree good-dog graph; no adapter, no daemon, no production contamination.
**Data:** `baselines/ontology_sensitivity_good_dog_2026-05-31.json`
**Harness:** `sme/eval/ontology_sensitivity.py` · conditions `sme/corpora/good_dog_ontologies.py` · runner `scripts/run_ontology_sensitivity.py`

---

## The question

Cat 4 (ingestion integrity) and Cat 5 (structural gap detection) both read on top of an ontology that is **exogenous** to SME — the entity types and edge types come from the adapter's underlying graph, not from SME. That makes ontology design an **unmeasured confound** for cross-system or cross-corpus comparison: *"system X scores 0.4 on Cat 4 vs Y at 0.8"* is partly a statement about ontology granularity, not adapter quality.

The experiment (the issue's own design): take one corpus, run it through three deliberately-different ontologies — **flat** (one type), **moderate** (the corpus as-authored), **fine-grained** (split types) — same corpus, same graph, and measure how far Cat 4 / Cat 5 move.

## Method

- **Corpus:** good-dog-corpus (the issue names jp-realm *or* good-dog; good-dog is the graph-bearing one — jp-realm-v0.1 ships only flat text). 97 entities, 164 edges, authored at 8 entity types / 10 edge types.
- **Topology preservation is the load-bearing invariant.** The sweep remaps only `entity_type` / `edge_type`; the node set, edge set, and every node/edge identity are untouched. So the *graph* is identical across conditions — any Cat 5 movement would be a type-driven signal, not a different graph. (Pinned by `test_remap_preserves_node_and_edge_sets`.)
- **Three conditions** (`good_dog_ontologies.py`, all deterministic pure functions — no model calls):
  - `flat` — every node → `node`, every edge → `related` (1 type each).
  - `moderate` — identity (8 entity / 10 edge types as authored).
  - `fine_grained` — `person`/`organization`/`publication` split by the corpus source-domain folder (researcher vs journalist vs official; kennel_club vs research_org vs regulatory_body; study vs breed_standard vs bylaw vs article); the catch-all `mentions` edge split by its evidence string. 15 entity / 12 edge types. Fine types are strict subtypes of moderate types (pinned by `test_..._splits_are_subtypes_of_moderate`).

## Results

| metric | flat | moderate | fine-grained | spread | rel. | stable |
|---|---:|---:|---:|---:|---:|:--:|
| **Cat 4** canonical_collisions | 1 | 0 | 0 | 1.0 | 300% | ✗ |
| **Cat 4** edge_type_entropy_normalized | 0.000 | 0.842 | 0.856 | 0.856 | 151% | ✗ |
| **Cat 4** dominant_edge_type_fraction | 1.000 | 0.348 | 0.293 | 0.707 | 129% | ✗ |
| **Cat 5** components | 4 | 4 | 4 | 0 | 0% | ✓ |
| **Cat 5** largest_component_size | 44 | 44 | 44 | 0 | 0% | ✓ |
| **Cat 5** isolated_nodes | 0 | 0 | 0 | 0 | — | ✓ |
| **Cat 5** betti_0_largest | 1 | 1 | 1 | 0 | 0% | ✓ |
| **Cat 5** betti_1_largest | 9 | 9 | 9 | 0 | 0% | ✓ |

## Verdict — a *split* finding, both halves publishable

**Cat 5 (The Missing Room) is ROBUST to ontology choice.** Components, largest-component size, isolate count, and both Betti numbers are **byte-identical** across flat / moderate / fine-grained. This is not luck — Cat 5's signals are functions of graph *topology*, and the topology is the same graph under every ontology. **Cross-system Cat 5 comparison is valid even when ontologies differ.**

**Cat 4 (The Threshold) is SENSITIVE to ontology choice — by construction.** Its monoculture signals are *definitionally* a function of the type vocabulary:

- `edge_type_entropy_normalized` is `H / log2(n_types)`; with one edge type it is **0.0 by definition**, and it climbs to 0.84–0.86 once real types exist. This is the metric *restating the ontology granularity*, not measuring adapter quality.
- `dominant_edge_type_fraction` is 1.0 under flat (one type holds 100%) and falls to ~0.29–0.35 as types split. The "edge-type monoculture" alarm Cat 4 raises is therefore only interpretable **relative to a fixed ontology**.
- `canonical_collisions` shows the subtler hazard: flat reports **1 false collision** because two distinct entities with the same *name* canonicalize together once `entity_type` is stripped (`default_canonical_key` is name+type scoped). A too-coarse ontology **manufactures** an ingestion defect that isn't real.

**The methodological caveat this lands:** Cat 4 readings MUST be reported alongside the ontology choice; a cross-system Cat 4 comparison is only valid when the ontologies are matched in granularity. This is exactly the confound flagged in the prod Cat 4 re-map work (the `other`-sink collapse) — that re-map moved the *vocabulary*, and this sweep shows the headline metric moves with it. Cat 5, by contrast, can be compared across systems with confidence.

## Spec implication (suggested, not yet applied)

`docs/sme_spec_v8.md` should carry, near the Cat 4 definition:

> Cat 4's monoculture/entropy signals are a function of the system's ontology granularity and are only comparable across systems at matched granularity. Report Cat 4 with the entity/edge type counts. Cat 5's topology signals are ontology-robust and are cross-comparable.

An explicit `--ontology-condition` reporting flag is a natural follow-up if cross-system Cat 4 tables are published.

## Reproduce

```bash
./venv/bin/python scripts/run_ontology_sensitivity.py --corpus good-dog \
    --out baselines/ontology_sensitivity_good_dog_2026-05-31.json
```

## Limits

- One corpus (good-dog, 97 nodes). The *direction* of the finding (Cat 4 type-driven, Cat 5 topology-driven) is structural and will generalize; the exact magnitudes are corpus-specific. Re-running against a HotpotQA/MINE graph once those land (#43) would widen the evidence base, as the issue notes.
- good-dog has zero isolates, so `isolated_by_type` (which *is* type-keyed and could move) doesn't surface here. A corpus with isolates would exercise that one Cat 5 sub-signal's ontology sensitivity; flagged for the #43 follow-up.
