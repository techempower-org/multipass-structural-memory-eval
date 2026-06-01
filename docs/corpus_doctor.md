# corpus-doctor — synthetic defect injection

> Upstream: `M0nkeyFl0wer/multipass-structural-memory-eval#27`.
> Status: **first slice** — 3 of 6 issue-#27 pathologies implemented,
> verification harness live, manifest schema frozen. The remaining
> pathologies are tracked in `PATHOLOGY_BACKLOG` (see [Backlog](#backlog)).

## Why

SME's public corpora are hand-authored and integrity-clean by
construction. Across every shipped corpus the structural probes report
zero collisions, zero orphans, full edge-type diversity. That leaves
**Cat 4 (The Threshold)** and **Cat 5 (The Missing Room)**
*un-calibrated*: they always read clean on clean input, so a passing
reading proves nothing about the detector's sensitivity. There was no
tool to inject a *known* defect and confirm the cat catches it.

corpus-doctor closes that loop. It takes a clean `(entities, edges)`
snapshot, injects a known defect, and emits a PROV-O-shaped
`defects.jsonl` manifest stating exactly what it did. The verification
harness then runs the relevant category against the dirtied snapshot and
**grades the reading against the manifest** — turning "reads clean on
clean corpora" into a falsifiable, measured sensitivity.

## Pathologies (first slice)

Each pathology is a pure transform over the core dataclasses
(`(entities, edges) -> DoctorResult`) and maps to an existing detector
with **no new dependency** (numpy / networkx only, via the scorers):

| Pathology | Detector | Signal | What it does |
|---|---|---|---|
| `duplicate_evidence` | Cat 4a | `canonical_collisions` (+1 each) | Clones entities under fresh IDs but identical name+type, so they collapse onto one canonical key — the low-degree straggler that "missed canonicalization". |
| `orphan_inflation` | Cat 5 | `isolated_nodes` (+1 each) | Strips every edge touching a sampled set of entities, leaving them as single-node components (orphans). |
| `monoculture_edge_type` | Cat 4c | `dominant_edge_type_fraction` (↑) | Rewrites a sampled fraction of edges onto a single dominant type, collapsing edge-type entropy toward 0. Defaults to amplifying the corpus's *existing* dominant type. |

`severity` (0..1) scales the count/fraction of injected defects; `seed`
makes injection deterministic.

### Collateral isolation (orphan_inflation)

Stripping a node's edges can also strand a degree-1 *neighbour* that only
connected through it. The manifest lists exactly the nodes corpus-doctor
**targeted**, so Cat 5 may legitimately report *more* isolates than there
are defects. The harness treats this as full recovery (it caps the
observed delta at the expected one) — the detector is right about the
collateral, not penalised for it. The over-shoot is surfaced as a note,
not a failure.

## CLI

```bash
# Verify all three pathologies on the in-tree good-dog corpus baseline
sme-eval corpus-doctor

# One pathology, write the dirtied snapshot + manifest
sme-eval corpus-doctor --pathology orphan_inflation --severity 0.5 \
    --out-dir /tmp/dirty-good-dog

# Dirty a real system's graph instead of the in-tree corpus
sme-eval corpus-doctor --from-adapter mempalace --db /path/to/palace.db \
    --json report.json
```

Exit code is **non-zero** when any pathology goes undetected, so the
command doubles as a CI calibration gate.

## Manifest schema (`defects.jsonl`)

One PROV-O-aligned JSON record per injected defect, newline-delimited.
This reuses the repo's existing PROV-O-JSON-shape convention (see
`docs/industry_standards_integration.md`) — no new dependency.

```json
{
  "defect_id": "dup::brand_hills_science_diet::0000",
  "pathology": "duplicate_evidence",
  "prov:activity": "inject_defect",
  "prov:wasAttributedTo": "corpus-doctor/0.1",
  "severity": 0.3,
  "seed": 0,
  "target": {
    "kind": "entity",
    "ids": ["brand_hills_science_diet__dupe0000"],
    "source_id": "brand_hills_science_diet"
  },
  "expect": {
    "category": "cat4",
    "field": "canonical_collisions",
    "delta": 1
  },
  "detail": {"entity_type": "product", "name": "Hill's Science Diet"}
}
```

- **`expect`** states the category/field the defect should move and by
  how much (`delta` for count defects; `direction: "increase"` for
  direction-only defects like monoculture), so the harness grades against
  the manifest without re-deriving the expectation.
- **`target`** identifies the injected artefact (`entity` clone, orphaned
  `entity`, or rewritten `edge`).
- **`detail`** carries the reversible record (removed edges, original
  edge type) so the dirtying can be undone.

Round-trips via `write_manifest` / `load_manifest`.

## Recovery rate

The headline number per pathology:

- **Count-delta** (`duplicate_evidence`, `orphan_inflation`):
  `recovery = min(observed_delta, expected_delta) / expected_delta`,
  comparing the dirty reading minus the clean baseline against the summed
  expected delta. Capped at 1.0 (collateral over-shoot is not rewarded).
- **Direction-only** (`monoculture_edge_type`): recovered iff the reading
  moved up by a non-trivial margin.

A pathology is `detected` when recovery ≥ `detection_threshold`
(default 0.99 — the harness is a calibration gate, not a soft score).

## Validation reading

Against the in-tree **good-dog corpus** (164 edges, dominant edge type
`mentions` at 34.8%), severity 0.3, seed 0:

```
duplicate_evidence     canonical_collisions         clean=0 dirty=29 obs=+29 expected +29 recovery=100% DETECTED
monoculture_edge_type  dominant_edge_type_fraction  clean=0.348 dirty=0.543 obs=+0.195 expected ↑ recovery=100% DETECTED
orphan_inflation       isolated_nodes               clean=0 dirty=34 obs=+34 expected +29 recovery=100% DETECTED
3/3 pathologies recovered (categories exercised: cat4, cat5)
```

All three implemented pathologies are recovered on a real public corpus —
the cats genuinely detect what corpus-doctor injects.

## Backlog

The remaining issue-#27 pathologies are deferred (named in
`sme.corpus_doctor.PATHOLOGY_BACKLOG` so the gap is discoverable in code):

- `zipfian_degree` — YCSB power-law degree re-weight (parameter `s`).
- `hotspot_entity` — YCSB N% of entities answer M% of queries.
- `stale_facts` — later-dated contradicting fact (exercises Cat 3 + Cat 6).
- `phantom_edge` — edge with no source support. **Blocked on the Cat
  phantom-edge detector** (upstream
  `M0nkeyFl0wer/multipass-structural-memory-eval#4`): once that detector
  lands, a `phantom_edge` pathology can target it directly.

`TextAttack` adversarial text perturbation remains an opt-in
`[text-perturb]` extra (follow-up).

## Files

- `sme/corpus_doctor.py` — injection engine + manifest IO.
- `sme/categories/corpus_doctor_harness.py` — verification harness.
- `sme/cli.py` — `corpus-doctor` subcommand.
- `tests/test_corpus_doctor.py` — inject → detect round-trip tests.
