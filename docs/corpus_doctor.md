# corpus-doctor — synthetic defect injection

*Upstream issue: M0nkeyFl0wer/multipass-structural-memory-eval#27 (genuinely-open).*

A diagnostic instrument is only trustworthy if it actually **detects** the
defects it claims to. corpus-doctor injects **known** structural defects into
a clean graph snapshot so the SME categories can be *verified* to catch them:
the **inject → run-category → assert-detection** loop. Without it, "Cat 4
reported 3 collisions" is unscored; with it, "Cat 4 recalled 3/3 *injected*
collisions" is a verification.

## Scope

Upstream #27 frames a large defect battery (YCSB / TPC-H / CheckList-style
framings). This module covers **five** defect types whose detector categories
exist, spanning **all three SME detection shapes** — id-recall, edge-key
recall, and scalar-signal. It is not yet the whole battery; the expansion plan
is below.

| Defect type | What's injected | Detected by | Shape |
|---|---|---|---|
| `duplicate_entity` | a clone of an existing entity under a fresh id, with a name that canonicalizes to the **same** key (a case/whitespace variant) | **Cat 4a** canonical-collision dedup (`ingestion_integrity`) | id-recall |
| `orphan_node` | a new entity with **no incident edges** (a structural isolate) | **Cat 5** isolated-node detection (`gap_detection`) | id-recall |
| `broken_ref` | a new edge whose `target_id` is an entity id that **does not exist** (a dangling reference); the source endpoint is real, isolating the defect | referential-integrity check (`detectors.detect_broken_refs`) | id-recall |
| `phantom_edge` | a new edge between **two real entities** whose `source_note` body **does not name either endpoint** — an assertion with no source support | phantom-edge grounding check (`phantom_edge.score_phantom_edges`, upstream #4) | edge-key recall |
| `edge_type_monoculture` | many edges of a **single dominant type** between real entities, collapsing the edge-type distribution | **Cat 4c** monoculture (`dominant_edge_type_fraction` ↑, normalized entropy ↓) | scalar signal |

### phantom_edge vs broken_ref

Both inject a defective edge, but they target different checks and are
deliberately distinct:

- **`broken_ref`** points at a **missing node** — caught by *endpoint
  membership* (the target id isn't in the entity set). A referential-integrity
  failure.
- **`phantom_edge`** has **both endpoints real** but **no source support** —
  caught by *grounding* (neither endpoint's name appears in the prose body the
  edge was extracted alongside). The phantom-edge category (upstream #4) is the
  graph→source diagnostic; `broken_ref` would pass it (the endpoints exist) and
  `phantom_edge` would pass a referential check (the nodes exist), so each
  isolates its own category.

## Design

Three small modules under `sme/corpus_doctor/`:

- **`injector.py`** — `CorpusDoctor` injects defects into a `(entities, edges)`
  snapshot. It is **deterministic** under a seed, **never mutates** its inputs
  (deep-copies first), and returns an `InjectionResult` carrying both the
  corrupted graph and a **manifest** (`InjectedDefect` list) of exactly what
  was injected. The manifest is the ground truth detection is scored against.
- **`detectors.py`** — maps each defect type to the SME signal that catches it.
  The three id-recall defects share a uniform `(entities, edges) → set[str]`
  shape (the flagged ids); Cat 4a and Cat 5 are wrapped, the referential check
  lives here. Two defects have non-uniform signatures the harness special-cases:
  `detect_phantom_edges` needs the source prose bodies and returns flagged
  `(src, dst, type)` **edge keys** (a phantom is a defect of the edge, not
  either endpoint); `monoculture_signal` returns a **scalar reading**
  (Cat 4c dominant fraction + normalized entropy), not a set.
- **`harness.py`** — the inject → detect → assert loop, dispatching by detection
  shape. For **id-recall** and **edge-key** defects, `verify_defect` scores
  **recall** against the manifest and **delta-precision** against the *injection
  delta* — flags on the corrupted graph but not the clean one. This is the key
  subtlety: a real corpus has its own native isolates/collisions/ungrounded
  edges, so penalizing the detector for correctly flagging those would be wrong;
  only injection-attributable flags count. For the **scalar-signal** monoculture
  defect there is no per-id set — detection means the Cat 4c reading **moved in
  the expected direction** (dominant fraction up, entropy down) vs the clean
  baseline.

  > **Degenerate case (documented, not papered over):** an already-monoculture
  > corpus (every edge one type → dominant fraction already 1.0) has no headroom
  > for `edge_type_monoculture` to move the signal, so the harness honestly
  > reports it as *not detected* there. On any corpus with ≥2 edge types it
  > moves; on good-dog (10 types) even 5 injected edges nudge it correctly, and
  > the move scales with count.

## Running it

```bash
# Default: good-dog clean graph, 5 defects/type, all three types.
python scripts/corpus_doctor.py

# Tune count/seed; restrict to one defect; emit JSON.
python scripts/corpus_doctor.py --count 10 --seed 3
python scripts/corpus_doctor.py --defect orphan_node --json
```

Exit code is non-zero if **any** injected defect goes undetected, so the script
doubles as a CI guard that the categories still detect what they claim to.
Lightweight and locally runnable — no daemon, no API key, no download
(constitutional principle).

Programmatic use:

```python
from sme.corpus_doctor import run_all_defects
from sme.corpora import good_dog_graph

entities, edges = good_dog_graph.load_graph()
results = run_all_defects(entities, edges, count=5, seed=7)
assert all(r.detected_all for r in results.values())
```

## Verification

On both a hand-built clean fixture and the real good-dog graph (97 entities /
164 edges), the four recall-shaped categories (`duplicate_entity`,
`orphan_node`, `broken_ref`, `phantom_edge`) recall **5/5** injected defects at
delta-precision **1.0**, and `edge_type_monoculture` moves the Cat 4c signal in
the expected direction (`tests/test_corpus_doctor.py`). The good-dog graph has
native isolates/collisions/ungrounded edges; the harness's delta-precision
correctly does not count them against the detectors.

```
$ python scripts/corpus_doctor.py --count 5 --seed 7
  [PASS] duplicate_entity: recall 5/5 (1.00), Δprecision 1.00 (5/5 new flags injected)
  [PASS] orphan_node: recall 5/5 (1.00), Δprecision 1.00 (5/5 new flags injected)
  [PASS] broken_ref: recall 5/5 (1.00), Δprecision 1.00 (5/5 new flags injected)
  [PASS] phantom_edge: recall 5/5 (1.00), Δprecision 1.00 (5/5 new flags injected)
  [PASS] edge_type_monoculture: signal moved (dominant_edge_type_fraction 0.348->0.367, ...)
  ALL INJECTED DEFECTS DETECTED
```

## Expansion plan (toward the full #27 battery)

The injector/detector/harness split is built to grow. With all three detection
shapes now exercised (id-recall, edge-key, scalar-signal), remaining work is:

1. **More defect types** — field-coverage gaps (empty `name`/`entity_type` →
   Cat 4b), contradiction injection (→ Cat 3), supersession/provenance breaks
   (→ Cat 6 / 6b). Each reuses an existing detection shape.
2. **Defect-rate sweeps** — inject at increasing rates and report the
   detection curve (the YCSB-style "vary one knob" framing), so a category's
   sensitivity floor is measurable, not assumed. The scalar-signal monoculture
   defect already shows this: the signal move scales with `--count`.
3. **Mixed-defect corpora** — inject several defect types at once and confirm
   each category isolates its own kind (the CheckList "behavioral test suite"
   framing — one test per capability).
4. **Adapter-level injection** — inject into a corpus *before* ingestion and
   verify detection survives the extraction pipeline, not just the graph
   snapshot.

Each is an additive `inject_*` + detector + harness parametrization; none
requires reworking the loop.
