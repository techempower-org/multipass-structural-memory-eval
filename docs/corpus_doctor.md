# corpus-doctor — synthetic defect injection

*Upstream issue: M0nkeyFl0wer/multipass-structural-memory-eval#27 (genuinely-open).*

A diagnostic instrument is only trustworthy if it actually **detects** the
defects it claims to. corpus-doctor injects **known** structural defects into
a clean graph snapshot so the SME categories can be *verified* to catch them:
the **inject → run-category → assert-detection** loop. Without it, "Cat 4
reported 3 collisions" is unscored; with it, "Cat 4 recalled 3/3 *injected*
collisions" is a verification.

## Scope (first slice)

Upstream #27 frames a large defect battery (YCSB / TPC-H / CheckList-style
framings). This module is the **first slice** — a working injector + a
verification harness + tests, covering three defect types whose detector
categories already exist. It is deliberately not the whole battery; the
expansion plan is below.

| Defect type | What's injected | Detected by |
|---|---|---|
| `duplicate_entity` | a clone of an existing entity under a fresh id, with a name that canonicalizes to the **same** key (a case/whitespace variant) | **Cat 4a** canonical-collision dedup (`sme.categories.ingestion_integrity`) |
| `orphan_node` | a new entity with **no incident edges** (a structural isolate) | **Cat 5** isolated-node detection (`sme.categories.gap_detection`) |
| `broken_ref` | a new edge whose `target_id` is an entity id that **does not exist** (a dangling reference); the source endpoint is real, isolating the defect | referential-integrity check (`sme.corpus_doctor.detectors.detect_broken_refs`) |

### Why not phantom-edge?

A phantom edge (an edge with no *source support* in the corpus) is the subject
of upstream #4 / SME task #184, which is building that detector **category**
itself. Injecting phantom edges here would collide with that work. It is a
natural follow-up once that category lands — `broken_ref` (an edge pointing at
a missing *node*) is the adjacent, non-colliding referential defect, and is
distinct: a broken ref is caught by endpoint membership, a phantom edge by
source attribution.

## Design

Three small modules under `sme/corpus_doctor/`:

- **`injector.py`** — `CorpusDoctor` injects defects into a `(entities, edges)`
  snapshot. It is **deterministic** under a seed, **never mutates** its inputs
  (deep-copies first), and returns an `InjectionResult` carrying both the
  corrupted graph and a **manifest** (`InjectedDefect` list) of exactly what
  was injected. The manifest is the ground truth detection is scored against.
- **`detectors.py`** — maps each defect type to the SME signal that catches it,
  in a uniform `(entities, edges) → set[str]` shape (the flagged ids). Cat 4a
  and Cat 5 are wrapped; the referential check is small enough to live here.
- **`harness.py`** — the inject → detect → assert loop. `verify_defect`
  injects, detects, and scores **recall** against the manifest. It also runs
  the detector on the **clean** graph first so **precision** is measured
  against the *injection delta* — ids flagged on the corrupted graph but not
  the clean one. This is the key subtlety: a real corpus has its own native
  isolates and collisions, so penalizing the detector for correctly flagging
  those would be wrong. Only injection-attributable flags count toward
  delta-precision.

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
164 edges), all three categories recall **5/5** injected defects at
delta-precision **1.0** (`tests/test_corpus_doctor.py`, 17 tests). The
good-dog graph has native isolates/collisions; the harness's delta-precision
correctly does not count them against the detectors.

## Expansion plan (toward the full #27 battery)

The injector/detector/harness split is built to grow:

1. **More defect types** — phantom edges (once #184 lands the category),
   field-coverage gaps (empty `name`/`entity_type` → Cat 4b), edge-type
   monoculture amplification (→ Cat 4c), contradiction injection (→ Cat 3),
   provenance breaks (→ Cat 6b).
2. **Defect-rate sweeps** — inject at increasing rates and report the
   detection curve (the YCSB-style "vary one knob" framing), so a category's
   sensitivity floor is measurable, not assumed.
3. **Mixed-defect corpora** — inject several defect types at once and confirm
   each category isolates its own kind (the CheckList "behavioral test suite"
   framing — one test per capability).
4. **Adapter-level injection** — inject into a corpus *before* ingestion and
   verify detection survives the extraction pipeline, not just the graph
   snapshot.

Each is an additive `inject_*` + detector + harness parametrization; none
requires reworking the loop.
