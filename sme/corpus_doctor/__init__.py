"""corpus-doctor — synthetic defect injection for SME (upstream #27).

A diagnostic instrument is only trustworthy if it actually *detects* the
defects it claims to. corpus-doctor injects KNOWN structural defects into
a clean graph snapshot so the SME categories can be verified to catch
them: the inject → run-category → assert-detection loop.

Upstream M0nkeyFl0wer#27 is large — it frames a whole YCSB/TPC-H/
CheckList-style defect battery. This package covers five defect types
whose detector categories exist, spanning all three SME detection shapes:

  id-recall (a set of defective entity/target ids):
  - ``duplicate_entity``  → Cat 4a canonical-collision dedup
    (sme.categories.ingestion_integrity)
  - ``orphan_node``       → Cat 5 isolated-node / gap detection
    (sme.categories.gap_detection)
  - ``broken_ref``        → referential-integrity check
    (sme.corpus_doctor.detectors.detect_broken_refs)

  edge-key recall (a defect of the EDGE, both endpoints real):
  - ``phantom_edge``      → phantom-edge grounding check
    (sme.categories.phantom_edge.score_phantom_edges, upstream #4)

  scalar signal (a distribution moved, no per-id defect set):
  - ``edge_type_monoculture`` → Cat 4c monoculture
    (dominant_edge_type_fraction up, normalized entropy down)

Each injection returns the corrupted graph PLUS a manifest of exactly
what was injected (the ground truth), so the harness can assert the
category's detection against a known answer rather than a guess.

The first three (duplicate/orphan/broken_ref) shipped in #27; phantom_edge
and edge_type_monoculture were added once the phantom-edge detector (#4)
and the Cat 4c monoculture reading landed on main, closing the
inject→detect loop for the remaining structural-integrity categories.

See ``docs/corpus_doctor.md`` for the full design and the expansion plan
toward the YCSB/TPC-H/CheckList framings named in #27.
"""

from sme.corpus_doctor.detectors import (
    DETECTORS,
    ID_RECALL_DEFECTS,
    detect_phantom_edges,
    monoculture_signal,
)
from sme.corpus_doctor.harness import (
    DetectionResult,
    run_all_defects,
    verify_defect,
)
from sme.corpus_doctor.injector import (
    DEFECT_TYPES,
    CorpusDoctor,
    InjectedDefect,
    InjectionResult,
)

__all__ = [
    "DEFECT_TYPES",
    "DETECTORS",
    "ID_RECALL_DEFECTS",
    "detect_phantom_edges",
    "monoculture_signal",
    "CorpusDoctor",
    "InjectedDefect",
    "InjectionResult",
    "DetectionResult",
    "verify_defect",
    "run_all_defects",
]
