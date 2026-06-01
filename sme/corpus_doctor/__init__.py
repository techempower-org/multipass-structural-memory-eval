"""corpus-doctor — synthetic defect injection for SME (upstream #27).

A diagnostic instrument is only trustworthy if it actually *detects* the
defects it claims to. corpus-doctor injects KNOWN structural defects into
a clean graph snapshot so the SME categories can be verified to catch
them: the inject → run-category → assert-detection loop.

This is the FIRST SLICE (upstream M0nkeyFl0wer#27 is large — it frames a
whole YCSB/TPC-H/CheckList-style defect battery). Scope here is a working
injector + verification harness covering three defect types whose
detector categories already exist:

  - ``duplicate_entity``  → Cat 4a canonical-collision dedup
    (sme.categories.ingestion_integrity)
  - ``orphan_node``       → Cat 5 isolated-node / gap detection
    (sme.categories.gap_detection)
  - ``broken_ref``        → referential-integrity check
    (sme.corpus_doctor.detectors.referential_integrity)

Each injection returns the corrupted graph PLUS a manifest of exactly
what was injected (the ground truth), so the harness can assert the
category's detection recall against a known answer rather than a guess.

The phantom-edge defect (edges with no source support) is deliberately
NOT included here — it is the subject of upstream #4 / SME task #184
(Phantasm), which is building the detector category itself. Adding it
here would collide; it is a natural follow-up once that category lands.

See ``docs/corpus_doctor.md`` for the full design and the expansion plan
toward the YCSB/TPC-H/CheckList framings named in #27.
"""

from sme.corpus_doctor.detectors import DETECTORS
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
    "CorpusDoctor",
    "InjectedDefect",
    "InjectionResult",
    "DetectionResult",
    "verify_defect",
    "run_all_defects",
]
