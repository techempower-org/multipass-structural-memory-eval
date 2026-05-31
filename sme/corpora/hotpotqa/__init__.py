"""HotpotQA corpus loader for SME cross-validation.

HotpotQA (Yang et al., EMNLP 2018; arXiv 1809.09600) is a Wikipedia-based
multi-hop question-answering benchmark with sentence-level annotated
supporting facts. This package loads the released benchmark JSON and produces
SME-shape question records so SME's **Cat 2c** (multi-hop retrieval recall by
depth) reading can be cross-validated against a public, 1000s-scale corpus
with known multi-hop evidence — the construct-validity demonstration the
hand-authored corpora cannot give (upstream
M0nkeyFl0wer/multipass-structural-memory-eval#43, Phase 1).

PINNED SUBSET: ``dev_distractor`` — ``hotpot_dev_distractor_v1.json``, the
**distractor** setting (10-paragraph haystack: 2 gold + 8 distractor),
**7405** questions. Every question is 2-hop by construction, so the loader
assigns ``min_hops = 2`` to every record (the Cat 2c key) and exposes the
``type`` field (comparison vs bridge) as the multi-hop shape. See loader.py's
module docstring for the comparability contract and the hop-depth rationale.

The dataset itself is NOT committed to this repo. Download with:

    mkdir -p sme/corpora/hotpotqa/data
    cd sme/corpora/hotpotqa/data
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

See README.md for details and the SME mapping table.
"""

from sme.corpora.hotpotqa.loader import (
    HOTPOT_MIN_HOPS,
    HOTPOT_SME_CATEGORY,
    HOTPOT_TYPE_NAMES,
    SETTING,
    SUBSET,
    SUBSET_QUESTION_COUNT,
    HotpotParagraph,
    HotpotQuestion,
    load_questions,
    materialize_sme_corpus,
)

__all__ = [
    "HOTPOT_MIN_HOPS",
    "HOTPOT_SME_CATEGORY",
    "HOTPOT_TYPE_NAMES",
    "SETTING",
    "SUBSET",
    "SUBSET_QUESTION_COUNT",
    "HotpotParagraph",
    "HotpotQuestion",
    "load_questions",
    "materialize_sme_corpus",
]
