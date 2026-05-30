"""LoCoMo corpus loader for SME cross-validation.

LoCoMo ("Long Conversational Memory"; Maharana et al., ACL 2024;
arXiv 2402.17753) is a benchmark of very long-term, multi-session
two-persona dialogues plus a 5-category question-answering suite. This
package loads the released benchmark JSON and produces SME-shape
question records so SME's category readings can be cross-validated
against the LoCoMo QA leaderboard (EverOS 93.05% / True Memory 93.0% /
Mem0 92.5% / Hindsight 89.61% all report LoCoMo QA accuracy).

PINNED SUBSET: ``locomo10`` — the canonical released benchmark, all
**1986** QA across **10** conversations, **adversarial category
included**. See loader.py's module docstring for the comparability
contract and the category-numbering caveat (the released JSON uses the
official scorer's numbering, where category 1 = multi-hop, NOT the
paper-prose numbering).

The dataset itself is NOT committed to this repo. Download with:

    cd sme/corpora/locomo
    mkdir -p data && cd data
    wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

See README.md for details and the SME mapping table.
"""

from sme.corpora.locomo.loader import (
    ADVERSARIAL_INCLUDED,
    LOCOMO_CATEGORY_NAMES,
    LOCOMO_CATEGORY_TO_SME,
    SUBSET,
    SUBSET_QA_COUNT,
    SUBSET_SAMPLE_COUNT,
    LoCoMoQuestion,
    LoCoMoSession,
    LoCoMoTurn,
    load_questions,
    materialize_sme_corpus,
)

__all__ = [
    "ADVERSARIAL_INCLUDED",
    "LOCOMO_CATEGORY_NAMES",
    "LOCOMO_CATEGORY_TO_SME",
    "SUBSET",
    "SUBSET_QA_COUNT",
    "SUBSET_SAMPLE_COUNT",
    "LoCoMoQuestion",
    "LoCoMoSession",
    "LoCoMoTurn",
    "load_questions",
    "materialize_sme_corpus",
]
