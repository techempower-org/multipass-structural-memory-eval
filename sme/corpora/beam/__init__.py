"""BEAM corpus loader for SME cross-validation.

BEAM ("Beyond a Million Tokens: Benchmarking and Enhancing Long-Term
Memory in LLMs"; ICLR 2026; arXiv 2510.27246) is a long-term memory
benchmark of very long single-user/assistant conversations (100K–10M
tokens) plus a 10-ability probing-question suite, scored end-to-end
with rubric *nuggets*. This package loads the released benchmark and
produces SME-shape question records so SME's category readings can be
cross-validated against the BEAM QA leaderboard (Mem0 reports 64.1 /
48.6 at the 1M / 10M buckets).

PINNED CONTRACT: BEAM is graded at token **buckets** (100K / 500K / 1M
/ 10M). The same conversation at a different bucket is a different
retrieval problem, so every record carries its `bucket` and any
published reading MUST state it. BEAM ships 20 probing questions per
conversation (10 ability types x 2). See loader.py's module docstring
for the full schema and the SME mapping table.

The dataset itself is NOT committed to this repo. It lives on
HuggingFace (``Mohammadta/BEAM`` splits 100K/500K/1M, and
``Mohammadta/BEAM-10M`` for the 10M bucket). See README.md for how to
fetch and cache it into per-bucket JSON files.
"""

from sme.corpora.beam.loader import (
    ABILITY_TYPES,
    BEAM_ABILITY_TO_SME,
    QUESTIONS_PER_CONVERSATION,
    VALID_BUCKETS,
    BEAMQuestion,
    BEAMSession,
    BEAMTurn,
    load_questions,
    materialize_sme_corpus,
)

__all__ = [
    "ABILITY_TYPES",
    "BEAM_ABILITY_TO_SME",
    "QUESTIONS_PER_CONVERSATION",
    "VALID_BUCKETS",
    "BEAMQuestion",
    "BEAMSession",
    "BEAMTurn",
    "load_questions",
    "materialize_sme_corpus",
]
