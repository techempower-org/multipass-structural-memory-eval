"""Pluggable entity extractors for graph-augmented retrieval.

Each backend exposes a single function:

    extract(text: str) -> list[Entity]

where Entity is the lightweight dataclass below. Backends are intentionally
thin so the extractor choice (regex / spacy / LLM) can be swapped without
touching the adapter or the bench harness.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    name: str  # canonicalized (lowercase, trimmed)
    type: str  # extractor-defined, e.g. "PROPER_NOUN", "PERSON", "ORG"
    count: int = 1  # how many times this entity appears in the source text
