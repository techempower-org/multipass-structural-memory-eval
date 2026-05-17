"""Regex-based entity extractor.

Reuses mempalace.entity_detector's locale-aware pattern files (candidate
single-word + multi-word patterns) without the 3+ occurrence frequency
threshold that filters out per-chunk mentions. Intended for write-through
AGE-population spikes where we want every proper-noun mention captured,
not just frequent ones.

Latency: ~5-20ms per drawer-sized chunk. Quality: precision >> recall
(catches capitalized proper nouns; misses concepts, hyphenated identifiers
that don't capitalize, lowercase acronyms).

Better-quality backends (spacy.py, llm.py) drop in here without changing
the adapter signature.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from sme.extractors import Entity

_MP_PATH = "/home/jp/Projects/memorypalace"
if _MP_PATH not in sys.path:
    sys.path.insert(0, _MP_PATH)


def _patterns(languages: Iterable[str] = ("en",)):
    from mempalace.entity_detector import _normalize_langs, _get_stopwords, get_entity_patterns

    langs = _normalize_langs(tuple(languages))
    return get_entity_patterns(langs), _get_stopwords(langs)


_PATTERNS_CACHE: dict | None = None
_STOPWORDS_CACHE: set | None = None


def _get_cached(languages: tuple = ("en",)):
    global _PATTERNS_CACHE, _STOPWORDS_CACHE
    if _PATTERNS_CACHE is None:
        _PATTERNS_CACHE, _STOPWORDS_CACHE = _patterns(languages)
    return _PATTERNS_CACHE, _STOPWORDS_CACHE


# Technical identifier patterns — hyphenated names, version strings, slashed paths.
# These catch domain entities the proper-noun regex misses: pgvector,
# nakata-app, FT-300, chat-ce-v3, mempalace_traverse, v3.3.5, etc.
_TECH_PATTERNS = [
    # hyphenated lowercase with digits: ft-300, chat-ce-v3, nakata-app, pgvector-cutover
    re.compile(r"\b[a-z][a-z0-9]*(?:[-_][a-z0-9]+){1,4}\b"),
    # version strings: v3.3.5, 1.5.4, 2026-05-14
    re.compile(r"\bv?\d+(?:\.\d+){1,3}\b"),
    # FT- / fT prefixed model variants: FT-300, FT-100, ft-v4
    re.compile(r"\b[Ff][Tt][- ]?\w+\b"),
    # repo-style identifiers: owner/repo, M0nkeyFl0wer/mempalace
    re.compile(r"\b[A-Za-z][A-Za-z0-9-_]*/[A-Za-z0-9-_.]+\b"),
]

# Identifiers to skip — common stopwords appearing as hyphenated forms.
_TECH_STOPWORDS = {
    "do-not", "follow-up", "follow-ups", "as-is", "real-time", "left-over",
    "open-source", "one-shot", "self-contained", "off-the-shelf", "in-flight",
}


def extract(text: str, *, languages: tuple = ("en",), min_len: int = 3) -> list[Entity]:
    """Return per-chunk entity list with counts, no frequency threshold."""
    if not text:
        return []
    patterns, stopwords = _get_cached(languages)
    counts: Counter[str] = Counter()
    types: dict[str, str] = {}

    for wrapped in patterns["candidate_patterns"]:
        try:
            rx = re.compile(wrapped)
        except re.error:
            continue
        for word in rx.findall(text):
            if word.lower() in stopwords or len(word) < min_len:
                continue
            key = word.strip()
            counts[key] += 1
            types.setdefault(key, "PROPER_NOUN")

    for wrapped in patterns["multi_word_patterns"]:
        try:
            rx = re.compile(wrapped)
        except re.error:
            continue
        for phrase in rx.findall(text):
            if any(w.lower() in stopwords for w in phrase.split()):
                continue
            key = phrase.strip()
            if len(key) < min_len:
                continue
            counts[key] += 1
            types[key] = "PROPER_NOUN_MULTI"

    # Second pass: technical identifiers (hyphenated names, version strings)
    for rx in _TECH_PATTERNS:
        for match in rx.findall(text):
            key = match.strip()
            if len(key) < min_len:
                continue
            kl = key.lower()
            if kl in stopwords or kl in _TECH_STOPWORDS:
                continue
            counts[key] += 1
            types.setdefault(key, "TECH_IDENT")

    return [
        Entity(name=k.lower(), type=types[k], count=v)
        for k, v in counts.most_common()
    ]
