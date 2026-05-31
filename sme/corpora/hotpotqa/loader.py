"""Load HotpotQA JSON records into SME-shape Python objects.

HotpotQA (Yang et al., EMNLP 2018; arXiv 1809.09600) is a Wikipedia-based
multi-hop question-answering benchmark. Each question is engineered so the
answer cannot be found in a single paragraph — the solver must combine facts
from two distinct gold paragraphs, and the dataset annotates the exact
*supporting facts* (sentence-level) that justify the answer. That annotated
multi-hop structure is what makes HotpotQA the natural public calibration
surface for SME **Cat 2c** (multi-hop retrieval recall by depth): unlike the
hand-authored jp-realm-v0.1 corpus, it provides a 1000s-scale corpus with
known multi-hop evidence to demonstrate (not just design) construct validity
(upstream M0nkeyFl0wer/multipass-structural-memory-eval#43, Phase 1).

Schema reference: the released ``hotpot_dev_distractor_v1.json`` and
``hotpot_train_v1.1.json`` from https://github.com/hotpotqa/hotpot. Each
top-level element is one *question* with the shape::

    {
      "_id": "5a8b57f25542995d1e6f1371",
      "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
      "answer": "yes",
      "type": "comparison",            # "comparison" | "bridge"
      "level": "hard",                 # "easy" | "medium" | "hard"
      "supporting_facts": [            # [title, sent_id], sent_id 0-based
        ["Scott Derrickson", 0],
        ["Ed Wood", 0]
      ],
      "context": [                     # [title, [sentence, ...]] per paragraph
        ["Scott Derrickson", ["Scott Derrickson (born ...) is an ...", ...]],
        ["Ed Wood", ["Edward Davis Wood Jr. ... was an American ...", ...]],
        ...                            # ~10 paragraphs in distractor setting
      ]
    }

Unlike LoCoMo (one shared conversation per *sample*), each HotpotQA question
owns its own haystack — the ~10 ``context`` paragraphs (2 gold + 8
distractors in the distractor setting). This mirrors LongMemEval's
per-question haystack scoping. The loader returns a stream of
``HotpotQuestion`` dataclasses and does not write to disk by default; the
helper ``materialize_sme_corpus`` writes per-question vault directories for
adapters that ingest a corpus ahead of querying.


PINNED SUBSET (the comparability contract)
-------------------------------------------
HotpotQA cross-comparisons are unreliable unless the exact split and
retrieval setting are pinned. This loader pins:

    SUBSET   = "dev_distractor"   # hotpot_dev_distractor_v1.json
    SETTING  = "distractor"       # 10-paragraph haystack (2 gold + 8 distractor)
    SUBSET_QUESTION_COUNT = 7405  # questions in hotpot_dev_distractor_v1.json

i.e. the **dev distractor split**, the standard reporting surface for the
supervised multi-hop task (the fullwiki setting retrieves over all of
Wikipedia and is a different, IR-heavy task — out of scope for a loader).
Any reading published from this loader MUST state the split + setting. The
7405 count is the size of ``hotpot_dev_distractor_v1.json`` (verified against
the upstream release). The train split (``hotpot_train_v1.1.json``, 90,447
questions) is also loadable by this loader — pass its path explicitly — but
the *pinned* comparability subset for cross-validation is the dev distractor
split.


HOP DEPTH (the Cat 2c join)
---------------------------
SME Cat 2c groups questions by ``min_hops``. HotpotQA does not annotate an
explicit integer hop depth, but every released question is 2-hop by
construction (two gold supporting paragraphs), so this loader assigns
``min_hops = 2`` to every question and exposes the ``type`` field
(``comparison`` vs ``bridge``) as the qualitative multi-hop *shape*:

  - ``bridge``     — sequential: resolve a bridge entity in paragraph A,
                     then use it to answer from paragraph B (true chaining).
  - ``comparison`` — parallel: retrieve a fact from each of two paragraphs
                     and compare them (both must be found, no chaining).

Both require ≥2 distinct gold paragraphs, hence ``min_hops = 2``. A future
deeper-hop corpus (e.g. MuSiQue) would extend the depth axis; HotpotQA pins
the 2-hop calibration point.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# --- Pinned subset (the comparability contract; see module docstring) ------
SUBSET = "dev_distractor"
SETTING = "distractor"
SUBSET_QUESTION_COUNT = 7405

# Every released HotpotQA question is 2-hop by construction (two gold
# supporting paragraphs). SME Cat 2c keys on this integer.
HOTPOT_MIN_HOPS = 2

# type -> human-readable multi-hop shape. The released JSON labels each
# question "comparison" or "bridge"; both are 2-hop, they differ in whether
# the two facts are chained (bridge) or compared (comparison).
HOTPOT_TYPE_NAMES = {
    "comparison": "comparison (parallel 2-hop)",
    "bridge": "bridge (sequential 2-hop)",
}

# All HotpotQA questions map to SME Cat 2c (multi-hop retrieval by depth).
# There is no single-hop / temporal / adversarial split the way LoCoMo has,
# so the mapping is uniform — the calibration value is entirely on the
# multi-hop axis.
HOTPOT_SME_CATEGORY = "cat_2c"


@dataclass(frozen=True)
class HotpotParagraph:
    """One context paragraph: a Wikipedia title plus its sentences.

    ``is_gold`` marks paragraphs that carry at least one annotated supporting
    fact — the gold evidence the multi-hop answer is built from. In the
    distractor setting exactly two paragraphs are gold; the rest are
    distractors retrieved to look plausible.
    """

    title: str
    sentences: list[str] = field(default_factory=list)
    is_gold: bool = False

    @property
    def text(self) -> str:
        """The full paragraph as a single string (sentences joined)."""
        return " ".join(self.sentences)

    def supporting_sentence_texts(self, sent_ids: list[int]) -> list[str]:
        """Texts of the sentences at the given 0-based ids (bounds-checked).

        Out-of-range ids (a few exist in the upstream annotation) are
        skipped rather than raising, mirroring the loader's tolerant
        evidence handling elsewhere.
        """
        out: list[str] = []
        for i in sent_ids:
            if 0 <= i < len(self.sentences):
                out.append(self.sentences[i])
        return out


@dataclass(frozen=True)
class HotpotQuestion:
    """One HotpotQA evaluation instance, parsed from a top-level record.

    Each question owns its own haystack — the ``paragraphs`` list (the ~10
    distractor-setting context paragraphs). ``supporting_facts`` are the
    annotated gold evidence as ``(title, sentence_id)`` pairs; the gold
    paragraph titles are the natural ``expected_sources`` target for SME's
    substring-on-filename matcher.
    """

    question_id: str        # the upstream "_id"
    question: str
    answer: str             # gold answer; "yes"/"no" for comparison questions
    qtype: str              # raw "comparison" | "bridge"
    type_name: str          # HOTPOT_TYPE_NAMES[qtype]
    level: str              # "easy" | "medium" | "hard" ("" if absent)
    min_hops: int           # always HOTPOT_MIN_HOPS (2) — the Cat 2c key
    supporting_facts: list[tuple[str, int]]  # [(title, sent_id), ...]
    paragraphs: list[HotpotParagraph]        # the per-question haystack
    sme_category: str       # always HOTPOT_SME_CATEGORY (cat_2c)

    @property
    def gold_answer(self) -> str:
        return self.answer

    @property
    def gold_titles(self) -> list[str]:
        """Unique titles of the gold (supporting-fact) paragraphs,
        order-preserving — the multi-hop evidence set."""
        seen: list[str] = []
        for title, _ in self.supporting_facts:
            if title not in seen:
                seen.append(title)
        return seen

    def expected_sources_paragraph_level(self) -> list[str]:
        """Gold paragraph titles — the SME ``expected_sources`` target.

        A question is multi-hop iff this list has ≥2 entries; for a
        well-formed HotpotQA item it is exactly the two gold paragraphs.
        """
        return self.gold_titles

    def expected_sources_sentence_level(self) -> list[str]:
        """Texts of the exact supporting-fact sentences (one per fact).

        Stronger signal than titles but less compatible with SME's
        substring-on-filename matcher. Use against an adapter that returns
        chunked text rather than whole-paragraph matches.
        """
        by_title: dict[str, list[int]] = {}
        for title, sid in self.supporting_facts:
            by_title.setdefault(title, []).append(sid)
        out: list[str] = []
        for p in self.paragraphs:
            if p.title in by_title:
                out.extend(p.supporting_sentence_texts(by_title[p.title]))
        return out

    def to_sme_question(self) -> dict:
        """Render as a single SME ``questions.yaml`` entry."""
        return {
            "id": self.question_id,
            "text": self.question,
            "expected_sources": self.expected_sources_paragraph_level(),
            "gold_answer": self.gold_answer,
            "min_hops": self.min_hops,
            "hotpotqa": {
                "type": self.qtype,
                "type_name": self.type_name,
                "level": self.level,
                "supporting_facts": [
                    [title, sid] for title, sid in self.supporting_facts
                ],
                "gold_titles": self.gold_titles,
            },
            "sme_category": self.sme_category,
        }


def _parse_record(record: dict) -> HotpotQuestion:
    qtype = record.get("type", "")
    level = record.get("level", "")

    # supporting_facts: [[title, sent_id], ...] — coerce to (str, int) tuples,
    # skipping malformed entries rather than raising.
    raw_sf = record.get("supporting_facts", []) or []
    supporting_facts: list[tuple[str, int]] = []
    gold_titles: set[str] = set()
    for entry in raw_sf:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], int)
        ):
            supporting_facts.append((entry[0], entry[1]))
            gold_titles.add(entry[0])

    # context: [[title, [sentence, ...]], ...] — one paragraph per entry. A
    # paragraph is gold iff its title carries a supporting fact.
    raw_ctx = record.get("context", []) or []
    paragraphs: list[HotpotParagraph] = []
    for entry in raw_ctx:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            continue
        title, sentences = entry[0], entry[1]
        if not isinstance(title, str):
            continue
        sents = [s for s in sentences if isinstance(s, str)] if isinstance(
            sentences, (list, tuple)
        ) else []
        paragraphs.append(
            HotpotParagraph(
                title=title,
                sentences=sents,
                is_gold=title in gold_titles,
            )
        )

    return HotpotQuestion(
        question_id=record["_id"],
        question=record.get("question", ""),
        answer=str(record.get("answer", "")),
        qtype=qtype,
        type_name=HOTPOT_TYPE_NAMES.get(qtype, qtype or "unknown"),
        level=level,
        min_hops=HOTPOT_MIN_HOPS,
        supporting_facts=supporting_facts,
        paragraphs=paragraphs,
        sme_category=HOTPOT_SME_CATEGORY,
    )


def load_questions(path: Path | str) -> Iterator[HotpotQuestion]:
    """Parse a HotpotQA JSON file and yield ``HotpotQuestion`` records.

    Args:
        path: Path to ``hotpot_dev_distractor_v1.json`` (the pinned subset;
            see module-level SUBSET constants) or any HotpotQA-shaped file
            (the train split is also accepted). The file is a top-level JSON
            array of question records; this generator yields one record per
            question with its per-question context haystack attached.

    Yields:
        One ``HotpotQuestion`` per top-level record.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected top-level JSON array of question records, got "
            f"{type(raw).__name__}"
        )
    for record in raw:
        yield _parse_record(record)


def materialize_sme_corpus(
    questions: Iterator[HotpotQuestion] | list[HotpotQuestion],
    output_dir: Path | str,
    *,
    max_questions: int | None = None,
    gold_only: bool = False,
) -> dict:
    """Write a per-question SME-shape corpus to disk.

    Each HotpotQA question owns its own haystack, so the context paragraphs
    are written under ``<output_dir>/vault/<question_id>/<title>.md`` — one
    vault per question, mirroring LongMemEval's per-question scoping. All of
    a question's paragraphs go in that question's vault; ingest the vault,
    then query that single question.

    ``<output_dir>/questions.yaml`` is written as a single index covering all
    materialized questions, each carrying its ``question_id`` so the caller
    knows which vault to query.

    Args:
        questions: iterable of HotpotQuestion records (typically from
            load_questions).
        output_dir: where to write ``vault/`` and ``questions.yaml``.
        max_questions: cap on how many questions to materialize.
        gold_only: when True, write only the gold (supporting-fact)
            paragraphs to each vault, dropping the distractors. Useful for an
            oracle-retrieval upper bound; defaults to False (the full
            distractor haystack, the standard setting).

    Returns:
        A summary dict with counts and the materialized question ids.
    """
    output_dir = Path(output_dir)
    vault_dir = output_dir / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    questions_yaml: list[dict] = []
    n = 0
    paragraphs_written = 0
    for q in questions:
        if max_questions is not None and n >= max_questions:
            break
        n += 1
        q_dir = vault_dir / _safe_dirname(q.question_id)
        q_dir.mkdir(exist_ok=True)
        for p in q.paragraphs:
            if gold_only and not p.is_gold:
                continue
            note_path = q_dir / f"{_safe_filename(p.title)}.md"
            note_path.write_text(_render_paragraph_md(q, p))
            paragraphs_written += 1
        questions_yaml.append(q.to_sme_question())

    # Defer import so the loader module itself doesn't pull pyyaml unless
    # materialization is actually requested.
    import yaml

    (output_dir / "questions.yaml").write_text(
        yaml.safe_dump(
            {
                "version": f"hotpotqa-{SUBSET}-v1",
                "source": "https://github.com/hotpotqa/hotpot",
                "license": "CC BY-SA 4.0",
                "subset": SUBSET,
                "setting": SETTING,
                "gold_only": gold_only,
                "questions": questions_yaml,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )

    return {
        "output_dir": str(output_dir),
        "questions_count": n,
        "paragraphs_written": paragraphs_written,
        "question_ids": [q["id"] for q in questions_yaml],
        "vault_dir": str(vault_dir),
        "gold_only": gold_only,
    }


def _safe_dirname(name: str) -> str:
    """Make a question id safe as a directory name (ids are hex, so this is
    a light guard rather than a real sanitizer)."""
    return name.replace("/", "_").replace("\\", "_") or "q"


def _safe_filename(title: str) -> str:
    """Make a Wikipedia title safe as a filename. Titles can contain slashes,
    colons, and other path-hostile characters; collapse them to underscores
    and trim length so the path stays valid."""
    safe = "".join(c if c.isalnum() or c in " -_.,()" else "_" for c in title)
    safe = safe.strip().replace(" ", "_") or "untitled"
    return safe[:120]


def _render_paragraph_md(q: HotpotQuestion, p: HotpotParagraph) -> str:
    """Render one context paragraph for embedding (sme-rich rendering).

    YAML frontmatter (question_id, title, gold flag, source), a ``# <title>``
    header, then the paragraph body as one sentence per line so a chunker
    that splits on lines preserves the sentence boundaries the supporting
    facts are annotated against.
    """
    lines = [
        "---",
        f"question_id: {q.question_id}",
        f"title: {p.title!r}",
        f"is_gold: {str(p.is_gold).lower()}",
        "source: hotpotqa",
        "---",
        "",
        f"# {p.title}",
        "",
    ]
    lines.extend(p.sentences)
    lines.append("")
    return "\n".join(lines)
