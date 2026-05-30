"""Load LoCoMo JSON records into SME-shape Python objects.

LoCoMo ("Long Conversational Memory"; Maharana et al., ACL 2024;
arXiv 2402.17753) is a benchmark of very long-term, multi-session
two-persona dialogues plus a question-answering suite over them. This
loader consumes the released benchmark JSON and produces SME-shape
question records so SME's category readings can be cross-validated
against the LoCoMo QA leaderboard (EverOS, True Memory, Mem0,
Hindsight all report LoCoMo QA accuracy).

Schema reference: the released ``data/locomo10.json`` in
https://github.com/snap-research/locomo. Each top-level element is one
*sample* (one conversation) with the shape::

    {
      "sample_id": "conv-26",
      "conversation": {
        "speaker_a": str,
        "speaker_b": str,
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
          {"speaker": str, "dia_id": "D1:1", "text": str,
           # optional multimodal fields:
           "img_url": [str], "blip_caption": str, "query": str}
        ],
        "session_2_date_time": ..., "session_2": [...], ...
      },
      "qa": [
        {"question": str, "answer": str|int, "evidence": ["D1:3"],
         "category": int}                       # categories 1-4
        | {"question": str, "adversarial_answer": str,
           "evidence": [...], "category": 5}     # adversarial
      ],
      "event_summary": ..., "observation": ..., "session_summary": ...
    }

This loader returns a stream of ``LoCoMoQuestion`` dataclasses. It does
not write to disk by default; the helper ``materialize_sme_corpus``
writes per-sample vault directories for adapters that ingest a corpus
ahead of querying.


PINNED SUBSET (the comparability contract)
-------------------------------------------
LoCoMo cross-comparisons are unreliable unless the exact question
subset and adversarial inclusion are pinned (docs/research/
2026-05-29-comparison-readiness.md §1.3, §3.4). This loader pins:

    SUBSET           = "locomo10"   # the canonical released benchmark
    SUBSET_QA_COUNT  = 1986         # all QA across all 10 conversations
    ADVERSARIAL_INCLUDED = True     # category-5 items are loaded, flagged

i.e. the **full LoCoMo-10 release, adversarial category included**. Any
reading published from this loader MUST state these three facts. The
1986 count is the sum over the 10 conversations (199+105+193+260+242+
158+190+239+196+204) of ``data/locomo10.json``; it is NOT the paper's
7,512-QA figure (that counts the larger 50-conversation construction
set, which was never publicly released).

Verified against ``locomo10.json`` (raw.githubusercontent.com/
snap-research/locomo/main/data/locomo10.json, downloaded 2026-05-29) —
10 samples, 1986 QA, schema matches.


CATEGORY NUMBERING CAVEAT (read before trusting question_type)
--------------------------------------------------------------
LoCoMo has a notorious numbering discrepancy between its *paper prose*
and its *released JSON*:

  - The paper (arXiv 2402.17753) lists the five categories in prose as
    (1) single-hop, (2) multi-hop, (3) temporal, (4) open-domain,
    (5) adversarial.
  - The official scorer ``task_eval/evaluation.py`` — which consumes
    the released ``locomo10.json`` — uses a DIFFERENT integer mapping:
    category **1 = multi-hop** (it is the branch that splits the answer
    into sub-answers and computes partial F1), categories **2/3/4 =
    single-hop / temporal / open-domain** (grouped, scored identically),
    and **5 = adversarial**.

The released JSON labels its data with the *scorer's* numbering, not
the prose numbering. We verified this empirically on locomo10.json:
category 1 carries the highest mean evidence-reference count (3.13
refs/question vs 1.0-2.1 for cats 2/3/4) — the signature of
multi-session synthesis, i.e. multi-hop. We therefore pin to the
**scorer numbering** (``LOCOMO_CATEGORY_NAMES`` below). Anyone
comparing against a system that used the prose numbering for cats 1-4
must reconcile this first.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# --- Pinned subset (the comparability contract; see module docstring) ------
SUBSET = "locomo10"
SUBSET_QA_COUNT = 1986
ADVERSARIAL_INCLUDED = True
SUBSET_SAMPLE_COUNT = 10

# Integer-category -> human name, per the official scorer
# (task_eval/evaluation.py), which is the numbering the released
# locomo10.json actually uses. See the "CATEGORY NUMBERING CAVEAT" in
# the module docstring — this is NOT the paper-prose order.
LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "single-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}

# Mapping from LoCoMo categories to SME categories, with the
# adversarial caveat baked in. Mirrors LME_TO_SME_CATEGORY in the
# longmemeval loader.
#
# - single-hop  -> Cat 1 (factual retrieval): exact primitive match.
# - multi-hop   -> Cat 2c (multi-hop by depth): partial — LoCoMo does
#   not break out hop depth (1/2/3) the way SME Cat 2c does.
# - temporal    -> Cat 6 (temporal reasoning): strong match on the
#   time-point side; LoCoMo does not test Cat 6b provenance chains.
# - open-domain -> unmapped: requires fusing dialogue with external
#   world knowledge; no SME category tests this primitive.
# - adversarial -> Cat 1 negative class: the system is supposed to
#   ABSTAIN ("no information available"), not retrieve. The `adversarial_
#   answer` field is the *wrong* answer the question is engineered to
#   bait; a correct system refuses it. This mirrors LongMemEval's `_abs`
#   handling (cat_1_negative).
LOCOMO_CATEGORY_TO_SME = {
    1: "cat_2c",          # multi-hop
    2: "cat_1",           # single-hop
    3: "cat_6",           # temporal
    4: "unmapped",        # open-domain knowledge — no SME analogue
    5: "cat_1_negative",  # adversarial — abstain, don't retrieve
}

# dia_id format: "D<session>:<turn>", e.g. "D1:3" -> session 1, turn 3.
_DIA_ID_RE = re.compile(r"^D(\d+):(\d+)$")
_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


@dataclass(frozen=True)
class LoCoMoTurn:
    """One turn within a session.

    `img_url` / `blip_caption` carry LoCoMo's multimodal payload (a
    shared image plus its BLIP caption). They are preserved so a
    multimodal adapter can use them, but the default text rendering
    folds the caption into the turn body so text-only retrieval still
    sees the image's semantic content.
    """

    speaker: str
    text: str
    dia_id: str
    img_url: list[str] = field(default_factory=list)
    blip_caption: str = ""

    @property
    def has_image(self) -> bool:
        return bool(self.img_url) or bool(self.blip_caption)


@dataclass(frozen=True)
class LoCoMoSession:
    """One dialogue session in a conversation.

    `session_index` is the integer N from `session_N` — it is also the
    `D<N>` prefix used in QA `evidence` references, so it is the natural
    join key between a question's evidence and its source session.
    """

    session_index: int
    date: str  # raw upstream format, e.g. "1:56 pm on 8 May, 2023"
    turns: list[LoCoMoTurn] = field(default_factory=list)

    @property
    def session_id(self) -> str:
        """`D<N>` — matches the dia_id session prefix used in evidence."""
        return f"D{self.session_index}"

    def is_evidence(self, evidence_dia_ids: list[str]) -> bool:
        return self.session_id in _evidence_session_ids(evidence_dia_ids)

    def evidence_turn_texts(self, evidence_dia_ids: list[str]) -> list[str]:
        """Texts of turns whose dia_id appears in `evidence_dia_ids`."""
        wanted = set(evidence_dia_ids)
        return [t.text for t in self.turns if t.dia_id in wanted]


@dataclass(frozen=True)
class LoCoMoQuestion:
    """One LoCoMo evaluation instance, parsed from a sample's `qa` list.

    Unlike LongMemEval (each question owns its haystack), LoCoMo shares
    one conversation across all of a sample's questions. Each
    `LoCoMoQuestion` therefore carries a reference to its sample's
    sessions (the shared haystack) plus the `sample_id` so callers can
    ingest the conversation once per sample rather than once per
    question.
    """

    question_id: str        # synthesized "<sample_id>::q<index>"
    sample_id: str          # the conversation this question belongs to
    category: int           # raw LoCoMo integer category (1-5)
    question_type: str      # LOCOMO_CATEGORY_NAMES[category]
    question: str
    answer: str             # gold answer ("" for pure adversarial items)
    adversarial_answer: str  # the baited wrong answer ("" if non-adversarial)
    is_adversarial: bool
    evidence: list[str]     # raw dia_id refs, e.g. ["D1:3"]
    speaker_a: str
    speaker_b: str
    sessions: list[LoCoMoSession]  # shared haystack for this sample
    sme_category: str       # derived per LOCOMO_CATEGORY_TO_SME

    @property
    def gold_answer(self) -> str:
        """The correct answer for QA scoring.

        For adversarial items the correct behavior is abstention, so the
        gold answer is empty/abstain — the `adversarial_answer` is the
        wrong answer the question baits, exposed separately.
        """
        return self.answer

    def expected_sources_session_level(self) -> list[str]:
        """`D<N>` session ids whose turns carry the answer evidence —
        the SME `expected_sources` substring matcher's natural target."""
        return _evidence_session_ids(self.evidence)

    def expected_sources_turn_level(self) -> list[str]:
        """Texts of the exact evidence turns (one entry per evidence dia_id).

        Stronger signal than session ids but less compatible with SME's
        substring-on-filename matcher. Use when running against an
        adapter that returns chunked text rather than whole-file matches.
        """
        out: list[str] = []
        for s in self.sessions:
            out.extend(s.evidence_turn_texts(self.evidence))
        return out

    def to_sme_question(self) -> dict:
        """Render as a single SME `questions.yaml` entry."""
        return {
            "id": self.question_id,
            "text": self.question,
            "expected_sources": self.expected_sources_session_level(),
            "gold_answer": self.gold_answer,
            "locomo": {
                "sample_id": self.sample_id,
                "category": self.category,
                "question_type": self.question_type,
                "is_adversarial": self.is_adversarial,
                "adversarial_answer": self.adversarial_answer,
                "evidence": list(self.evidence),
            },
            "sme_category": self.sme_category,
        }


def _evidence_session_ids(evidence_dia_ids: list[str]) -> list[str]:
    """Collapse a list of `D<N>:<turn>` dia_ids to the unique `D<N>`
    session ids they reference, order-preserving. Non-conforming
    evidence entries (a handful exist in locomo10.json) are skipped."""
    seen: list[str] = []
    for e in evidence_dia_ids:
        if not isinstance(e, str):
            continue
        m = _DIA_ID_RE.match(e)
        if not m:
            continue
        sid = f"D{m.group(1)}"
        if sid not in seen:
            seen.append(sid)
    return seen


def _parse_session(conv: dict, session_index: int) -> LoCoMoSession:
    raw_turns = conv[f"session_{session_index}"]
    date = conv.get(f"session_{session_index}_date_time", "")
    turns = [
        LoCoMoTurn(
            speaker=t.get("speaker", ""),
            text=t.get("text", ""),
            dia_id=t.get("dia_id", ""),
            img_url=list(t["img_url"]) if isinstance(t.get("img_url"), list) else [],
            blip_caption=t.get("blip_caption", ""),
        )
        for t in raw_turns
    ]
    return LoCoMoSession(session_index=session_index, date=date, turns=turns)


def _parse_sample(sample: dict) -> Iterator[LoCoMoQuestion]:
    sample_id = sample["sample_id"]
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "")
    speaker_b = conv.get("speaker_b", "")

    # Sessions in numeric order — session_N is the haystack, shared by
    # every question in this sample.
    session_indices = sorted(
        int(m.group(1))
        for k in conv
        if (m := _SESSION_KEY_RE.match(k))
    )
    sessions = [_parse_session(conv, i) for i in session_indices]

    for idx, qa in enumerate(sample["qa"]):
        category = qa["category"]
        is_adv = category == 5
        # Adversarial items use `adversarial_answer` (the baited wrong
        # answer); a few also carry a real `answer`. Non-adversarial
        # items always carry `answer` (str or int — coerce to str).
        raw_answer = qa.get("answer", "")
        answer = "" if raw_answer == "" else str(raw_answer)
        adv_answer = str(qa.get("adversarial_answer", "")) if is_adv else ""
        yield LoCoMoQuestion(
            question_id=f"{sample_id}::q{idx}",
            sample_id=sample_id,
            category=category,
            question_type=LOCOMO_CATEGORY_NAMES.get(category, "unknown"),
            question=qa["question"],
            answer=answer,
            adversarial_answer=adv_answer,
            is_adversarial=is_adv,
            evidence=[e for e in qa.get("evidence", [])],
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            sme_category=LOCOMO_CATEGORY_TO_SME.get(category, "unmapped"),
        )


def load_questions(path: Path | str) -> Iterator[LoCoMoQuestion]:
    """Parse a LoCoMo JSON file and yield `LoCoMoQuestion` records.

    Args:
        path: Path to ``locomo10.json`` (the pinned subset; see
            module-level SUBSET constants). The file is a top-level JSON
            array of *samples*; this generator flattens it to one record
            per QA item, attaching each question's shared conversation
            sessions.

    Yields:
        One `LoCoMoQuestion` per (sample, qa) pair. Sessions are parsed
        once per sample and shared by reference across that sample's
        questions, so the full file materializes to SUBSET_QA_COUNT
        records without re-parsing the conversation per question.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected top-level JSON array of samples, got "
            f"{type(raw).__name__}"
        )
    for sample in raw:
        yield from _parse_sample(sample)


def materialize_sme_corpus(
    questions: Iterator[LoCoMoQuestion] | list[LoCoMoQuestion],
    output_dir: Path | str,
    *,
    max_questions: int | None = None,
) -> dict:
    """Write a per-sample SME-shape corpus to disk.

    Unlike LongMemEval (per-question vaults — each question has its own
    haystack), LoCoMo shares one conversation across all of a sample's
    questions, so the haystack is written ONCE per sample under
    ``<output_dir>/vault/<sample_id>/<session_id>.md``. All of that
    sample's questions query the same vault. This is the per-sample
    scoping a LoCoMo run requires: ingest the sample's vault once, then
    iterate that sample's questions.

    ``<output_dir>/questions.yaml`` is written as a single index covering
    all materialized questions, each carrying its `sample_id` so the
    caller knows which vault to query.

    Args:
        questions: iterable of LoCoMoQuestion records (typically from
            load_questions).
        output_dir: where to write `vault/` and `questions.yaml`.
        max_questions: cap on how many QA records to materialize. The
            cap is on questions, not samples — a sample's vault is
            written the first time any of its questions is seen.

    Returns:
        A summary dict with counts and the materialized sample ids.
    """
    output_dir = Path(output_dir)
    vault_dir = output_dir / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    questions_yaml: list[dict] = []
    written_samples: set[str] = set()
    n = 0
    for q in questions:
        if max_questions is not None and n >= max_questions:
            break
        n += 1
        if q.sample_id not in written_samples:
            written_samples.add(q.sample_id)
            sample_dir = vault_dir / q.sample_id
            sample_dir.mkdir(exist_ok=True)
            for s in q.sessions:
                note_path = sample_dir / f"{s.session_id}.md"
                note_path.write_text(_render_session_md(q, s))
        questions_yaml.append(q.to_sme_question())

    # Defer import so the loader module itself doesn't pull pyyaml
    # unless materialization is actually requested.
    import yaml

    (output_dir / "questions.yaml").write_text(
        yaml.safe_dump(
            {
                "version": f"locomo-{SUBSET}-v1",
                "source": "https://github.com/snap-research/locomo",
                "license": "see snap-research/locomo LICENSE.txt",
                "subset": SUBSET,
                "subset_qa_count": SUBSET_QA_COUNT,
                "adversarial_included": ADVERSARIAL_INCLUDED,
                "questions": questions_yaml,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )

    return {
        "output_dir": str(output_dir),
        "questions_count": n,
        "samples_count": len(written_samples),
        "sample_ids": sorted(written_samples),
        "vault_dir": str(vault_dir),
    }


def _render_session_md(q: LoCoMoQuestion, s: LoCoMoSession) -> str:
    """Render one session for embedding (sme-rich rendering).

    YAML frontmatter (sample_id, session_id, date, speakers, source),
    a `# Session <id>` header, then each turn as a `## <speaker>` block.
    Multimodal turns fold their BLIP caption into the body as an
    italicized `[shared image: ...]` line so text-only retrieval sees
    the image's semantic content; the raw img_url is preserved in an
    HTML comment for a multimodal adapter to recover.
    """
    lines = [
        "---",
        f"sample_id: {q.sample_id}",
        f"session_id: {s.session_id}",
        f"session_index: {s.session_index}",
        f"date: {s.date!r}",
        f"speaker_a: {q.speaker_a}",
        f"speaker_b: {q.speaker_b}",
        "source: locomo",
        "---",
        "",
        f"# Session {s.session_id}",
        "",
        f"_Date: {s.date}_",
        "",
    ]
    for t in s.turns:
        lines.append(f"## {t.speaker}")
        lines.append("")
        lines.append(t.text)
        if t.blip_caption:
            lines.append("")
            lines.append(f"_[shared image: {t.blip_caption}]_")
        if t.img_url:
            lines.append(f"<!-- img_url: {', '.join(t.img_url)} -->")
        lines.append("")
    return "\n".join(lines)
