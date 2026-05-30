"""Load BEAM JSON records into SME-shape Python objects.

BEAM ("Beyond a Million Tokens: Benchmarking and Enhancing Long-Term
Memory in LLMs"; ICLR 2026; arXiv 2510.27246) is a long-term memory
benchmark of very long single-user/assistant conversations (100K–10M
tokens each) plus a 10-ability probing-question suite. Unlike LoCoMo /
LongMemEval, BEAM is an end-to-end QA pass-rate benchmark scored with
rubric *nuggets* (0 / 0.5 / 1.0 per nugget), and it is graded at
context "buckets" (100K, 500K, 1M, 10M tokens) — the same conversation
truncated to different lengths exposes whether a memory system holds up
as the haystack grows. This loader consumes the released benchmark and
produces SME-shape question records so SME's category readings can be
cross-validated against the BEAM QA leaderboard (Mem0 reports 64.1 /
48.6 at 1M / 10M).

Schema reference: the HuggingFace dataset ``Mohammadta/BEAM`` (splits
``100K`` / ``500K`` / ``1M``) and ``Mohammadta/BEAM-10M`` (split
``10M``), and the upstream runner ``benchmarks/beam/run.py`` in
https://github.com/mem0ai/memory-benchmarks. Each dataset row is one
*conversation* with the shape::

    {
      "conversation_id": "1",
      "conversation_seed": {"category": str, "id": int,
                            "subtopics": [str], "theme": str,
                            "title": str},
      "narratives": str,
      "user_profile": {"user_info": str, "user_relationships": str},
      "conversation_plan": str,
      "user_questions": [...],
      "chat": [                       # list of sessions (batches)
        [ {"content": str, "id": int, "index": str,
           "question_type": str, "role": "user"|"assistant",
           "time_anchor": "March-15-2024"} , ... ],
        ...
      ],
      "probing_questions": str        # a JSON/repr STRING — see below
    }

``probing_questions`` is stored as a *string* (Python-repr or JSON) of
a dict keyed by the 10 ability types. Each value is a list of question
dicts. The question dicts vary slightly by type but share these
load-bearing fields::

    {"question": str,
     "answer": str,               # gold answer; ABSTENTION items use
                                  #   "ideal_response" instead
     "rubric": [str, ...],        # nuggets — the gold-answer key facts
     "source_chat_ids": [int]}    # chat turn `id`s carrying the evidence
                                  #   (absent on abstention items)

This loader returns a stream of ``BEAMQuestion`` dataclasses. It does
not write to disk by default; the helper ``materialize_sme_corpus``
writes per-conversation vault directories for adapters that ingest a
corpus ahead of querying.


PINNED SUBSET (the comparability contract)
-------------------------------------------
BEAM cross-comparisons are unreliable unless the **bucket** is pinned —
the same conversation at 100K vs 10M tokens is a different retrieval
problem, and a published number means nothing without its bucket. This
loader records the bucket on every record and in the materialized
``questions.yaml``. The released split sizes (HF dataset card,
downloaded 2026-05-29):

    BUCKET   examples   questions (20/conv)
    100K     20         400
    500K     35         700
    1M       35         700
    10M      (separate HF dataset Mohammadta/BEAM-10M)

Any reading published from this loader MUST state the bucket and the
conversation count. BEAM ships 2 questions per ability type = 20
questions per conversation; ``QUESTIONS_PER_CONVERSATION = 20`` and
``ABILITY_TYPES`` pin that contract.

Verified against the ``Mohammadta/BEAM`` 100K split (datasets-server
row 0, downloaded 2026-05-29): conversation_id "1", 3 sessions / 188
turns, ``probing_questions`` a 24KB repr-string parsing to a 10-key
dict of 20 total questions. Schema matches.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# --- Pinned subset (the comparability contract; see module docstring) ------
# Token buckets BEAM grades at. The same conversation truncated to each
# length is a distinct retrieval problem, so a reading is meaningless
# without its bucket.
VALID_BUCKETS = ("100K", "500K", "1M", "10M")
QUESTIONS_PER_CONVERSATION = 20  # 10 ability types x 2 questions each

# The 10 memory-ability types BEAM probes, keyed exactly as they appear
# as the top-level keys of the (string-encoded) `probing_questions`
# dict. Verbatim from benchmarks/beam/prompts.py:BEAM_QUESTION_TYPES.
ABILITY_TYPES = {
    "abstention": "Withholding answers when evidence is absent from the conversation",
    "contradiction_resolution": "Detecting and reconciling inconsistent statements across dialogue turns",
    "event_ordering": "Reconstructing the chronological sequence of events and developments",
    "information_extraction": "Recalling specific entities, dates, numbers, and factual details",
    "instruction_following": "Sustained adherence to user-specified constraints and formatting preferences",
    "knowledge_update": "Revising stored facts when new or corrected information appears",
    "multi_session_reasoning": "Integrating evidence scattered across non-adjacent dialogue segments",
    "preference_following": "Adapting responses to evolving user preferences and personal choices",
    "summarization": "Abstracting and compressing dialogue content into concise summaries",
    "temporal_reasoning": "Reasoning about explicit and implicit time relations, durations, and sequences",
}

# Mapping from BEAM ability types to SME categories, with caveats baked
# in. Mirrors LME_TO_SME_CATEGORY (longmemeval) and
# LOCOMO_CATEGORY_TO_SME (locomo).
#
# - information_extraction -> Cat 1 (factual retrieval): exact primitive
#   match (recall a specific entity/date/number).
# - multi_session_reasoning -> Cat 2c (multi-hop): integrating evidence
#   across non-adjacent segments. Partial — BEAM doesn't break out hop
#   depth (1/2/3) the way SME Cat 2c does.
# - knowledge_update -> Cat 3 (contradiction surfacing): PARTIAL, with
#   the same divergence as LongMemEval KU — BEAM rewards returning the
#   *revised* value, whereas SME Cat 3 rewards *flagging* old vs new. A
#   silent-overwriter scores better on KU than a contradiction-surfacing
#   system. See docs/related_work/longmemeval.md for the full analysis.
# - contradiction_resolution -> Cat 3 (contradiction surfacing): the
#   closer match to Cat 3 than KU — BEAM explicitly tests detecting AND
#   reconciling inconsistent statements. Still partial (BEAM grades the
#   reconciled answer, not the surfacing of both sides).
# - temporal_reasoning -> Cat 6 (temporal reasoning): strong match on
#   time-point/duration queries; BEAM does not test Cat 6b provenance.
# - event_ordering -> Cat 6 (temporal reasoning): chronological
#   reconstruction is a temporal primitive; scored with Kendall tau-b
#   upstream, not a substring match.
# - abstention -> Cat 1 negative class: the system must ABSTAIN ("no
#   information available"), not retrieve. Mirrors LongMemEval `_abs`
#   and LoCoMo's adversarial category. Gold is a refusal.
# - preference_following / instruction_following / summarization ->
#   unmapped: no SME category tests these primitives (they grade
#   generation behavior — adherence, compression — rather than the
#   structural retrieval SME diagnoses).
BEAM_ABILITY_TO_SME = {
    "information_extraction": "cat_1",
    "multi_session_reasoning": "cat_2c",
    "knowledge_update": "cat_3_partial",      # see KU divergence caveat
    "contradiction_resolution": "cat_3_partial",
    "temporal_reasoning": "cat_6",
    "event_ordering": "cat_6",
    "abstention": "cat_1_negative",           # abstain, don't retrieve
    "preference_following": "unmapped",
    "instruction_following": "unmapped",
    "summarization": "unmapped",
}


@dataclass(frozen=True)
class BEAMTurn:
    """One turn within a session (batch) of a BEAM conversation.

    `turn_id` is the integer `id` field — it is the join key referenced
    by a probing question's `source_chat_ids`, so it is how a question's
    evidence resolves back to its source turns.
    """

    role: str  # "user" or "assistant"
    content: str
    turn_id: int
    index: str = ""           # upstream "<conv>,<turn>" locator, e.g. "1,1"
    question_type: str = ""   # upstream turn-level tag (e.g. "main_question")
    time_anchor: str = ""     # raw upstream format, e.g. "March-15-2024"


@dataclass(frozen=True)
class BEAMSession:
    """One session (batch) of turns within a BEAM conversation.

    BEAM's `chat` is a list of sessions; `session_index` is the 0-based
    position in that list. It is the natural per-session scoping unit
    for rendering one markdown note per session.
    """

    session_index: int
    turns: list[BEAMTurn] = field(default_factory=list)

    @property
    def session_id(self) -> str:
        """`S<N>` — stable id used as the materialized note filename."""
        return f"S{self.session_index}"

    @property
    def turn_ids(self) -> set[int]:
        return {t.turn_id for t in self.turns}

    def evidence_turn_texts(self, source_chat_ids: list[int]) -> list[str]:
        """Texts of turns whose `turn_id` appears in `source_chat_ids`."""
        wanted = set(source_chat_ids)
        return [t.content for t in self.turns if t.turn_id in wanted]


@dataclass(frozen=True)
class BEAMQuestion:
    """One BEAM evaluation instance, parsed from a conversation's
    ``probing_questions``.

    Like LoCoMo (and unlike LongMemEval), BEAM shares one conversation
    across all of a conversation's questions. Each `BEAMQuestion`
    therefore carries a reference to its conversation's sessions (the
    shared haystack) plus the `conversation_id` so callers can ingest
    the conversation once per conversation rather than once per question.

    `bucket` is the token bucket this conversation was loaded from — it
    is part of the comparability contract (a BEAM number is meaningless
    without its bucket).
    """

    question_id: str         # synthesized "<bucket>::<conv_id>::q<index>"
    conversation_id: str     # the conversation this question belongs to
    bucket: str              # "100K" | "500K" | "1M" | "10M"
    ability_type: str        # the outer probing_questions key (ABILITY_TYPES)
    question_subtype: str    # the item's own `question_type` (e.g. "context_date/time")
    question: str
    answer: str              # gold answer ("" for pure abstention items)
    ideal_response: str      # abstention items' ideal refusal ("" otherwise)
    is_abstention: bool
    rubric: list[str]        # nugget descriptions — the gold key facts
    source_chat_ids: list[int]  # chat turn ids carrying the evidence
    category: str            # conversation_seed category (e.g. "Coding")
    sessions: list[BEAMSession]  # shared haystack for this conversation
    sme_category: str        # derived per BEAM_ABILITY_TO_SME

    @property
    def gold_answer(self) -> str:
        """The correct answer for QA scoring.

        For abstention items the correct behavior is refusal, so the
        gold answer is the ideal refusal text; the `rubric` carries the
        nuggets a judge scores against.
        """
        return self.ideal_response if self.is_abstention else self.answer

    @property
    def ground_truth_nuggets(self) -> str:
        """Rubric nuggets joined by ' | ' — the upstream runner's
        ``ground_truth_answer`` (benchmarks/beam/run.py)."""
        return " | ".join(self.rubric)

    def expected_sources_session_level(self) -> list[str]:
        """`S<N>` session ids whose turns carry the answer evidence —
        the SME `expected_sources` substring matcher's natural target.

        Resolved by mapping each `source_chat_ids` turn id to the
        session that contains it. Order-preserving over sessions.
        Abstention items have no source chat ids (the evidence is
        absent by construction) and return []."""
        wanted = set(self.source_chat_ids)
        out: list[str] = []
        for s in self.sessions:
            if s.turn_ids & wanted and s.session_id not in out:
                out.append(s.session_id)
        return out

    def expected_sources_turn_level(self) -> list[str]:
        """Texts of the exact evidence turns (one per source_chat_id).

        Stronger signal than session ids but less compatible with SME's
        substring-on-filename matcher. Use when running against an
        adapter that returns chunked text rather than whole-file matches.
        """
        out: list[str] = []
        for s in self.sessions:
            out.extend(s.evidence_turn_texts(self.source_chat_ids))
        return out

    def to_sme_question(self) -> dict:
        """Render as a single SME `questions.yaml` entry."""
        return {
            "id": self.question_id,
            "text": self.question,
            "expected_sources": self.expected_sources_session_level(),
            "gold_answer": self.gold_answer,
            "beam": {
                "conversation_id": self.conversation_id,
                "bucket": self.bucket,
                "ability_type": self.ability_type,
                "question_subtype": self.question_subtype,
                "is_abstention": self.is_abstention,
                "rubric": list(self.rubric),
                "ground_truth_nuggets": self.ground_truth_nuggets,
                "source_chat_ids": list(self.source_chat_ids),
                "category": self.category,
            },
            "sme_category": self.sme_category,
        }


def _coerce_probing_questions(raw: Any) -> dict:
    """Parse BEAM's `probing_questions` field into a dict.

    Upstream stores it as a *string* (Python-repr first, JSON second) of
    a dict keyed by ability type. ``benchmarks/beam/run.py`` tries
    ``ast.literal_eval`` then ``json.loads``; we mirror that exactly so
    we accept the same range of releases. An already-parsed dict (e.g.
    from a local fixture) passes through unchanged.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_chat(chat: Any) -> list[BEAMSession]:
    """Parse BEAM's `chat` into a list of sessions.

    The released 100K–1M splits store `chat` as a 2D list (list of
    sessions, each a list of turn dicts). We parse that canonical shape;
    a flat list of turn dicts is treated as a single session so a
    minimal fixture need not nest.
    """
    if not isinstance(chat, list) or not chat:
        return []

    # Flat list of turn dicts -> wrap as a single session.
    if isinstance(chat[0], dict):
        chat = [chat]

    sessions: list[BEAMSession] = []
    for s_idx, raw_session in enumerate(chat):
        if not isinstance(raw_session, list):
            continue
        turns: list[BEAMTurn] = []
        for t in raw_session:
            if not isinstance(t, dict):
                continue
            turns.append(
                BEAMTurn(
                    role=t.get("role", "user"),
                    content=t.get("content", ""),
                    turn_id=int(t["id"]) if t.get("id") is not None else -1,
                    index=str(t.get("index", "")),
                    question_type=str(t.get("question_type", "")),
                    time_anchor=str(t.get("time_anchor", "")),
                )
            )
        sessions.append(BEAMSession(session_index=s_idx, turns=turns))
    return sessions


def _parse_conversation(conv: dict, bucket: str) -> Iterator[BEAMQuestion]:
    conv_id = str(conv.get("conversation_id", ""))
    seed = conv.get("conversation_seed", {})
    category = seed.get("category", "unknown") if isinstance(seed, dict) else "unknown"
    sessions = _parse_chat(conv.get("chat", []))
    pq = _coerce_probing_questions(conv.get("probing_questions", {}))

    q_index = 0
    # Iterate ability types in the pinned order so question ids are
    # stable across runs regardless of dict insertion order.
    for ability in ABILITY_TYPES:
        items = pq.get(ability, [])
        if not isinstance(items, list):
            # A few releases store a single dict instead of a list.
            items = [items] if isinstance(items, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            is_abs = ability == "abstention"
            raw_answer = item.get("answer", "")
            answer = "" if raw_answer in (None, "") else str(raw_answer)
            ideal = str(item.get("ideal_response", "")) if is_abs else ""
            rubric_raw = item.get("rubric", [])
            if isinstance(rubric_raw, list):
                rubric = [str(n) for n in rubric_raw]
            elif rubric_raw:
                rubric = [str(rubric_raw)]
            else:
                rubric = []
            source_ids = [
                int(c) for c in item.get("source_chat_ids", [])
                if isinstance(c, (int, float)) or (isinstance(c, str) and c.isdigit())
            ]
            yield BEAMQuestion(
                question_id=f"{bucket}::{conv_id}::q{q_index}",
                conversation_id=conv_id,
                bucket=bucket,
                ability_type=ability,
                question_subtype=str(item.get("question_type", "")),
                question=str(item.get("question", "")),
                answer=answer,
                ideal_response=ideal,
                is_abstention=is_abs,
                rubric=rubric,
                source_chat_ids=source_ids,
                category=str(category),
                sessions=sessions,
                sme_category=BEAM_ABILITY_TO_SME.get(ability, "unmapped"),
            )
            q_index += 1


def load_questions(path: Path | str, *, bucket: str) -> Iterator[BEAMQuestion]:
    """Parse a BEAM JSON file and yield `BEAMQuestion` records.

    Args:
        path: Path to a cached BEAM split JSON — a top-level JSON array
            of conversation dicts, as written by
            ``benchmarks/beam/run.py``'s ``download_dataset`` (one file
            per bucket, e.g. ``beam_100K.json``). See README.md for how
            to produce these from the HuggingFace dataset.
        bucket: the token bucket this file represents (one of
            VALID_BUCKETS). Required and recorded on every record — a
            BEAM reading is meaningless without its bucket.

    Yields:
        One `BEAMQuestion` per (conversation, probing-question) pair.
        Sessions are parsed once per conversation and shared by
        reference across that conversation's questions, so the file
        materializes without re-parsing the chat per question.
    """
    if bucket not in VALID_BUCKETS:
        raise ValueError(f"bucket must be one of {VALID_BUCKETS}; got {bucket!r}")
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected top-level JSON array of conversations, got "
            f"{type(raw).__name__}"
        )
    for conv in raw:
        yield from _parse_conversation(conv, bucket)


def materialize_sme_corpus(
    questions: Iterator[BEAMQuestion] | list[BEAMQuestion],
    output_dir: Path | str,
    *,
    max_questions: int | None = None,
) -> dict:
    """Write a per-conversation SME-shape corpus to disk.

    Like LoCoMo (and unlike LongMemEval's per-question vaults), BEAM
    shares one conversation across all of a conversation's questions, so
    the haystack is written ONCE per conversation under
    ``<output_dir>/vault/<conversation_id>/<session_id>.md``. All of that
    conversation's questions query the same vault: ingest the
    conversation's vault once, then iterate its questions.

    ``<output_dir>/questions.yaml`` is written as a single index covering
    all materialized questions, each carrying its `conversation_id` and
    `bucket` so the caller knows which vault to query and at what scale.

    Args:
        questions: iterable of BEAMQuestion records (from load_questions).
        output_dir: where to write `vault/` and `questions.yaml`.
        max_questions: cap on how many QA records to materialize. The cap
            is on questions, not conversations — a conversation's vault is
            written the first time any of its questions is seen.

    Returns:
        A summary dict with counts, the materialized conversation ids,
        and the bucket(s) covered.
    """
    output_dir = Path(output_dir)
    vault_dir = output_dir / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    questions_yaml: list[dict] = []
    written_convs: set[str] = set()
    buckets_seen: set[str] = set()
    n = 0
    for q in questions:
        if max_questions is not None and n >= max_questions:
            break
        n += 1
        buckets_seen.add(q.bucket)
        if q.conversation_id not in written_convs:
            written_convs.add(q.conversation_id)
            conv_dir = vault_dir / q.conversation_id
            conv_dir.mkdir(exist_ok=True)
            for s in q.sessions:
                note_path = conv_dir / f"{s.session_id}.md"
                note_path.write_text(_render_session_md(q, s))
        questions_yaml.append(q.to_sme_question())

    # Defer import so the loader module itself doesn't pull pyyaml unless
    # materialization is actually requested.
    import yaml

    (output_dir / "questions.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "beam-v1",
                "source": "https://github.com/mem0ai/memory-benchmarks",
                "dataset": "https://huggingface.co/datasets/Mohammadta/BEAM",
                "license": "CC-BY-SA-4.0 (arXiv 2510.27246, ICLR 2026)",
                "buckets": sorted(buckets_seen),
                "questions_per_conversation": QUESTIONS_PER_CONVERSATION,
                "questions": questions_yaml,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )

    return {
        "output_dir": str(output_dir),
        "questions_count": n,
        "conversations_count": len(written_convs),
        "conversation_ids": sorted(written_convs),
        "buckets": sorted(buckets_seen),
        "vault_dir": str(vault_dir),
    }


def _render_session_md(q: BEAMQuestion, s: BEAMSession) -> str:
    """Render one session for embedding (sme-rich rendering).

    YAML frontmatter (conversation_id, session_id, bucket, category,
    source), a `# Session <id>` header, then each turn as a
    `## <role>` block. The turn's `time_anchor` is folded into the body
    as an italicized date line so a text-only retriever sees the
    temporal signal, and the turn id is preserved in an HTML comment so
    a turn-level evaluator can recover the `source_chat_ids` join.
    """
    lines = [
        "---",
        f"conversation_id: {q.conversation_id}",
        f"session_id: {s.session_id}",
        f"session_index: {s.session_index}",
        f"bucket: {q.bucket}",
        f"category: {q.category}",
        "source: beam",
        "---",
        "",
        f"# Session {s.session_id}",
        "",
    ]
    for t in s.turns:
        lines.append(f"## {t.role}")
        lines.append("")
        if t.time_anchor:
            lines.append(f"_Time anchor: {t.time_anchor}_")
            lines.append("")
        lines.append(t.content)
        lines.append(f"<!-- turn_id: {t.turn_id} -->")
        lines.append("")
    return "\n".join(lines)
