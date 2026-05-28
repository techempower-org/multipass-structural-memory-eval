"""Load LongMemEval JSON records into SME-shape Python objects.

Schema reference: §3 of arXiv 2410.10813 ("Dataset Format" in the
upstream README at https://github.com/xiaowu0162/LongMemEval). Each
record in the oracle / S / M JSON files has the same shape:

    {
      "question_id": str,
      "question_type": str,        # one of LME_QUESTION_TYPES (or *_abs)
      "question": str,
      "answer": str,
      "question_date": str,        # "YYYY/MM/DD (DDD) HH:MM"
      "haystack_session_ids": [str],
      "haystack_dates": [str],
      "haystack_sessions": [
        [{"role": "user"|"assistant",
          "content": str,
          "has_answer": bool}]
      ],
      "answer_session_ids": [str],
    }

This loader returns a stream of `LMEQuestion` dataclasses. It does not
write to disk by default; the helper `materialize_sme_corpus` writes
per-question vault directories for adapters that ingest a corpus
ahead of querying.

Verified against `longmemeval_oracle.json` (Hugging Face
xiaowu0162/longmemeval-cleaned, downloaded 2026-05-01) — 500 records,
schema matches.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Question types declared in §3 of the LongMemEval paper, with the
# `_abs` suffix denoting the abstention variant (verified against
# upstream README: "If `question_id` ends with `_abs`, then the
# question is an `abstention` question.")
LME_QUESTION_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
}

# Mapping from LongMemEval question types to SME categories, with the
# semantic-divergence caveat baked in for `knowledge-update`.
#
# Verified primary-source against arXiv 2410.10813 §3 (lines 359-368
# of the pdftotext extract) on 2026-05-01. The KU mapping is partial,
# not exact: KU rewards *returning the new value* after overwrite,
# whereas SME Cat 3 rewards *flagging both old and new*. A
# silent-overwriter scores better on KU than a contradiction-surfacing
# system. See docs/related_work/longmemeval.md for the full divergence
# analysis.
LME_TO_SME_CATEGORY = {
    "single-session-user": "cat_1",
    "single-session-assistant": "cat_1",
    "single-session-preference": "cat_1",
    "multi-session": "cat_2c",
    "knowledge-update": "cat_3_partial",  # see caveat above
    "temporal-reasoning": "cat_6",
    # `_abs` suffix is handled separately by treating the question as a
    # negative-class Cat 1 (system should abstain rather than retrieve)
    "abstention": "cat_1_negative",
}


@dataclass(frozen=True)
class LMETurn:
    """One turn within a haystack session."""

    role: str  # "user" or "assistant"
    content: str
    has_answer: bool = False  # turn-level evidence marker


@dataclass(frozen=True)
class LMESession:
    """One chat session in the haystack."""

    session_id: str
    date: str  # raw upstream format, e.g. "2023/04/10 (Mon) 23:07"
    turns: list[LMETurn] = field(default_factory=list)

    def is_evidence(self, answer_session_ids: list[str]) -> bool:
        return self.session_id in set(answer_session_ids)

    def evidence_turn_texts(self) -> list[str]:
        """Concat of all turn contents marked with has_answer=True."""
        return [t.content for t in self.turns if t.has_answer]


@dataclass(frozen=True)
class LMEQuestion:
    """One LongMemEval evaluation instance, parsed from JSON."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    is_abstention: bool
    haystack_sessions: list[LMESession]
    answer_session_ids: list[str]
    sme_category: str  # derived per LME_TO_SME_CATEGORY

    def expected_sources_session_level(self) -> list[str]:
        """Session ids whose content should appear in retrieval — the
        SME `expected_sources` substring matcher's natural target."""
        return list(self.answer_session_ids)

    def expected_sources_turn_level(self) -> list[str]:
        """Concat of evidence-turn contents, one entry per evidence turn.

        Stronger signal than session ids but less compatible with
        SME's substring-on-filename matcher. Use this when running
        against an adapter that returns chunked text rather than
        whole-file matches.
        """
        out: list[str] = []
        for s in self.haystack_sessions:
            if s.is_evidence(self.answer_session_ids):
                out.extend(s.evidence_turn_texts())
        return out

    def to_sme_question(self) -> dict:
        """Render as a single SME `questions.yaml` entry."""
        return {
            "id": self.question_id,
            "text": self.question,
            "expected_sources": self.expected_sources_session_level(),
            "gold_answer": self.answer,
            "longmemeval": {
                "question_type": self.question_type,
                "is_abstention": self.is_abstention,
                "question_date": self.question_date,
            },
            "sme_category": self.sme_category,
        }


def _parse_record(record: dict) -> LMEQuestion:
    qid = record["question_id"]
    is_abs = qid.endswith("_abs")
    qtype = record["question_type"]
    if is_abs:
        sme_cat = LME_TO_SME_CATEGORY["abstention"]
    else:
        sme_cat = LME_TO_SME_CATEGORY.get(qtype, "unmapped")

    sessions: list[LMESession] = []
    for sid, date, turns in zip(
        record["haystack_session_ids"],
        record["haystack_dates"],
        record["haystack_sessions"],
        strict=True,
    ):
        sessions.append(
            LMESession(
                session_id=sid,
                date=date,
                turns=[
                    LMETurn(
                        role=t["role"],
                        content=t["content"],
                        has_answer=bool(t.get("has_answer", False)),
                    )
                    for t in turns
                ],
            )
        )

    return LMEQuestion(
        question_id=qid,
        question_type=qtype,
        question=record["question"],
        answer=record["answer"],
        question_date=record["question_date"],
        is_abstention=is_abs,
        haystack_sessions=sessions,
        answer_session_ids=list(record["answer_session_ids"]),
        sme_category=sme_cat,
    )


def load_questions(path: Path | str) -> Iterator[LMEQuestion]:
    """Parse a LongMemEval JSON file and yield `LMEQuestion` records.

    Args:
        path: Path to longmemeval_oracle.json, longmemeval_s.json, or
            longmemeval_m.json. All three files share the same record
            schema; only the haystack length differs.

    Yields:
        One `LMEQuestion` per dataset record. Records are yielded
        lazily — the JSON file is fully parsed up front (it's a list,
        not line-delimited), but record construction is per-yield so
        callers can stop iterating early without paying for unused
        records.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected top-level JSON array of records, got "
            f"{type(raw).__name__}"
        )
    for record in raw:
        yield _parse_record(record)


CONTENT_RULES_SME_RICH = "sme-rich"
CONTENT_RULES_UPSTREAM_EXACT = "upstream-exact"
_VALID_CONTENT_RULES = (CONTENT_RULES_SME_RICH, CONTENT_RULES_UPSTREAM_EXACT)


def materialize_sme_corpus(
    questions: Iterator[LMEQuestion] | list[LMEQuestion],
    output_dir: Path | str,
    *,
    max_questions: int | None = None,
    content_rules: str = CONTENT_RULES_SME_RICH,
) -> dict:
    """Write a per-question SME-shape corpus to disk.

    Each LongMemEval question gets its own subdirectory under
    `<output_dir>/vault/<question_id>/` containing one markdown file
    per haystack session. This is the per-question scoping that
    LongMemEval requires (each question has its own haystack); SME's
    standard adapter pattern assumes one shared vault, so callers that
    want to run all 500 questions against a single adapter must
    iterate per-question, ingest the small per-question vault, then
    query.

    `<output_dir>/questions.yaml` is written as a single index covering
    all materialized questions, so callers that want the SME-shape
    questions list have it ready.

    Args:
        questions: iterable of LMEQuestion records (typically from
            load_questions).
        output_dir: where to write `vault/` and `questions.yaml`.
        max_questions: cap on how many records to materialize. Useful
            for smoke-test runs (e.g. first 10 questions).
        content_rules: how each session is rendered for embedding.
            "sme-rich" (default) writes YAML frontmatter + role headers
            + both user and assistant turns. "upstream-exact" writes
            only user turns concatenated by newline — matches upstream
            longmemeval_bench.py's raw protocol. The two modes produce
            measurably different retrieval (-2.2pp R@5 absorbed into
            sme-rich's richer rendering vs upstream-exact); the choice
            should be visible to anyone comparing readings across
            systems. See docs/benchmarks/2026-05-17-longmemeval-
            substrate-parity.md.

    Returns:
        A summary dict with counts and the questions.yaml content.
    """
    if content_rules not in _VALID_CONTENT_RULES:
        raise ValueError(
            f"content_rules must be one of {_VALID_CONTENT_RULES}; got {content_rules!r}"
        )
    output_dir = Path(output_dir)
    vault_dir = output_dir / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    questions_yaml: list[dict] = []
    n = 0
    for q in questions:
        if max_questions is not None and n >= max_questions:
            break
        n += 1
        q_dir = vault_dir / q.question_id
        q_dir.mkdir(exist_ok=True)
        for s in q.haystack_sessions:
            note_path = q_dir / f"{s.session_id}.md"
            note_path.write_text(_render_session_md(q, s, content_rules=content_rules))
        questions_yaml.append(q.to_sme_question())

    # Defer import so the loader module itself doesn't pull pyyaml
    # unless materialization is actually requested.
    import yaml

    (output_dir / "questions.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "longmemeval-oracle-v1",
                "source": "https://github.com/xiaowu0162/LongMemEval",
                "license": "MIT (Wu et al., ICLR 2025)",
                "questions": questions_yaml,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )

    return {
        "output_dir": str(output_dir),
        "questions_count": n,
        "vault_dir": str(vault_dir),
    }


def _render_session_md(
    q: LMEQuestion,
    s: LMESession,
    *,
    content_rules: str = CONTENT_RULES_SME_RICH,
) -> str:
    """Render one haystack session for embedding.

    Two rule sets are supported:

    - "sme-rich" (default): YAML frontmatter (session_id, date, question_id,
      is_evidence, source), `# Session <id>` header, then `## user` / `##
      assistant` blocks with all turns. Carries useful metadata + assistant
      semantics for downstream consumption, but produces embeddings that
      drift -2.2pp on R@5 vs upstream's content rules.

    - "upstream-exact": `"\\n".join(t.content for t in s.turns if t.role ==
      "user")`. No metadata, no role markers, no assistant turns. Matches
      upstream longmemeval_bench.py's raw mode and reproduces upstream's
      published 96.6% R@5 exactly (per substrate-parity bench, #51).
    """
    if content_rules == CONTENT_RULES_UPSTREAM_EXACT:
        return "\n".join(t.content for t in s.turns if t.role == "user")

    is_evidence = s.is_evidence(q.answer_session_ids)
    lines = [
        "---",
        f"session_id: {s.session_id}",
        f"date: {s.date!r}",
        f"question_id: {q.question_id}",
        f"is_evidence: {str(is_evidence).lower()}",
        "source: longmemeval",
        "---",
        "",
        f"# Session {s.session_id}",
        "",
        f"_Date: {s.date}_",
        "",
    ]
    for t in s.turns:
        marker = "  <!-- evidence -->" if t.has_answer else ""
        lines.append(f"## {t.role}{marker}")
        lines.append("")
        lines.append(t.content)
        lines.append("")
    return "\n".join(lines)
