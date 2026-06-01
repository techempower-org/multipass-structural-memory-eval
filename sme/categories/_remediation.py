"""Diagnostic actionability — the 'fix this and re-run' half of a reading.

SME's constitutional posture is *diagnostic, not leaderboard* — but a
diagnostic that only says "here's what's wrong" stops one step short of
useful. This module supplies the small, shared shape that lets each
category attach the *actionable* half to its findings: what to change,
and how to re-verify that the change worked (upstream
M0nkeyFl0wer#44).

A ``Remediation`` is deliberately tiny — three strings and a band — so
that wiring it into a category is a few lines, not a refactor. The
pattern a new category follows:

  1. In the scorer, after computing the interpretive band for a signal,
     append a ``Remediation`` to ``report.remediations`` *only when the
     band is not healthy* (a clean reading needs no fix).
  2. In ``format_report``, call ``render_remediations(report.remediations)``
     after the Reading section.

Keeping the structure identical across categories is the point — a
reader who has seen one category's remediation block can read every
category's, and the case-study catalog (``docs/case-studies/``) can
quote them verbatim as "fix → re-run" entries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Remediation:
    """One actionable 'fix this and re-run' item attached to a finding.

    Fields:
        finding: the defect this addresses, in the operator's terms
            ("248 canonical collisions, 0.2% of entities").
        fix: the concrete change to make. Names the layer/knob, not a
            vague exhortation ("normalize case+whitespace in the
            entity-ID function, or merge by highest-degree ID").
        reverify: how to confirm the fix landed — ideally the exact
            command or metric to re-read ("re-run `sme-eval ingest`;
            expect canonical_collisions → 0").
        band: which interpretive band triggered this — "warning" or
            "concerning". Healthy readings never produce a Remediation.
    """

    finding: str
    fix: str
    reverify: str
    band: str = "warning"

    def to_dict(self) -> dict:
        return {
            "finding": self.finding,
            "fix": self.fix,
            "reverify": self.reverify,
            "band": self.band,
        }


def render_remediations(remediations: list[Remediation]) -> list[str]:
    """Render a Remediation block as report lines (no trailing newline).

    Returns an empty list when there's nothing to remediate, so callers
    can ``lines.extend(render_remediations(...))`` unconditionally and a
    clean reading simply adds nothing. When there ARE items, the block
    is self-titled so it reads correctly wherever it's appended.
    """
    if not remediations:
        return []

    lines = [
        "",
        "Remediation — fix this and re-run",
        "─" * 60,
    ]
    for i, rem in enumerate(remediations, 1):
        lines.append(f"  {i}. [{rem.band}] {rem.finding}")
        lines.append(f"       Fix:      {rem.fix}")
        lines.append(f"       Re-run:   {rem.reverify}")
    return lines
