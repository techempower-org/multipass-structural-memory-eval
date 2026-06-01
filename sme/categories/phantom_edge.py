"""Proposed category: Phantom Edges — graph edges with no source support.

Upstream proposal: ``M0nkeyFl0wer/multipass-structural-memory-eval#4``.
This module implements the capability so the proposal has a concrete,
deterministic, locally-runnable detector to react to. It is NOT yet
wired into the canonical Cat 1–9 CLI sequence — canonization (and the
"Cat N" number) is JP's call at review.

The shape (from #4)
-------------------
The existing ingestion line (Cat 4) checks that the graph is *well-
formed* — clean canonicalization, populated fields, balanced edge
vocabulary. Cat 8 checks *declared-vs-built* — does the graph match the
architecture the README claims. Neither checks whether each individual
edge is **grounded**: that the source files actually say what the edge
asserts. Phantom-edge detection is the inverse-of-Cat-8 unidirectional
diagnostic — graph → source — that #4 carves out:

  * **Cat 3 (contradiction):** two *source* assertions disagree.
  * **Cat 5 (gaps):** an assertion is *missing* from the graph.
  * **Phantom edges:** an assertion *exists in the graph* that *no
    source supports*. Excess structure, not missing structure.

Failure modes it catches (verbatim from #4): auto-tunnel-detection
firing on coincidental keyword overlap; stale edges left behind after a
drawer mutation invalidated their basis; embedding-induced links
materialized from a cosine threshold whose endpoints actually disagree.

The grounding check (first slice — deterministic, lexical)
----------------------------------------------------------
#4 flags the threshold + overlap function as the parts that need
design, and warns that a *substring-overlap* grounding check is circular
for graphs whose edges were themselves created by substring overlap. We
sidestep that two ways:

1. **Ground against the prose body, not the frontmatter.** good-dog
   edges are declared in YAML frontmatter; the prose body is an
   independent signal. An edge is grounded when both of its endpoint
   entities are textually present in the body of the note that declared
   the edge (matched by canonical name OR any alias, via token-overlap
   so multi-word names tolerate minor variation). The relation *verb* is
   a secondary signal — present-verb strengthens grounding but its
   absence alone does not condemn an edge (English has many ways to say
   "authored_by").

2. **Endpoint-presence, not relation-paraphrase.** Requiring both
   endpoints to appear in the source body is the cheapest defensible
   "this edge could plausibly have come from this text" test. It is
   deliberately a *lower bound* on phantom-ness: an edge can clear it and
   still be semantically unsupported (both entities are named but the
   text never relates them). A stricter relation-grounding pass is the
   obvious next slice; #4 explicitly scopes the first cut to lexical.

Calibration signal
-------------------
good-dog-corpus pre-flags weakly-grounded edges with
``needs_grounding: true`` in frontmatter (the maintainer already knows
these lean on an alias registry or a reframing rather than verbatim
source text). A detector that works should flag a *strictly higher*
phantom rate among ``needs_grounding`` edges than among the rest — the
report surfaces both rates so a maintainer can see the detector is
tracking the maintainer's own judgement, not noise.

Measured on good-dog (97 entities / 164 edges), the calibration delta
holds across the usable threshold band:

    min_overlap   flagged phantom   unflagged phantom   lift
    0.50          21.4% (3/14)       3.3% (5/150)        6.5×
    0.60          21.4% (3/14)       6.7% (10/150)       3.2×
    0.75          28.6% (4/14)      12.7% (19/150)       2.3×

The band has hard edges, and both are documented limitations rather than
swept under the rug:

* At the strict ``1.0`` reading the signal **inverts** (28.6% flagged vs
  30.0% unflagged). A long canonical title ("Expression Studies on
  Wolves...") almost never appears token-complete in prose that refers to
  it as "Schenkel's 1947 monograph", so strict mode condemns legitimate
  alias-named edges and the noise swamps the flagged signal.
* At the permissive ``0.34`` reading **every** edge grounds (0 phantom),
  so there's no separation to read.

The default ``min_overlap=0.5`` sits in the usable band. The per-edge-
type breakdown is the companion view: on good-dog at 0.5, ``regulates``
and ``contradicts`` carry the highest phantom rates — exactly the
heuristic / cross-note edge types #4 predicts, while ``subject_of`` /
``located_in`` / ``alias_of`` ground cleanly.

Known false-negative (the #4 'coincidental keyword overlap' mode in
reverse): an absent endpoint whose tokens happen to collide with a
PRESENT entity's tokens can ground spuriously (e.g. "American Kennel
Club" grounding off "American Pit Bull Terrier" + "United Kennel Club").
Per-edge-type thresholds or an IDF-weighted token model (a future slice
#4 itself sketches) would tighten this; the first cut is plain coverage.

This stays within the constitutional bound: numpy/networkx-free, no LLM,
no server. Lexical grounding only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from sme.adapters.base import Edge, Entity

log = logging.getLogger(__name__)


# --- Tokenization -----------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tokens too generic to count as evidence that a specific entity is
# present — they collide across unrelated entities. Kept deliberately
# small (the issue's "stop-word collision" failure mode): an entity
# whose ONLY name tokens are stop-words can't be lexically grounded, and
# the report flags that as un-checkable rather than silently grounded.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "at",
    "by", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "dog", "dogs", "pet", "canine",  # corpus-universal — every note has them
})


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, stop-words removed."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _name_forms(name: str, aliases: list[str]) -> list[set[str]]:
    """One content-token set per surface form (canonical name + each
    alias). Forms whose tokens are all stop-words drop out. An entity
    grounds when ANY ONE of these forms is sufficiently covered — pooling
    every form's tokens into one set would penalize an entity for having
    a rich alias list (a body that names it by one alias shouldn't have
    to also satisfy the canonical title's tokens)."""
    forms: list[set[str]] = []
    for raw in [name or ""] + list(aliases or []):
        toks = _tokens(raw)
        if toks:
            forms.append(toks)
    return forms


def _has_checkable_form(entity: Entity) -> bool:
    """False when every surface form is all stop-words — un-checkable."""
    return bool(
        _name_forms(entity.name, entity.properties.get("aliases") or [])
    )


def _entity_present(
    entity: Entity,
    body_tokens: set[str],
    *,
    min_overlap: float,
) -> bool:
    """Is ``entity`` textually grounded in a body with these tokens?

    An entity is present when at least one of its surface forms (canonical
    name or an alias) has ``min_overlap`` fraction of its tokens in the
    body. A single-distinctive-token form (e.g. a proper noun or a
    capitalized acronym alias) grounds on that one token; a multi-word
    form tolerates a missing word or two. Entities whose every form is all
    stop-words have no checkable form and are reported separately as
    *un-checkable* rather than counted either way.
    """
    forms = _name_forms(entity.name, entity.properties.get("aliases") or [])
    if not forms:
        return False  # caller treats no-checkable-form entities as un-checkable
    for form_tokens in forms:
        hits = len(form_tokens & body_tokens)
        if (hits / len(form_tokens)) >= min_overlap:
            return True
    return False


# --- Data shapes ------------------------------------------------------


@dataclass
class PhantomEdge:
    """One edge whose endpoints aren't both grounded in its source body."""

    source_id: str
    target_id: str
    edge_type: str
    source_note: str
    # Which endpoint(s) failed to ground: "source", "target", or "both".
    missing: str
    # True when an endpoint couldn't be checked (all-stop-word name) —
    # the edge is reported as a phantom *candidate* but flagged uncheckable
    # so a maintainer doesn't read it as a confirmed coincidental edge.
    uncheckable: bool = False
    needs_grounding_flag: bool = False
    evidence: str = ""


@dataclass
class PhantomEdgeReport:
    edges_total: int
    edges_checked: int  # had a recoverable source body
    edges_missing_source: int  # source_note not in the bodies map
    grounded: int
    phantom: int
    phantom_rate: float
    phantom_edges: list[PhantomEdge] = field(default_factory=list)

    # Per-edge-type breakdown: {edge_type: (phantom, total)}
    per_type: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Calibration against the corpus's own needs_grounding flag.
    flagged_total: int = 0
    flagged_phantom: int = 0
    unflagged_total: int = 0
    unflagged_phantom: int = 0

    # Edges with an un-checkable endpoint (all-stop-word name).
    uncheckable: int = 0

    # Echo the knobs the reading was taken with, so a JSON consumer
    # knows the threshold the rate is relative to.
    min_overlap: float = 1.0

    @property
    def flagged_phantom_rate(self) -> float:
        return self.flagged_phantom / self.flagged_total if self.flagged_total else 0.0

    @property
    def unflagged_phantom_rate(self) -> float:
        return (
            self.unflagged_phantom / self.unflagged_total
            if self.unflagged_total
            else 0.0
        )


# --- Scorer -----------------------------------------------------------


def score_phantom_edges(
    entities: list[Entity],
    edges: list[Edge],
    source_bodies: dict[str, str],
    *,
    min_overlap: float = 0.5,
    example_limit: int = 20,
    source_note_key: str = "source_note",
) -> PhantomEdgeReport:
    """Produce a phantom-edge diagnostic reading for a graph snapshot.

    Args:
        entities, edges: the graph snapshot (adapter-agnostic).
        source_bodies: ``{source_note: body_text}`` — the prose the edges
            should be grounded in. Keys must match the ``source_note``
            property stamped on each edge. For good-dog, get this from
            :func:`sme.corpora.good_dog_graph.load_source_bodies`.
        min_overlap: fraction of an entity's content tokens that must
            appear in the body for the endpoint to count as present.
            ``0.5`` (default) means "at least half the distinctive name/
            alias tokens show up" — chosen because the strict ``1.0``
            reading floods false positives on this corpus: a publication
            whose canonical name is a long title ("Expression Studies on
            Wolves...") is referred to in prose as "Schenkel's 1947
            monograph", so every title token never appears verbatim even
            though the edge is legitimate. ``1.0`` is available for the
            strict reading; ``0.34`` for a permissive one-distinctive-
            token-suffices reading. The absolute rate IS threshold- and
            corpus-shape-dependent — read the per-type breakdown and the
            ``needs_grounding`` calibration delta, not the bare number.
        example_limit: cap on retained phantom-edge examples.
        source_note_key: the edge property holding the note id. Defaults
            to ``"source_note"`` (good-dog's convention).

    An edge is *phantom* when, in the body of the note that declared it,
    one or both endpoint entities are not textually present. Edges whose
    ``source_note`` is absent from ``source_bodies`` can't be checked and
    are excluded from the rate (counted under ``edges_missing_source``)
    rather than assumed phantom.
    """
    by_id: dict[str, Entity] = {e.id: e for e in entities}

    grounded = 0
    phantom_list: list[PhantomEdge] = []
    per_type_phantom: dict[str, int] = {}
    per_type_total: dict[str, int] = {}
    flagged_total = flagged_phantom = 0
    unflagged_total = unflagged_phantom = 0
    uncheckable_count = 0
    checked = 0
    missing_source = 0

    # Cache body tokenization — many edges share a source note.
    body_tokens_cache: dict[str, set[str]] = {}

    for edge in edges:
        note = edge.properties.get(source_note_key)
        body = source_bodies.get(note) if note else None
        if body is None:
            missing_source += 1
            continue
        checked += 1

        etype = edge.edge_type or "unknown"
        per_type_total[etype] = per_type_total.get(etype, 0) + 1
        needs_flag = bool(edge.properties.get("needs_grounding"))
        if needs_flag:
            flagged_total += 1
        else:
            unflagged_total += 1

        if note not in body_tokens_cache:
            body_tokens_cache[note] = _tokens(body)
        body_toks = body_tokens_cache[note]

        src_ent = by_id.get(edge.source_id)
        dst_ent = by_id.get(edge.target_id)

        # An endpoint with no resolvable entity record can't be grounded;
        # treat as missing. An endpoint with an all-stop-word name is
        # un-checkable (flagged on the result).
        def _check(ent: Optional[Entity]) -> tuple[bool, bool]:
            """Returns (present, uncheckable)."""
            if ent is None:
                return (False, False)
            if not _has_checkable_form(ent):
                return (False, True)
            return (
                _entity_present(ent, body_toks, min_overlap=min_overlap),
                False,
            )

        src_present, src_uncheckable = _check(src_ent)
        dst_present, dst_uncheckable = _check(dst_ent)
        edge_uncheckable = src_uncheckable or dst_uncheckable

        if src_present and dst_present:
            grounded += 1
        else:
            per_type_phantom[etype] = per_type_phantom.get(etype, 0) + 1
            if needs_flag:
                flagged_phantom += 1
            else:
                unflagged_phantom += 1
            if edge_uncheckable:
                uncheckable_count += 1
            if not src_present and not dst_present:
                missing = "both"
            elif not src_present:
                missing = "source"
            else:
                missing = "target"
            if len(phantom_list) < example_limit:
                phantom_list.append(
                    PhantomEdge(
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        edge_type=etype,
                        source_note=note,
                        missing=missing,
                        uncheckable=edge_uncheckable,
                        needs_grounding_flag=needs_flag,
                        evidence=str(edge.properties.get("evidence") or ""),
                    )
                )

    phantom = checked - grounded
    rate = (phantom / checked) if checked else 0.0
    per_type = {
        t: (per_type_phantom.get(t, 0), per_type_total[t]) for t in per_type_total
    }

    return PhantomEdgeReport(
        edges_total=len(edges),
        edges_checked=checked,
        edges_missing_source=missing_source,
        grounded=grounded,
        phantom=phantom,
        phantom_rate=rate,
        phantom_edges=phantom_list,
        per_type=per_type,
        flagged_total=flagged_total,
        flagged_phantom=flagged_phantom,
        unflagged_total=unflagged_total,
        unflagged_phantom=unflagged_phantom,
        uncheckable=uncheckable_count,
        min_overlap=min_overlap,
    )


# --- Interpretive bands -----------------------------------------------
#
# Phantom-edge rate is corpus-shape dependent (a graph with many cross-
# note edges whose endpoints are only named in OTHER notes will read
# high under endpoint-presence grounding even when the edges are
# legitimate). The bands describe where the reading sits, not pass/fail.

_PHANTOM_HEALTHY = 0.02   # < 2% of edges ungrounded
_PHANTOM_WARN = 0.10      # 2-10% warning
# > 10% concerning


def _band(rate: float) -> str:
    if rate <= _PHANTOM_HEALTHY:
        return "healthy"
    if rate <= _PHANTOM_WARN:
        return "warning"
    return "concerning"


def format_report(report: PhantomEdgeReport) -> str:
    lines = [
        "Phantom Edges — graph assertions with no source support",
        "═══════════════════════════════════════════════════════",
        "  (proposed category — upstream #4; not yet canonical)",
        "",
        "Measurements",
        "─" * 60,
        f"  Edges total:               {report.edges_total:,}",
        f"  Edges checked:             {report.edges_checked:,}"
        f"  (min_overlap={report.min_overlap:g})",
    ]
    if report.edges_missing_source:
        lines.append(
            f"  Edges w/o source body:     {report.edges_missing_source:,}"
            "  (excluded — can't be grounded)"
        )
    lines += [
        f"  Grounded:                  {report.grounded:,}",
        f"  Phantom:                   {report.phantom:,}",
        f"  Phantom rate:              {report.phantom_rate:.1%}",
    ]
    if report.uncheckable:
        lines.append(
            f"  Un-checkable endpoints:    {report.uncheckable:,}"
            "  (all-stop-word entity name)"
        )

    if report.per_type:
        lines.append("")
        lines.append("  Per-edge-type phantom rate:")
        for etype, (ph, tot) in sorted(
            report.per_type.items(), key=lambda kv: (-kv[1][0], kv[0])
        ):
            pct = 100 * ph / tot if tot else 0.0
            lines.append(f"    {etype:24s} {ph:>4}/{tot:<4}  ({pct:5.1f}%)")

    if report.flagged_total or report.unflagged_total:
        lines.append("")
        lines.append("  Calibration vs corpus `needs_grounding` flag:")
        lines.append(
            f"    flagged edges:    {report.flagged_phantom}/{report.flagged_total}"
            f"  ({report.flagged_phantom_rate:.1%} phantom)"
        )
        lines.append(
            f"    unflagged edges:  {report.unflagged_phantom}/{report.unflagged_total}"
            f"  ({report.unflagged_phantom_rate:.1%} phantom)"
        )

    if report.phantom_edges:
        lines.append("")
        lines.append(f"  Phantom-edge examples ({len(report.phantom_edges)} shown):")
        for pe in report.phantom_edges[:10]:
            flag = " [needs_grounding]" if pe.needs_grounding_flag else ""
            uncheck = " [uncheckable]" if pe.uncheckable else ""
            lines.append(
                f"    {pe.source_id} -[{pe.edge_type}]-> {pe.target_id}"
                f"  (missing: {pe.missing}){flag}{uncheck}"
            )
            lines.append(f"        in: {pe.source_note}")
        if len(report.phantom_edges) > 10:
            lines.append(f"    ... +{len(report.phantom_edges) - 10} more")

    # --- Reading ------------------------------------------------------

    lines.append("")
    lines.append("Reading")
    lines.append("─" * 60)

    if report.edges_checked == 0:
        lines.append("  No checkable edges — nothing to read.")
        return "\n".join(lines)

    band = _band(report.phantom_rate)
    lines.append(
        f"  ● Phantom-edge rate: {report.phantom_rate:.1%} of checked edges "
        f"have an endpoint not grounded in their source body [{band}]."
    )
    if band == "healthy":
        lines.append(
            "      Nearly every edge's endpoints are named in the prose the "
            "edge was extracted alongside. The graph's structure tracks the "
            "source."
        )
    elif band == "warning":
        lines.append(
            "      A non-trivial slice of edges assert a relation between "
            "entities the source body doesn't both name. Inspect the "
            "highest-rate edge type — a heuristic (auto-tunnel, cosine "
            "threshold) may be firing without a textual basis."
        )
    else:
        lines.append(
            "      Large fraction of ungrounded edges. Either the edge-"
            "creation rule materializes links the source never asserts "
            "(phantom edges in the #4 sense), or the grounding check is "
            "mismatched to this corpus (e.g. cross-note edges whose "
            "endpoints are only named in OTHER notes). Check the per-type "
            "breakdown before concluding."
        )

    # Calibration reading — the headline validity check.
    if report.flagged_total and report.unflagged_total:
        if report.flagged_phantom_rate > report.unflagged_phantom_rate:
            lines.append(
                "  ● Calibration: the detector flags `needs_grounding` edges "
                f"at {report.flagged_phantom_rate:.1%} vs "
                f"{report.unflagged_phantom_rate:.1%} for the rest — it is "
                "tracking the maintainer's own weak-grounding judgement."
            )
        else:
            lines.append(
                "  ● Calibration: WARNING — the detector does NOT flag "
                "`needs_grounding` edges more often than the rest "
                f"({report.flagged_phantom_rate:.1%} vs "
                f"{report.unflagged_phantom_rate:.1%}). Either the grounding "
                "check is mis-tuned or the flagged edges happen to ground "
                "lexically anyway (their weakness is semantic, not lexical)."
            )

    if report.uncheckable:
        lines.append(
            f"  ● {report.uncheckable} edge(s) have an un-checkable endpoint "
            "(entity name is all stop-words). These are reported as phantom "
            "candidates but a maintainer should resolve them by hand — the "
            "lexical check can't speak to them."
        )

    return "\n".join(lines)
