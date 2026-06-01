# Proposed category: Phantom Edges — graph assertions with no source support

**Status:** proposal + working capability. Implemented in
`sme/categories/phantom_edge.py` with tests in `tests/test_phantom_edge.py`.
**Not** yet assigned a canonical "Cat N" number or wired into the main
CLI cat-run sequence — canonization is deferred to review.

**Upstream:** `M0nkeyFl0wer/multipass-structural-memory-eval#4`.

## What it is

The existing categories cover three failure modes around the
graph-vs-source relationship, but leave a fourth uncovered:

| Category | Diagnostic | Direction |
|----------|-----------|-----------|
| **Cat 3** (contradiction) | two *source* assertions disagree | source → source |
| **Cat 5** (gaps) | an assertion is *missing* from the graph | source → graph |
| **Cat 8** (declared-vs-built) | the graph doesn't match the *declared architecture* | declaration → graph |
| **Phantom edges** (this) | an assertion *exists in the graph* that *no source supports* | **graph → source** |

Phantom edges are *excess* structure: an edge the graph asserts that the
underlying source files don't justify. It is the inverse of Cat 5 (which
surfaces missing structure) and finer-grained than Cat 8 (which checks
the architecture claim, not whether each individual edge landed on top of
real source content).

Failure modes it targets (from #4):

- **Auto-tunnel-detection firing on coincidental keyword overlap** — a
  heuristic linking rooms by shared lexical tokens where the overlap was
  incidental (a project name in a session log, a stop-word collision).
- **Stale edges after drawer mutation** — a drawer's content changed such
  that the edge's original basis no longer holds, but the edge persists.
- **Embedding-induced phantom links** — edges materialized from a
  cosine-similarity threshold whose endpoints' source content actually
  disagrees.

## The grounding check (first slice — deterministic, lexical)

#4 flags the threshold + overlap function as the parts needing design,
and warns that a *substring-overlap* check is circular for graphs whose
edges were themselves created by substring overlap. Two design choices
sidestep that:

1. **Ground against the prose body, not the frontmatter.** good-dog edges
   are declared in YAML frontmatter; the prose body is an *independent*
   signal. `sme.corpora.good_dog_graph.load_source_bodies()` returns
   `{source_note → body}` with the frontmatter stripped, so grounding an
   edge never reads the block that declared it.

2. **Endpoint-presence, not relation-paraphrase.** An edge is *grounded*
   when **both** of its endpoint entities are textually present in the
   body of the note that declared it (matched by canonical name **or any
   alias**, via per-form token coverage so an entity grounds when any one
   surface form clears `min_overlap`). The relation *verb* is a secondary
   signal — its absence alone doesn't condemn an edge (English has many
   ways to say "authored_by"). This is deliberately a **lower bound** on
   phantom-ness: an edge can clear it and still be semantically
   unsupported. A relation-grounding pass is the obvious next slice.

The detector is numpy/networkx-free, no LLM, no server — constitutional.

## Validity: calibration against the corpus's own flag

good-dog pre-flags weakly-grounded edges with `needs_grounding: true`
(14 of 164 edges) — the maintainer already knows these lean on an alias
registry or a reframing rather than verbatim source text. A working
detector should flag a *strictly higher* phantom rate among flagged edges
than among the rest. Measured (97 entities / 164 edges):

| `min_overlap` | flagged phantom | unflagged phantom | lift |
|---------------|-----------------|-------------------|------|
| 0.50 (default) | 21.4% (3/14) | 3.3% (5/150) | **6.5×** |
| 0.60 | 21.4% (3/14) | 6.7% (10/150) | 3.2× |
| 0.75 | 28.6% (4/14) | 12.7% (19/150) | 2.3× |

The detector tracks the maintainer's own weak-grounding judgement across
the usable band. The companion view — per-edge-type phantom rate at
`min_overlap=0.5` — points at exactly the heuristic / cross-note edge
types #4 predicts:

```
regulates      6/17   (35%)   ← cross-entity authority claims
contradicts    3/6    (50%)   ← cross-note framing edges
authored_by    4/23
mentions       4/57
subject_of     0/27   (clean)
located_in     0/11   (clean)
alias_of       0/5    (clean)
```

## Known limitations (documented, not swept under)

- **Threshold band has hard edges.** At the strict `1.0` reading the
  calibration signal *inverts* (28.6% flagged vs 30.0% unflagged): a long
  canonical title ("Expression Studies on Wolves...") almost never
  appears token-complete in prose that refers to it as "Schenkel's 1947
  monograph", so strict mode condemns legitimate alias-named edges and
  the noise swamps the flagged signal. At the permissive `0.34` reading
  every edge grounds (0 phantom) — no separation. The default `0.5` sits
  in the usable middle. **Read the per-type breakdown and the calibration
  delta, not the bare absolute rate**, which is threshold- and
  corpus-shape-dependent.

- **Token-collision false-negative** (the #4 "coincidental overlap" mode
  in reverse): an *absent* endpoint whose tokens collide with a *present*
  entity's tokens can ground spuriously (e.g. "American Kennel Club"
  grounding off "American Pit Bull Terrier" + "United Kennel Club"). The
  two ways out #4 sketches both address this: per-edge-type thresholds,
  or an IDF-weighted token model. The first cut is plain coverage.

- **Cross-note edges** whose endpoints are only named in *other* notes
  read high under endpoint-presence grounding even when legitimate. This
  is why the absolute rate alone is not the headline.

## Where it could land (deferred to review)

#4 asks whether this is its own category or a Cat 8 sub-test ("8b" /
"8-inverse"). The capability is the same either way; the implementation
keeps a small surface (`score_phantom_edges` + `format_report`, mirroring
the Cat 4 / Cat 5 scorer shape) so it can be wired into whichever slot
review blesses. Naming follows #4: "phantom edge" is the term from the
originating MemPalace discussion.

## API

```python
from sme.corpora.good_dog_graph import load_graph, load_source_bodies
from sme.categories.phantom_edge import score_phantom_edges, format_report

entities, edges = load_graph()
bodies = load_source_bodies()
report = score_phantom_edges(entities, edges, bodies, min_overlap=0.5)
print(format_report(report))
```

`source_bodies` is `{source_note → body_text}`; keys must match the
`source_note` property each edge carries. Any adapter that exposes
per-edge source text in this shape can be probed — the scorer is
adapter-agnostic, like the Cat 4 / Cat 5 scorers.
