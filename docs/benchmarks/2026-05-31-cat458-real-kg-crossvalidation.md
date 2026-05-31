# Cat 4/5/8 real-KG cross-validation — independent diagnosis + direct-cypher check

**Date:** 2026-05-31
**Branch:** `docs/cat458-real-kg-xval`
**Method:** read-only (direct `POST /cypher` + `GET /graph?kg_only` against prod familiar; no writes)
**Result:** An independent SME-projection-side diagnosis that **cross-validates** the merged Cat 4/5/8
measurement fix (#147) and extraction fix (mempalace#336) from a third vantage point — plus a live
direct-cypher cross-check showing that **Cat 4 has an exact full-graph ground truth (verified), but the
SME `--real-kg` adapter's Cat 4/5/8 absolute numbers are sampling artifacts** that drift with
`--graph-limit`. Publish Cat 4/5/8 from full-graph computation, not the adapter sample.

## Background — three converging findings

The SME Cat 4/5/8 structural readings on mempalace went through several corrections this cycle. This
note records an independent diagnosis that converged with the two fixes already merged, then stress-tests
the *corrected* measurement path.

1. **Original artifact (measurement).** Cat 4 reported "98.98% one edge type, normalized entropy 0.020."
   Root cause, found independently from the SME code side: `sme/adapters/_graph_mapping.py` generated
   `tunnel` edges **combinatorially** — for each room shared across wings, one edge per *wing-pair*
   sharing it, i.e. O(k²) per shared room. Popular rooms (`references`, `discoveries`, `architecture`,
   `general`) span hundreds of wings, producing 167,645 of 169,372 edges (98.98%). `ingestion_integrity.py`
   then computed entropy over **all** edges with no layer filter, folding this combinatorial structural
   layer in with the semantic KG-triple layer. The `_table` property (`"structural"` vs `"kg_triple"`)
   already discriminated the layers. → Fixed by #147 (Sage, PR #210): `project_graph` gained a `kg_only`
   mode that skips the structural projection, plus `--real-kg`/`--graph-limit` CLI flags on cat4/5/8.

2. **Real extraction defect.** Below the tunnel noise sat a genuine problem: on the real AGE graph,
   **55.05% of `:RELATION` edges carry `relation_type = 'other'`** — the canonical predicate mapper
   (`kg_canonical_vocab.py`, embedding gate threshold 0.45) dumping unbindable predicates into a fallback
   bucket, with originals preserved in `raw_relation_type`. → Fixed by mempalace#336 (somnia-2): expanded
   the deterministic synonym/drop pre-pass; `other` 55.05% → 26.9%, KG-layer normalized entropy
   0.340 → 0.640. Prod re-map flag = palace-daemon #208/#150 (deterministic SQL on the preserved raw type,
   no LLM re-call).

Independent capped-sample cross-check (this work) gave `other ≈ 49.7%` on a 1,000-edge slice — which
**cross-validates** the full-graph 55.05% (sample vs full graph), confirming the defect is real, not a
projection artifact.

## Live cross-validation — adapter vs direct cypher

Ground truth via `POST /cypher` (`MATCH ()-[r:RELATION]->() RETURN r.relation_type, count(*)`):

- **1,581,282 nodes ; 1,921,600 RELATION edges.**
- Cat 4: `other` = 1,057,935 = **55.05%**, entropy 2.6837 bits (**normalized 0.3402**), 237 distinct types.
- This reproduces mempalace#336's validated "before" (`other` 0.5505490…, entropy 0.3402) **to the digit**,
  confirming the full-graph aggregate is the trustworthy method.

The SME `--real-kg` adapter reads `GET /graph?kg_only=true&limit=N` — a non-representative, order-dependent
slice of `kg_triples`. Its numbers **drift with `--graph-limit`** rather than converging to ground truth:

### Cat 4 — `other%` and entropy

| `--graph-limit` | edges seen | `other%` | entropy (normalized) |
|----------------:|-----------:|---------:|---------------------:|
| 5,000           | 10,000     | 47.3%    | 0.56                 |
| 10,000          | 20,000     | 57.8%    | 0.46                 |
| 25,000          | 50,000     | 67.8%    | 0.37                 |
| **full (cypher)** | **1,921,600** | **55.05%** | **0.34**         |

Confirmed daemon-side (`GET /graph?kg_only` directly shows the same drift), so it's the daemon's `/graph`
KG sampler, not an SME-side bug. **None** of the adapter readings equal the ground truth.

### Cat 5 — components / isolates

| `--graph-limit` | nodes seen | components |
|----------------:|-----------:|-----------:|
| 5,000           | 12,849     | 6,072      |
| 25,000          | 52,005     | 27,025     |

The adapter sees ≤52k of 1,581,282 real nodes. A node whose `:RELATION` edges all fall outside the sampled
slice appears **isolated** — so component/isolate counts are dominated by sampling and scale with
`--graph-limit`. They are not real-graph connectivity.

### Cat 8 — modularity

| `--graph-limit` | modularity | inter-community |
|----------------:|-----------:|----------------:|
| 5,000           | 0.962      | 2.2%            |
| 25,000          | 0.912      | 6.6%            |

Louvain modularity on the sampled subgraph also drifts with the limit.

## The asymmetry that matters

- **Cat 4 has an exact full-graph ground truth** — a cheap cypher aggregate over `relation_type`. Verified
  (55.05% / 0.34). Cat 4 is safe to publish *if computed from the full-graph aggregate*.
- **Cat 5 / Cat 8 have no API-computable full-graph ground truth.** The full `:RELATION` edge-list pull
  (needed for local WCC/Louvain), a global `count(DISTINCT n)`, and the undirected anonymous
  `(n)-[:RELATION]-()` pattern all fail (daemon statement-timeout / AGE syntax). The daemon's statement
  timeout correctly protects prod; it was not bypassed. So Cat 5/8 real-graph values need **server-side
  WCC/Louvain or a direct `psql` connection**, not the HTTP API.

## Recommendation

1. **Publish Cat 4 from the full-graph aggregate** (cypher `relation_type` histogram), not the `--real-kg`
   adapter sample.
2. **Do not publish Cat 5/8 absolute numbers from the `--real-kg` adapter sample** — they are
   `--graph-limit`-dependent artifacts. Either compute server-side over the full graph, or mark the Cat 5/8
   matrix cells "not measurable via the sampled adapter."
3. **Default-path hardening (already routed to Sage):** `--real-kg` is opt-in; the default cat4/5/8 path
   still folds in the tunnel scaffold and reports the ~0.020 artifact. Flip the default or emit a loud
   warning so a plain run cannot silently publish the artifact number. Same footgun class as the #140
   silent candidate-strategy no-op.

## Post-re-map prod state (2026-05-31) — recommendations actioned

All three recommendations above have since landed, and the JP-gated prod re-map +
the server-side full-graph compute have run. These are the **current published
numbers**, each verified directly against the live AGE graph by Sage (read-only)
on 2026-05-31 — not the sampled adapter.

**Cat 4 (canonical re-map applied, relabel-only — 0 deletions, 520,043 edges
relabeled out of the `other` sink):** verified via the full-graph
`relation_type` GROUP BY.

| metric | before (pre-re-map) | **after (current prod)** |
|---|---|---|
| total RELATION edges | 1,921,600 | **1,921,600** (unchanged) |
| `other` fraction | 0.5505 | **0.2818** (541,438 edges) |
| normalized entropy | 0.3402 | **0.4378** |
| distinct relation types | 237 | **236** |

> Optional JP-gated junk-DELETE pass (~48K content-free edges) would further shift
> these to ≈0.2689 / ≈0.64 / ≈41 types. Not yet decided; a one-line follow-up if greenlit.

**Cat 5 (exact full-graph WCC via `GET /graph/structural-stats`, palace-daemon#211 +
SME#223):** replaces the bogus capped-`/graph` artifact ("44.8% isolates / 498
components"). Verified from the cached structural-stats response.

| metric | **current prod (exact)** |
|---|---|
| entities | 1,156,277 |
| RELATION edges | 1,921,600 |
| largest component | 733,753 (**63.46%** — well-connected giant component) |
| isolates | 236,169 (**20.4%**) |
| components | 305,975 |

**Cat 8:**
- **Introspection 0 → 1** — `GET /ontology` is live on the prod familiar daemon
  (palace-daemon#205, restarted). The system now self-reports declared-vs-effective drift.
- **Modularity NOT computed** — `networkx` is not installed on the familiar daemon, so
  the structural-stats response carries: *"connectivity stats (Cat 5) are exact and
  dependency-free; modularity (Cat 8) not computed."* Cat 8 publishes the **hierarchy
  verdict** with modularity marked *pending-networkx* — no fabricated modularity number.

## Provenance

- #147 (Sage, PR #210) — `kg_only` measurement fix (merged).
- mempalace#336 (somnia-2) — `other`-sink extraction fix (merged); palace-daemon #208/#150 re-map flag.
- This note — independent SME-projection-side diagnosis + live direct-cypher cross-validation (#151).
- palace-daemon#211 + SME#223 (Sage) — server-side full-graph Cat 5 WCC + Cat 8 modularity endpoint + consumer.
- Prod re-map + structural-stats POST driven by team-lead 2026-05-31; numbers verified read-only by Sage.
- Working artifacts: `scratch/cat4-131/diagnosis.md`, `scratch/cat458-xval/`, familiar `/tmp/structstats.json`.
