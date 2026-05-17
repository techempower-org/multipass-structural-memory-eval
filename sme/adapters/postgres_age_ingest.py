"""PostgresIngestAdapter + AGE graph write-through.

Subclasses PostgresIngestAdapter to add an AGE-write side effect on every
upsert: extract entities from the document, MERGE entity nodes, CREATE
[:MENTIONED_IN] edges from each entity back to the drawer id. On query,
optionally consult the graph for entity-overlap candidates and fuse with
vector retrieval.

Designed as a stepping stone toward write-through entity extraction in
mempalace itself. The architecture here mirrors what a real
mempalace.backends.postgres write-through middleware would look like:
the extractor is pluggable, the AGE write is idempotent (MERGE),
and the read-side fusion strategy is configurable.

Two retrieval modes exposed:
  - "fusion" (default): vector top-K ⊕ graph entity-overlap candidates
    fused by reciprocal-rank-fusion.
  - "graph_only": ignore vectors; rank purely by entity-overlap count.
    Diagnostic mode for measuring what graph contributes vs what vector
    already had.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2 import sql

from sme.adapters.base import Entity, QueryResult
from sme.adapters.postgres_ingest import PostgresIngestAdapter, DEFAULT_DSN
from sme.extractors import Entity as ExtractedEntity
from sme.extractors.regex import extract as regex_extract

log = logging.getLogger(__name__)

DEFAULT_GRAPH_NAME = "sme_spike_kg"


class PostgresAgeIngestAdapter(PostgresIngestAdapter):
    """Postgres + pgvector ingest with AGE write-through and read-side fusion."""

    def __init__(
        self,
        *,
        dsn: str = DEFAULT_DSN,
        table_name: str = "lme_bench_drawers",
        n_results: int = 5,
        graph_name: str = DEFAULT_GRAPH_NAME,
        extractor=regex_extract,
        retrieval_mode: str = "fusion",  # "fusion" | "graph_only" | "vector_only"
        fusion_k: int = 60,  # RRF k constant
        graph_top_k: int = 50,  # how many graph candidates to fetch
        **kwargs,
    ) -> None:
        super().__init__(dsn=dsn, table_name=table_name, n_results=n_results, **kwargs)
        self.graph_name = graph_name
        self.extractor = extractor
        self.retrieval_mode = retrieval_mode
        self.fusion_k = fusion_k
        self.graph_top_k = graph_top_k
        # Use autocommit so each Cypher statement is its own transaction.
        # The alternative — wrapping the whole bulk-ingest in one transaction
        # — produces a too-large transaction when extracting ~12K entities
        # across 238 docs, and any syntax error in one entity name (special
        # chars in the dictionary) aborts the whole transaction. Per-statement
        # commits cost ~1ms each in latency but make error recovery trivial.
        self._age_conn = psycopg2.connect(dsn)
        self._age_conn.autocommit = True
        self._init_age()

    def _init_age(self) -> None:
        cur = self._age_conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS age CASCADE")
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, public")
        # Create graph if missing (idempotent on retry).
        cur.execute(
            "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s",
            (self.graph_name,),
        )
        if cur.fetchone() is None:
            cur.execute(f"SELECT create_graph('{self.graph_name}')")
        # autocommit=True — no explicit commit

    def _cypher(self, query: str, fetch: bool = True) -> list:
        """Execute a Cypher statement; return result rows (parsed from agtype).

        With autocommit=True (set in __init__), each call is its own implicit
        transaction. A syntax error raises but doesn't poison subsequent calls.
        """
        cur = self._age_conn.cursor()
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, public")
        cur.execute(
            f"SELECT * FROM cypher('{self.graph_name}', $${query}$$) AS (r agtype)"
        )
        if fetch:
            return [row[0] for row in cur.fetchall()]
        return []

    def truncate_graph(self) -> None:
        """Drop and recreate the graph — clears all nodes + edges.

        With autocommit=True each statement self-commits.
        """
        cur = self._age_conn.cursor()
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, public")
        cur.execute(
            "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s",
            (self.graph_name,),
        )
        if cur.fetchone() is not None:
            cur.execute(f"SELECT drop_graph('{self.graph_name}', true)")
        cur.execute(f"SELECT create_graph('{self.graph_name}')")

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        """TRUNCATE + bulk upsert + 3-pass AGE batched write-through.

        AGE's Cypher dialect doesn't support ON CREATE SET / ON MATCH SET
        on MERGE statements. We avoid them by truncating the graph fresh
        and using plain CREATE in three passes:
          1. CREATE Entity nodes (unique names across all docs)
          2. CREATE Drawer nodes (one per corpus entry)
          3. CREATE MENTIONED_IN edges
        """
        self.truncate_graph()
        result = super().ingest_corpus(corpus)

        # Pass 0: extract entities from every doc, build global inventories.
        all_entities: dict[str, str] = {}  # name -> type
        drawer_entities: dict[str, list] = {}  # drawer_id -> [(name, count)]
        for entry in corpus:
            es = self.extractor(entry["document"])
            drawer_entities[entry["id"]] = [(e.name, e.type, e.count) for e in es]
            for e in es:
                all_entities.setdefault(e.name, e.type)

        # Pass 1: CREATE Entity nodes (unique).
        for name, etype in all_entities.items():
            name_esc = name.replace("'", "\\'")
            etype_esc = etype.replace("'", "\\'")
            self._cypher(
                f"CREATE (:Entity {{name: '{name_esc}', type: '{etype_esc}'}})",
                fetch=False,
            )

        # Pass 2: CREATE Drawer nodes.
        for drawer_id in drawer_entities:
            d_esc = drawer_id.replace("'", "\\'")
            self._cypher(
                f"CREATE (:Drawer {{id: '{d_esc}'}})",
                fetch=False,
            )

        # Pass 3: CREATE MENTIONED_IN edges.
        for drawer_id, entities in drawer_entities.items():
            d_esc = drawer_id.replace("'", "\\'")
            for name, _etype, count in entities:
                name_esc = name.replace("'", "\\'")
                self._cypher(
                    f"""
                    MATCH (e:Entity {{name: '{name_esc}'}}), (d:Drawer {{id: '{d_esc}'}})
                    CREATE (e)-[:MENTIONED_IN {{count: {count}}}]->(d)
                    """,
                    fetch=False,
                )

        # autocommit=True — no explicit commit needed; each statement self-commits.
        return result

    def _graph_candidates(self, question: str) -> dict[str, float]:
        """Extract entities from the query, walk the graph, return
        {drawer_id: score} where score is sum-of-mention-counts across
        all query-anchor entities. Returns empty dict if no anchors found.

        Uses direct cursor.execute (not _cypher) because we need a 2-column
        agtype return (drawer_id, mention_count) which _cypher's
        '(r agtype)' annotation can't express.
        """
        query_entities = self.extractor(question)
        if not query_entities:
            return {}
        # AGE rejects multi-column RETURN and list literals inside cypher() —
        # return drawer ids only and weight by query-entity overlap count.
        # Cost: lose mention-count weighting (a drawer mentioning an entity 3x
        # ranks same as 1x); gain: works against current AGE.
        scores: defaultdict[str, float] = defaultdict(float)
        for qe in query_entities:
            name_esc = qe.name.replace("'", "\\'")
            cur = self._age_conn.cursor()
            try:
                cur.execute("LOAD 'age'")
                cur.execute("SET search_path = ag_catalog, public")
                cur.execute(
                    f"""
                    SELECT * FROM cypher('{self.graph_name}', $$
                        MATCH (e:Entity {{name: '{name_esc}'}})-[r:MENTIONED_IN]->(d:Drawer)
                        RETURN d.id
                    $$) AS (drawer_id agtype)
                    """
                )
                for (raw,) in cur.fetchall():
                    s = str(raw).strip('"')
                    if "::" in s:
                        s = s.rsplit("::", 1)[0].strip('"')
                    scores[s] += 1
            except Exception:
                # Skip this entity; autocommit means no transaction to roll back.
                continue
        return dict(sorted(scores.items(), key=lambda x: -x[1])[: self.graph_top_k])

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        route: bool = False,
    ) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        mode = self.retrieval_mode

        # Get vector ranking (parent class behavior).
        vector_result = super().query(question, n_results=max(k * 4, 20))
        # Build {drawer_id: rank} from vector result.
        vector_ranks: dict[str, int] = {}
        for i, ent in enumerate(vector_result.retrieved_entities or []):
            did = ent.id.removeprefix("chunk:")
            vector_ranks[did] = i

        if mode == "vector_only":
            return self._wrap_query_result(vector_result.retrieved_entities[:k])

        # Get graph candidates.
        graph_scores = self._graph_candidates(question)
        graph_ranks = {did: i for i, did in enumerate(graph_scores.keys())}

        if mode == "graph_only":
            ranked = sorted(graph_scores.items(), key=lambda x: -x[1])[:k]
            entities = [self._entity_for_drawer(did, sim=score / 100) for did, score in ranked]
            return self._wrap_query_result(entities)

        # Fusion mode: RRF combine vector_ranks + graph_ranks.
        union = set(vector_ranks) | set(graph_ranks)
        rrf: dict[str, float] = {}
        for did in union:
            score = 0.0
            if did in vector_ranks:
                score += 1.0 / (self.fusion_k + vector_ranks[did])
            if did in graph_ranks:
                score += 1.0 / (self.fusion_k + graph_ranks[did])
            rrf[did] = score
        fused = sorted(rrf.items(), key=lambda x: -x[1])[:k]
        # Re-look-up entity objects from vector_result (it has the docs+metas).
        vector_by_id = {
            ent.id.removeprefix("chunk:"): ent
            for ent in vector_result.retrieved_entities or []
        }
        out_entities: list[Entity] = []
        for did, score in fused:
            if did in vector_by_id:
                out_entities.append(vector_by_id[did])
            else:
                out_entities.append(self._entity_for_drawer(did, sim=score))
        return self._wrap_query_result(out_entities)

    def _entity_for_drawer(self, drawer_id: str, sim: float = 0.0) -> Entity:
        """Build an Entity reference for a drawer found via graph but not in vector top-K."""
        return Entity(
            id=f"chunk:{drawer_id}",
            name=drawer_id,
            entity_type="chunk",
            properties={
                "_table": self.table_name,
                "similarity": float(sim),
                "source_file": drawer_id,
                "via": "graph",
            },
        )

    def _wrap_query_result(self, entities: list[Entity]) -> QueryResult:
        """Rebuild context_string + answer from a list of Entity objects."""
        if not entities:
            return QueryResult(answer="", context_string="", error="NO_RESULTS")
        parts: list[str] = []
        for i, e in enumerate(entities):
            src = (e.properties or {}).get("source_file") or e.name
            parts.append(f"[{i + 1}] {src}")
        ctx = "\n\n".join(parts)
        return QueryResult(
            answer=ctx,
            context_string=ctx,
            retrieved_entities=entities,
        )

    def close(self) -> None:
        return  # no-op, see parent's docstring

    def shutdown(self) -> None:
        super().shutdown()
        try:
            if self._age_conn:
                self._age_conn.close()
        except Exception:
            pass
        self._age_conn = None
