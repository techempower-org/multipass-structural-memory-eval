"""PostgreSQL-backed SME adapter that ingests a corpus per-query.

Designed for LongMemEval-shape benchmarks where each question has its own
small haystack and the adapter must reset between questions. Uses
mempalace.backends.postgres.PostgresCollection so the retrieval path
matches the fork's production substrate (postgres + pgvector + the same
MiniLM-L6-v2 embedding as upstream MemPalace's raw baseline).

Comparison anchor: upstream's longmemeval_bench --mode raw lands at
R@5=0.966 on longmemeval_s_cleaned.json using ChromaDB with the same
MiniLM embeddings. Running this adapter on the same dataset isolates
the backend swap (chroma -> postgres+pgvector) from every other
variable. Equal recall validates the migration; a gap quantifies the
regression.

Not intended for production use. The adapter TRUNCATEs the working table
on every ingest_corpus call so per-question vaults don't bleed into
each other.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from sme.adapters.base import Entity, QueryResult, SMEAdapter

log = logging.getLogger(__name__)

DEFAULT_TABLE = "lme_bench_drawers"
_DSN_ENV_VAR = "SME_POSTGRES_DSN"


def _resolve_dsn(dsn: Optional[str] = None) -> str:
    """Resolve the postgres DSN from arg → env var → fail.

    Never hardcode credentials in source. The previous default DSN
    embedded a live percent-encoded password and internal IP and was
    flagged as a security issue (closed #1, refiled if reintroduced).
    """
    if dsn:
        return dsn
    env_dsn = os.environ.get(_DSN_ENV_VAR)
    if env_dsn:
        return env_dsn
    raise RuntimeError(
        f"PostgresIngestAdapter requires either an explicit `dsn=` kwarg "
        f"or the {_DSN_ENV_VAR} environment variable; no default DSN is "
        f"hardcoded in source (see closed fork issue #1)."
    )


class PostgresIngestAdapter(SMEAdapter):
    """Per-question ingest into a postgres+pgvector collection."""

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        table_name: str = DEFAULT_TABLE,
        n_results: int = 5,
        mempalace_path: Optional[str] = None,
        read_only: bool = False,
    ) -> None:
        mp_root = mempalace_path or "/home/jp/Projects/memorypalace"
        if mp_root not in sys.path:
            sys.path.insert(0, mp_root)

        from mempalace.backends.postgres import PostgresCollection

        resolved_dsn = _resolve_dsn(dsn)
        self._collection_cls = PostgresCollection
        self.dsn = resolved_dsn
        self.table_name = table_name
        self.n_results = n_results
        self._collection = PostgresCollection(dsn=resolved_dsn, table_name=table_name)
        self._collection._ensure_setup(create=True)

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        self._truncate()
        ids = [c["id"] for c in corpus]
        documents = [c["document"] for c in corpus]
        metadatas = [c.get("metadata") or {} for c in corpus]
        if not documents:
            return {"ingested": 0}
        self._collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )
        return {"ingested": len(documents)}

    def ingest_from_vault(self, vault_dir: Path) -> dict:
        vault_dir = Path(vault_dir)
        corpus: list[dict] = []
        for md_file in sorted(vault_dir.rglob("*.md")):
            if not md_file.is_file():
                continue
            corpus.append(
                {
                    "id": md_file.stem,
                    "document": md_file.read_text(encoding="utf-8", errors="replace"),
                }
            )
        return self.ingest_corpus(corpus)

    def _truncate(self) -> None:
        conn = self._collection._get_conn()
        cur = conn.cursor()
        cur.execute(f"TRUNCATE TABLE {self.table_name}")
        conn.commit()
        self._collection._local_row_estimate = 0
        self._collection._vector_index_ready = False

    def query(
        self,
        question: str,
        *,
        n_results: Optional[int] = None,
        route: bool = False,
    ) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        try:
            result = self._collection.query(
                query_texts=[question],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            return QueryResult(answer="", context_string="", error=f"INTERNAL: {e}")

        ids = (result.ids or [[]])[0]
        docs = (result.documents or [[]])[0]
        metas = (result.metadatas or [[]])[0]
        dists = (result.distances or [[]])[0]

        if not docs:
            return QueryResult(answer="", context_string="", error="NO_RESULTS")

        context_parts: list[str] = []
        retrieved: list[Entity] = []
        for i, (doc, meta, doc_id, dist) in enumerate(
            zip(docs, metas or [{}] * len(docs), ids, dists)
        ):
            meta = meta or {}
            source = meta.get("source_file") or meta.get("source") or doc_id
            source_label = Path(str(source)).name if source else f"hit{i}"
            context_parts.append(f"[{i + 1}] {source_label}\n{doc}")
            retrieved.append(
                Entity(
                    id=f"chunk:{doc_id}",
                    name=source_label,
                    entity_type="chunk",
                    properties={
                        "_table": self.table_name,
                        "similarity": 1.0 - float(dist),
                        "source_file": source,
                    },
                )
            )

        context_string = "\n\n".join(context_parts)
        return QueryResult(
            answer=context_string,
            context_string=context_string,
            retrieved_entities=retrieved,
        )

    def get_graph_snapshot(self):
        return [], []

    def get_flat_retrieval(self, question: str) -> QueryResult:
        return self.query(question)

    def get_ontology_source(self) -> dict:
        return {
            "type": "inferred",
            "schema": [],
            "documentation": (
                "Postgres-pgvector ingest adapter. Uses ChromaDB's MiniLM-L6-v2 "
                "embeddings for parity with the upstream raw baseline."
            ),
        }

    def close(self) -> None:
        # No-op: the singleton pattern in cross_validate_longmemeval reuses
        # this adapter across all questions to amortize connection + DDL.
        # Use shutdown() for the real close.
        return

    def shutdown(self) -> None:
        try:
            if self._collection and self._collection._conn:
                self._collection._conn.close()
        except Exception:
            pass
        self._collection = None
