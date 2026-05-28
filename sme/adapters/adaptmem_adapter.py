"""AdaptMem SME adapter (closes #81).

Wraps a pre-trained ``adaptmem.AdaptMem`` model as an SMEAdapter so the
candidate-strategy / cross-validate harnesses can drive AdaptMem under
the same contract as every other system under test.

Design choice (mirrors ``FlatBaselineAdapter``): training is out of
band. Load a model trained via the AdaptMem CLI (or `am.save(path)`)
and point the adapter at it. ``ingest_corpus`` is NotImplementedError —
use AdaptMem's own scripts to train + persist, then run benches
against the saved artifact.

Usage:
    adapter = AdaptMemAdapter("/path/to/saved-encoder", n_results=5)
    result = adapter.query("what did Susan say about Apache Kafka?")
    for ent in result.retrieved_entities:
        print(ent.id, ent.properties.get("score"), ent.name)

``adaptmem`` is an optional install. Install via:

    pip install adaptmem
    # or
    pip install -e /path/to/adaptmem-repo
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sme.adapters.base import Edge, Entity, QueryResult, SMEAdapter

log = logging.getLogger(__name__)


class AdaptMemAdapter(SMEAdapter):
    """Pre-trained AdaptMem encoder as a flat-retrieval SMEAdapter.

    No graph traversal, no metadata filtering. Returns top-K nearest
    neighbours under the domain-tuned encoder's embedding space.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_results: int = 5,
        rerank: bool = False,
    ) -> None:
        try:
            from adaptmem import AdaptMem
        except ImportError as e:
            raise RuntimeError(
                "AdaptMemAdapter requires the `adaptmem` package. "
                "Install with `pip install adaptmem` or `pip install -e "
                "/path/to/adaptmem-repo`."
            ) from e

        self.model_path = str(model_path)
        self.n_results = n_results
        self.rerank = rerank

        log.info("loading AdaptMem model from %s (rerank=%s)",
                 self.model_path, self.rerank)
        self._am = AdaptMem.load(self.model_path)
        if self.rerank and not getattr(self._am, "rerank_enabled", False):
            log.warning(
                "rerank=True but AdaptMem.load() returned an instance with "
                "rerank disabled; the saved artifact may need re-saving "
                "with rerank=True. Proceeding without rerank."
            )

    # --- required SMEAdapter methods ----------------------------------

    def ingest_corpus(self, corpus: list[dict]) -> dict:
        raise NotImplementedError(
            "AdaptMemAdapter reads a pre-trained model. To ingest a new "
            "corpus, train via AdaptMem's CLI/scripts (e.g. "
            "`adaptmem train --corpus ... --labelled ...`) and pass the "
            "saved model path to AdaptMemAdapter(model_path=...)."
        )

    def query(self, question: str, n_results: Optional[int] = None) -> QueryResult:
        k = n_results if n_results is not None else self.n_results
        try:
            hits = self._am.search(question, top_k=k)
        except Exception as e:  # noqa: BLE001 — adapter must not crash the harness
            log.error("AdaptMem search failed: %s", e)
            return QueryResult(
                answer="",
                context_string="",
                retrieved_entities=[],
                retrieved_edges=[],
                error=f"AdaptMem.search raised {type(e).__name__}: {e}",
            )

        entities: list[Entity] = []
        context_parts: list[str] = []
        for i, h in enumerate(hits):
            chunk_id = str(getattr(h, "chunk_id", f"adaptmem_hit:{i}"))
            text = getattr(h, "text", "") or ""
            score = float(getattr(h, "score", 0.0))
            entities.append(
                Entity(
                    id=chunk_id,
                    name=chunk_id,
                    entity_type="adaptmem_hit",
                    properties={
                        "_table": "adaptmem_hit",
                        "score": score,
                        "rank": i + 1,
                        "text": text,
                    },
                )
            )
            context_parts.append(f"[{i + 1}] {chunk_id}\n{text}")

        context_string = "\n\n".join(context_parts)
        return QueryResult(
            answer="",
            context_string=context_string,
            retrieved_entities=entities,
            retrieved_edges=[],
        )

    def get_graph_snapshot(self) -> tuple[list[Entity], list[Edge]]:
        # Flat retrieval — no graph. Mirrors FlatBaselineAdapter.
        return [], []
