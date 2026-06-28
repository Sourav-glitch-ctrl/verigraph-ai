from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from embeddings.embedding_model import EmbeddingModel
from vectorstore.chroma_db import ChromaDBManager


def _default_score_fn(distance: float) -> float:
    """Convert a non-negative distance to a similarity-like score in (0, 1].

    Uses the function score = 1 / (1 + distance) which is robust for many
    distance metrics returned by vectorstores.
    """
    try:
        return 1.0 / (1.0 + float(distance))
    except Exception:
        return 0.0


class SimilaritySearch:
    """Performs semantic similarity search on ChromaDB.

    Upgrades over the simple wrapper:
    - Accepts optional `where` metadata filters and `include` passthrough
    - Supports batch queries (list of strings) or single query
    - Returns structured, normalized result items with `score` computed from
      the returned `distance` using a stable default function
    """

    def __init__(
        self,
        collection_name: str = "verigraph_collection",
        db: Optional[ChromaDBManager] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.db = db if db is not None else ChromaDBManager(collection_name=collection_name)

    def _prepare_query_embedding(self, query: Union[str, Sequence[str]]) -> List[List[float]]:
        if isinstance(query, str):
            return [self.embedding_model.encode(query)]
        return [self.embedding_model.encode(q) for q in query]

    def search(
        self,
        query: Union[str, Sequence[str]],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        score_fn: Optional[Callable[[float], float]] = None,
    ) -> Dict[str, Any]:
        """Search for the most similar document chunks.

        Args:
            query: single query string or sequence of query strings
            top_k: number of results to return per query
            where: optional metadata filter passed to ChromaDB
            include: optional include list passed to ChromaDB
            score_fn: optional callable(distance) -> score. If omitted a
                stable default is used.

        Returns:
            A dictionary with keys:
            - `query_embeddings`: list of embeddings used
            - `results`: list (per query) of result item dicts with keys
              `id`, `document`, `metadata`, `distance`, `score`, `rank`
        """

        score_fn = score_fn or _default_score_fn

        query_embeddings = self._prepare_query_embedding(query)

        raw = self.db.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where=where,
            include=include,
        )

        # Chroma returns parallel lists for ids/documents/metadatas/distances
        results: List[List[Dict[str, Any]]] = []

        ids_lists = raw.get("ids") or raw.get("ids", [])
        documents_lists = raw.get("documents") or []
        metadatas_lists = raw.get("metadatas") or []
        distances_lists = raw.get("distances") or []

        # Normalize shapes: chroma returns nested lists (queries x results)
        for q_index in range(len(query_embeddings)):
            q_ids = ids_lists[q_index] if q_index < len(ids_lists) else []
            q_docs = documents_lists[q_index] if q_index < len(documents_lists) else []
            q_metas = metadatas_lists[q_index] if q_index < len(metadatas_lists) else []
            q_dists = distances_lists[q_index] if q_index < len(distances_lists) else []

            q_results: List[Dict[str, Any]] = []
            for rank, (rid, doc, meta, dist) in enumerate(zip(q_ids, q_docs, q_metas, q_dists), start=1):
                score = score_fn(dist) if dist is not None else 0.0
                q_results.append(
                    {
                        "rank": rank,
                        "id": rid,
                        "document": doc,
                        "metadata": meta,
                        "distance": dist,
                        "score": score,
                    }
                )

            results.append(q_results)

        return {"query_embeddings": query_embeddings, "results": results, "raw": raw}

    # convenience alias for single-query flows
    def search_one(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        score_fn: Optional[Callable[[float], float]] = None,
    ) -> List[Dict[str, Any]]:
        return self.search(query, top_k=top_k, where=where, include=include, score_fn=score_fn)["results"][0]
