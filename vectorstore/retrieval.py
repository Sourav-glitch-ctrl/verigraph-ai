import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """
    Represents a retrieved text chunk with its identifier, content,
    relevance score, and associated metadata.
    """
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the retrieved chunk to a dictionary format."""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


class SearchBackend(ABC):
    """
    Abstract interface for search backends (e.g. vector databases, keyword search indexes).
    """

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Query the search backend for relevant documents.

        Args:
            query: The search query text.
            top_k: The number of results to retrieve.
            where: Optional metadata filter parameters.
            **kwargs: Backend-specific arguments.

        Returns:
            A list of dictionary objects representing retrieved hits.
            Each dictionary must contain:
            - 'id': str
            - 'document': str
            - 'score': float
            - 'metadata': dict
        """
        pass


class BasePostProcessor(ABC):
    """
    Abstract interface for post-retrieval processing steps,
    such as re-ranking, duplicate filtering, or chunk merging.
    """

    @abstractmethod
    def process(
        self, chunks: List[RetrievedChunk], query: str
    ) -> List[RetrievedChunk]:
        """
        Post-process retrieved chunks.

        Args:
            chunks: List of retrieved chunks.
            query: The original search query.

        Returns:
            Processed list of retrieved chunks.
        """
        pass


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers in the application.
    Defines the unified retrieval public API.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks relevant to the query.

        Args:
            query: The search query.
            top_k: Max number of chunks to return.
            filters: Optional metadata filters.
            **kwargs: Extra parameters passed down to backends or postprocessors.

        Returns:
            A list of RetrievedChunk objects.
        """
        pass


class StandardRetriever(BaseRetriever):
    """
    Production-ready, database-agnostic retriever that validates queries,
    delegates retrieval to a SearchBackend, and optionally runs post-processors.
    """

    def __init__(
        self,
        backend: SearchBackend,
        post_processors: Optional[List[BasePostProcessor]] = None,
    ):
        """
        Initialize the retriever.

        Args:
            backend: Concrete search database backend implementation.
            post_processors: Ordered list of post-retrieval processing steps (e.g. rerankers).
        """
        self.backend = backend
        self.post_processors = post_processors or []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant document chunks, validating parameters and handling backend exceptions.

        Args:
            query: The query text string.
            top_k: Number of results to return (must be > 0).
            filters: Optional dictionary of metadata filters.
            **kwargs: Additional parameters passed to backend or post-processors.

        Returns:
            A list of RetrievedChunk instances.

        Raises:
            ValueError: If query is invalid or top_k is non-positive.
            RuntimeError: If the search backend raises an exception.
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        clean_query = query.strip()
        if not clean_query:
            logger.warning("Empty search query provided to retrieve(). Returning empty list.")
            return []

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        # If post-processors exist, fetch a larger pool of candidates to allow re-ranking/filtering
        candidate_k = top_k * 2 if self.post_processors else top_k

        try:
            logger.info("Executing search query using backend...")
            raw_hits = self.backend.search(
                query=clean_query, top_k=candidate_k, where=filters, **kwargs
            )
        except Exception as exc:
            logger.exception("Search backend query execution failed.")
            raise RuntimeError(
                "Failed to retrieve documents due to an error in the search backend."
            ) from exc

        chunks: List[RetrievedChunk] = []
        for hit in raw_hits:
            if not isinstance(hit, dict):
                logger.error(f"Search backend returned invalid item type: {type(hit)}")
                continue

            try:
                chunk = RetrievedChunk(
                    id=str(hit.get("id", "")),
                    text=str(hit.get("document", "")),
                    score=float(hit.get("score", 0.0)),
                    metadata=dict(hit.get("metadata") or {}),
                )
                chunks.append(chunk)
            except (ValueError, TypeError) as exc:
                logger.warning(f"Skipping malformed search hit dictionary: {hit}. Error: {exc}")

        # Execute post-processing pipeline sequentially
        for processor in self.post_processors:
            try:
                chunks = processor.process(chunks, clean_query)
            except Exception:
                logger.exception(f"Post-processor {processor.__class__.__name__} failed. Continuing...")

        # Final slice to limit the results to the requested top_k
        return chunks[:top_k]
