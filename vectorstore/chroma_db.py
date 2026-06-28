import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import chromadb
from chromadb import Collection
from chromadb.api.types import Metadata
from chromadb.config import Settings

from ingestion.models import Document

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    Production-style ChromaDB wrapper for persistent vector storage.

    Responsibilities:
    - Initialize a persistent ChromaDB client
    - Create or reuse a collection with sensible metadata
    - Add or upsert documents with embeddings and metadata
    - Query stored vectors by text or embeddings
    - Support ingestion directly from project Document chunks
    """

    def __init__(
        self,
        persist_directory: str = "data/embeddings",
        collection_name: str = "verigraph_collection",
        collection_metadata: Optional[Dict[str, str]] = None,
        embedding_function: Optional[Any] = None,
    ):
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection_name = collection_name
        self.collection_metadata = collection_metadata or {"hnsw:space": "cosine"}
        self.embedding_function = embedding_function
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """Create or retrieve the target collection with safe defaults."""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=self.collection_metadata,
                embedding_function=self.embedding_function,
            )
        except Exception as exc:
            logger.exception("Failed to initialize Chroma collection '%s'", self.collection_name)
            raise RuntimeError(
                f"Unable to initialize Chroma collection '{self.collection_name}'"
            ) from exc

    def _normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure metadata is a plain dictionary and remove any embedding payloads."""
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be provided as a dictionary.")

        # pyrefly: ignore [unnecessary-type-conversion]
        normalized = {str(k): v for k, v in metadata.items() if k != "embedding"}
        return normalized

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> int:
        """Add documents and embeddings to ChromaDB."""
        if not ids:
            return 0

        if not (len(ids) == len(documents) == len(embeddings)):
            raise ValueError("ids, documents, and embeddings must have the same length.")

        if metadatas is None:
            metadatas = [{} for _ in ids]
        elif len(metadatas) != len(ids):
            raise ValueError("Metadata length must match ids.")

        # pyrefly: ignore [unnecessary-type-conversion]
        normalized_ids = [str(item) for item in ids]
        normalized_metadatas = cast(List[Metadata], [self._normalize_metadata(meta) for meta in metadatas])

        self.collection.add(
            ids=normalized_ids,
            documents=list(documents),
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=normalized_metadatas,
        )
        return len(normalized_ids)

    def upsert_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> int:
        """Upsert documents and embeddings into the collection."""
        if not ids:
            return 0

        if not (len(ids) == len(documents) == len(embeddings)):
            raise ValueError("ids, documents, and embeddings must have the same length.")

        if metadatas is None:
            metadatas = [{} for _ in ids]
        elif len(metadatas) != len(ids):
            raise ValueError("Metadata length must match ids.")

        # pyrefly: ignore [unnecessary-type-conversion]
        normalized_ids = [str(item) for item in ids]
        normalized_metadatas = cast(List[Metadata], [self._normalize_metadata(meta) for meta in metadatas])

        self.collection.upsert(
            ids=normalized_ids,
            documents=list(documents),
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=normalized_metadatas,
        )
        return len(normalized_ids)

    def add_chunk_documents(
        self,
        chunks: Sequence[Document],
        embeddings: Optional[List[List[float]]] = None,
        embedding_service: Optional[Any] = None,
    ) -> int:
        """Ingest project Document chunks and optionally generate embeddings through the embedding service."""
        if not chunks:
            return 0

        documents = list(chunks)
        if embeddings is None:
            if embedding_service is None:
                raise ValueError("Provide embeddings or an embedding_service to ingest chunks.")

            embedded_documents = embedding_service.process_documents(documents)
            documents = embedded_documents
            embeddings = [doc.metadata.get("embedding", []) for doc in documents]

        if len(embeddings) != len(documents):
            raise ValueError("Embedding count must match chunk count.")

        ids = []
        chunk_documents = []
        chunk_metadatas = []

        for index, chunk in enumerate(documents):
            metadata = self._normalize_metadata(getattr(chunk, "metadata", None))
            metadata.setdefault("source", "unknown")
            metadata.setdefault("chunk_index", index)
            ids.append(str(metadata.get("id") or f"chunk_{index}"))
            chunk_documents.append(chunk.page_content)
            chunk_metadatas.append(metadata)

        return self.add_documents(
            ids=ids,
            documents=chunk_documents,
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=chunk_metadatas,
        )

    def query(
        self,
        query_texts: Optional[Sequence[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query the collection for similar documents."""
        if query_texts is None and query_embeddings is None:
            raise ValueError("Provide either query_texts or query_embeddings.")

        payload: Dict[str, Any] = {
            "n_results": n_results,
            "where": where,
            "include": include or ["documents", "metadatas", "distances"],
        }
        if query_texts is not None:
            payload["query_texts"] = list(query_texts)
        if query_embeddings is not None:
            payload["query_embeddings"] = [list(item) for item in query_embeddings]

        return self.collection.query(**payload) # type: ignore

    def get_collection(self) -> Collection:
        """Return the active collection."""
        return self.collection

    def count(self) -> int:
        """Return the number of stored vectors."""
        return self.collection.count()

    def get(self, ids: Sequence[str]) -> Dict[str, Any]:
        """Fetch documents by id."""
        # pyrefly: ignore [unnecessary-type-conversion]
        return cast(Dict[str, Any], self.collection.get(ids=[str(item) for item in ids]))

    def delete(self, ids: Optional[Sequence[str]] = None, where: Optional[Dict[str, Any]] = None) -> None:
        """Delete items from the collection by id or metadata filter."""
        # pyrefly: ignore [unnecessary-type-conversion]
        self.collection.delete(ids=[str(item) for item in ids] if ids else None, where=where)

    def delete_collection(self) -> None:
        """Delete the current collection from disk."""
        self.client.delete_collection(self.collection_name)

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.delete_collection()
        self.collection = self._get_or_create_collection()