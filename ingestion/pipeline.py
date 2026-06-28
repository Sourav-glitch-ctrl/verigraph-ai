import copy
import json
import logging
from pathlib import Path
from typing import List, Optional

from embeddings.embedding_service import EmbeddingService
from ingestion.models import Document
from ingestion.pdf_loader import PDFLoader
from ingestion.docx_loader import DOCXLoader
from ingestion.preprocess import TextPreprocessor
from ingestion.chunker import TextChunker

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    A unified factory for discovering, loading, preprocessing,
    and chunking documents from a directory.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, output_dir: Optional[str] = None):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "data" / "processed data"

    def load_directory(self, directory_path: str, save_chunks: bool = True) -> List[Document]:
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        all_chunks: List[Document] = []

        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                chunks = self.process_file(str(file_path))
                all_chunks.extend(chunks)

        if save_chunks and all_chunks:
            self.save_chunks_to_json(all_chunks)

        return all_chunks

    def process_file(self, file_path: str) -> List[Document]:
        """
        Loads a single file, cleans it, and splits it into chunked Documents.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        try:
            if ext == ".pdf":
                loader = PDFLoader(file_path)
            elif ext == ".docx":
                loader = DOCXLoader(file_path)
            else:
                logger.debug(f"Unsupported file extension skipped: {file_path}")
                return []

            # 1. Load the raw documents
            docs = loader.load()

            # 2. Preprocess and chunk
            file_chunks = []
            for doc in docs:
                cleaned_doc = TextPreprocessor.clean_document(doc)
                chunks = self.chunker.split_document(cleaned_doc)
                file_chunks.extend(chunks)

            return file_chunks

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return []

    def save_chunks_to_json(self, documents: List[Document], output_dir: Optional[str] = None, filename: str = "chunks.json") -> str:
        """
        Persist chunk text and metadata to disk without storing embeddings.
        """
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        payload = []
        for doc in documents:
            metadata = copy.deepcopy(doc.metadata)
            metadata.pop("embedding", None)
            payload.append({
                "page_content": doc.page_content,
                "metadata": metadata,
            })

        output_path = target_dir / filename
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        return str(output_path)

    def load_chunks_from_json(self, json_path: Optional[str] = None) -> List[Document]:
        """
        Load previously saved chunks from disk.
        """
        path = Path(json_path) if json_path else self.output_dir / "chunks.json"
        if not path.exists():
            raise FileNotFoundError(f"Chunk file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return [
            Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {}))
            for item in payload
        ]

    def create_embeddings_from_saved_chunks(
        self,
        json_path: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> List[Document]:
        """
        Load saved chunks, generate embeddings in memory, and do not write embeddings to disk.

        Args:
            json_path: path to the saved chunks JSON file
            embedding_service: optional service instance to reuse embedding model loading
        """
        chunks = self.load_chunks_from_json(json_path)
        if not chunks:
            return []

        service = embedding_service or EmbeddingService()
        return service.process_documents(chunks)
