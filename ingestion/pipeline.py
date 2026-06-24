import logging
from pathlib import Path
from typing import List

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

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_directory(self, directory_path: str) -> List[Document]:
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        all_chunks: List[Document] = []

        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                chunks = self.process_file(str(file_path))
                all_chunks.extend(chunks)

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
