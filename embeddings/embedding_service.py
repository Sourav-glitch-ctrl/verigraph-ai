import logging
from typing import List

from embeddings.embedding_model import EmbeddingModel
from ingestion.models import Document

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Generates embeddings for document chunks and binds them together 
    with their metadata for vector database ingestion.
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel()

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Takes a list of Document objects, generates embeddings for their text,
        and injects the resulting vectors directly into their metadata.
        """
        if not documents:
            return []

        logger.info(f"Generating embeddings for {len(documents)} document chunks...")
        
        try:
            # 1. Extract text from all documents
            texts = [doc.page_content for doc in documents]
            
            # 2. Generate embeddings in a single optimized batch
            embeddings = self.embedding_model.encode_batch(texts)
            
            # 3. Attach embeddings back into the document's metadata
            for doc, vector in zip(documents, embeddings):
                doc.metadata["embedding"] = vector

            logger.info("Embeddings successfully generated and attached.")
            return documents
        except Exception as e:
            logger.error(f"Failed to process documents through embedding service: {e}")
            raise

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Legacy method: Generate embeddings for a list of raw text strings.
        """
        return self.embedding_model.encode_batch(chunks)