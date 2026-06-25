import logging
from typing import List

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Handles robust loading and inference for the embedding model.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        logger.info(f"Loading SentenceTransformer model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            raise RuntimeError(f"Could not load SentenceTransformer model: {model_name}") from e

    def encode(self, text: str) -> List[float]:
        """
        Generate an embedding for a single piece of text.
        """
        if not text or not text.strip():
            return []

        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error encoding single text: {e}")
            raise

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts simultaneously.
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error encoding batch of {len(texts)} texts: {e}")
            raise