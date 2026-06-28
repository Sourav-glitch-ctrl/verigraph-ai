import logging
import os
from pathlib import Path
from typing import List, Optional

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Handles robust loading and inference for the embedding model.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", local_model_dir: Optional[str] = None):
        self.model_name = model_name
        self.local_model_dir = local_model_dir or self._resolve_local_model_dir()
        logger.info(f"Loading SentenceTransformer model: {model_name}...")
        try:
            if self.local_model_dir and Path(self.local_model_dir).exists():
                logger.info(f"Using local model directory: {self.local_model_dir}")
                self.model = SentenceTransformer(self.local_model_dir)
            else:
                self.model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            raise RuntimeError(f"Could not load SentenceTransformer model: {model_name}") from e

    def _resolve_local_model_dir(self) -> Optional[str]:
        """Try common local cache locations for a downloaded sentence-transformers model."""
        candidates = []
        env_overrides = [os.getenv("HF_HOME"), os.getenv("HUGGINGFACE_HUB_CACHE")]
        for value in env_overrides:
            if value:
                candidates.append(Path(value) / "models--BAAI--bge-small-en-v1.5")
                candidates.append(Path(value) / "models--BAAI--bge-small-en-v1.5" / "snapshots")
                candidates.append(Path(value) / "BAAI" / "bge-small-en-v1.5")

        candidates.extend([
            Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-en-v1.5",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-en-v1.5" / "snapshots",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-en-v1.5" / "snapshots" / "*",
            Path("models") / "bge-small-en-v1.5",
            Path("data") / "models" / "bge-small-en-v1.5",
        ])

        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.is_dir() and (candidate / "config.json").exists():
                return str(candidate)
            if candidate.is_dir() and any(child.is_dir() for child in candidate.iterdir()):
                for child in sorted(candidate.iterdir()):
                    if child.is_dir() and (child / "config.json").exists():
                        return str(child)

        return None

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