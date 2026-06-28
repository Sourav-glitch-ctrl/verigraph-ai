import json
import logging
from pathlib import Path
from typing import List, Optional

import argparse
import platform
import shutil
import sys
import tempfile

from embeddings.embedding_model import EmbeddingModel
from embeddings.embedding_service import EmbeddingService
from ingestion.pipeline import IngestionPipeline
from vectorstore.chroma_db import ChromaDBManager
from vectorstore.similarity_search import SimilaritySearch, ChromaSearchBackend
from vectorstore.retrieval import StandardRetriever


# Set up basic logging so we can see the internal info messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_pdf_pipeline(mock_embed: bool = False, persist_directory: Optional[str] = None, cleanup: bool = True):
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data" / "raw"

    print("\n--- 1. Loading documents from data/raw ---")
    pipeline = IngestionPipeline(chunk_size=300, chunk_overlap=50)
    all_chunks = []

    for file_path in sorted(raw_dir.glob("*")):
        if not file_path.is_file():
            continue
        print(f"Processing: {file_path.name}")
        chunks = pipeline.process_file(str(file_path))
        all_chunks.extend(chunks)

    print(f"Split into {len(all_chunks)} chunks while preserving metadata.")

    if not all_chunks:
        print("No raw files found or no supported files were processed. Exiting test.")
        return

    print("\n--- 2. Saving chunks to JSON ---")
    output_path = pipeline.save_chunks_to_json(all_chunks)
    print(f"Saved chunks to {output_path}")

    print("\n--- 3. Generating embeddings ---")

    if mock_embed:
        # lightweight deterministic embeddings for CI / fast local runs
        chunks = pipeline.load_chunks_from_json(output_path)
        for doc in chunks:
            vec = [float(len(doc.page_content) % 7 + 1) for _ in range(8)]
            doc.metadata["embedding"] = vec
        embedded_docs = chunks
        embedding_service = None
        print("Using mock embeddings (fast, deterministic).")
    else:
        embedding_service = EmbeddingService()
        embedded_docs = pipeline.create_embeddings_from_saved_chunks(output_path, embedding_service=embedding_service)

    # prepare persistence directory
    if persist_directory:
        persist_dir = persist_directory
    else:
        persist_dir = tempfile.mkdtemp(prefix="verigraph_embeddings_")

    print(f"\n--- 4. Inserting into ChromaDB (persist_directory={persist_dir}) ---")
    vector_store = ChromaDBManager(persist_directory=persist_dir)
    count_before = vector_store.count()
    inserted = vector_store.add_chunk_documents(
        chunks=embedded_docs,
        embeddings=[doc.metadata.get("embedding", []) for doc in embedded_docs],
    )
    count_after = vector_store.count()

    print(f"ChromaDB records before insert: {count_before}")
    print(f"ChromaDB records after insert: {count_after}")
    print(f"Inserted into ChromaDB: {inserted}")

    print("\n--- 5. Semantic Search Verification ---")

    if mock_embed:
        class MockEmbeddingModel(EmbeddingModel):
            def __init__(self):
                # Bypass parent class __init__ to avoid loading sentence-transformers model
                pass

            def encode(self, text: str) -> List[float]:
                return [float(len(text) % 7 + 1) for _ in range(8)]

            def encode_batch(self, texts: List[str]) -> List[List[float]]:
                return [self.encode(text) for text in texts]

        similarity_search = SimilaritySearch(db=vector_store, embedding_model=MockEmbeddingModel())
    else:
        assert embedding_service is not None
        similarity_search = SimilaritySearch(db=vector_store, embedding_model=embedding_service.embedding_model)

    backend = ChromaSearchBackend(similarity_search)
    retriever = StandardRetriever(backend)

    test_queries = [
        "How many paid leave days do employees receive?",
        "What is the password policy?",
        "Is remote work allowed?",
        "What is the confidentiality policy?",
        "How is overtime calculated?"
    ]

    for query in test_queries:
        print("\n" + "=" * 90)
        print(f"Query: {query}")

        results = retriever.retrieve(query, top_k=3)

        for rank, chunk in enumerate(results, start=1):
            preview = chunk.text.replace("\n", " ") if isinstance(chunk.text, str) else str(chunk.text)
            preview = preview[:300] + ("..." if len(preview) > 300 else "")

            print(f"\nResult {rank}")
            print(f"ID        : {chunk.id}")
            print(f"Score     : {chunk.score:.4f}")
            
            # Print metadata and document safely to avoid UnicodeEncodeError on Windows terminal
            metadata_str = f"Metadata  : {chunk.metadata}"
            doc_str = f"Document  : {preview}"
            for line in [metadata_str, doc_str]:
                try:
                    print(line)
                except UnicodeEncodeError:
                    encoding = sys.stdout.encoding or 'utf-8'
                    print(line.encode(encoding, errors='replace').decode(encoding))

    if cleanup and not persist_directory:
        print(f"\n--- Cleaning up temporary embed directory: {persist_dir}")
        try:
            # attempt to delete the chroma collection first to release file handles
            try:
                vector_store.delete_collection()
            except Exception:
                pass

            shutil.rmtree(persist_dir)
            print("Cleanup successful.")
        except Exception as e:
            print(f"Cleanup failed: {e}")


def _parse_args():
    p = argparse.ArgumentParser(description="Run end-to-end ingestion+embedding+persist test")
    p.add_argument("--mock-embed", action="store_true", help="Use deterministic mock embeddings (fast)")
    p.add_argument("--persist-dir", type=str, default=None, help="Custom Chroma persist directory (skip cleanup)")
    p.add_argument("--no-cleanup", dest="cleanup", action="store_false", help="Do not cleanup temp persist dir")
    # On Windows cleanup often fails due to file locks; default to no cleanup there
    default_cleanup = False if platform.system() == "Windows" else True
    p.set_defaults(cleanup=default_cleanup)
    return p.parse_args()


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            reconfigure = getattr(sys.stdout, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding='utf-8')
        except Exception:
            pass
    args = _parse_args()
    test_pdf_pipeline(mock_embed=args.mock_embed, persist_directory=args.persist_dir, cleanup=args.cleanup)
