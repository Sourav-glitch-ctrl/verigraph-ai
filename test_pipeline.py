import logging
import json
from ingestion.pdf_loader import PDFLoader
from ingestion.preprocess import TextPreprocessor
from ingestion.chunker import TextChunker
from embeddings.embedding_service import EmbeddingService

# Set up basic logging so we can see the internal info messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_pdf_pipeline():
    pdf_path = r"C:\Users\soura\Documents\Projects\VeriGraph_AI\data\raw\company_policy.pdf"

    print("\n--- 1. Loading PDF ---")
    loader = PDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages as Document objects.")

    print("\n--- 2. Preprocessing & Chunking ---")
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    all_chunks = []
    
    for doc in docs:
        cleaned_doc = TextPreprocessor.clean_document(doc)
        chunks = chunker.split_document(cleaned_doc)
        all_chunks.extend(chunks)

    print(f"Split pages into {len(all_chunks)} chunks while preserving metadata.")

    print("\n--- 3. Generating Embeddings ---")
    service = EmbeddingService()
    # This will batch process all chunks and attach the vectors to the metadata
    embedded_docs = service.process_documents(all_chunks)

    print("\n--- Final System Output (Sample) ---")
    if embedded_docs:
        sample = embedded_docs[0]
        print(f"Text Preview:\n{sample.page_content[:150]}...\n")
        
        # We temporarily remove the massive embedding vector from the printout just so we can read the other metadata
        meta_preview = {k: v for k, v in sample.metadata.items() if k != 'embedding'}
        print(f"Citation Metadata:\n{json.dumps(meta_preview, indent=2)}\n")
        
        # Verify the embedding vector exists
        embedding = sample.metadata.get('embedding', [])
        print(f"Embedding Vector Generated? {'Yes' if embedding else 'No'}")
        print(f"Vector Dimensions: {len(embedding)}")
        print(f"Vector Data Preview: {embedding[:5]} ...")

if __name__ == "__main__":
    test_pdf_pipeline()
