# VeriGraph AI

VeriGraph AI is a Python-based intelligent knowledge graph builder and validation system.

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <your-repository-url>
   cd VeriGraph_AI
   ```

2. **Set up the virtual environment:**

   ```bash
   python -m venv verigraph
   .\verigraph\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory and add your API keys:

   ```env
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key
   GROQ_API_KEY=your_groq_key
   ```

5. **Run the application:**

   ```bash
   python -m app.main
   ```

## Enterprise Ingestion Pipeline

VeriGraph AI includes an enterprise-ready document ingestion pipeline. It automatically handles PDF and DOCX files, cleans the text, and chunks it while strictly preserving metadata citations (such as source file paths and page numbers).

```python
from ingestion.pipeline import IngestionPipeline

# Initialize the pipeline
pipeline = IngestionPipeline(chunk_size=500, chunk_overlap=50)

# Automatically process all PDFs and DOCXs in a directory
documents = pipeline.load_directory("data/raw")

# You now have a list of Document objects!
# Each Document has `.page_content` (the text) and `.metadata` (dict with source, page)
```

## Enterprise Embedding Pipeline

Once documents are ingested and chunked, you can use the `EmbeddingService` to instantly convert the chunks into 384-dimensional mathematical vectors using the lightning-fast `BAAI/bge-small-en-v1.5` model. These vectors are automatically injected directly into the `Document` metadata, keeping your text and citations perfectly synced.

```python
from embeddings.embedding_service import EmbeddingService

# Initialize the embedding service
embedding_service = EmbeddingService()

# Batch process all document chunks 
embedded_docs = embedding_service.process_documents(documents)

# The vector is now securely attached to the metadata!
# embedded_docs[0].metadata["embedding"] -> [-0.028, 0.004, ...]
```
