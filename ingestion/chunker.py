import copy
from typing import List
from ingestion.models import Document

class TextChunker:
    """
    Splits cleaned text into chunks with overlap, avoiding splitting words in half.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, doc: Document) -> List[Document]:
        """
        Splits a single Document into multiple chunk Documents, 
        duplicating the original metadata into each chunk.
        """
        text_chunks = self.split_text(doc.page_content)
        doc_chunks = []
        for i, text_chunk in enumerate(text_chunks):
            new_metadata = copy.deepcopy(doc.metadata)
            new_metadata["chunk_index"] = i
            doc_chunks.append(Document(page_content=text_chunk, metadata=new_metadata))
        return doc_chunks

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Determine the theoretical end of the current chunk
            end = start + self.chunk_size

            # If we're at or past the end of the text, take the rest and finish
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Try to find a good breaking point (paragraph, newline, or space) 
            # by searching backwards from 'end' to 'start'
            break_point = -1
            for separator in ['\n\n', '\n', ' ']:
                break_point = text.rfind(separator, start, end)
                if break_point != -1:
                    break

            # If no separator found (e.g., an extremely long string of characters), 
            # force a split exactly at chunk_size
            if break_point == -1 or break_point <= start:
                break_point = end

            chunk = text[start:break_point].strip()
            if chunk:
                chunks.append(chunk)

            # Advance 'start' for the next chunk, subtracting the overlap.
            next_start = break_point - self.chunk_overlap
            
            # Ensure we always make forward progress (prevents infinite loops)
            if next_start <= start:
                start = break_point
            else:
                start = next_start

        return chunks