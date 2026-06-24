import logging
from pathlib import Path
from typing import List

from ingestion.models import Document

# pyrefly: ignore [missing-import]
from pypdf import PdfReader
# pyrefly: ignore [missing-import]
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

class PDFLoader:
    """
    Responsible only for extracting text
    from PDF documents.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        pdf_path = Path(self.file_path)

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {self.file_path}")
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        if not pdf_path.is_file():
            logger.error(f"Path is not a file: {self.file_path}")
            raise ValueError(f"Path is not a file: {self.file_path}")

        if pdf_path.suffix.lower() != '.pdf':
            logger.warning(f"File {self.file_path} does not have a .pdf extension.")

        docs = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            docs.append(Document(
                                page_content=text,
                                metadata={"source": str(pdf_path), "page": page_num + 1}
                            ))
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} of {self.file_path}: {e}")
        except PdfReadError as e:
            logger.error(f"Invalid or corrupted PDF file {self.file_path}: {e}")
            raise ValueError(f"Invalid or corrupted PDF file: {self.file_path}") from e
        except Exception as e:
            logger.error(f"Unexpected error while loading PDF {self.file_path}: {e}")
            raise RuntimeError(f"Unexpected error while loading PDF: {self.file_path}") from e

        return docs
