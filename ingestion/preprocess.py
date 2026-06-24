import re
import unicodedata
from ingestion.models import Document

class TextPreprocessor:
    """
    Cleans extracted text before chunking, preserving basic paragraph structures
    while normalizing unicode and removing excess whitespace/noise.
    """

    @staticmethod
    def clean_document(doc: Document) -> Document:
        """
        Cleans the page_content of a Document object, preserving its metadata.
        """
        doc.page_content = TextPreprocessor.clean_text(doc.page_content)
        return doc

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Normalize unicode characters (e.g. standardizing quotes, ligatures)
        text = unicodedata.normalize("NFKC", text)

        # 2. Replace multiple spaces and horizontal tabs with a single space
        #    (We deliberately don't use \s to avoid destroying \n newlines)
        text = re.sub(r"[ \t]+", " ", text)

        # 3. Clean up each line (stripping trailing/leading space)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 4. Replace 3 or more consecutive newlines with exactly 2 newlines (a paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 5. Remove any leading or trailing whitespace from the final string
        text = text.strip()

        return text