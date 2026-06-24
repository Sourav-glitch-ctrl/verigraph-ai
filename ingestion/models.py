from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Document:
    """
    Represents a chunk of text and its associated metadata (citations).
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
