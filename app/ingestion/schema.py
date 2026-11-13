"""schema.py — Document data model."""
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Document:
    content:  str
    metadata: Dict[str, Any] = field(default_factory=dict)
