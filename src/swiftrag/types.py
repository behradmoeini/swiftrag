"""Lightweight data containers used across swiftrag."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ``slots=True`` was added to ``dataclass`` in Python 3.10. Use it where available
# for lower per-object memory, and fall back gracefully on 3.9.
_SLOTS: dict[str, bool] = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_SLOTS)
class Document:
    """A raw source document before chunking."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(**_SLOTS)
class Chunk:
    """A chunk of a document, optionally carrying its embedding."""

    text: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass(**_SLOTS)
class ScoredChunk:
    """A retrieved chunk with its similarity score."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata


@dataclass(**_SLOTS)
class RAGResponse:
    """The final answer plus the evidence used to produce it."""

    answer: str
    sources: list[ScoredChunk]
    query: str
    usage: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # convenient: print(rag.query(...))
        return self.answer

    def format_sources(self, max_chars: int = 200) -> str:
        """Render a numbered, human-readable citation list of the sources used."""
        lines = []
        for i, s in enumerate(self.sources, 1):
            snippet = s.text.strip().replace("\n", " ")
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars].rstrip() + "…"
            src = s.metadata.get("source")
            tag = f" ({src})" if src else ""
            lines.append(f"[{i}] score={s.score:.3f}{tag}: {snippet}")
        return "\n".join(lines)
