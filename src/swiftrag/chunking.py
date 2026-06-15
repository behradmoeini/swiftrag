"""Text chunking.

Strategy (aimed at retrieval quality and speed):

1. If ``tiktoken`` is available we chunk by *tokens*, which matches how the
   downstream embedding/LLM models actually see text. This gives uniform chunk
   sizes regardless of language/whitespace.
2. Otherwise we fall back to a fast, dependency-free splitter that respects
   natural boundaries (paragraphs -> sentences -> words) and approximates token
   counts with a characters-per-token heuristic.

Both paths support configurable overlap so context isn't lost at boundaries.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Rough average for English text with common BPE tokenizers.
_CHARS_PER_TOKEN = 4

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# Split on sentence terminators while keeping reasonable behavior on edge cases.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@lru_cache(maxsize=4)
def _get_tiktoken_encoding(name: str):
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in ``text``.

    Uses tiktoken when available for an exact count; otherwise falls back to a
    fast characters-per-token estimate so callers never need to special-case it.
    """
    if not text:
        return 0
    enc = _get_tiktoken_encoding(encoding_name)
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // _CHARS_PER_TOKEN)


def chunk_text(
    text: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """Split ``text`` into overlapping chunks of ~``chunk_size`` tokens.

    Args:
        text: The input text.
        chunk_size: Target chunk length in tokens.
        chunk_overlap: Number of tokens shared between consecutive chunks.
        encoding_name: tiktoken encoding to use when available.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    enc = _get_tiktoken_encoding(encoding_name)
    if enc is not None:
        return _chunk_by_tokens(text, enc, chunk_size, chunk_overlap)
    return _chunk_by_chars(text, chunk_size * _CHARS_PER_TOKEN, chunk_overlap * _CHARS_PER_TOKEN)


def _chunk_by_tokens(text: str, enc, chunk_size: int, chunk_overlap: int) -> list[str]:
    tokens = enc.encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(enc.decode(window).strip())
        if start + chunk_size >= len(tokens):
            break
    return [c for c in chunks if c]


def _chunk_by_chars(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    # Build semantic units: paragraphs, then sentences for oversized paragraphs.
    units: list[str] = []
    for para in _PARAGRAPH_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(_split_oversized(para, max_chars))

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for unit in units:
        unit_len = len(unit) + 1
        if buf and buf_len + unit_len > max_chars:
            chunks.append("\n".join(buf))
            buf, buf_len = _carry_overlap(buf, overlap_chars)
        buf.append(unit)
        buf_len += unit_len
    if buf:
        chunks.append("\n".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def _split_oversized(para: str, max_chars: int) -> list[str]:
    """Split a paragraph that is itself larger than ``max_chars``."""
    pieces: list[str] = []
    sentences = _SENTENCE_RE.split(para)
    buf: list[str] = []
    buf_len = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if buf:
                pieces.append(" ".join(buf))
                buf, buf_len = [], 0
            pieces.extend(_split_words(sent, max_chars))
            continue
        if buf and buf_len + len(sent) + 1 > max_chars:
            pieces.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(sent)
        buf_len += len(sent) + 1
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def _split_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for word in words:
        if buf and buf_len + len(word) + 1 > max_chars:
            pieces.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(word)
        buf_len += len(word) + 1
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def _carry_overlap(buf: list[str], overlap_chars: int) -> tuple[list[str], int]:
    """Keep trailing units from ``buf`` to seed the next chunk's overlap."""
    if overlap_chars <= 0:
        return [], 0
    carried: list[str] = []
    length = 0
    for unit in reversed(buf):
        if length >= overlap_chars:
            break
        carried.insert(0, unit)
        length += len(unit) + 1
    return carried, length


def chunk_documents(
    docs,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding_name: str = "cl100k_base",
) -> list:
    """Chunk an iterable of :class:`~swiftrag.types.Document` into chunks."""
    from .types import Chunk  # local import to avoid cycle at module import time

    chunks: list[Chunk] = []
    for doc in docs:
        parts = chunk_text(
            doc.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
        )
        for i, part in enumerate(parts):
            chunks.append(
                Chunk(
                    text=part,
                    doc_id=doc.id or "",
                    chunk_index=i,
                    metadata=dict(doc.metadata),
                )
            )
    return chunks


__all__ = ["chunk_text", "chunk_documents", "count_tokens"]
