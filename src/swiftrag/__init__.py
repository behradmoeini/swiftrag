"""swiftrag: retrieval-augmented generation from your text in one call.

Pass your text and, optionally, a model, and you get a RAG-backed LLM.

    from swiftrag import RAG

    rag = RAG(
        documents="your knowledge as a string (or a list of strings/dicts)",
        embedding_model="openai:text-embedding-3-small",
        llm_model="openai:gpt-4o-mini",
    )
    print(rag.query("your question"))
"""

from __future__ import annotations

from .chunking import chunk_text, count_tokens
from .core import RAG
from .embeddings import (
    EmbeddingProvider,
    HashEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
)
from .exceptions import (
    ConfigurationError,
    DependencyError,
    EmptyCorpusError,
    SwiftRagError,
)
from .llms import AnthropicLLM, CallableLLM, EchoLLM, LLMProvider, OpenAILLM
from .types import Chunk, Document, RAGResponse, ScoredChunk

__version__ = "0.1.1"

__all__ = [
    "RAG",
    "chunk_text",
    "count_tokens",
    "Document",
    "Chunk",
    "ScoredChunk",
    "RAGResponse",
    "EmbeddingProvider",
    "HashEmbeddings",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    "LLMProvider",
    "OpenAILLM",
    "AnthropicLLM",
    "EchoLLM",
    "CallableLLM",
    "SwiftRagError",
    "ConfigurationError",
    "DependencyError",
    "EmptyCorpusError",
    "__version__",
]
