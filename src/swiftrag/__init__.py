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

from .chat import Conversation
from .chunking import chunk_text, count_tokens
from .core import RAG
from .embeddings import (
    CohereEmbeddings,
    EmbeddingProvider,
    GeminiEmbeddings,
    HashEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
)
from .evaluation import (
    EvalExample,
    RetrievalReport,
    evaluate_retrieval,
    groundedness,
)
from .exceptions import (
    ConfigurationError,
    DependencyError,
    EmptyCorpusError,
    SwiftRagError,
)
from .llms import (
    AnthropicLLM,
    CallableLLM,
    EchoLLM,
    GeminiLLM,
    LLMProvider,
    OllamaLLM,
    OpenAILLM,
)
from .loaders import html_to_text, load_docx, load_file, load_html, load_pdf, load_url
from .types import Chunk, Document, RAGResponse, ScoredChunk

__version__ = "0.1.1"

__all__ = [
    "RAG",
    "Conversation",
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
    "OllamaEmbeddings",
    "CohereEmbeddings",
    "GeminiEmbeddings",
    "LLMProvider",
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    "GeminiLLM",
    "EchoLLM",
    "CallableLLM",
    "SwiftRagError",
    "ConfigurationError",
    "DependencyError",
    "EmptyCorpusError",
    "evaluate_retrieval",
    "groundedness",
    "EvalExample",
    "RetrievalReport",
    "load_file",
    "load_pdf",
    "load_docx",
    "load_html",
    "load_url",
    "html_to_text",
    "__version__",
]
