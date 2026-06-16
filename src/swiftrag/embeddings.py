"""Embedding providers.

A provider turns text into vectors. swiftrag includes three backends and a thin
protocol so you can plug in anything else:

* ``openai``  -> OpenAI / Azure-compatible embedding endpoints (needs ``openai``).
* ``st`` / ``local`` -> sentence-transformers, fully local (needs ``sentence-transformers``).
* ``hash`` -> a deterministic, dependency-free hashing embedder. It is limited
  semantically, but it lets the pipeline run offline with no setup, which is
  useful for tests, demos, and CI.

Any object exposing ``embed_documents(list[str]) -> np.ndarray`` and
``embed_query(str) -> np.ndarray`` is accepted directly.
"""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._retry import DEFAULT_MAX_RETRIES, retry_call
from .exceptions import ConfigurationError, DependencyError


class EmbeddingProvider(ABC):
    """Abstract embedding backend."""

    #: Set by subclasses once known; used to pre-size structures.
    dim: int | None = None

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of documents -> ``(len(texts), dim)`` float32 array."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Override if a provider has a query-specific path."""
        return self.embed_documents([text])[0]


class HashEmbeddings(EmbeddingProvider):
    """Deterministic hashing embedder (offline, zero dependencies).

    Uses the hashing trick over word unigrams + bigrams with signed buckets,
    then L2-normalizes. Good enough for keyword-ish retrieval and for making
    the library work out of the box without any API key.
    """

    _token_re = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._token_re.findall(text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI (and OpenAI-compatible) embeddings with batching + concurrency."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 128,
        max_workers: int = 4,
        dimensions: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise DependencyError("openai", "openai") from e

        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.dimensions = dimensions
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def _embed_batch(self, batch: Sequence[str]) -> np.ndarray:
        kwargs = {"model": self.model, "input": list(batch)}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = retry_call(
            lambda: self._client.embeddings.create(**kwargs), retries=self.max_retries
        )
        data = sorted(resp.data, key=lambda d: d.index)
        return np.asarray([d.embedding for d in data], dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim or 1), dtype=np.float32)
        batches = [
            texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
        ]
        if len(batches) == 1:
            out = self._embed_batch(batches[0])
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(self._embed_batch, batches))
            out = np.vstack(results)
        self.dim = out.shape[1]
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local embeddings via sentence-transformers."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise DependencyError("sentence-transformers", "local") from e
        self._model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        self._normalize = normalize
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        arr = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(arr, dtype=np.float32)


class OllamaEmbeddings(EmbeddingProvider):
    """Embeddings via a local (or remote) Ollama server. No extra dependencies.

    Talks to Ollama's HTTP API directly, so a fully offline, free embedding +
    LLM stack works with just the core install. Point at a different host with
    the ``OLLAMA_HOST`` environment variable or the ``host`` argument.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        host: str | None = None,
        timeout: float = 60.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        from ._http import post_json

        if not texts:
            return np.empty((0, self.dim or 1), dtype=np.float32)
        data = retry_call(
            lambda: post_json(
                f"{self.host}/api/embed",
                {"model": self.model, "input": list(texts)},
                timeout=self.timeout,
            ),
            retries=self.max_retries,
        )
        embeddings = data.get("embeddings")
        if not embeddings:
            raise ConfigurationError(
                f"Ollama returned no embeddings for model '{self.model}'. "
                f"Is it pulled? Try:  ollama pull {self.model}"
            )
        arr = np.asarray(embeddings, dtype=np.float32)
        self.dim = arr.shape[1]
        return arr


class CohereEmbeddings(EmbeddingProvider):
    """Cohere embeddings (needs ``cohere``).

    Uses the correct ``input_type`` for documents vs. queries, which Cohere's
    retrieval models are trained to distinguish.
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        *,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        try:
            import cohere
        except ImportError as e:
            raise DependencyError("cohere", "cohere") from e
        self.model = model
        self.max_retries = max_retries
        self._client = cohere.Client(api_key or os.getenv("COHERE_API_KEY"))

    def _embed(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        resp = retry_call(
            lambda: self._client.embed(
                texts=list(texts), model=self.model, input_type=input_type
            ),
            retries=self.max_retries,
        )
        arr = np.asarray(resp.embeddings, dtype=np.float32)
        self.dim = arr.shape[1]
        return arr

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim or 1), dtype=np.float32)
        return self._embed(texts, "search_document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "search_query")[0]


class GeminiEmbeddings(EmbeddingProvider):
    """Google Gemini embeddings (needs ``google-generativeai``)."""

    def __init__(
        self,
        model: str = "text-embedding-004",
        *,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise DependencyError("google-generativeai", "gemini") from e
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        self._genai = genai
        self.max_retries = max_retries
        self.model = model if model.startswith("models/") else f"models/{model}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim or 1), dtype=np.float32)
        resp = retry_call(
            lambda: self._genai.embed_content(
                model=self.model, content=list(texts), task_type="retrieval_document"
            ),
            retries=self.max_retries,
        )
        arr = np.asarray(resp["embedding"], dtype=np.float32)
        self.dim = arr.shape[1]
        return arr

    def embed_query(self, text: str) -> np.ndarray:
        resp = retry_call(
            lambda: self._genai.embed_content(
                model=self.model, content=text, task_type="retrieval_query"
            ),
            retries=self.max_retries,
        )
        return np.asarray(resp["embedding"], dtype=np.float32)


def resolve_embeddings(spec, **kwargs) -> EmbeddingProvider:
    """Build an :class:`EmbeddingProvider` from a spec.

    ``spec`` may be:
    * an :class:`EmbeddingProvider` (returned as-is),
    * any object with ``embed_documents``/``embed_query`` (wrapped),
    * a string ``"provider:model"`` or ``"provider"``, e.g. ``"openai:text-embedding-3-small"``,
      ``"st:all-MiniLM-L6-v2"``, ``"local"``, or ``"hash"``.
    """
    if isinstance(spec, EmbeddingProvider):
        return spec
    if hasattr(spec, "embed_documents") and hasattr(spec, "embed_query"):
        return spec  # duck-typed custom provider
    if not isinstance(spec, str):
        raise ConfigurationError(f"Unsupported embedding spec: {spec!r}")

    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()

    if provider in ("openai", "azure", "oai"):
        return OpenAIEmbeddings(model or "text-embedding-3-small", **kwargs)
    if provider in ("st", "local", "sentence-transformers", "hf"):
        return SentenceTransformerEmbeddings(model or "all-MiniLM-L6-v2", **kwargs)
    if provider in ("ollama",):
        return OllamaEmbeddings(model or "nomic-embed-text", **kwargs)
    if provider in ("cohere", "co"):
        return CohereEmbeddings(model or "embed-english-v3.0", **kwargs)
    if provider in ("gemini", "google", "googleai"):
        return GeminiEmbeddings(model or "text-embedding-004", **kwargs)
    if provider in ("hash", "none", "offline"):
        dim = int(model) if model.isdigit() else kwargs.get("dim", 512)
        return HashEmbeddings(dim=dim)
    raise ConfigurationError(
        f"Unknown embedding provider '{provider}'. Use one of: openai, st/local, "
        "ollama, cohere, gemini, hash, or pass a custom provider object."
    )


__all__ = [
    "EmbeddingProvider",
    "HashEmbeddings",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    "OllamaEmbeddings",
    "CohereEmbeddings",
    "GeminiEmbeddings",
    "resolve_embeddings",
]
