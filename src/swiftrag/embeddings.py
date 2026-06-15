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
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise DependencyError("openai", "openai") from e

        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.dimensions = dimensions
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def _embed_batch(self, batch: Sequence[str]) -> np.ndarray:
        kwargs = {"model": self.model, "input": list(batch)}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self._client.embeddings.create(**kwargs)
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
    if provider in ("hash", "none", "offline"):
        dim = int(model) if model.isdigit() else kwargs.get("dim", 512)
        return HashEmbeddings(dim=dim)
    raise ConfigurationError(
        f"Unknown embedding provider '{provider}'. "
        "Use one of: openai, st/local, hash, or pass a custom provider object."
    )


__all__ = [
    "EmbeddingProvider",
    "HashEmbeddings",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    "resolve_embeddings",
]
