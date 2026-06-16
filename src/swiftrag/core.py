"""The :class:`RAG` orchestrator, the one class most users ever touch."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import pickle
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Union

import numpy as np

from .chunking import chunk_documents, count_tokens
from .embeddings import EmbeddingProvider, resolve_embeddings
from .exceptions import (
    ConfigurationError,
    DependencyError,
    EmptyCorpusError,
    SwiftRagError,
)
from .llms import LLMProvider, resolve_llm
from .store import VectorStore
from .types import Chunk, Document, RAGResponse, ScoredChunk

logger = logging.getLogger("swiftrag")

#: Bumped when the on-disk save format changes incompatibly.
_SAVE_FORMAT_VERSION = 1

#: File extensions treated as plain text when loading a directory.
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".text", ".log", ".csv"}

DocsInput = Union[str, Document, dict[str, Any], Sequence[Union[str, Document, dict[str, Any]]]]

# A metadata filter: either a mapping of exact key==value matches, or a
# predicate callable evaluated against each candidate chunk.
Where = Union[Mapping[str, Any], Callable[[Chunk], bool]]

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the "
    "provided context. If the context is insufficient, say you don't know. "
    "Be concise and cite facts from the context."
)


def _coerce_documents(documents: DocsInput) -> list[Document]:
    """Normalize the flexible ``documents`` argument into ``Document`` objects."""
    if documents is None:
        return []
    if isinstance(documents, (str, Document, dict)):
        documents = [documents]

    out: list[Document] = []
    for i, item in enumerate(documents):
        if isinstance(item, Document):
            doc = item
        elif isinstance(item, str):
            doc = Document(text=item)
        elif isinstance(item, dict):
            doc = Document(
                text=item.get("text") or item.get("content") or "",
                metadata=item.get("metadata", {k: v for k, v in item.items() if k not in ("text", "content", "id")}),
                id=item.get("id"),
            )
        else:
            raise TypeError(f"Unsupported document type at index {i}: {type(item)!r}")
        if doc.id is None:
            doc.id = uuid.uuid4().hex[:12]
        out.append(doc)
    return out


def _text_hash(text: str) -> str:
    return hashlib.blake2b(text.strip().encode("utf-8"), digest_size=16).hexdigest()


def _make_predicate(where: Where | None):
    """Turn a ``where`` filter into a ``Chunk -> bool`` predicate (or ``None``)."""
    if where is None:
        return None
    if callable(where):
        return where
    items = list(where.items())

    def predicate(chunk: Chunk) -> bool:
        md = chunk.metadata
        return all(md.get(k) == v for k, v in items)

    return predicate


class RAG:
    """Retrieval-augmented generation in one call.

    Example
    -------
    >>> from swiftrag import RAG
    >>> rag = RAG(
    ...     documents="The Eiffel Tower is 330 metres tall and located in Paris.",
    ...     embedding_model="openai:text-embedding-3-small",
    ...     llm_model="openai:gpt-4o-mini",
    ... )
    >>> print(rag.query("How tall is the Eiffel Tower?"))

    No API key? It still runs end-to-end offline using a hashing embedder and an
    extractive answer:

    >>> rag = RAG(documents="...")          # embedding_model defaults to "hash"
    >>> rag.query("...")
    """

    def __init__(
        self,
        documents: DocsInput | None = None,
        *,
        embedding_model: str | EmbeddingProvider = "hash",
        llm_model: str | LLMProvider | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 4,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        use_mmr: bool = False,
        use_hybrid: bool = False,
        use_faiss: bool = False,
        min_score: float | None = None,
        max_context_tokens: int | None = None,
        dedup: bool = True,
        query_cache_size: int = 128,
        reranker: Callable[[str, list[ScoredChunk]], list[ScoredChunk]] | None = None,
        embedding_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ConfigurationError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ConfigurationError("chunk_overlap must be smaller than chunk_size")
        if top_k < 1:
            raise ConfigurationError("top_k must be >= 1")

        self.embedder: EmbeddingProvider = resolve_embeddings(
            embedding_model, **(embedding_kwargs or {})
        )
        self.llm: LLMProvider = resolve_llm(llm_model, **(llm_kwargs or {}))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.use_mmr = use_mmr
        self.use_hybrid = use_hybrid
        self.reranker = reranker
        self.min_score = min_score
        self.max_context_tokens = max_context_tokens
        self.dedup = dedup
        self.store = VectorStore(use_faiss=use_faiss)
        self._query_cache_size = max(0, query_cache_size)
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._seen_hashes: set[str] = set()

        if documents is not None:
            self.add(documents)

    # ------------------------------------------------------------------ ingest
    def add(self, documents: DocsInput) -> RAG:
        """Chunk, embed, and index more documents. Returns ``self`` for chaining.

        Exact-duplicate chunks are skipped when ``dedup`` is enabled (default),
        which avoids paying to embed and store repeated text.
        """
        docs = _coerce_documents(documents)
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if self.dedup:
            unique = []
            for c in chunks:
                h = _text_hash(c.text)
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
                unique.append(c)
            chunks = unique
        if not chunks:
            return self
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        self._query_cache.clear()  # results may change; invalidate stale answers
        logger.debug("Indexed %d chunks (total=%d)", len(chunks), len(self.store))
        return self

    @classmethod
    def from_files(
        cls,
        paths: str | Path | Sequence[str | Path],
        *,
        glob: str = "**/*",
        encoding: str = "utf-8",
        **kwargs,
    ) -> RAG:
        """Build a :class:`RAG` from text files and/or directories.

        Each file becomes a document with ``metadata={"source": <path>}``.
        Directories are expanded using ``glob`` and filtered to common text
        extensions (``.txt``, ``.md``, ``.rst``, ``.csv``, ...).

        Example
        -------
        >>> rag = RAG.from_files("docs/", embedding_model="openai:text-embedding-3-small")
        """
        from .loaders import EXTENSION_LOADERS, load_file

        if isinstance(paths, (str, Path)):
            paths = [paths]

        supported = _TEXT_EXTENSIONS | set(EXTENSION_LOADERS)
        files: list[Path] = []
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                files.extend(
                    f for f in sorted(p.glob(glob))
                    if f.is_file() and f.suffix.lower() in supported
                )
            elif p.is_file():
                files.append(p)
            else:
                raise FileNotFoundError(f"No such file or directory: {p}")

        documents: list[Document] = []
        for f in files:
            try:
                text = load_file(f, encoding=encoding)
            except DependencyError:
                raise  # missing optional parser is actionable; surface it
            except (UnicodeDecodeError, OSError, SwiftRagError) as exc:
                logger.warning("Skipping %s: %s", f, exc)
                continue
            if text.strip():
                documents.append(Document(text=text, metadata={"source": str(f)}))

        logger.debug("from_files loaded %d documents from %d paths", len(documents), len(files))
        return cls(documents=documents, **kwargs)

    @classmethod
    def from_url(
        cls,
        urls: str | Sequence[str],
        *,
        timeout: float = 30.0,
        **kwargs,
    ) -> RAG:
        """Build a :class:`RAG` from one or more web pages (or PDF URLs).

        Each URL is fetched and reduced to text, then tagged with
        ``metadata={"source": <url>}``. HTML works with no extra dependencies;
        PDF URLs require ``pypdf`` (``pip install 'swiftrag[loaders]'``).

        Example
        -------
        >>> rag = RAG.from_url("https://example.com", llm_model="openai:gpt-4o-mini")
        """
        from .loaders import load_url

        urls = [urls] if isinstance(urls, str) else list(urls)

        documents: list[Document] = []
        for url in urls:
            text = load_url(url, timeout=timeout)
            if text.strip():
                documents.append(Document(text=text, metadata={"source": url}))

        logger.debug("from_url loaded %d documents from %d urls", len(documents), len(urls))
        return cls(documents=documents, **kwargs)

    def clear(self) -> RAG:
        """Remove all indexed documents (keeps the configured providers)."""
        self.store.clear()
        self._seen_hashes.clear()
        self._query_cache.clear()
        return self

    def _embed_query_cached(self, question: str) -> np.ndarray:
        if self._query_cache_size == 0:
            return self.embedder.embed_query(question)
        cached = self._query_cache.get(question)
        if cached is not None:
            self._query_cache.move_to_end(question)
            return cached
        vec = self.embedder.embed_query(question)
        self._query_cache[question] = vec
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return vec

    # --------------------------------------------------------------- retrieval
    def _maybe_rerank(
        self, question: str, results: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Apply the optional reranker, then cap to ``top_k``."""
        if self.reranker is not None and results:
            results = list(self.reranker(question, results))
        return results[:top_k]

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        min_score: float | None = None,
        where: Where | None = None,
        hybrid: bool | None = None,
    ) -> list[ScoredChunk]:
        """Return the most relevant chunks for ``question`` (no generation).

        Args:
            top_k: Number of chunks to return (defaults to the instance setting).
            min_score: Drop chunks whose cosine score is below this threshold.
            where: Metadata filter, a dict of exact ``key == value`` matches or
                a ``Chunk -> bool`` predicate.
            hybrid: Override the instance ``use_hybrid`` setting for this call.
                When on, dense and BM25 rankings are fused with Reciprocal Rank
                Fusion (helps exact-term matches the embedder misses).
        """
        if len(self.store) == 0:
            raise EmptyCorpusError("No documents indexed. Call .add(...) or pass documents=...")
        effective_k = top_k or self.top_k
        q_emb = self._embed_query_cached(question)
        results = self.store.search(
            q_emb,
            top_k=effective_k,
            mmr=self.use_mmr,
            min_score=self.min_score if min_score is None else min_score,
            predicate=_make_predicate(where),
            hybrid=self.use_hybrid if hybrid is None else hybrid,
            query_text=question,
        )
        return self._maybe_rerank(question, results, effective_k)

    # -------------------------------------------------------------- generation
    def _fit_to_budget(self, sources: list[ScoredChunk]) -> list[ScoredChunk]:
        """Trim lowest-ranked sources so the context fits ``max_context_tokens``."""
        budget = self.max_context_tokens
        if not budget or not sources:
            return sources
        kept: list[ScoredChunk] = []
        used = 0
        for s in sources:  # sources are already ordered best-first
            cost = count_tokens(s.text) + 8  # +overhead for the "[n] " wrapper
            if kept and used + cost > budget:
                break
            kept.append(s)
            used += cost
        return kept

    def _build_messages(self, question: str, sources: list[ScoredChunk]) -> list[dict[str, str]]:
        sources = self._fit_to_budget(sources)
        context = "\n\n---\n\n".join(f"[{i + 1}] {s.text}" for i, s in enumerate(sources))
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        where: Where | None = None,
        hybrid: bool | None = None,
        **llm_kwargs,
    ) -> RAGResponse:
        """Retrieve relevant context and generate a grounded answer."""
        sources = self.retrieve(
            question, top_k=top_k, min_score=min_score, where=where, hybrid=hybrid
        )
        messages = self._build_messages(question, sources)
        answer = self.llm.generate(messages, **llm_kwargs)
        return RAGResponse(answer=answer, sources=sources, query=question)

    def stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        where: Where | None = None,
        hybrid: bool | None = None,
        **llm_kwargs,
    ) -> Iterator[str]:
        """Like :meth:`query`, but yields answer tokens as they arrive."""
        sources = self.retrieve(
            question, top_k=top_k, min_score=min_score, where=where, hybrid=hybrid
        )
        messages = self._build_messages(question, sources)
        yield from self.llm.stream(messages, **llm_kwargs)

    # ------------------------------------------------------------------- batch
    def retrieve_many(
        self,
        questions: Sequence[str],
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        where: Where | None = None,
        hybrid: bool | None = None,
    ) -> list[list[ScoredChunk]]:
        """Retrieve for many questions, embedding them all in one batched call."""
        if not questions:
            return []
        if len(self.store) == 0:
            raise EmptyCorpusError("No documents indexed. Call .add(...) or pass documents=...")
        questions = list(questions)
        embeddings = self.embedder.embed_documents(questions)
        predicate = _make_predicate(where)
        effective_min = self.min_score if min_score is None else min_score
        effective_k = top_k or self.top_k
        effective_hybrid = self.use_hybrid if hybrid is None else hybrid
        results = []
        for question, emb in zip(questions, embeddings):
            hits = self.store.search(
                emb,
                top_k=effective_k,
                mmr=self.use_mmr,
                min_score=effective_min,
                predicate=predicate,
                hybrid=effective_hybrid,
                query_text=question,
            )
            results.append(self._maybe_rerank(question, hits, effective_k))
        return results

    def query_many(
        self,
        questions: Sequence[str],
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        where: Where | None = None,
        hybrid: bool | None = None,
        max_workers: int = 4,
        **llm_kwargs,
    ) -> list[RAGResponse]:
        """Answer many questions. Embeddings are batched; generation is parallelized."""
        questions = list(questions)
        sources_list = self.retrieve_many(
            questions, top_k=top_k, min_score=min_score, where=where, hybrid=hybrid
        )

        def _answer(item: tuple[str, list[ScoredChunk]]) -> RAGResponse:
            question, sources = item
            messages = self._build_messages(question, sources)
            answer = self.llm.generate(messages, **llm_kwargs)
            return RAGResponse(answer=answer, sources=sources, query=question)

        items = list(zip(questions, sources_list))
        if max_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(_answer, items))
        return [_answer(it) for it in items]

    # ------------------------------------------------------------------- async
    async def aretrieve(self, question: str, **kwargs) -> list[ScoredChunk]:
        """Async wrapper around :meth:`retrieve` (runs in a worker thread)."""
        return await asyncio.to_thread(self.retrieve, question, **kwargs)

    async def aquery(self, question: str, **kwargs) -> RAGResponse:
        """Async wrapper around :meth:`query` (runs in a worker thread)."""
        return await asyncio.to_thread(self.query, question, **kwargs)

    async def astream(self, question: str, **kwargs) -> AsyncIterator[str]:
        """Async streaming: yields tokens from :meth:`stream` without blocking the loop."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _worker() -> None:
            try:
                for token in self.stream(question, **kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:  # surface provider errors to the consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Persist the index (chunks + embeddings) to ``path``.

        Providers aren't pickled, reload with the same ``embedding_model`` /
        ``llm_model`` you used originally.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _SAVE_FORMAT_VERSION,
            "chunks": self.store._chunks,
            "matrix": self.store._matrix,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "system_prompt": self.system_prompt,
                "use_mmr": self.use_mmr,
                "use_hybrid": self.use_hybrid,
                "min_score": self.min_score,
                "max_context_tokens": self.max_context_tokens,
                "dedup": self.dedup,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embedding_model: str | EmbeddingProvider = "hash",
        llm_model: str | LLMProvider | None = None,
        **kwargs,
    ) -> RAG:
        """Load an index saved by :meth:`save`.

        Security note: this uses ``pickle``; only load files you created/trust.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        version = payload.get("format_version", 1)
        if version > _SAVE_FORMAT_VERSION:
            raise ConfigurationError(
                f"Index was saved with a newer swiftrag format (v{version}); "
                f"this version supports up to v{_SAVE_FORMAT_VERSION}. Please upgrade swiftrag."
            )
        cfg = payload.get("config", {})
        rag = cls(
            embedding_model=embedding_model,
            llm_model=llm_model,
            chunk_size=cfg.get("chunk_size", 512),
            chunk_overlap=cfg.get("chunk_overlap", 64),
            top_k=cfg.get("top_k", 4),
            system_prompt=cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            use_mmr=cfg.get("use_mmr", False),
            use_hybrid=cfg.get("use_hybrid", False),
            min_score=cfg.get("min_score"),
            max_context_tokens=cfg.get("max_context_tokens"),
            dedup=cfg.get("dedup", True),
            **kwargs,
        )
        rag.store._chunks = payload["chunks"]
        rag.store._matrix = payload["matrix"]
        rag._seen_hashes = {_text_hash(c.text) for c in rag.store._chunks}
        return rag

    # ------------------------------------------------------------------- misc
    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        return (
            f"RAG(chunks={len(self.store)}, embedder={type(self.embedder).__name__}, "
            f"llm={type(self.llm).__name__}, top_k={self.top_k})"
        )


__all__ = ["RAG"]
