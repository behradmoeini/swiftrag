# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI (lint and test matrix across Python 3.9 to 3.12 on Linux, macOS, and Windows).
- GitHub Actions release workflow that publishes to PyPI via Trusted Publishing.
- Metadata filtering on retrieval via `where=` (exact-match dict or `Chunk -> bool` callable).
- `min_score` cosine threshold to drop weak matches (per-call or instance default).
- LRU cache for query embeddings (`query_cache_size`) so repeated queries skip re-embedding.
- `max_context_tokens` to cap how much retrieved context is packed into the prompt
  (token-aware via tiktoken when available), which prevents context-window overflows.
- Exact-duplicate chunk de-duplication on ingest (`dedup`, on by default).
- `RAG.clear()` to reset the index while keeping configured providers.
- Public `swiftrag.chunking.count_tokens` helper.
- Batch APIs: `query_many` (batched embeddings and parallel generation) and `retrieve_many`.
- Async APIs: `aquery`, `aretrieve`, and `astream` (non-blocking, run in a worker thread).
- `RAG.from_files(...)` to build an index directly from files or directories (auto `source` metadata).
- `RAGResponse.format_sources()` for numbered citations you can print directly.
- Constructor input validation with clear `ConfigurationError` messages.
- Versioned save format with a clear error when loading a newer index.
- Library logging under the `swiftrag` logger.

## [0.1.0] - 2026-06-15

### Added
- Initial release of `swiftrag`.
- One-call `RAG` class: pass documents (str, list, or dicts) and a model spec.
- Embedding providers: OpenAI, sentence-transformers (local), and an offline hashing embedder.
- LLM providers: OpenAI, Anthropic, custom callables, and an offline echo answerer.
- Vector store with normalized embeddings, BLAS matmul search, `argpartition`
  top-k, optional MMR re-ranking, and an optional FAISS backend.
- Token-aware chunking with overlap.
- Save and load of the index.
