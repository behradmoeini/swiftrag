# swiftrag

[![CI](https://github.com/behradmoeini/swiftrag/actions/workflows/ci.yml/badge.svg)](https://github.com/behradmoeini/swiftrag/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/swiftrag.svg)](https://pypi.org/project/swiftrag/)
[![Python versions](https://img.shields.io/pypi/pyversions/swiftrag.svg)](https://pypi.org/project/swiftrag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

swiftrag sets up a retrieval-augmented generation (RAG) pipeline from your own text in one call. You pass your documents and a model, and it handles the chunking, embedding, vector search, retrieval, and prompt building.

```python
from swiftrag import RAG

rag = RAG(
    documents="The Eiffel Tower is 330 metres tall and located in Paris.",
    embedding_model="openai:text-embedding-3-small",
    llm_model="openai:gpt-4o-mini",
)

print(rag.query("How tall is the Eiffel Tower?"))
# -> "The Eiffel Tower is 330 metres tall."
```

That is the whole API. You bring documents (a string, a list of strings, or dicts) and a model spec, and swiftrag does the rest.

## What you get

- One call to a working RAG pipeline, with no extra glue code.
- A small core that depends only on `numpy`, so it installs quickly.
- Search that normalizes embeddings once and runs as a single BLAS matrix multiply, with top-k selection via `argpartition` instead of a full sort. Embedding requests are batched and run concurrently, chunking is token-aware, and there is an optional FAISS backend for large corpora.
- Offline operation by default. With no API key it uses a built-in hashing embedder and an extractive answerer, so the pipeline runs in tests, CI, and demos.
- Support for OpenAI, Anthropic, local sentence-transformers, or any callable or object you pass in.
- An MIT license.

## Install

```bash
pip install swiftrag                 # core (numpy only), works offline
pip install "swiftrag[openai]"       # OpenAI embeddings and LLM
pip install "swiftrag[anthropic]"    # Claude LLM
pip install "swiftrag[local]"        # local sentence-transformers embeddings
pip install "swiftrag[faiss]"        # FAISS backend for big corpora
pip install "swiftrag[all]"          # everything
```

## Usage

### Pick your models with a `"provider:model"` string

```python
# OpenAI (needs OPENAI_API_KEY)
RAG(documents=text, embedding_model="openai:text-embedding-3-small", llm_model="openai:gpt-4o-mini")

# Anthropic for generation, local embeddings (no embedding API calls)
RAG(documents=text, embedding_model="local:all-MiniLM-L6-v2", llm_model="anthropic:claude-3-5-sonnet-latest")

# Fully offline (default), no keys required
RAG(documents=text)
```

### Build straight from files or a folder

```python
rag = RAG.from_files("docs/", embedding_model="openai:text-embedding-3-small")
# each file becomes a document tagged with metadata={"source": <path>}

resp = rag.query("What's our deployment process?")
print(resp.answer)
print(resp.format_sources())   # numbered, readable citations
```

### Documents can be a string, a list, or dicts with metadata

```python
rag = RAG(documents=[
    "Plain string document.",
    {"text": "Document with metadata.", "metadata": {"source": "handbook", "page": 12}},
])
```

### Query, stream, or just retrieve

```python
resp = rag.query("What does the handbook say about refunds?")
print(resp.answer)
for s in resp.sources:
    print(s.score, s.metadata, s.text[:80])

# Token streaming
for token in rag.stream("Summarize the refund policy."):
    print(token, end="", flush=True)

# Retrieval only (no LLM call)
chunks = rag.retrieve("refunds", top_k=5)
```

### Filter by metadata and score

```python
# Only consider chunks whose metadata matches, and drop weak matches.
resp = rag.query(
    "What is the refund window?",
    where={"source": "handbook"},     # exact metadata match, or a Chunk -> bool callable
    min_score=0.25,                    # cosine threshold; weaker chunks are ignored
    top_k=3,
)
```

Repeated queries reuse a cached query embedding (LRU, configurable via
`query_cache_size`), so re-asking the same question skips the embedding call.

### Add documents incrementally, save, and reload

```python
rag = RAG().add("first batch").add("second batch")
rag.save("index.pkl")

rag = RAG.load("index.pkl", embedding_model="openai:text-embedding-3-small",
               llm_model="openai:gpt-4o-mini")
```

### Batch and async

```python
# Answer many questions at once. Embeddings are batched and generation is parallelized.
responses = rag.query_many(["q1?", "q2?", "q3?"], max_workers=8)

# Async API for non-blocking use, for example in a web server:
resp = await rag.aquery("your question")
async for token in rag.astream("your question"):
    print(token, end="", flush=True)
```

### Bring your own provider

```python
# Any callable fn(prompt) -> str works as an LLM:
rag = RAG(documents=text, llm_model=lambda prompt: my_model.generate(prompt))

# Any object with embed_documents(list[str]) and embed_query(str) works as an embedder.
```

## Configuration

| Argument | Default | Description |
| --- | --- | --- |
| `documents` | `None` | str / list[str] / list[dict] / `Document`(s) to index. |
| `embedding_model` | `"hash"` | `"provider:model"` string or a custom provider. |
| `llm_model` | `None` (offline) | `"provider:model"` string, callable, or provider. |
| `chunk_size` | `512` | Target chunk size in tokens. |
| `chunk_overlap` | `64` | Token overlap between chunks. |
| `top_k` | `4` | Chunks retrieved per query. |
| `use_mmr` | `False` | Maximal Marginal Relevance re-ranking for diverse results. |
| `use_faiss` | `False` | Use FAISS index (install `swiftrag[faiss]`). |
| `min_score` | `None` | Default cosine threshold for dropping weak matches. |
| `max_context_tokens` | `None` | Cap the tokens of retrieved context packed into the prompt. |
| `dedup` | `True` | Skip exact-duplicate chunks on ingest. |
| `query_cache_size` | `128` | LRU size for cached query embeddings (`0` disables). |
| `system_prompt` | grounded default | System prompt for the LLM. |

## How it works

On ingest, each document is chunked (token-aware), embedded in batches, normalized, and stored in the vector index. On a query, the question is embedded, compared to the index with a cosine similarity matrix multiply, reduced to the top-k chunks with `argpartition`, and passed as context to the LLM, which returns the answer.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
