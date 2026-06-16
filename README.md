# swiftrag

[![CI](https://github.com/behradmoeini/swiftrag/actions/workflows/ci.yml/badge.svg)](https://github.com/behradmoeini/swiftrag/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/swiftrag.svg)](https://pypi.org/project/swiftrag/)
[![Python versions](https://img.shields.io/pypi/pyversions/swiftrag.svg)](https://pypi.org/project/swiftrag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/behradmoeini/swiftrag/blob/main/LICENSE)

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
pip install "swiftrag[cohere]"       # Cohere embeddings
pip install "swiftrag[gemini]"       # Google Gemini embeddings and LLM
pip install "swiftrag[loaders]"      # PDF, DOCX, and richer HTML/URL loading
pip install "swiftrag[faiss]"        # FAISS backend for big corpora
pip install "swiftrag[all]"          # everything

# Ollama needs no extra package — just a running Ollama server.
```

## Usage

### Pick your models with a `"provider:model"` string

```python
# OpenAI (needs OPENAI_API_KEY)
RAG(documents=text, embedding_model="openai:text-embedding-3-small", llm_model="openai:gpt-4o-mini")

# Anthropic for generation, local embeddings (no embedding API calls)
RAG(documents=text, embedding_model="local:all-MiniLM-L6-v2", llm_model="anthropic:claude-3-5-sonnet-latest")

# Google Gemini, or Cohere embeddings
RAG(documents=text, embedding_model="cohere:embed-english-v3.0", llm_model="gemini:gemini-1.5-flash")

# Fully local and free via Ollama (no API keys, no extra packages)
RAG(documents=text, embedding_model="ollama:nomic-embed-text", llm_model="ollama:llama3")

# Fully offline (default), no keys required
RAG(documents=text)
```

Supported provider prefixes: embeddings — `openai`, `local`/`st`, `ollama`,
`cohere`, `gemini`, `hash` (offline default); LLMs — `openai`, `anthropic`,
`ollama`, `gemini`, `echo` (offline default). Set `OLLAMA_HOST` to target a
remote Ollama server.

### Build straight from files, PDFs, or the web

```python
# Folders and plain-text files work out of the box.
rag = RAG.from_files("docs/", embedding_model="openai:text-embedding-3-small")
# each file becomes a document tagged with metadata={"source": <path>}

resp = rag.query("What's our deployment process?")
print(resp.answer)
print(resp.format_sources())   # numbered, readable citations
```

`from_files` also reads PDF, DOCX, and HTML when you install the loaders extra
(`pip install "swiftrag[loaders]"`); HTML works even without it via a stdlib
fallback. You can also pull straight from the web:

```python
# Mixed folder: .txt/.md/.pdf/.docx/.html are all picked up and parsed.
rag = RAG.from_files(["handbook.pdf", "notes/", "page.html"])

# Fetch and index web pages (or PDF URLs).
rag = RAG.from_url("https://example.com/docs", llm_model="openai:gpt-4o-mini")

# Use a single loader directly if you want the raw text.
from swiftrag import load_pdf, load_url
text = load_pdf("handbook.pdf")
text = load_url("https://example.com")
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

### Hybrid search and reranking

Dense embeddings capture meaning but often miss exact-term matches (names, IDs,
error codes, rare words). Turn on hybrid retrieval to fuse dense similarity with
a built-in BM25 lexical score using Reciprocal Rank Fusion. It needs no extra
dependencies and noticeably helps the offline hash embedder too.

```python
rag = RAG(documents=docs, use_hybrid=True)
rag.query("error code E1042")        # exact tokens now pull their way up

# Or decide per call:
rag.retrieve("Zorblax42", hybrid=True)
```

Need a stronger final ordering? Plug in any reranker — a callable
`(query, list[ScoredChunk]) -> list[ScoredChunk]` runs after retrieval:

```python
def my_reranker(query, hits):
    # e.g. a cross-encoder or a hosted rerank API; return reordered hits
    return sorted(hits, key=lambda h: my_cross_encoder(query, h.text), reverse=True)

rag = RAG(documents=docs, use_hybrid=True, reranker=my_reranker)
```

### Add, update, and delete documents

```python
rag = RAG().add("first batch").add("second batch")

# Delete by metadata filter (or a Chunk -> bool predicate). Returns the count removed.
rag.delete({"source": "outdated.pdf"})

# Update = delete matching + add fresh (handy when a source changes).
rag.update("the new handbook text", where={"source": "handbook"})
```

### Save and reload (pickle-free by default)

```python
rag.save("index.swiftrag")          # safe zip format (meta.json + matrix.npy)
rag = RAG.load("index.swiftrag", embedding_model="openai:text-embedding-3-small",
               llm_model="openai:gpt-4o-mini")
```

The default `"safe"` format carries no executable payload, so it's safe to
share and load from untrusted sources. The format is auto-detected on load, and
the legacy pickle format still works (`.pkl`/`.pickle` paths use it
automatically, or pass `format="pickle"`).

### Multi-turn conversations

`rag.chat()` returns a stateful conversation that remembers history and rewrites
follow-ups into standalone queries before retrieval, so references like "it" or
"that" resolve correctly across turns.

```python
chat = rag.chat()
print(chat.ask("Who owns the deployment runbook?").answer)
print(chat.ask("And when was it last updated?").answer)  # "it" understood from context

for token in chat.stream("Summarize what we just discussed."):
    print(token, end="", flush=True)

chat.reset()   # clear history, keep the index
```

Query rewriting uses the configured LLM and is skipped automatically in offline
mode. Disable it with `rag.chat(rewrite_queries=False)`, and cap remembered
turns with `max_history_turns`.

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
| `use_hybrid` | `False` | Fuse dense + BM25 lexical ranking with Reciprocal Rank Fusion. |
| `reranker` | `None` | Optional `(query, sources) -> sources` callable applied after retrieval. |
| `on_retrieve` | `None` | Hook `(query, sources) -> None` called after retrieval. |
| `on_generate` | `None` | Hook `(query, answer, usage) -> None` called after generation. |
| `use_faiss` | `False` | Use FAISS index (install `swiftrag[faiss]`). |
| `min_score` | `None` | Default cosine threshold for dropping weak matches. |
| `max_context_tokens` | `None` | Cap the tokens of retrieved context packed into the prompt. |
| `dedup` | `True` | Skip exact-duplicate chunks on ingest. |
| `query_cache_size` | `128` | LRU size for cached query embeddings (`0` disables). |
| `system_prompt` | grounded default | System prompt for the LLM. |

## Production niceties

Provider API calls (OpenAI, Anthropic, Gemini, Cohere, Ollama) are automatically
retried with exponential backoff and jitter on transient failures. Tune it per
provider with `max_retries`:

```python
RAG(documents=docs, llm_model="openai:gpt-4o-mini", llm_kwargs={"max_retries": 5})
```

Token usage is attached to each response when the provider reports it, and you
can hook into retrieval and generation for logging, tracing, or cost tracking:

```python
resp = rag.query("...")
print(resp.usage)   # e.g. {"prompt_tokens": 812, "completion_tokens": 95, ...}

rag = RAG(
    documents=docs,
    llm_model="openai:gpt-4o-mini",
    on_retrieve=lambda q, sources: log.info("retrieved %d chunks for %r", len(sources), q),
    on_generate=lambda q, answer, usage: meter.record(usage),
)
```

Hooks are best-effort: an exception inside a callback is logged and never breaks
the query.

## Evaluation

Tune your pipeline with a small labelled set instead of guessing. `evaluate_retrieval`
reports hit-rate, MRR, and precision@k, and `groundedness` estimates how well an
answer is supported by its sources.

```python
from swiftrag import evaluate_retrieval, groundedness

examples = [
    {"query": "What's our refund window?", "relevant": ["handbook"]},  # match by metadata source
    {"query": "How do we deploy?", "relevant": ["runbook"]},
]
report = evaluate_retrieval(rag, examples, top_k=5)
print(report)   # RetrievalReport(n=2, top_k=5, hit_rate=1.000, mrr=0.875, precision=0.300)

resp = rag.query("What's our refund window?")
print(groundedness(resp.answer, resp.sources))   # 0.0–1.0 lexical-coverage estimate
# Pass llm=... for a model-judged groundedness score instead.
```

## How it works

On ingest, each document is chunked (token-aware), embedded in batches, normalized, and stored in the vector index. On a query, the question is embedded, compared to the index with a cosine similarity matrix multiply, reduced to the top-k chunks with `argpartition`, and passed as context to the LLM, which returns the answer.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT. See [LICENSE](https://github.com/behradmoeini/swiftrag/blob/main/LICENSE).
