import numpy as np

from swiftrag.store import VectorStore
from swiftrag.types import Chunk


def _chunk(text):
    return Chunk(text=text, doc_id="d", chunk_index=0)


def test_search_returns_most_similar():
    store = VectorStore()
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    store.add(chunks, embeddings)

    results = store.search(np.array([0.9, 0.1, 0.0], dtype=np.float32), top_k=2)
    assert results[0].text == "a"
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_empty_store_returns_empty():
    store = VectorStore()
    assert store.search(np.array([1.0, 0.0]), top_k=3) == []


def test_mmr_diversifies():
    store = VectorStore()
    chunks = [_chunk(x) for x in ["a", "a2", "b"]]
    embeddings = np.array([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    store.add(chunks, embeddings)
    results = store.search(np.array([1.0, 0.2], dtype=np.float32), top_k=2, mmr=True, mmr_lambda=0.3)
    texts = {r.text for r in results}
    assert "b" in texts  # diversity pulls in the orthogonal chunk
