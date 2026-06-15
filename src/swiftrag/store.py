"""In-memory vector store with a brute-force backend and an optional FAISS
backend for large corpora.

Design notes
------------
* Embeddings are stored as a single contiguous ``float32`` matrix. A query is
  a single ``matrix @ vector`` call, which runs through BLAS rather than a
  Python loop.
* All vectors are L2-normalized on insert, so cosine similarity reduces to a
  dot product (no per-query normalization of the corpus).
* Top-k uses ``np.argpartition`` (O(n)) instead of a full sort (O(n log n)).
* Optional Maximal Marginal Relevance (MMR) re-ranking for diverse results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .types import Chunk, ScoredChunk

Predicate = Callable[[Chunk], bool]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows stay zero (avoids divide-by-zero)."""
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


class VectorStore:
    """Brute-force cosine-similarity store backed by a normalized matrix."""

    def __init__(self, use_faiss: bool = False) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # (n, dim) normalized float32
        self._use_faiss = use_faiss
        self._faiss_index = None

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        """Drop all indexed chunks and embeddings."""
        self._chunks = []
        self._matrix = None
        self._faiss_index = None

    @property
    def dim(self) -> int | None:
        return None if self._matrix is None else self._matrix.shape[1]

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if len(chunks) == 0:
            return
        normalized = _normalize(np.asarray(embeddings))

        for chunk, vec in zip(chunks, normalized):
            chunk.embedding = vec
        self._chunks.extend(chunks)

        if self._matrix is None:
            self._matrix = normalized
        else:
            self._matrix = np.vstack([self._matrix, normalized])

        if self._use_faiss:
            self._build_faiss()

    def _build_faiss(self) -> None:
        try:
            import faiss
        except ImportError:
            self._use_faiss = False
            self._faiss_index = None
            return
        dim = self._matrix.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product == cosine on normalized vecs
        index.add(self._matrix)
        self._faiss_index = index

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        *,
        mmr: bool = False,
        mmr_lambda: float = 0.5,
        candidate_pool: int = 20,
        min_score: float | None = None,
        predicate: Predicate | None = None,
    ) -> list[ScoredChunk]:
        if self._matrix is None or len(self._chunks) == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn

        top_k = min(top_k, len(self._chunks))

        # FAISS can't express metadata filters / MMR, so fall back to the
        # brute-force path (still a single BLAS matmul) when those are requested.
        use_faiss = self._faiss_index is not None and not mmr and predicate is None

        if mmr:
            results = self._search_mmr(q, top_k, mmr_lambda, candidate_pool, predicate)
        elif use_faiss:
            scores, idx = self._faiss_index.search(q.reshape(1, -1), top_k)
            results = [
                ScoredChunk(self._chunks[i], float(s))
                for i, s in zip(idx[0], scores[0])
                if i != -1
            ]
        else:
            sims = self._matrix @ q
            if predicate is not None:
                sims = self._apply_predicate(sims, predicate)
            idx = self._top_k_indices(sims, top_k)
            results = [
                ScoredChunk(self._chunks[i], float(sims[i]))
                for i in idx
                if np.isfinite(sims[i])
            ]

        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results

    def _apply_predicate(self, sims: np.ndarray, predicate: Predicate) -> np.ndarray:
        sims = sims.copy()
        for i, chunk in enumerate(self._chunks):
            if not predicate(chunk):
                sims[i] = -np.inf
        return sims

    @staticmethod
    def _top_k_indices(sims: np.ndarray, k: int) -> np.ndarray:
        if k >= len(sims):
            return np.argsort(sims)[::-1]
        part = np.argpartition(sims, -k)[-k:]
        return part[np.argsort(sims[part])[::-1]]

    def _search_mmr(
        self,
        q: np.ndarray,
        top_k: int,
        lambda_: float,
        pool: int,
        predicate: Predicate | None = None,
    ) -> list[ScoredChunk]:
        sims = self._matrix @ q
        if predicate is not None:
            sims = self._apply_predicate(sims, predicate)
        pool = min(max(pool, top_k), len(self._chunks))
        cand = self._top_k_indices(sims, pool)
        cand = cand[np.isfinite(sims[cand])]
        if len(cand) == 0:
            return []
        cand_vecs = self._matrix[cand]
        cand_sims = sims[cand]

        selected: list[int] = []
        selected_mask = np.zeros(len(cand), dtype=bool)
        # Precompute candidate-candidate similarity (small pool, cheap).
        cc = cand_vecs @ cand_vecs.T

        for _ in range(min(top_k, len(cand))):
            if not selected:
                pick = int(np.argmax(cand_sims))
            else:
                redundancy = cc[:, selected].max(axis=1)
                mmr_score = lambda_ * cand_sims - (1 - lambda_) * redundancy
                mmr_score[selected_mask] = -np.inf
                pick = int(np.argmax(mmr_score))
            selected.append(pick)
            selected_mask[pick] = True

        return [ScoredChunk(self._chunks[cand[i]], float(cand_sims[i])) for i in selected]


__all__ = ["VectorStore"]
