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

from .lexical import BM25
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
        self._bm25: BM25 | None = None  # lazily (re)built for hybrid search

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        """Drop all indexed chunks and embeddings."""
        self._chunks = []
        self._matrix = None
        self._faiss_index = None
        self._bm25 = None

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
        self._bm25 = None  # corpus changed; rebuild lexical index on next hybrid query

        if self._matrix is None:
            self._matrix = normalized
        else:
            self._matrix = np.vstack([self._matrix, normalized])

        if self._use_faiss:
            self._build_faiss()

    def delete(self, predicate: Predicate) -> int:
        """Remove every chunk for which ``predicate`` returns True.

        Returns the number of chunks removed. The embedding matrix and any
        lexical/FAISS indexes are rebuilt to match.
        """
        if not self._chunks:
            return 0
        keep = [i for i, c in enumerate(self._chunks) if not predicate(c)]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0

        self._chunks = [self._chunks[i] for i in keep]
        if keep and self._matrix is not None:
            self._matrix = np.ascontiguousarray(self._matrix[keep])
        else:
            self._matrix = None

        self._bm25 = None  # lexical index is rebuilt lazily on next hybrid query
        self._faiss_index = None
        if self._use_faiss and self._matrix is not None:
            self._build_faiss()
        return removed

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
        hybrid: bool = False,
        query_text: str | None = None,
        rrf_k: int = 60,
    ) -> list[ScoredChunk]:
        if self._matrix is None or len(self._chunks) == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn

        top_k = min(top_k, len(self._chunks))

        # FAISS can't express metadata filters / MMR / hybrid, so fall back to
        # the brute-force path (still a single BLAS matmul) when those are used.
        use_faiss = (
            self._faiss_index is not None and not mmr and not hybrid and predicate is None
        )

        if hybrid:
            results = self._search_hybrid(
                q, query_text or "", top_k, candidate_pool, predicate, rrf_k
            )
        elif mmr:
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

    def _search_hybrid(
        self,
        q: np.ndarray,
        query_text: str,
        top_k: int,
        pool: int,
        predicate: Predicate | None,
        rrf_k: int,
    ) -> list[ScoredChunk]:
        """Fuse dense (cosine) and BM25 (lexical) rankings via Reciprocal Rank Fusion.

        The reported ``score`` stays the dense cosine similarity so ``min_score``
        and downstream consumers keep a consistent, interpretable meaning; only
        the *ordering* reflects the fused dense+lexical signal.
        """
        dense = self._matrix @ q
        if predicate is not None:
            dense = self._apply_predicate(dense, predicate)

        if self._bm25 is None:
            self._bm25 = BM25([c.text for c in self._chunks])
        lexical = self._bm25.scores(query_text)
        if predicate is not None:
            # Drop predicate-filtered docs from the lexical ranking too.
            lexical = np.where(np.isfinite(dense), lexical, 0.0)

        pool = min(max(pool, top_k), len(self._chunks))

        dense_rank = self._top_k_indices(dense, pool)
        dense_rank = [i for i in dense_rank if np.isfinite(dense[i])]
        lex_rank = self._top_k_indices(lexical, pool)
        lex_rank = [i for i in lex_rank if lexical[i] > 0.0]

        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense_rank):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, idx in enumerate(lex_rank):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        if not fused:
            return []
        order = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]
        return [ScoredChunk(self._chunks[i], float(dense[i])) for i in order]


__all__ = ["VectorStore"]
