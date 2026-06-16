"""Pure-Python BM25 lexical scoring for hybrid retrieval.

Dense embeddings capture meaning, but they often miss exact-term matches:
names, identifiers, error codes, and rare words. BM25 is a classic lexical
ranking function that nails those. swiftrag fuses dense and BM25 rankings with
Reciprocal Rank Fusion (see :mod:`swiftrag.store`), which tends to beat either
signal alone. This module has no third-party dependencies.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer shared by the lexical index."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed corpus, using a postings index.

    Scoring only touches documents that contain a query term, so a query costs
    O(sum of postings for its terms) rather than O(corpus).
    """

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_len: list[int] = []

        total_tokens = 0
        for i, text in enumerate(documents):
            tokens = tokenize(text)
            self._doc_len.append(len(tokens))
            total_tokens += len(tokens)
            freqs: dict[str, int] = {}
            for tok in tokens:
                freqs[tok] = freqs.get(tok, 0) + 1
            for tok, freq in freqs.items():
                self._postings.setdefault(tok, []).append((i, freq))

        self.N = len(self._doc_len)
        self._avgdl = (total_tokens / self.N) if self.N else 0.0

    def __len__(self) -> int:
        return self.N

    def scores(self, query: str) -> np.ndarray:
        """Return a BM25 score per document for ``query`` (shape ``(N,)``)."""
        out = np.zeros(self.N, dtype=np.float32)
        if self.N == 0 or self._avgdl == 0:
            return out

        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for doc_idx, freq in postings:
                dl = self._doc_len[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                out[doc_idx] += idf * (freq * (self.k1 + 1)) / denom
        return out


__all__ = ["BM25", "tokenize"]
