"""Exact cosine retrieval over L2-normalised embeddings.

The default backend is a dense NumPy inner product (exact, dependency-free). FAISS
(``IndexFlatIP``, also exact) is used only when requested *and* importable; if it is
missing the index falls back to NumPy with a warning. FAISS is never required.
"""

from __future__ import annotations

import warnings

import numpy as np


class EmbeddingIndex:
    def __init__(self, embeddings: np.ndarray, backend: str = "numpy") -> None:
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("EmbeddingIndex expects L2-normalised embeddings")
        self.backend = backend
        self._faiss_index = None
        if backend == "faiss":
            try:
                import faiss  # type: ignore[import-not-found]

                self._faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self._faiss_index.add(self.embeddings)
            except ImportError:
                warnings.warn("faiss not installed; falling back to exact NumPy search", stacklevel=2)
                self.backend = "numpy"
        elif backend != "numpy":
            raise ValueError(f"unknown backend {backend!r}")

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def scores(self, queries: np.ndarray) -> np.ndarray:
        """Dense ``(Q, C)`` cosine similarities (always exact NumPy)."""
        return np.asarray(queries, dtype=np.float32) @ self.embeddings.T

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` ``(indices, scores)`` per query, best first (ties: lower index first)."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        k = min(k, len(self))
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(queries, k)
            return indices, scores
        sims = self.scores(queries)
        indices = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        return indices, np.take_along_axis(sims, indices, axis=1)


def rotation_max_scores(
    query_embeddings_by_rotation: np.ndarray, candidate_embeddings: np.ndarray
) -> np.ndarray:
    """Test-time transposition: max over the 12 rotated-query embeddings, ``(R, Q, D)`` -> ``(Q, C)``."""
    return np.max(np.einsum("rqd,cd->rqc", query_embeddings_by_rotation, candidate_embeddings), axis=0)
