"""Cross-similarity (and cost) matrices between two ``(12, T)`` feature sequences."""

from __future__ import annotations

import numpy as np


def _unit_frames(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=0, keepdims=True)
    return np.where(norms > eps, x / np.maximum(norms, eps), 0.0)


def cross_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Frame-wise cosine similarity ``S[i, j] = cos(x[:, i], y[:, j])``, shape ``(T_x, T_y)``.

    Zero (silent) frames have similarity 0 to everything.
    """
    xn = _unit_frames(x)
    yn = _unit_frames(y)
    return np.clip(xn.T @ yn, -1.0, 1.0)


def cosine_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cosine distance ``C = 1 - S``. For non-negative HPCP, ``C`` lies in [0, 1]."""
    return 1.0 - cross_similarity_matrix(x, y)
