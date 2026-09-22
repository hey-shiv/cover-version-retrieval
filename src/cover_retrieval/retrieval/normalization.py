"""Hubness correction for retrieval scores.

Some tracks are "hubs": they score highly against almost every query. In this
project the mechanism is measurable — tonally static tracks get uniformly low DTW
cost, and tonal dispersion correlates with top-10 false-positive count at
Spearman -0.77 on the development pool (``reports/error_analysis.md``).

The correction penalises a candidate by how well it scores *in general*::

    corrected[q, c] = score[q, c] - lam * reference[c]

``reference[c]`` is estimated from the score matrix itself, using no labels:

* ``mean``  – the mean score of candidate ``c`` over all queries;
* ``topk``  – the mean of candidate ``c``'s ``k`` highest scores, which is the
  candidate-side term of CSLS (Conneau et al., 2018) and is more robust when only
  a few queries are relevant to ``c``.

The query-side CSLS term is deliberately omitted: it is constant within a query's
ranking and therefore cannot change that ranking.

``lam`` is a single scalar and must be calibrated on held-out calibration works,
never on evaluation queries.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ReferenceMethod = Literal["mean", "topk"]


def hub_reference(
    scores: np.ndarray,
    method: ReferenceMethod = "topk",
    k: int = 10,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Per-candidate "general popularity" reference from a ``(Q, C)`` score matrix.

    ``exclude`` (``(Q, C)`` bool, typically the self-match mask) marks entries that
    must not contribute. Returns ``(C,)`` float64.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a (Q, C) score matrix, got {values.shape}")
    if exclude is not None:
        values = np.where(np.asarray(exclude, dtype=bool), np.nan, values)
    if method == "mean":
        return np.nanmean(values, axis=0)
    if method == "topk":
        if k < 1:
            raise ValueError("k must be >= 1")
        n_queries = values.shape[0]
        k = min(k, n_queries)
        # partial sort per column: the k largest, ignoring excluded entries
        filled = np.where(np.isnan(values), -np.inf, values)
        top = np.partition(filled, n_queries - k, axis=0)[n_queries - k :]
        top = np.where(np.isfinite(top), top, np.nan)
        return np.nanmean(top, axis=0)
    raise ValueError(f"unknown reference method {method!r}")


def apply_hub_correction(scores: np.ndarray, reference: np.ndarray, lam: float) -> np.ndarray:
    """``scores - lam * reference`` broadcast over queries (ranking-equivalent to CSLS)."""
    reference = np.nan_to_num(np.asarray(reference, dtype=np.float64), nan=0.0)
    if reference.shape[0] != scores.shape[-1]:
        raise ValueError("reference must have one entry per candidate")
    return np.asarray(scores, dtype=np.float64) - float(lam) * reference[None, :]
