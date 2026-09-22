"""Ranked-retrieval metrics (Manning, Raghavan & Schütze, IR book ch. 8; MIREX CSI).

Every function takes a *ranked relevance vector*: element ``r`` is True when the
candidate at rank ``r + 1`` is relevant (same WID as the query). The query itself
must already be excluded from the ranking; ``ranking_from_scores`` does that.

In a protocol where every query has exactly one relevant item (the development
pair protocol), AP equals reciprocal rank by construction, so MAP equals MRR.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_KS: tuple[int, ...] = (1, 10, 100)


def _as_bool(relevance: Sequence[bool] | np.ndarray) -> np.ndarray:
    array = np.asarray(relevance, dtype=bool)
    if array.ndim != 1:
        raise ValueError("relevance must be a 1-D ranked vector")
    return array


def first_relevant_rank(relevance: Sequence[bool] | np.ndarray) -> int | None:
    """1-based rank of the first relevant item, or ``None`` when there is none."""
    hits = np.flatnonzero(_as_bool(relevance))
    return int(hits[0]) + 1 if hits.size else None


def reciprocal_rank(relevance: Sequence[bool] | np.ndarray) -> float:
    rank = first_relevant_rank(relevance)
    return 0.0 if rank is None else 1.0 / rank


def average_precision(relevance: Sequence[bool] | np.ndarray, n_relevant: int | None = None) -> float:
    """AP = (1/R) * sum over relevant ranks k of precision@k.

    ``n_relevant`` (R) defaults to the number of relevant items in the ranking; pass
    the collection total if the ranking is truncated so missed items count as zero.
    """
    rel = _as_bool(relevance)
    total = int(rel.sum()) if n_relevant is None else int(n_relevant)
    if total == 0:
        return 0.0
    ranks = np.flatnonzero(rel) + 1
    precisions = np.arange(1, ranks.size + 1) / ranks
    return float(precisions.sum() / total)


def recall_at_k(relevance: Sequence[bool] | np.ndarray, k: int, n_relevant: int | None = None) -> float:
    """Fraction of all relevant items retrieved within the top ``k``."""
    rel = _as_bool(relevance)
    total = int(rel.sum()) if n_relevant is None else int(n_relevant)
    return 0.0 if total == 0 else float(rel[:k].sum() / total)


def hit_at_k(relevance: Sequence[bool] | np.ndarray, k: int) -> float:
    """1.0 if at least one relevant item is in the top ``k`` (a.k.a. Top-k accuracy)."""
    return float(_as_bool(relevance)[:k].any())


def ranking_from_scores(
    scores: np.ndarray, relevant: np.ndarray, exclude: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sort candidates by descending score and return ``(order, ranked_relevance)``.

    ``exclude`` marks candidates to drop (the query itself). Ties are broken by the
    candidate's position in the input, which manifests fix to PID order, so rankings
    are deterministic.
    """
    scores = np.asarray(scores, dtype=np.float64)
    keep = np.ones_like(scores, dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool)
    candidates = np.flatnonzero(keep)
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    return order, np.asarray(relevant, dtype=bool)[order]


@dataclass
class MetricsSummary:
    n_queries: int
    map: float
    mrr: float
    recall: dict[int, float]
    hit: dict[int, float]
    mean_first_rank: float
    median_first_rank: float
    mean_relevant_per_query: float

    def as_dict(self) -> dict:
        data = asdict(self)
        data["recall"] = {f"@{k}": v for k, v in self.recall.items()}
        data["hit"] = {f"@{k}": v for k, v in self.hit.items()}
        return data


@dataclass
class QueryMetrics:
    ap: float
    rr: float
    first_rank: int | None
    n_relevant: int
    recall: dict[int, float]
    hit: dict[int, float]


def query_metrics(relevance: Sequence[bool] | np.ndarray, ks: Iterable[int] = DEFAULT_KS) -> QueryMetrics:
    rel = _as_bool(relevance)
    ks = tuple(ks)
    return QueryMetrics(
        ap=average_precision(rel),
        rr=reciprocal_rank(rel),
        first_rank=first_relevant_rank(rel),
        n_relevant=int(rel.sum()),
        recall={k: recall_at_k(rel, k) for k in ks},
        hit={k: hit_at_k(rel, k) for k in ks},
    )


def summarize(per_query: Sequence[QueryMetrics], ks: Iterable[int] = DEFAULT_KS) -> MetricsSummary:
    ks = tuple(ks)
    if not per_query:
        raise ValueError("no queries to summarise")
    if any(q.n_relevant == 0 for q in per_query):
        raise ValueError("every evaluated query needs at least one relevant candidate")
    first = np.array([q.first_rank for q in per_query], dtype=np.float64)
    return MetricsSummary(
        n_queries=len(per_query),
        map=float(np.mean([q.ap for q in per_query])),
        mrr=float(np.mean([q.rr for q in per_query])),
        recall={k: float(np.mean([q.recall[k] for q in per_query])) for k in ks},
        hit={k: float(np.mean([q.hit[k] for q in per_query])) for k in ks},
        mean_first_rank=float(first.mean()),
        median_first_rank=float(np.median(first)),
        mean_relevant_per_query=float(np.mean([q.n_relevant for q in per_query])),
    )


def evaluate_rankings(
    rankings: Iterable[Sequence[bool] | np.ndarray], ks: Iterable[int] = DEFAULT_KS
) -> MetricsSummary:
    """Summarise a collection of ranked relevance vectors (one per query)."""
    ks = tuple(ks)
    return summarize([query_metrics(r, ks) for r in rankings], ks)
