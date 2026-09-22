"""Two-stage hybrid retrieval: global embedding shortlist -> alignment reranking.

For each query:

1. Stage 1 ranks every candidate (self excluded) by global embedding cosine.
2. Stage 2 takes the top ``K`` of that ranking, scores each with the classical
   transposition-aware subsequence-DTW aligner, and orders the shortlist by

       hybrid = alpha * z(global) + (1 - alpha) * z(alignment)

   where ``z`` standardises each score within the query's shortlist (per-query,
   label-free, so the two scales become comparable). ``alpha = 1`` reproduces
   Stage 1; ``alpha = 0`` is pure alignment reranking of the shortlist.
3. The final ranking is the reranked shortlist followed by the remaining candidates
   in their untouched Stage-1 order, so it is always a complete ranking.

``alpha`` is chosen on the calibration protocol only (``calibrate_alpha``) and then
frozen for the development and benchmark evaluations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cover_retrieval.data.manifests import Protocol
from cover_retrieval.evaluation.metrics import DEFAULT_KS, MetricsSummary, query_metrics, ranking_from_scores
from cover_retrieval.retrieval.rank import ClassicalAligner, RankedQuery, summarize_ranked


def zscore(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros_like(values)
    std = values.std()
    return np.zeros_like(values) if std < eps else (values - values.mean()) / std


@dataclass
class Shortlist:
    query_index: int
    global_order: np.ndarray  # full Stage-1 order (self excluded)
    global_relevance: np.ndarray
    shortlist: np.ndarray  # first K candidate indices of global_order
    global_scores: np.ndarray  # (K,)
    align_scores: np.ndarray  # (K,)
    align_shifts: np.ndarray  # (K,)


@dataclass
class HybridRun:
    shortlists: list[Shortlist]
    k: int
    timings: dict[str, float] = field(default_factory=dict)

    def rank(self, alpha: float, ks: Sequence[int] = DEFAULT_KS) -> list[RankedQuery]:
        ranked = []
        for s in self.shortlists:
            k = s.shortlist.size
            hybrid = alpha * zscore(s.global_scores) + (1.0 - alpha) * zscore(s.align_scores)
            head = np.argsort(-hybrid, kind="stable")  # ties keep Stage-1 order
            order = np.concatenate([s.shortlist[head], s.global_order[k:]])
            relevance = np.concatenate([s.global_relevance[:k][head], s.global_relevance[k:]])
            ranked.append(RankedQuery(s.query_index, order, relevance, query_metrics(relevance, ks)))
        return ranked

    def evaluate(self, alpha: float, ks: Sequence[int] = DEFAULT_KS) -> MetricsSummary:
        return summarize_ranked(self.rank(alpha, ks), ks)

    def shortlist_recall(self) -> float:
        """Fraction of all relevant items that survive Stage 1 (upper bound for reranking)."""
        kept = sum(int(s.global_relevance[: s.shortlist.size].sum()) for s in self.shortlists)
        total = sum(int(s.global_relevance.sum()) for s in self.shortlists)
        return kept / total if total else 0.0


def build_shortlists(
    global_scores: np.ndarray,
    protocol: Protocol,
    query_features: np.ndarray,
    candidate_features: np.ndarray,
    aligner: ClassicalAligner,
    k: int,
) -> HybridRun:
    """Run Stage 1 for every query and align each shortlist once (alpha-independent)."""
    relevance = protocol.relevance_matrix()
    self_mask = protocol.self_mask()
    shortlists = []
    t_rank = t_align = 0.0
    for i in range(len(protocol.queries)):
        t0 = time.perf_counter()
        order, rel = ranking_from_scores(global_scores[i], relevance[i], exclude=self_mask[i])
        head = order[:k]
        t1 = time.perf_counter()
        pair = aligner.score_pairs(query_features[i], candidate_features[head])
        t2 = time.perf_counter()
        t_rank += t1 - t0
        t_align += t2 - t1
        shortlists.append(Shortlist(i, order, rel, head, global_scores[i, head], pair.score, pair.shift))
    return HybridRun(shortlists, k, {"shortlist_seconds": t_rank, "rerank_seconds": t_align})


@dataclass
class Calibration:
    alpha: float
    k: int
    table: list[dict[str, float]]


def calibrate_alpha(run: HybridRun, step: float = 0.05) -> Calibration:
    """Grid-search alpha in [0, 1] on a calibration run.

    Selection: highest MAP; ties -> lower mean first-relevant rank; then alpha closest
    to 0.5 (least extreme blend).
    """
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 6)
    table = []
    for alpha in grid:
        summary = run.evaluate(float(alpha))
        table.append({"alpha": float(alpha), "map": summary.map, "mean_first_rank": summary.mean_first_rank})
    best = max(table, key=lambda r: (round(r["map"], 12), -r["mean_first_rank"], -abs(r["alpha"] - 0.5)))
    return Calibration(alpha=best["alpha"], k=run.k, table=table)
