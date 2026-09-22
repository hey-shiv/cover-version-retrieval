"""Classical structure-aware pair scoring and protocol-level ranking.

Pair score (query ``Q``, candidate ``Y``; both ``(12, T)`` preprocessed HPCP):

1. score all 12 cyclic rotations of the candidate by cosine between time-averaged
   chroma profiles; keep the best rotation (``profile_cosine``). Ablations: run DTW
   for every rotation and keep the cheapest (``exhaustive_dtw``), or disable key
   handling entirely (``none``, shift 0);
2. cosine-distance cost matrix between query frames and rotated candidate frames;
3. subsequence DTW with slope-constrained steps, cost normalised by weighted path
   length;
4. retrieval score = 1 - normalised cost (higher is better).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cover_retrieval.alignment.dtw import batch_subsequence_dtw
from cover_retrieval.alignment.transposition import batch_profile_rotation_scores, chroma_profile
from cover_retrieval.data.manifests import Protocol
from cover_retrieval.evaluation.metrics import (
    DEFAULT_KS,
    MetricsSummary,
    QueryMetrics,
    query_metrics,
    ranking_from_scores,
    summarize,
)


@dataclass
class AlignmentSettings:
    rotation_selection: str = "profile_cosine"
    step_sizes: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2))
    step_weights: tuple[float, ...] = (1.0, 2.0, 1.0)
    batch_size: int = 4096
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AlignmentSettings:
        align = config["alignment"]
        if not align.get("subsequence", True):
            raise ValueError("the retrieval pipeline uses subsequence DTW (alignment.subsequence: true)")
        return cls(
            rotation_selection=align["rotation_selection"],
            step_sizes=tuple(tuple(s) for s in align["step_sizes"]),
            step_weights=tuple(float(w) for w in align["step_weights"]),
            batch_size=int(config["retrieval"].get("dtw_batch_size", 4096)),
        )


@dataclass
class PairScores:
    score: np.ndarray  # (B,) 1 - normalised DTW cost; higher = more similar
    shift: np.ndarray  # (B,) rotation applied to each candidate
    cost: np.ndarray  # (B,) normalised DTW cost


class ClassicalAligner:
    """Transposition-aware subsequence-DTW scorer (the classical baseline / reranker)."""

    def __init__(self, settings: AlignmentSettings) -> None:
        if settings.rotation_selection not in ("profile_cosine", "exhaustive_dtw", "none"):
            raise ValueError(f"unknown rotation_selection {settings.rotation_selection!r}")
        self.settings = settings

    def _dtw_cost(
        self, query_rolled: torch.Tensor, candidates: torch.Tensor, shifts: torch.Tensor
    ) -> torch.Tensor:
        # Rotating the candidate by k equals rotating the query by -k (a permutation of
        # the pitch axis), so gather the pre-rolled query: (B, N, 12) @ (B, 12, M).
        q = query_rolled[(-shifts) % 12]
        cost = (1.0 - torch.bmm(q, candidates)).clamp_(0.0, 2.0)
        normalized, _ = batch_subsequence_dtw(cost, self.settings.step_sizes, self.settings.step_weights)
        return normalized

    def score_pairs(self, query: np.ndarray, candidates: np.ndarray) -> PairScores:
        """Score one ``(12, N)`` query against ``(B, 12, M)`` candidates."""
        s = self.settings
        if candidates.shape[0] == 0:
            empty = np.zeros(0)
            return PairScores(empty, empty.astype(int), empty)
        q_profile = chroma_profile(query)
        c_profiles = chroma_profile(candidates)
        rotation_scores = batch_profile_rotation_scores(q_profile, c_profiles)  # (B, 12)
        q = torch.as_tensor(query, dtype=s.dtype, device=s.device)
        query_rolled = torch.stack([torch.roll(q, k, dims=0).T for k in range(12)])  # (12, N, 12)

        costs, shifts = [], []
        for start in range(0, candidates.shape[0], s.batch_size):
            block = torch.as_tensor(candidates[start : start + s.batch_size], dtype=s.dtype, device=s.device)
            if s.rotation_selection in ("profile_cosine", "none"):
                if s.rotation_selection == "none":
                    shift = torch.zeros(block.shape[0], dtype=torch.long, device=s.device)
                else:
                    shift = torch.as_tensor(
                        np.argmax(rotation_scores[start : start + s.batch_size], axis=1), device=s.device
                    )
                cost = self._dtw_cost(query_rolled, block, shift)
            else:
                all_costs = torch.stack(
                    [
                        self._dtw_cost(query_rolled, block, torch.full((block.shape[0],), k, device=s.device))
                        for k in range(12)
                    ],
                    dim=1,
                )
                cost, shift = torch.min(all_costs, dim=1)
            costs.append(cost.cpu().double().numpy())
            shifts.append(shift.cpu().numpy())
        cost = np.concatenate(costs)
        return PairScores(score=1.0 - cost, shift=np.concatenate(shifts).astype(int), cost=cost)

    def score_matrix(self, queries: np.ndarray, candidates: np.ndarray, verbose: bool = False):
        """Dense ``(Q, C)`` score (float32) and shift (int8) matrices."""
        scores = np.zeros((queries.shape[0], candidates.shape[0]), dtype=np.float32)
        shifts = np.zeros((queries.shape[0], candidates.shape[0]), dtype=np.int8)
        for i, query in enumerate(queries):
            pair = self.score_pairs(query, candidates)
            scores[i], shifts[i] = pair.score, pair.shift
            if verbose and (i + 1) % 100 == 0:
                print(f"  aligned {i + 1}/{queries.shape[0]} queries", flush=True)
        return scores, shifts


@dataclass
class RankedQuery:
    query_index: int
    order: np.ndarray  # candidate indices, best first, self excluded
    relevance: np.ndarray  # relevance along ``order``
    metrics: QueryMetrics


def rank_protocol(
    scores: np.ndarray, protocol: Protocol, ks: Sequence[int] = DEFAULT_KS
) -> list[RankedQuery]:
    """Rank every candidate for every query (self excluded) from a ``(Q, C)`` score matrix."""
    relevance = protocol.relevance_matrix()
    self_mask = protocol.self_mask()
    ranked = []
    for i in range(len(protocol.queries)):
        order, rel = ranking_from_scores(scores[i], relevance[i], exclude=self_mask[i])
        ranked.append(RankedQuery(i, order, rel, query_metrics(rel, ks)))
    return ranked


def summarize_ranked(ranked: Sequence[RankedQuery], ks: Sequence[int] = DEFAULT_KS) -> MetricsSummary:
    return summarize([r.metrics for r in ranked], ks)


def ranking_rows(
    ranked: Sequence[RankedQuery],
    protocol: Protocol,
    scores: np.ndarray,
    top_n: int | None = None,
    extra: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Flatten rankings to CSV rows (top ``top_n`` plus every relevant item)."""
    rows = []
    for r in ranked:
        q = protocol.queries[r.query_index]
        for position, (cand, rel) in enumerate(zip(r.order, r.relevance, strict=True), 1):
            if top_n is not None and position > top_n and not rel:
                continue
            c = protocol.candidates[cand]
            row = {
                "query_pid": q.pid,
                "query_wid": q.wid,
                "rank": position,
                "candidate_pid": c.pid,
                "candidate_wid": c.wid,
                "score": float(scores[r.query_index, cand]),
                "is_relevant": bool(rel),
            }
            for name, values in (extra or {}).items():
                row[name] = values[r.query_index, cand].item()
            rows.append(row)
    return rows
