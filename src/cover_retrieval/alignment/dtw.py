"""Dynamic time warping: a transparent reference implementation and a batched one.

``dtw`` is the readable dynamic program with backtracking (FMP, Müller 2021, §3.2),
used for tests, plots and error analysis. ``batch_subsequence_dtw`` computes the same
recurrence for many cost matrices at once with PyTorch and is what the retrieval
pipeline uses; tests check that both agree exactly on random inputs.

Recurrence for step sizes Σ and step weights w::

    D[i, j] = min_{(di, dj) in Σ} D[i - di, j - dj] + w(di, dj) * C[i, j]

* full DTW:        D[0, 0] = C[0, 0]; the path ends at (N-1, M-1).
* subsequence DTW: D[0, j] = C[0, j] for every j (the query may start anywhere in
  the candidate); the path ends at argmin_j D[N-1, j].

Normalisation (D-006): the returned ``normalized_cost`` is the accumulated weighted
cost divided by the accumulated step weight along the chosen path ("weighted path
length"). With the project default Σ = {(1,1), (2,1), (1,2)} and weights
{1, 2, 1}, every admissible subsequence path has weighted length exactly N (each
query frame is paid for once), so the dynamic program and the normalised score
optimise the same objective. The slope constraint (between 1/2 and 2) forbids the
degenerate paths that plain {(1,0), (0,1), (1,1)} steps allow, such as matching the
whole query against one candidate frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

CLASSIC_STEPS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1))
SLOPE_STEPS: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2))
SLOPE_WEIGHTS: tuple[float, ...] = (1.0, 2.0, 1.0)


@dataclass
class DTWResult:
    normalized_cost: float  # total_cost / path_weight
    total_cost: float  # accumulated weighted cost at the end cell
    path_weight: float  # accumulated step weight along the path
    path: np.ndarray  # (L, 2) array of (query index, candidate index)
    accumulated: np.ndarray  # (N, M) accumulated cost matrix D

    @property
    def start(self) -> tuple[int, int]:
        return int(self.path[0, 0]), int(self.path[0, 1])

    @property
    def end(self) -> tuple[int, int]:
        return int(self.path[-1, 0]), int(self.path[-1, 1])


def _check_steps(step_sizes: Sequence[Sequence[int]], step_weights: Sequence[float] | None):
    steps = [tuple(int(v) for v in s) for s in step_sizes]
    if any(di < 0 or dj < 0 or di + dj == 0 for di, dj in steps):
        raise ValueError(f"invalid step sizes {steps}")
    weights = [1.0] * len(steps) if step_weights is None else [float(w) for w in step_weights]
    if len(weights) != len(steps):
        raise ValueError("step_weights must match step_sizes")
    return steps, weights


def dtw(
    cost: np.ndarray,
    step_sizes: Sequence[Sequence[int]] = CLASSIC_STEPS,
    step_weights: Sequence[float] | None = None,
    subsequence: bool = False,
) -> DTWResult:
    """Reference DTW / subsequence DTW with backtracking. ``cost`` is ``(N, M)``."""
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or 0 in c.shape:
        raise ValueError(f"cost matrix must be non-empty 2-D, got {c.shape}")
    steps, weights = _check_steps(step_sizes, step_weights)
    n, m = c.shape
    acc = np.full((n, m), np.inf)
    acc_weight = np.zeros((n, m))
    back = np.full((n, m), -1, dtype=np.int64)
    if subsequence:
        acc[0, :] = c[0, :]
        acc_weight[0, :] = 1.0
    else:
        acc[0, 0] = c[0, 0]
        acc_weight[0, 0] = 1.0

    for i in range(n):
        for j in range(m):
            if i == 0 and (subsequence or j == 0):
                continue  # initialised above
            best, best_step = np.inf, -1
            for s, (di, dj) in enumerate(steps):
                pi, pj = i - di, j - dj
                if pi < 0 or pj < 0 or not np.isfinite(acc[pi, pj]):
                    continue
                candidate = acc[pi, pj] + weights[s] * c[i, j]
                if candidate < best:  # strict: ties keep the earlier step
                    best, best_step = candidate, s
            if best_step >= 0:
                acc[i, j] = best
                back[i, j] = best_step
                di, dj = steps[best_step]
                acc_weight[i, j] = acc_weight[i - di, j - dj] + weights[best_step]

    end_i = n - 1
    end_j = int(np.argmin(acc[end_i])) if subsequence else m - 1
    if not np.isfinite(acc[end_i, end_j]):
        raise ValueError("no admissible warping path for these step sizes and shapes")

    path = [(end_i, end_j)]
    i, j = end_i, end_j
    while back[i, j] >= 0:
        di, dj = steps[back[i, j]]
        i, j = i - di, j - dj
        path.append((i, j))
    path_arr = np.array(path[::-1], dtype=np.int64)
    total = float(acc[end_i, end_j])
    weight = float(acc_weight[end_i, end_j])
    return DTWResult(total / weight, total, weight, path_arr, acc)


def batch_subsequence_dtw(
    cost: torch.Tensor,
    step_sizes: Sequence[Sequence[int]] = SLOPE_STEPS,
    step_weights: Sequence[float] | None = SLOPE_WEIGHTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsequence DTW for a batch of cost matrices ``(B, N, M)``.

    Returns ``(normalized_cost (B,), end_column (B,))``; unreachable ends give ``inf``.
    Every step must advance the query index (``di >= 1``) so each row depends only on
    earlier rows and can be computed for all columns and batch items at once.
    Tie-breaking matches ``dtw`` (earliest step in ``step_sizes``, earliest end column).
    """
    steps, weights = _check_steps(step_sizes, step_weights)
    if any(di < 1 for di, _ in steps):
        raise ValueError("batch_subsequence_dtw requires every step to advance the query (di >= 1)")
    if cost.ndim != 3:
        raise ValueError(f"expected (B, N, M) costs, got {tuple(cost.shape)}")
    batch, n, m = cost.shape
    max_di = max(di for di, _ in steps)
    inf = torch.tensor(float("inf"), dtype=cost.dtype, device=cost.device)

    def shift_right(row: torch.Tensor, dj: int, fill: torch.Tensor) -> torch.Tensor:
        if dj == 0:
            return row
        pad = fill.expand(batch, dj) if fill.ndim == 0 else fill[:, :dj]
        return torch.cat([pad, row[:, :-dj]], dim=1) if dj < m else fill.expand(batch, m).clone()

    acc_rows: list[torch.Tensor] = [cost[:, 0, :].clone()]
    weight_rows: list[torch.Tensor] = [torch.ones(batch, m, dtype=cost.dtype, device=cost.device)]
    zero = torch.tensor(0.0, dtype=cost.dtype, device=cost.device)
    for i in range(1, n):
        c_i = cost[:, i, :]
        cand_acc, cand_weight = [], []
        for (di, dj), w in zip(steps, weights, strict=True):
            if i - di < 0:
                cand_acc.append(inf.expand(batch, m))
                cand_weight.append(zero.expand(batch, m))
                continue
            prev_acc = acc_rows[-di]
            prev_weight = weight_rows[-di]
            cand_acc.append(shift_right(prev_acc, dj, inf) + w * c_i)
            cand_weight.append(shift_right(prev_weight, dj, zero) + w)
        stacked = torch.stack(cand_acc, dim=0)  # (S, B, M)
        best, which = torch.min(stacked, dim=0)  # first minimum on ties
        new_weight = torch.gather(torch.stack(cand_weight, dim=0), 0, which.unsqueeze(0)).squeeze(0)
        acc_rows.append(best)
        weight_rows.append(new_weight)
        if len(acc_rows) > max_di:
            acc_rows.pop(0)
            weight_rows.pop(0)

    last_acc, last_weight = acc_rows[-1], weight_rows[-1]
    end_value, end_col = torch.min(last_acc, dim=1)
    end_weight = torch.gather(last_weight, 1, end_col.unsqueeze(1)).squeeze(1)
    normalized = torch.where(torch.isfinite(end_value), end_value / end_weight.clamp_min(1e-12), inf)
    return normalized, end_col
