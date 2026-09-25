"""Statistics for research/: work-level (clique) bootstrap, as in the repository (D-009).

All intervals resample WORKS (the benchmark's 1,000 query cliques), never individual
queries, because the 13 recordings of a work are not independent. Percentile bootstrap,
10,000 resamples, seed 20260817 unless stated. See research/statistics.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SEED = 20260817
N_BOOT = 10_000


def _groups(groups: Sequence[str]) -> tuple[np.ndarray, int]:
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return inverse, int(inverse.max()) + 1


def boot_mean(values, groups, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95) -> dict[str, float]:
    """Mean of a per-query quantity with a work-level percentile interval."""
    v = np.asarray(values, dtype=np.float64)
    inv, g = _groups(groups)
    sums = np.bincount(inv, weights=v, minlength=g)
    counts = np.bincount(inv, minlength=g).astype(np.float64)
    draws = np.random.default_rng(seed).integers(0, g, size=(n_boot, g))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    a = (1 - level) / 2
    return {"estimate": float(v.mean()), "ci_low": float(np.quantile(boot, a)), "ci_high": float(np.quantile(boot, 1 - a)), "n_groups": g}


def boot_delta(a, b, groups, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95) -> dict[str, float]:
    """Paired mean(b) - mean(a), work-level. Same estimator as evaluation.analysis.bootstrap_delta_ci."""
    d = boot_mean(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64), groups, n_boot, seed, level)
    return {"delta": d["estimate"], "ci_low": d["ci_low"], "ci_high": d["ci_high"], "n_groups": d["n_groups"],
            "excludes_zero": bool(d["ci_low"] > 0 or d["ci_high"] < 0)}


def boot_ratio(num, den, groups, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95) -> dict[str, float]:
    """Ratio of sums (e.g. a conditional rate), work-level."""
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    inv, g = _groups(groups)
    sn, sd = np.bincount(inv, weights=n, minlength=g), np.bincount(inv, weights=d, minlength=g)
    draws = np.random.default_rng(seed).integers(0, g, size=(n_boot, g))
    boot = sn[draws].sum(1) / np.maximum(sd[draws].sum(1), 1e-12)
    a = (1 - level) / 2
    return {"estimate": float(n.sum() / max(d.sum(), 1e-12)), "ci_low": float(np.quantile(boot, a)), "ci_high": float(np.quantile(boot, 1 - a)), "n_groups": g}


def auroc(scores, labels) -> float:
    """Area under the ROC curve (Mann-Whitney), ties averaged."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size)
    ranks[order] = np.arange(1, s.size + 1)
    # average ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def group_folds(groups, n_folds: int = 5, seed: int = SEED) -> np.ndarray:
    """Fold id per item; every work lands in exactly one fold (work-disjoint cross-fitting)."""
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    perm = np.random.default_rng(seed).permutation(uniq.size)
    fold_of_group = np.empty(uniq.size, dtype=np.int64)
    fold_of_group[perm] = np.arange(uniq.size) % n_folds
    return fold_of_group[inv]


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton's method; returns weights incl. bias (last)."""
    x1 = np.hstack([x, np.ones((x.shape[0], 1))])
    w = np.zeros(x1.shape[1])
    reg = np.full(x1.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(x1 @ w, -30, 30)))
        grad = x1.T @ (p - y) + reg * w
        hess = (x1 * (p * (1 - p))[:, None]).T @ x1 + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logistic(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(np.hstack([x, np.ones((x.shape[0], 1))]) @ w, -30, 30)))
