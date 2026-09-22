"""Statistical comparison, error-case selection and diagnostic plots.

Bootstrap confidence intervals resample *works* (WIDs), not individual queries: all
queries of a resampled work move together, because recordings of the same work are
not independent (Smucker, Allan & Carterette, 2007; D-009).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ------------------------------------------------------------------ statistics
def bootstrap_delta_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    groups: Sequence[str],
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Paired, group-level percentile bootstrap for mean(b) - mean(a)."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.shape[0] != len(groups):
        raise ValueError("values_a, values_b and groups must align")
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = unique.size
    diff_sum = np.bincount(inverse, weights=b - a, minlength=n_groups)
    counts = np.bincount(inverse, minlength=n_groups).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    boot = diff_sum[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    return {
        "delta": float((b - a).mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
        "n_groups": int(n_groups),
        "n_boot": int(n_boot),
    }


# ------------------------------------------------------------------ error cases
@dataclass
class ErrorCases:
    successes: list[int]  # query indices, relevant at rank <= 10, best first
    false_positives: list[int]  # rank-1 candidate is not relevant
    false_negatives: list[int]  # first relevant item ranked > 10, worst first


def select_error_cases(first_ranks: Sequence[int], n: int = 3, success_cutoff: int = 10) -> ErrorCases:
    """Deterministic selection of illustrative cases (no cherry-picking by eye)."""
    ranks = np.asarray(first_ranks)
    order = np.argsort(ranks, kind="stable")
    successes = [int(i) for i in order if ranks[i] <= success_cutoff][:n]
    false_pos = [int(i) for i in order[::-1] if ranks[i] > 1]
    false_neg = [int(i) for i in order[::-1] if ranks[i] > success_cutoff][:n]
    # false positives: queries whose top result is wrong; prefer those not already shown as FN
    fp = [i for i in false_pos if i not in false_neg][:n]
    fp += [i for i in false_pos if i not in fp][: max(0, n - len(fp))]
    return ErrorCases(successes, fp[:n], false_neg)


# ------------------------------------------------------------------ plots
def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_feature_matrices(matrices: Sequence[np.ndarray], titles: Sequence[str], path: str | Path) -> None:
    """Row of ``(12, T)`` HPCP matrices."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.2 * len(matrices), 2.8), squeeze=False)
    for ax, matrix, title in zip(axes[0], matrices, titles, strict=True):
        ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("pitch class")
        ax.set_yticks(range(0, 12, 2))
    _save(fig, path)


def plot_alignment(
    cost: np.ndarray, path_ij: np.ndarray | None, title: str, out_path: str | Path, ax: plt.Axes | None = None
) -> None:
    """Cost matrix with the DTW path overlaid (query on the vertical axis)."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
    image = ax.imshow(cost, origin="lower", aspect="auto", cmap="viridis_r", vmin=0.0, vmax=1.0)
    if path_ij is not None:
        ax.plot(path_ij[:, 1], path_ij[:, 0], color="red", linewidth=1.2)
    ax.set_xlabel("candidate frame")
    ax.set_ylabel("query frame")
    ax.set_title(title, fontsize=9)
    if own:
        fig.colorbar(image, ax=ax, label="cosine distance")
        _save(fig, out_path)


def plot_alignment_grid(items: Sequence[tuple[np.ndarray, np.ndarray, str]], path: str | Path) -> None:
    fig, axes = plt.subplots(1, len(items), figsize=(4.0 * len(items), 3.8), squeeze=False)
    for ax, (cost, path_ij, title) in zip(axes[0], items, strict=True):
        plot_alignment(cost, path_ij, title, path, ax=ax)
    _save(fig, path)


def plot_rotation_scores(score_sets: Sequence[np.ndarray], labels: Sequence[str], path: str | Path) -> None:
    fig, axes = plt.subplots(1, len(score_sets), figsize=(3.4 * len(score_sets), 2.6), squeeze=False)
    for ax, scores, label in zip(axes[0], score_sets, labels, strict=True):
        best = int(np.argmax(scores))
        colors = ["tab:red" if k == best else "tab:gray" for k in range(12)]
        ax.bar(range(12), scores, color=colors)
        ax.set_ylim(min(0.0, float(np.min(scores))), 1.0)
        ax.set_xticks(range(12))
        ax.set_xlabel("candidate rotation (semitones)")
        ax.set_title(f"{label}\nbest shift = {best}", fontsize=9)
    axes[0][0].set_ylabel("profile cosine")
    _save(fig, path)


def plot_training_history(history: Sequence[dict[str, float]], path: str | Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(6, 3.4))
    ax1.plot(epochs, [h["train_loss"] for h in history], color="tab:blue", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("SupCon loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_map"] for h in history], color="tab:orange", label="validation MAP")
    ax2.set_ylabel("validation MAP", color="tab:orange")
    ax1.set_title("Encoder training (validation = disjoint WIDs)", fontsize=10)
    _save(fig, path)


def plot_alpha_curve(table: Sequence[dict[str, float]], chosen: float, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot([r["alpha"] for r in table], [r["map"] for r in table], marker="o", markersize=3)
    ax.axvline(chosen, color="tab:red", linestyle="--", label=f"chosen alpha = {chosen:.2f}")
    ax.set_xlabel("alpha (1 = global only, 0 = alignment only)")
    ax.set_ylabel("MAP")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, path)
