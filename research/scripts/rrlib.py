"""Shared helpers for the recall-aware retrieval experiments (research/).

Design: Stage 1 and the expensive alignment are run ONCE per query at the largest
shortlist size K_max. Every smaller K is derived from those scores exactly, because
the hybrid score of a shortlist depends only on the shortlist's own global and
alignment scores (per-query z-scores are recomputed for each K, as in
``cover_retrieval.retrieval.hybrid.HybridRun.rank``). ``tests/test_research_lib.py``
checks this equivalence against ``HybridRun``.

Metrics only need the ranks of the relevant items, so everything below works on rank
positions instead of full 15,000-long rankings.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ metrics from ranks


def ap_from_ranks(ranks: Sequence[int] | np.ndarray, n_relevant: int | None = None) -> float:
    """Average precision given the 1-based ranks of the relevant items in a full ranking."""
    r = np.sort(np.asarray(ranks, dtype=np.int64))
    n = len(r) if n_relevant is None else n_relevant
    if n == 0:
        return 0.0
    return float((np.arange(1, len(r) + 1) / r).sum() / n)


def zscore(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Identical to ``cover_retrieval.retrieval.hybrid.zscore``."""
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros_like(values)
    std = values.std()
    return np.zeros_like(values) if std < eps else (values - values.mean()) / std


def rerank_ranks(
    stage1_rel_ranks: np.ndarray,
    head_relevance: np.ndarray,
    global_head: np.ndarray,
    align_head: np.ndarray,
    k: int,
    alpha: float,
) -> tuple[np.ndarray, int]:
    """Ranks of relevant items after reranking the first ``k`` Stage-1 candidates.

    ``head_relevance``/``global_head``/``align_head`` cover the first K_max >= k Stage-1
    positions. Items beyond ``k`` keep their Stage-1 ranks. Returns (sorted relevant
    ranks, index into the head of the new rank-1 item).
    """
    hybrid = alpha * zscore(global_head[:k]) + (1.0 - alpha) * zscore(align_head[:k])
    order = np.argsort(-hybrid, kind="stable")  # ties keep Stage-1 order
    new_head_rel = head_relevance[:k][order]
    head_ranks = np.flatnonzero(new_head_rel) + 1
    tail_ranks = stage1_rel_ranks[stage1_rel_ranks > k]
    return np.concatenate([head_ranks, tail_ranks]).astype(np.int64), int(order[0])


# ------------------------------------------------------------------ label-free Stage-1 signals

SIGNAL_TAU = 0.05  # softmax temperature for score entropy, fixed a priori (not tuned)


def stage1_signals(row: np.ndarray, self_index: int | None) -> dict[str, float]:
    """Difficulty signals from one query's Stage-1 cosine row. Uses no labels."""
    s = np.asarray(row, dtype=np.float64).copy()
    if self_index is not None:
        s[self_index] = -np.inf
    valid = s[np.isfinite(s)]
    top = np.sort(np.partition(valid, -100)[-100:])[::-1] if valid.size > 100 else np.sort(valid)[::-1]
    p = np.exp((top - top[0]) / SIGNAL_TAU)
    p /= p.sum()
    mean, std = float(valid.mean()), float(valid.std())
    return {
        "top1": float(top[0]),
        "top2": float(top[1]),
        "margin_1_2": float(top[0] - top[1]),
        "margin_1_10": float(top[0] - top[9]),
        "mean_top10": float(top[:10].mean()),
        "std_top10": float(top[:10].std()),
        "mean_top100": float(top.mean()),
        "entropy_top100": float(-(p * np.log(p + 1e-300)).sum()),
        "row_mean": mean,
        "row_std": std,
        "z_top1": float((top[0] - mean) / (std + 1e-12)),
    }


def k_occurrence(scores: np.ndarray, self_mask: np.ndarray, k: int = 10, chunk: int = 1000) -> np.ndarray:
    """How often each candidate appears in the top ``k`` of any query (label-free hubness)."""
    counts = np.zeros(scores.shape[1], dtype=np.int64)
    for start in range(0, scores.shape[0], chunk):
        block = np.where(self_mask[start : start + chunk], -np.inf, scores[start : start + chunk])
        top = np.argpartition(-block, k, axis=1)[:, :k]
        counts += np.bincount(top.ravel(), minlength=scores.shape[1])
    return counts


# ------------------------------------------------------------------ provenance / outputs


def git_state() -> dict[str, str]:
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": str(bool(run("git", "status", "--porcelain")))}


def environment() -> dict[str, str]:
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count()),
        "torch_threads": str(torch.get_num_threads()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def evidence_label(config: dict) -> str:
    """SYNTHETIC-MECHANICS for the smoke fixture, LOCAL-FULL for real Da-TACOS paths."""
    root = str(config["paths"]["datacos_root"])
    return "SYNTHETIC-MECHANICS" if "smoke" in root else "LOCAL-FULL"


def prepare_out(out: Path, force: bool) -> Path:
    """Create the output directory; refuse to overwrite a finished experiment."""
    out = Path(out)
    if (out / "metrics.json").exists() and not force:
        raise SystemExit(f"{out} already holds a finished experiment; refusing to overwrite (use a new --out)")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False, default=float) + "\n")


def write_meta(out: Path, experiment_id: str, config: dict, args: dict, label: str) -> None:
    """config.yaml (resolved config + CLI) and env.json for every experiment."""
    import yaml

    def plain(v):
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)

    (out / "config.yaml").write_text(
        yaml.safe_dump(plain({"experiment_id": experiment_id, "evidence": label, "args": args, "config": config}), sort_keys=False)
    )
    write_json(out / "env.json", {"experiment_id": experiment_id, "evidence": label, "git": git_state(), **environment()})


def summarize(first: np.ndarray, ap: np.ndarray, extra: dict | None = None) -> dict[str, float]:
    first = np.asarray(first, dtype=np.float64)
    out = {
        "n_queries": int(first.size),
        "MAP": float(np.mean(ap)),
        "MRR": float(np.mean(1.0 / first)),
        "Hit@1": float(np.mean(first <= 1)),
        "Hit@10": float(np.mean(first <= 10)),
        "Hit@100": float(np.mean(first <= 100)),
        "mean_first_rank": float(first.mean()),
        "median_first_rank": float(np.median(first)),
    }
    if extra:
        out.update(extra)
    return out


def ranks_to_str(ranks: np.ndarray) -> str:
    return ";".join(str(int(r)) for r in np.sort(ranks))


def str_to_ranks(s: str) -> np.ndarray:
    return np.array([int(x) for x in str(s).split(";") if x != ""], dtype=np.int64)
