"""End-to-end building blocks shared by the command-line scripts.

Every function takes the resolved config dict; paths inside it are relative to the
repository root (or absolute for tests).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cover_retrieval.data.datacos import features_root
from cover_retrieval.data.manifests import Protocol
from cover_retrieval.features.preprocessing import FeatureStore, Track, build_feature_store
from cover_retrieval.models.tcn_encoder import TCNEncoder
from cover_retrieval.models.training import embed
from cover_retrieval.retrieval.hybrid import HybridRun, build_shortlists
from cover_retrieval.retrieval.index import EmbeddingIndex, rotation_max_scores
from cover_retrieval.retrieval.rank import (
    AlignmentSettings,
    ClassicalAligner,
    RankedQuery,
    rank_protocol,
    summarize_ranked,
)
from cover_retrieval.utils.io import write_csv


def results_dir(config: dict[str, Any]) -> Path:
    path = Path(config["paths"]["reports_dir"]) / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(config: dict[str, Any]) -> Path:
    path = Path(config["paths"]["reports_dir"]) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(config: dict[str, Any], name: str) -> Path:
    path = Path(config["paths"]["runs_dir"]) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def protocol_store(
    config: dict[str, Any], protocol: Protocol, subset: str, nonfinite_policy: str | None = None
) -> FeatureStore:
    """Features for every candidate of ``protocol`` (queries are always candidates too)."""
    candidate_pids = {t.pid for t in protocol.candidates}
    missing = [t.pid for t in protocol.queries if t.pid not in candidate_pids]
    if missing:
        raise ValueError(f"{protocol.name}: queries missing from candidate pool: {missing[:3]}")
    return build_feature_store(
        protocol.candidates,
        features_root(config, subset),  # type: ignore[arg-type]
        config,
        cache_name=f"{subset}_{protocol.name}",
        nonfinite_policy=nonfinite_policy,
    )


def split_store(config: dict[str, Any], tracks: Sequence[Track], name: str) -> FeatureStore:
    return build_feature_store(
        tracks, features_root(config, "coveranalysis"), config, cache_name=f"coveranalysis_{name}"
    )


def aligner_from_config(config: dict[str, Any], **overrides: Any) -> ClassicalAligner:
    settings = AlignmentSettings.from_config(config)
    for key, value in overrides.items():
        setattr(settings, key, value)
    return ClassicalAligner(settings)


@dataclass
class ScoredRun:
    name: str
    scores: np.ndarray  # (Q, C) higher = better
    ranked: list[RankedQuery]
    timings: dict[str, float]
    shifts: np.ndarray | None = None


def run_classical(
    config: dict[str, Any], protocol: Protocol, store: FeatureStore, **aligner_overrides: Any
) -> ScoredRun:
    """Alignment-only retrieval: every query against every candidate."""
    aligner = aligner_from_config(config, **aligner_overrides)
    queries = store.get("classical", [t.pid for t in protocol.queries])
    t0 = time.perf_counter()
    scores, shifts = aligner.score_matrix(
        queries, store.views["classical"], verbose=len(protocol.queries) > 500
    )
    align_seconds = time.perf_counter() - t0
    ranked = rank_protocol(scores, protocol)
    n_pairs = len(protocol.queries) * (len(protocol.candidates) - 1)
    return ScoredRun(
        f"classical_{aligner.settings.rotation_selection}",
        scores,
        ranked,
        {
            "alignment_seconds": align_seconds,
            "n_pairs": n_pairs,
            "ms_per_pair": 1e3 * align_seconds / max(n_pairs, 1),
        },
        shifts,
    )


def global_scores(
    config: dict[str, Any], model: TCNEncoder, protocol: Protocol, store: FeatureStore, device: str = "cpu"
) -> tuple[np.ndarray, dict[str, float]]:
    """Stage-1 cosine scores ``(Q, C)`` plus embedding / search timings."""
    t0 = time.perf_counter()
    candidates = embed(model, store.views["encoder"], device)
    query_idx = store.index([t.pid for t in protocol.queries])
    if config["retrieval"].get("test_time_rotations", False):
        rotated = np.stack(
            [embed(model, store.views["encoder"][query_idx], device, rotation=r) for r in range(12)]
        )
    else:
        rotated = None
    t1 = time.perf_counter()
    index = EmbeddingIndex(candidates, backend=config["retrieval"].get("backend", "numpy"))
    scores = (
        rotation_max_scores(rotated, candidates)
        if rotated is not None
        else index.scores(candidates[query_idx])
    )
    t2 = time.perf_counter()
    return scores.astype(np.float32), {"embedding_seconds": t1 - t0, "global_search_seconds": t2 - t1}


def run_global(
    config: dict[str, Any], model: TCNEncoder, protocol: Protocol, store: FeatureStore
) -> ScoredRun:
    scores, timings = global_scores(config, model, protocol, store)
    t0 = time.perf_counter()
    ranked = rank_protocol(scores, protocol)
    timings["ranking_seconds"] = time.perf_counter() - t0
    return ScoredRun("global_embedding", scores, ranked, timings)


def run_hybrid_shortlists(
    config: dict[str, Any], global_run: ScoredRun, protocol: Protocol, store: FeatureStore, k: int
) -> HybridRun:
    aligner = aligner_from_config(config)
    query_features = store.get("classical", [t.pid for t in protocol.queries])
    return build_shortlists(global_run.scores, protocol, query_features, store.views["classical"], aligner, k)


def metrics_row(name: str, ranked: Sequence[RankedQuery], **extra: Any) -> dict[str, Any]:
    s = summarize_ranked(ranked)
    return {
        "system": name,
        "n_queries": s.n_queries,
        "MAP": s.map,
        "MRR": s.mrr,
        "Recall@1": s.recall[1],
        "Recall@10": s.recall[10],
        "Recall@100": s.recall[100],
        "Hit@1": s.hit[1],
        "Hit@10": s.hit[10],
        "Hit@100": s.hit[100],
        "mean_first_rank": s.mean_first_rank,
        "median_first_rank": s.median_first_rank,
        **extra,
    }


METRIC_COLUMNS = [
    "system",
    "n_queries",
    "MAP",
    "MRR",
    "Recall@1",
    "Recall@10",
    "Recall@100",
    "Hit@1",
    "Hit@10",
    "Hit@100",
    "mean_first_rank",
    "median_first_rank",
]


def per_query_rows(protocol: Protocol, runs: dict[str, Sequence[RankedQuery]]) -> list[dict[str, Any]]:
    rows = []
    for i, query in enumerate(protocol.queries):
        row: dict[str, Any] = {"query_pid": query.pid, "query_wid": query.wid}
        for name, ranked in runs.items():
            m = ranked[i].metrics
            row[f"{name}_ap"] = m.ap
            row[f"{name}_first_rank"] = m.first_rank
        rows.append(row)
    return rows


def write_per_query(path: Path, protocol: Protocol, runs: dict[str, Sequence[RankedQuery]]) -> None:
    rows = per_query_rows(protocol, runs)
    write_csv(path, rows, list(rows[0]))


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str], digits: int = 3) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def torch_threads(config: dict[str, Any]) -> None:
    torch.set_num_threads(int(config["training"].get("num_threads", 8)))


@dataclass
class PairDiagnostics:
    rotation_scores: np.ndarray
    shift: int
    cost: np.ndarray
    dtw: Any  # alignment.dtw.DTWResult


def pair_diagnostics(config: dict[str, Any], query: np.ndarray, candidate: np.ndarray) -> PairDiagnostics:
    """Reference (non-batched) alignment of one pair, for plots and error analysis."""
    from cover_retrieval.alignment.csm import cosine_cost_matrix
    from cover_retrieval.alignment.dtw import dtw
    from cover_retrieval.alignment.transposition import best_key_rotation

    rotation = best_key_rotation(query, candidate)
    cost = cosine_cost_matrix(query, rotation.rotated_candidate)
    align = config["alignment"]
    result = dtw(cost, align["step_sizes"], align["step_weights"], subsequence=True)
    return PairDiagnostics(rotation.scores, rotation.shift, cost, result)
