"""research/scripts/rrlib.py must agree exactly with the pipeline it shortcuts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "scripts"))
import rrlib  # noqa: E402
import stats  # noqa: E402

from cover_retrieval.evaluation.metrics import average_precision  # noqa: E402
from cover_retrieval.retrieval.hybrid import HybridRun, Shortlist  # noqa: E402


@pytest.mark.parametrize("seed", range(20))
def test_rerank_from_kmax_equals_hybridrun_at_every_k(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, k_max = 200, 40
    rel = np.zeros(n, dtype=bool)
    rel[rng.choice(n, size=rng.integers(1, 13), replace=False)] = True
    order = np.arange(n)
    g_all = np.sort(rng.random(n))[::-1]
    a_all = rng.random(n)
    a_all[rng.random(n) < 0.2] = 0.5  # force ties
    s1_ranks = np.flatnonzero(rel) + 1
    for k in (1, 5, 17, 30, 40):
        for alpha in (0.0, 0.1, 0.5):
            run = HybridRun([Shortlist(0, order, rel, order[:k], g_all[:k], a_all[:k], np.zeros(k, int))], k)
            ranked = run.rank(alpha)[0]
            ranks, _ = rrlib.rerank_ranks(s1_ranks, rel[:k_max], g_all[:k_max], a_all[:k_max], k, alpha)
            assert np.array_equal(np.sort(ranks), np.flatnonzero(ranked.relevance) + 1)
            assert rrlib.ap_from_ranks(ranks, rel.sum()) == pytest.approx(average_precision(ranked.relevance), abs=1e-12)


def test_ap_from_ranks_matches_pipeline() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        rel = rng.random(300) < 0.03
        if not rel.any():
            continue
        assert rrlib.ap_from_ranks(np.flatnonzero(rel) + 1) == pytest.approx(average_precision(rel), abs=1e-12)


def test_group_folds_are_work_disjoint() -> None:
    groups = [f"W_{i // 13}" for i in range(13 * 50)]
    folds = stats.group_folds(groups, 5)
    for g in set(groups):
        assert len({folds[i] for i, x in enumerate(groups) if x == g}) == 1
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]


def test_auroc_and_bootstrap_basics() -> None:
    assert stats.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert stats.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    groups = [f"W_{i // 4}" for i in range(400)]
    d = stats.boot_delta(np.zeros(400), np.ones(400), groups, n_boot=500)
    assert d["delta"] == 1.0 and d["ci_low"] == 1.0 and d["excludes_zero"]
