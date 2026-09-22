"""DTW / subsequence DTW on hand-checked toy matrices, plus reference-vs-batch equality."""

import numpy as np
import pytest
import torch

from cover_retrieval.alignment.csm import cosine_cost_matrix, cross_similarity_matrix
from cover_retrieval.alignment.dtw import (
    CLASSIC_STEPS,
    SLOPE_STEPS,
    SLOPE_WEIGHTS,
    batch_subsequence_dtw,
    dtw,
)


def _assert_valid_path(path, steps, n, m, subsequence):
    path = np.asarray(path)
    if subsequence:
        assert path[0, 0] == 0 and path[-1, 0] == n - 1
    else:
        assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (n - 1, m - 1)
    deltas = {tuple(d) for d in np.diff(path, axis=0)}
    assert deltas <= {tuple(s) for s in steps}, f"illegal steps {deltas}"
    assert (np.diff(path, axis=0) >= 0).all(), "path must be monotonic"


def test_full_dtw_hand_calculated():
    cost = np.array([[0.0, 2.0, 3.0], [1.0, 0.0, 2.0], [3.0, 1.0, 0.0]])
    result = dtw(cost, CLASSIC_STEPS)
    # Accumulated matrix by hand:
    # D = [[0, 2, 5], [1, 0, 2], [4, 1, 0]]
    np.testing.assert_allclose(result.accumulated, [[0, 2, 5], [1, 0, 2], [4, 1, 0]])
    assert result.path.tolist() == [[0, 0], [1, 1], [2, 2]]
    assert result.total_cost == 0.0
    assert result.path_weight == 3.0
    assert result.normalized_cost == 0.0


def test_full_dtw_with_warping():
    cost = np.array([[1.0, 1.0, 5.0], [5.0, 1.0, 1.0]])
    result = dtw(cost, CLASSIC_STEPS)
    # By hand: D = [[1, 2, 7], [6, 2, 3]]; D[1,2] = 3 is reached from (1,1) via the
    # (0,1) step, which comes before the tying (1,1) step from (0,1) in step order.
    assert result.total_cost == pytest.approx(3.0)
    assert result.path.tolist() == [[0, 0], [1, 1], [1, 2]]
    _assert_valid_path(result.path, CLASSIC_STEPS, 2, 3, subsequence=False)
    assert result.normalized_cost == pytest.approx(3.0 / len(result.path))


def test_subsequence_dtw_finds_embedded_query():
    rng = np.random.default_rng(0)
    candidate = rng.random((12, 40))
    query = candidate[:, 15:25]
    cost = cosine_cost_matrix(query, candidate)
    result = dtw(cost, SLOPE_STEPS, SLOPE_WEIGHTS, subsequence=True)
    assert result.start == (0, 15)
    assert result.end == (9, 24)
    assert result.normalized_cost == pytest.approx(0.0, abs=1e-9)
    assert result.path_weight == 10.0  # weighted length == query length
    _assert_valid_path(result.path, SLOPE_STEPS, 10, 40, subsequence=True)


def test_time_stretched_sequence_aligns_diagonally_scaled():
    rng = np.random.default_rng(1)
    base = rng.random((12, 20))
    stretched = np.repeat(base, 2, axis=1)  # tempo halved
    cost = cosine_cost_matrix(base, stretched)
    result = dtw(cost, SLOPE_STEPS, SLOPE_WEIGHTS, subsequence=True)
    assert result.normalized_cost == pytest.approx(0.0, abs=1e-9)
    _assert_valid_path(result.path, SLOPE_STEPS, 20, 40, subsequence=True)
    # the query frame i must be matched to one of its two stretched copies
    for i, j in result.path:
        assert j // 2 == i


def test_slope_steps_forbid_degenerate_vertical_path():
    cost = np.ones((6, 4))
    cost[:, 0] = 0.0  # a single perfect candidate column
    classic = dtw(cost, CLASSIC_STEPS, subsequence=True)
    slope = dtw(cost, SLOPE_STEPS, SLOPE_WEIGHTS, subsequence=True)
    assert classic.normalized_cost == 0.0  # whole query collapsed onto one frame
    assert slope.normalized_cost > 0.0


def test_no_admissible_path_raises():
    with pytest.raises(ValueError):
        dtw(np.zeros((10, 2)), SLOPE_STEPS, SLOPE_WEIGHTS, subsequence=True)


@pytest.mark.parametrize("shape", [(16, 16), (10, 25), (30, 18)])
def test_batch_matches_reference(shape):
    rng = np.random.default_rng(42)
    costs = rng.random((20, *shape))
    normalized, end = batch_subsequence_dtw(torch.tensor(costs))
    for b in range(costs.shape[0]):
        ref = dtw(costs[b], SLOPE_STEPS, SLOPE_WEIGHTS, subsequence=True)
        assert normalized[b].item() == pytest.approx(ref.normalized_cost, rel=1e-12)
        assert end[b].item() == ref.end[1]


def test_batch_reports_unreachable_as_inf():
    normalized, _ = batch_subsequence_dtw(torch.zeros(2, 10, 3, dtype=torch.float64))
    assert torch.isinf(normalized).all()


def test_batch_rejects_horizontal_steps():
    with pytest.raises(ValueError):
        batch_subsequence_dtw(torch.zeros(1, 4, 4), step_sizes=CLASSIC_STEPS, step_weights=None)


def test_cross_similarity_properties():
    rng = np.random.default_rng(3)
    x = rng.random((12, 8))
    x[:, 3] = 0.0  # silent frame
    s = cross_similarity_matrix(x, x)
    assert s.shape == (8, 8)
    np.testing.assert_allclose(np.diag(s)[[0, 1, 2, 4]], 1.0)
    assert (s[3] == 0).all()
    c = cosine_cost_matrix(x, x)
    assert c.min() >= 0.0 and c.max() <= 1.0 + 1e-12  # non-negative features
