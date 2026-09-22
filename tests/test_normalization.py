"""Hubness correction: reference estimation and its effect on rankings."""

import numpy as np
import pytest

from cover_retrieval.evaluation.metrics import ranking_from_scores
from cover_retrieval.retrieval.normalization import apply_hub_correction, hub_reference


def test_mean_reference_and_exclusion():
    scores = np.array([[1.0, 0.0], [3.0, 2.0]])
    np.testing.assert_allclose(hub_reference(scores, "mean"), [2.0, 1.0])
    exclude = np.array([[True, False], [False, False]])
    np.testing.assert_allclose(hub_reference(scores, "mean", exclude=exclude), [3.0, 1.0])


def test_topk_reference_uses_only_largest():
    scores = np.array([[0.0, 9.0], [10.0, 0.0], [8.0, 0.0]])
    np.testing.assert_allclose(hub_reference(scores, "topk", k=2), [9.0, 4.5])
    np.testing.assert_allclose(hub_reference(scores, "topk", k=1), [10.0, 9.0])
    # k larger than the number of queries degrades to the mean
    np.testing.assert_allclose(hub_reference(scores, "topk", k=99), hub_reference(scores, "mean"))


def test_topk_ignores_excluded_entries():
    scores = np.array([[5.0, 1.0], [0.0, 1.0]])
    exclude = np.array([[True, False], [False, False]])
    np.testing.assert_allclose(hub_reference(scores, "topk", k=1, exclude=exclude), [0.0, 1.0])


def test_lambda_zero_is_a_no_op():
    rng = np.random.default_rng(0)
    scores = rng.random((4, 6))
    np.testing.assert_allclose(apply_hub_correction(scores, hub_reference(scores), 0.0), scores)


def test_correction_demotes_a_hub_without_hurting_a_specific_match():
    # candidate 0 is a hub (high to everyone); candidate 2 matches query 1 specifically
    scores = np.array(
        [
            [0.90, 0.10, 0.20],
            [0.88, 0.15, 0.85],
            [0.91, 0.12, 0.10],
            [0.89, 0.11, 0.05],
        ]
    )
    relevant = np.array([False, False, True])
    order_before, _ = ranking_from_scores(scores[1], relevant)
    assert order_before[0] == 0  # the hub wins before correction

    reference = hub_reference(scores, "topk", k=2)
    corrected = apply_hub_correction(scores, reference, lam=1.0)
    order_after, rel_after = ranking_from_scores(corrected[1], relevant)
    assert order_after[0] == 2  # the specific match wins after
    assert rel_after[0]


def test_reference_shape_and_input_validation():
    with pytest.raises(ValueError):
        hub_reference(np.zeros(5))
    with pytest.raises(ValueError):
        hub_reference(np.zeros((2, 2)), "nonsense")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hub_reference(np.zeros((2, 2)), "topk", k=0)
    with pytest.raises(ValueError):
        apply_hub_correction(np.zeros((2, 3)), np.zeros(2), 1.0)


def test_correction_is_per_candidate_not_per_query():
    """Adding a constant to a query's row must not change that row's ranking."""
    rng = np.random.default_rng(1)
    scores = rng.random((5, 8))
    reference = hub_reference(scores)
    shifted = scores.copy()
    shifted[2] += 3.0
    a, _ = ranking_from_scores(apply_hub_correction(scores, reference, 0.5)[2], np.zeros(8, bool))
    b, _ = ranking_from_scores(apply_hub_correction(shifted, reference, 0.5)[2], np.zeros(8, bool))
    np.testing.assert_array_equal(a, b)
