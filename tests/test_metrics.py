"""Metric tests against hand-calculated values."""

import numpy as np
import pytest

from cover_retrieval.evaluation.metrics import (
    average_precision,
    evaluate_rankings,
    first_relevant_rank,
    hit_at_k,
    ranking_from_scores,
    recall_at_k,
    reciprocal_rank,
)


def test_relevant_at_rank_one():
    rel = [True, False, False, False]
    assert first_relevant_rank(rel) == 1
    assert reciprocal_rank(rel) == 1.0
    assert average_precision(rel) == 1.0
    assert recall_at_k(rel, 1) == 1.0


def test_relevant_at_rank_five():
    rel = [False] * 4 + [True] + [False] * 10
    assert first_relevant_rank(rel) == 5
    assert reciprocal_rank(rel) == pytest.approx(0.2)
    # single relevant item: AP == RR by construction
    assert average_precision(rel) == pytest.approx(0.2)
    assert recall_at_k(rel, 1) == 0.0
    assert recall_at_k(rel, 5) == 1.0
    assert hit_at_k(rel, 4) == 0.0 and hit_at_k(rel, 10) == 1.0


def test_multiple_relevant_average_precision():
    # relevant at ranks 1, 3, 6 -> precisions 1/1, 2/3, 3/6 -> AP = (1 + 2/3 + 1/2) / 3
    rel = [True, False, True, False, False, True, False]
    assert average_precision(rel) == pytest.approx((1 + 2 / 3 + 0.5) / 3)
    assert reciprocal_rank(rel) == 1.0
    assert recall_at_k(rel, 3) == pytest.approx(2 / 3)
    assert recall_at_k(rel, 1) == pytest.approx(1 / 3)
    assert hit_at_k(rel, 1) == 1.0


def test_ap_with_truncated_ranking_counts_missing_items():
    # 2 retrieved relevant (ranks 2, 4) out of 4 relevant in the collection
    rel = [False, True, False, True]
    assert average_precision(rel, n_relevant=4) == pytest.approx((1 / 2 + 2 / 4) / 4)


def test_no_relevant_item():
    rel = [False, False]
    assert first_relevant_rank(rel) is None
    assert reciprocal_rank(rel) == 0.0
    assert average_precision(rel) == 0.0


def test_map_equals_mrr_for_single_relevant_protocol():
    rankings = [[True, False, False], [False, False, True], [False, True, False]]
    summary = evaluate_rankings(rankings, ks=(1, 2))
    assert summary.map == pytest.approx(summary.mrr)
    assert summary.map == pytest.approx((1 + 1 / 3 + 1 / 2) / 3)
    assert summary.recall[1] == pytest.approx(1 / 3)
    assert summary.recall[2] == pytest.approx(2 / 3)
    assert summary.mean_first_rank == pytest.approx(2.0)
    assert summary.median_first_rank == 2.0


def test_map_differs_from_mrr_with_multiple_relevant():
    rankings = [[True, False, True], [False, True, True]]
    summary = evaluate_rankings(rankings, ks=(1,))
    assert summary.mrr == pytest.approx((1 + 0.5) / 2)
    assert summary.map == pytest.approx(((1 + 2 / 3) / 2 + (1 / 2 + 2 / 3) / 2) / 2)
    assert summary.map != pytest.approx(summary.mrr)


def test_evaluate_rejects_query_without_relevant_item():
    with pytest.raises(ValueError):
        evaluate_rankings([[False, False]])


def test_ranking_from_scores_excludes_self_and_breaks_ties_by_position():
    scores = np.array([0.9, 1.0, 0.5, 0.5, 0.1])
    relevant = np.array([False, False, False, True, False])
    exclude = np.array([False, True, False, False, False])  # the query itself scores highest
    order, rel = ranking_from_scores(scores, relevant, exclude)
    assert 1 not in order
    assert list(order) == [0, 2, 3, 4]  # tie 0.5/0.5 keeps input (PID) order
    assert list(rel) == [False, False, True, False]
    assert first_relevant_rank(rel) == 3
