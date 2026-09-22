"""Classical scoring, ranking with self-exclusion, the embedding index and hybrid reranking."""

import numpy as np
import pytest
import torch

from cover_retrieval.alignment.transposition import rotate_pitch_classes
from cover_retrieval.data.manifests import Protocol
from cover_retrieval.evaluation.analysis import bootstrap_delta_ci, select_error_cases
from cover_retrieval.features.hpcp import l2_normalize_frames, uniform_resample
from cover_retrieval.features.preprocessing import Track
from cover_retrieval.retrieval.hybrid import build_shortlists, calibrate_alpha, zscore
from cover_retrieval.retrieval.index import EmbeddingIndex, rotation_max_scores
from cover_retrieval.retrieval.rank import AlignmentSettings, ClassicalAligner, rank_protocol


def _song(rng, n=96):
    return l2_normalize_frames(rng.random((12, n)) ** 4)


@pytest.fixture()
def aligner():
    return ClassicalAligner(AlignmentSettings(batch_size=7, dtype=torch.float64))


def test_aligner_is_key_and_tempo_invariant(aligner):
    rng = np.random.default_rng(0)
    song = _song(rng, 120)
    query = l2_normalize_frames(uniform_resample(song, 96))
    cover = l2_normalize_frames(
        uniform_resample(rotate_pitch_classes(song[:, :100], 5), 96)
    )  # key + tempo/length
    others = np.stack([_song(rng) for _ in range(20)])
    candidates = np.concatenate([others, cover[None]])
    pair = aligner.score_pairs(query, candidates)
    assert int(np.argmax(pair.score)) == 20
    assert pair.shift[20] == 7  # transposed up by 5 -> rotate by 7 to undo
    assert np.all(pair.score <= 1.0 + 1e-9)


def test_rotation_ablation_none_hurts_transposed_copy(aligner):
    rng = np.random.default_rng(1)
    query = _song(rng)
    transposed = rotate_pitch_classes(query, 3)
    no_rotation = ClassicalAligner(AlignmentSettings(rotation_selection="none"))
    exhaustive = ClassicalAligner(AlignmentSettings(rotation_selection="exhaustive_dtw"))
    best = aligner.score_pairs(query, transposed[None]).score[0]
    assert best == pytest.approx(1.0, abs=1e-6)
    assert no_rotation.score_pairs(query, transposed[None]).score[0] < best - 0.1
    ex = exhaustive.score_pairs(query, transposed[None])
    assert ex.shift[0] == 9 and ex.score[0] == pytest.approx(1.0, abs=1e-5)


def test_batching_does_not_change_scores():
    rng = np.random.default_rng(2)
    query = _song(rng)
    candidates = np.stack([_song(rng) for _ in range(23)])
    a = ClassicalAligner(AlignmentSettings(batch_size=5)).score_pairs(query, candidates)
    b = ClassicalAligner(AlignmentSettings(batch_size=100)).score_pairs(query, candidates)
    np.testing.assert_allclose(a.score, b.score, rtol=1e-6)


def _protocol():
    tracks = [
        Track("P_1", "W_a"),
        Track("P_2", "W_a"),
        Track("P_3", "W_b"),
        Track("P_4", "W_b"),
        Track("P_5", "W_c"),
    ]
    return Protocol("toy", [tracks[0], tracks[2]], tracks)


def test_rank_protocol_excludes_self_and_uses_wid_relevance():
    protocol = _protocol()
    protocol.check()
    scores = np.array([[9.0, 1.0, 5.0, 4.0, 3.0], [1.0, 2.0, 9.0, 0.5, 8.0]])
    ranked = rank_protocol(scores, protocol)
    assert 0 not in ranked[0].order and 2 not in ranked[1].order
    assert len(ranked[0].order) == 4
    assert ranked[0].metrics.first_rank == 4  # P_2 is last among non-self
    assert ranked[1].metrics.first_rank == 4  # P_5 (8) > P_2 (2) > P_1 (1) > P_4 (0.5)


def test_protocol_check_rejects_unsorted_or_relevance_free():
    tracks = [Track("P_2", "W_a"), Track("P_1", "W_a")]
    with pytest.raises(ValueError):
        Protocol("x", tracks[:1], tracks).check()
    with pytest.raises(ValueError):
        Protocol("x", [Track("P_1", "W_z")], [Track("P_1", "W_z"), Track("P_2", "W_y")]).check()


def test_embedding_index_exact_search_and_faiss_fallback():
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(50, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    index = EmbeddingIndex(emb)
    idx, scores = index.search(emb[:3], 1)
    assert idx[:, 0].tolist() == [0, 1, 2]
    np.testing.assert_allclose(scores[:, 0], 1.0, rtol=1e-5)
    try:
        import faiss  # noqa: F401
    except ImportError:
        with pytest.warns(UserWarning):
            fallback = EmbeddingIndex(emb, backend="faiss")
        assert fallback.backend == "numpy"
    with pytest.raises(ValueError):
        EmbeddingIndex(emb * 2)


def test_rotation_max_scores():
    q = np.zeros((12, 1, 2), dtype=np.float32)
    q[:, 0, 0] = 1.0
    q[4, 0] = [0.0, 1.0]
    c = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(rotation_max_scores(q, c), [[1.0, 1.0]])


def test_zscore_edge_cases():
    assert zscore(np.array([3.0])).tolist() == [0.0]
    assert zscore(np.array([2.0, 2.0])).tolist() == [0.0, 0.0]
    np.testing.assert_allclose(zscore(np.array([1.0, 3.0])), [-1.0, 1.0])


def test_hybrid_reranks_shortlist_and_keeps_complete_ranking():
    rng = np.random.default_rng(4)
    songs = [_song(rng) for _ in range(6)]
    # candidates: P_0..P_7; P_7 is a transposed copy of the query song P_0
    features = np.stack([*songs, _song(rng), rotate_pitch_classes(songs[0], 2)])
    tracks = [Track(f"P_{i}", "W_q" if i in (0, 7) else f"W_{i}") for i in range(8)]
    protocol = Protocol("toy", [tracks[0]], tracks)
    # global scores put the true cover (index 7) at shortlist position 3 of K=4
    global_scores = np.array([[1.0, 0.9, 0.8, 0.7, 0.1, 0.05, 0.0, 0.6]])
    run = build_shortlists(
        global_scores, protocol, features[:1], features, ClassicalAligner(AlignmentSettings()), k=4
    )
    stage1 = run.rank(1.0)[0]
    assert stage1.order.tolist() == [1, 2, 3, 7, 4, 5, 6]
    reranked = run.rank(0.0)[0]
    assert reranked.order[0] == 7  # alignment finds the cover
    assert sorted(reranked.order.tolist()) == [1, 2, 3, 4, 5, 6, 7]
    assert reranked.order[4:].tolist() == [4, 5, 6]  # remainder untouched
    calibration = calibrate_alpha(run, step=0.5)
    assert calibration.alpha in (0.0, 0.5)
    assert run.shortlist_recall() == 1.0


def test_bootstrap_groups_and_zero_difference():
    a = [0.1, 0.2, 0.3, 0.4]
    ci = bootstrap_delta_ci(a, a, ["W_1", "W_1", "W_2", "W_3"], n_boot=200)
    assert ci["delta"] == ci["ci_low"] == ci["ci_high"] == 0.0
    assert ci["n_groups"] == 3
    better = bootstrap_delta_ci([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"], n_boot=200)
    assert better["ci_low"] == pytest.approx(1.0)


def test_error_case_selection_is_deterministic():
    cases = select_error_cases([1, 5, 30, 2, 12, 1, 60], n=3)
    assert cases.successes == [0, 5, 3]
    assert cases.false_negatives == [6, 2, 4]
    assert all(r > 1 for r in (np.array([1, 5, 30, 2, 12, 1, 60])[cases.false_positives]))


def test_tonal_dispersion_and_hub_counts():
    from cover_retrieval.evaluation.analysis import hub_counts, tonal_dispersion

    static = np.zeros((12, 10))
    static[0] = 1.0
    varied = l2_normalize_frames(np.eye(12)[:, np.arange(10) % 12] + 1e-3)
    d = tonal_dispersion(np.stack([static, varied]))
    assert d[0] == pytest.approx(0.0, abs=1e-12) and d[1] > 0.5
    rows = [
        {"rank": 1, "candidate_pid": "P_1", "is_relevant": "False"},
        {"rank": 2, "candidate_pid": "P_2", "is_relevant": "True"},
        {"rank": 11, "candidate_pid": "P_1", "is_relevant": "False"},
        {"rank": 3, "candidate_pid": "P_1", "is_relevant": "False"},
    ]
    assert hub_counts(rows) == {"P_1": 2}
