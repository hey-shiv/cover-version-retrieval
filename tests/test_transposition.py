"""Key invariance: rotation convention and recovery of all 12 transpositions."""

import numpy as np
import pytest

from cover_retrieval.alignment.transposition import (
    batch_profile_rotation_scores,
    best_key_rotation,
    chroma_profile,
    profile_rotation_scores,
    rotate_pitch_classes,
)
from cover_retrieval.features.hpcp import l2_normalize_frames


@pytest.fixture()
def sequence():
    rng = np.random.default_rng(7)
    x = rng.random((12, 50)) ** 3  # peaky, chroma-like
    return l2_normalize_frames(x)


def test_rotation_convention(sequence):
    rotated = rotate_pitch_classes(sequence, 3)
    for p in range(12):
        np.testing.assert_array_equal(rotated[p], sequence[(p - 3) % 12])


def test_shift_zero_and_twelve_are_identity(sequence):
    np.testing.assert_array_equal(rotate_pitch_classes(sequence, 0), sequence)
    np.testing.assert_array_equal(rotate_pitch_classes(sequence, 12), sequence)


@pytest.mark.parametrize("k", range(12))
def test_inverse_rotation_restores(sequence, k):
    np.testing.assert_array_equal(rotate_pitch_classes(rotate_pitch_classes(sequence, k), 12 - k), sequence)


@pytest.mark.parametrize("s", range(12))
def test_recovers_every_transposition(sequence, s):
    """A copy transposed up by s semitones needs candidate rotation (12 - s) % 12."""
    transposed = rotate_pitch_classes(sequence, s)
    result = best_key_rotation(sequence, transposed)
    assert result.shift == (12 - s) % 12
    np.testing.assert_allclose(result.rotated_candidate, sequence)
    assert result.scores[result.shift] == pytest.approx(1.0)


def test_rotation_acts_only_on_pitch_axis(sequence):
    rotated = rotate_pitch_classes(sequence, 5)
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=0), np.linalg.norm(sequence, axis=0), rtol=1e-6)


def test_batch_scores_match_pairwise(sequence):
    rng = np.random.default_rng(0)
    candidates = rng.random((6, 12, 30))
    batch = batch_profile_rotation_scores(chroma_profile(sequence), chroma_profile(candidates))
    for b in range(6):
        np.testing.assert_allclose(batch[b], profile_rotation_scores(sequence, candidates[b]))


def test_batch_rotation_of_3d_arrays():
    x = np.arange(2 * 12 * 3, dtype=float).reshape(2, 12, 3)
    np.testing.assert_array_equal(rotate_pitch_classes(x, 1)[1], np.roll(x[1], 1, axis=0))
