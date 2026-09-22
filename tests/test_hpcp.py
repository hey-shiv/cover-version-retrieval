"""HPCP orientation, normalisation and resampling."""

import numpy as np
import pytest

from cover_retrieval.features.hpcp import (
    l2_normalize_frames,
    preprocess_hpcp,
    to_pitch_by_time,
    uniform_resample,
)


def test_auto_orientation_transposes_time_by_pitch():
    x = np.random.default_rng(0).random((300, 12))
    out = to_pitch_by_time(x)
    assert out.shape == (12, 300)
    np.testing.assert_array_equal(out, x.T)


def test_auto_orientation_keeps_pitch_by_time():
    x = np.random.default_rng(0).random((12, 40))
    assert to_pitch_by_time(x) is x


def test_ambiguous_square_requires_explicit_layout():
    x = np.zeros((12, 12))
    with pytest.raises(ValueError):
        to_pitch_by_time(x)
    assert to_pitch_by_time(x, layout="time_by_pitch").shape == (12, 12)


@pytest.mark.parametrize("shape", [(10, 11), (12,), (3, 12, 4)])
def test_invalid_shapes_rejected(shape):
    with pytest.raises(ValueError):
        to_pitch_by_time(np.zeros(shape))


def test_explicit_layout_mismatch_rejected():
    with pytest.raises(ValueError):
        to_pitch_by_time(np.zeros((12, 50)), layout="time_by_pitch")


def test_l2_normalisation_unit_frames_and_silence():
    x = np.random.default_rng(1).random((12, 20))
    x[:, 5] = 0.0
    y = l2_normalize_frames(x)
    norms = np.linalg.norm(y, axis=0)
    np.testing.assert_allclose(np.delete(norms, 5), 1.0, rtol=1e-6)
    assert norms[5] == 0.0
    assert np.isfinite(y).all()


@pytest.mark.parametrize("length", [7, 96, 97, 250, 13728])
def test_resample_exact_length(length):
    x = np.random.default_rng(2).random((12, length))
    assert uniform_resample(x, 96).shape == (12, 96)


def test_downsampling_is_area_average():
    x = np.random.default_rng(3).random((12, 960))
    y = uniform_resample(x, 96)
    np.testing.assert_allclose(y, x.reshape(12, 96, 10).mean(axis=2), rtol=1e-5)
    # fractional bins still preserve the overall mean
    z = uniform_resample(x[:, :955], 96)
    np.testing.assert_allclose(z.mean(axis=1), x[:, :955].mean(axis=1), rtol=1e-5)


def test_constant_sequence_is_preserved():
    x = np.tile(np.arange(12, dtype=float)[:, None], (1, 33))
    np.testing.assert_allclose(uniform_resample(x, 96), np.tile(np.arange(12.0)[:, None], (1, 96)), rtol=1e-6)
    np.testing.assert_allclose(uniform_resample(x, 8), np.tile(np.arange(12.0)[:, None], (1, 8)), rtol=1e-6)


def test_preprocess_pipeline_outputs_unit_frames_without_nan():
    x = np.random.default_rng(4).random((500, 12))
    x[100:200] = 0.0  # silent stretch
    y = preprocess_hpcp(x, 96)
    assert y.shape == (12, 96) and y.dtype == np.float32
    assert np.isfinite(y).all()
    norms = np.linalg.norm(y, axis=0)
    assert ((np.abs(norms - 1) < 1e-5) | (norms == 0)).all()


def test_preprocess_rejects_nonfinite():
    x = np.ones((200, 12))
    x[3, 4] = np.nan
    with pytest.raises(ValueError):
        preprocess_hpcp(x, 96)
