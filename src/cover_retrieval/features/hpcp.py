"""HPCP representation utilities. Internal convention: arrays are ``(12, T)``.

Row ``p`` is pitch class ``p`` (Da-TACOS/Essentia HPCP bin order), column ``t`` is a
time frame. Nothing here reads metadata: key or tempo normalisation from metadata
would leak annotation information into the features.
"""

from __future__ import annotations

import numpy as np

N_PITCH_CLASSES = 12


def to_pitch_by_time(x: np.ndarray, layout: str = "auto") -> np.ndarray:
    """Return ``x`` as ``(12, T)``.

    ``layout`` is ``"time_by_pitch"`` (Da-TACOS storage order), ``"pitch_by_time"``
    or ``"auto"``. ``auto`` transposes when exactly one axis has size 12 and refuses
    the ambiguous ``(12, 12)`` case instead of guessing.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"HPCP must be 2-D, got shape {x.shape}")
    if layout == "time_by_pitch":
        if x.shape[1] != N_PITCH_CLASSES:
            raise ValueError(f"expected (T, 12), got {x.shape}")
        return x.T
    if layout == "pitch_by_time":
        if x.shape[0] != N_PITCH_CLASSES:
            raise ValueError(f"expected (12, T), got {x.shape}")
        return x
    if layout != "auto":
        raise ValueError(f"unknown layout {layout!r}")
    rows12, cols12 = x.shape[0] == N_PITCH_CLASSES, x.shape[1] == N_PITCH_CLASSES
    if rows12 and cols12:
        raise ValueError("ambiguous (12, 12) HPCP; pass an explicit layout")
    if cols12:
        return x.T
    if rows12:
        return x
    raise ValueError(f"no axis of size 12 in HPCP shape {x.shape}")


def l2_normalize_frames(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalise every frame (column) of a ``(12, T)`` array.

    All-zero (silent) frames stay exactly zero rather than becoming NaN.
    """
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=0, keepdims=True)
    return np.where(norms > eps, x / np.maximum(norms, eps), 0.0).astype(np.float32)


def uniform_resample(x: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample a ``(12, T)`` sequence to exactly ``n_frames`` uniformly spaced frames.

    Downsampling averages the frames inside each of ``n_frames`` equal-width time
    bins (fractional bin edges are weighted exactly), which avoids the aliasing that
    point sampling or linear interpolation would cause when T >> n_frames.
    Upsampling uses linear interpolation. Returns float32.
    """
    x = np.asarray(x, dtype=np.float64)
    n_pitch, length = x.shape
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if length == 0:
        raise ValueError("cannot resample an empty sequence")
    if length == n_frames:
        return x.astype(np.float32)
    if length < n_frames:
        source = (np.arange(length) + 0.5) / length
        target = (np.arange(n_frames) + 0.5) / n_frames
        out = np.stack([np.interp(target, source, row) for row in x])
        return out.astype(np.float32)
    # Exact area-weighted averaging via the cumulative sum at fractional edges.
    cumulative = np.concatenate([np.zeros((n_pitch, 1)), np.cumsum(x, axis=1)], axis=1)
    edges = np.linspace(0.0, length, n_frames + 1)
    lower = np.floor(edges).astype(int)
    frac = edges - lower
    upper = np.minimum(lower + 1, length)
    at_edges = cumulative[:, lower] + frac * (cumulative[:, upper] - cumulative[:, lower])
    widths = np.diff(edges)
    return (np.diff(at_edges, axis=1) / widths).astype(np.float32)


def preprocess_hpcp(x: np.ndarray, n_frames: int, layout: str = "auto") -> np.ndarray:
    """Full classical preprocessing: orient -> frame L2 -> resample -> frame L2.

    The second normalisation restores unit-norm frames after bin averaging, so the
    cosine cost between frames stays within [0, 2].
    """
    x = to_pitch_by_time(x, layout=layout)
    if not np.isfinite(x).all():
        raise ValueError("preprocess_hpcp received non-finite values; apply the nonfinite policy first")
    return l2_normalize_frames(uniform_resample(l2_normalize_frames(x), n_frames))
