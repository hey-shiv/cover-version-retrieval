"""Key (transposition) invariance by cyclic pitch-class rotation.

Convention (tested in tests/test_transposition.py)::

    rotate_pitch_classes(X, k)[p, t] == X[(p - k) % 12, t]

i.e. rotating by ``k`` moves every pitch-class bin *up* by ``k`` semitones. If ``Y``
is ``X`` transposed up by ``s`` semitones, the rotation that maps ``Y`` back onto
``X`` is ``k = (12 - s) % 12``.

The rotation is chosen from the audio features alone ("optimal transposition index"
style: cosine between time-averaged chroma profiles, Serrà et al. 2008). Metadata key
annotations are never used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_SHIFTS = 12


def rotate_pitch_classes(x: np.ndarray, semitones: int) -> np.ndarray:
    """Cyclically rotate the pitch-class axis (axis 0 for ``(12, T)``, axis -2 for batches)."""
    axis = 0 if x.ndim == 2 else -2
    if x.shape[axis] != 12:
        raise ValueError(f"expected 12 pitch classes on axis {axis}, got shape {x.shape}")
    return np.roll(x, int(semitones) % 12, axis=axis)


def chroma_profile(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Time-averaged, L2-normalised pitch-class profile of a ``(12, T)`` sequence."""
    profile = np.asarray(x, dtype=np.float64).mean(axis=-1)
    norm = np.linalg.norm(profile, axis=-1, keepdims=True)
    return profile / np.maximum(norm, eps)


def profile_rotation_scores(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Cosine between the query profile and the candidate profile rotated by 0..11."""
    q = chroma_profile(query)
    c = chroma_profile(candidate)
    return np.array([float(q @ np.roll(c, k)) for k in range(N_SHIFTS)])


def batch_profile_rotation_scores(query_profile: np.ndarray, candidate_profiles: np.ndarray) -> np.ndarray:
    """Vectorised form: ``(12,)`` x ``(B, 12)`` profiles -> ``(B, 12)`` scores."""
    # rolled[b, k, p] = candidate_profiles[b, (p - k) % 12]
    index = (np.arange(12)[None, :] - np.arange(N_SHIFTS)[:, None]) % 12
    rolled = candidate_profiles[:, index]
    return rolled @ query_profile


@dataclass
class RotationResult:
    shift: int  # rotation applied to the candidate
    scores: np.ndarray  # (12,) profile cosine for every shift
    rotated_candidate: np.ndarray


def best_key_rotation(query: np.ndarray, candidate: np.ndarray) -> RotationResult:
    """Pick the candidate rotation maximising profile cosine (ties -> smallest shift)."""
    scores = profile_rotation_scores(query, candidate)
    shift = int(np.argmax(scores))
    return RotationResult(shift, scores, rotate_pitch_classes(candidate, shift))
