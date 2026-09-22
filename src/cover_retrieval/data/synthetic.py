"""Synthetic Da-TACOS-shaped fixture for tests and the end-to-end smoke pipeline.

It writes metadata JSON and HPCP H5 files in exactly the Da-TACOS layout (``hpcp``
dataset of shape (T, 12), ``label``/``track_id`` attributes, per-frame max
normalisation), so every loader and script runs unchanged on it.

Musical model (deliberately simple, *not* a claim about real music): a work is a
random chord progression; a performance transposes it by a random number of
semitones, stretches its tempo, may drop or repeat a section, and adds noise.
Results on this data only show that the code runs; they say nothing about Da-TACOS.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from cover_retrieval.data.datacos import METADATA_FILES, feature_path

_TRIADS = [(0, 4, 7), (0, 3, 7), (0, 4, 7, 10), (0, 3, 7, 10), (0, 5, 7)]


def _chord_template(root: int, quality: tuple[int, ...]) -> np.ndarray:
    vector = np.zeros(12)
    for interval in quality:
        vector[(root + interval) % 12] += 1.0
    return vector


def _work(
    rng: np.random.Generator, n_sections: int = 4, chords_per_section: int = 4
) -> list[list[np.ndarray]]:
    return [
        [
            _chord_template(int(rng.integers(12)), _TRIADS[int(rng.integers(len(_TRIADS)))])
            for _ in range(chords_per_section)
        ]
        for _ in range(n_sections)
    ]


def _perform(work: list[list[np.ndarray]], rng: np.random.Generator, base_frames: int) -> np.ndarray:
    sections = list(work)
    if len(sections) > 2 and rng.random() < 0.3:
        del sections[int(rng.integers(len(sections)))]
    if rng.random() < 0.3:
        sections.insert(int(rng.integers(len(sections) + 1)), sections[int(rng.integers(len(sections)))])
    shift = int(rng.integers(12))
    tempo = float(rng.uniform(0.7, 1.4))
    frames = []
    for section in sections:
        for chord in section:
            length = max(2, int(round(base_frames * tempo * rng.uniform(0.8, 1.2))))
            frames.extend([np.roll(chord, shift)] * length)
    hpcp = np.array(frames) + 0.35 * rng.random((len(frames), 12))
    hpcp[rng.random(len(frames)) < 0.02] = 0.0  # occasional silent frames
    peak = hpcp.max(axis=1, keepdims=True)
    return np.where(peak > 0, hpcp / np.where(peak > 0, peak, 1), 0.0).astype(np.float32)


def _write_h5(path: Path, hpcp: np.ndarray, wid: str, pid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("hpcp", data=hpcp)
        handle.attrs["label"] = np.bytes_(wid)
        handle.attrs["track_id"] = np.bytes_(pid)


def make_synthetic_datacos(
    root: str | Path,
    *,
    n_coveranalysis_works: int = 120,
    n_benchmark_cliques: int = 12,
    benchmark_clique_size: int = 4,
    n_benchmark_noise: int = 16,
    base_frames: int = 24,
    seed: int = 0,
    features_dirs: tuple[str, str] = ("da-tacos_coveranalysis_subset_hpcp", "da-tacos_benchmark_subset_hpcp"),
    metadata_dir: str = "da-tacos_metadata",
) -> dict[str, int]:
    """Create the fixture under ``root``; returns counts."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    next_id = iter(range(100_000, 10_000_000))
    layout = {
        "coveranalysis": [2] * n_coveranalysis_works,
        "benchmark": [benchmark_clique_size] * n_benchmark_cliques + [1] * n_benchmark_noise,
    }
    counts = {}
    for (subset, sizes), features_dir in zip(layout.items(), features_dirs, strict=True):
        metadata: dict[str, dict[str, dict[str, str]]] = {}
        for size in sizes:
            wid = f"W_{next(next_id)}"
            work = _work(rng)
            metadata[wid] = {}
            for _ in range(size):
                pid = f"P_{next(next_id)}"
                metadata[wid][pid] = {"work_id": wid, "perf_id": pid, "perf_title": "synthetic"}
                _write_h5(
                    feature_path(root / features_dir, wid, pid), _perform(work, rng, base_frames), wid, pid
                )
        meta_path = root / metadata_dir / METADATA_FILES[subset]
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata))
        counts[subset] = sum(sizes)
    return counts
