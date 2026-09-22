"""Load, validate and preprocess HPCP for a list of tracks, with an on-disk cache.

Two views are produced from each raw ``(T, 12)`` Da-TACOS HPCP array:

* ``classical`` – ``(12, n_frames)`` (default 96): frame L2 -> area resample -> frame L2.
  Input to the alignment baseline and the hybrid reranker.
* ``encoder``   – ``(12, cache_frames)`` (default 512), same recipe at higher
  resolution; the encoder crops/pools it to its input length.

Caches live in ``data/cache`` (git-ignored) and are keyed by a fingerprint of the
track list and every preprocessing setting, so stale caches are never reused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cover_retrieval.data.datacos import feature_path, load_validated_hpcp
from cover_retrieval.features.hpcp import l2_normalize_frames, uniform_resample

PREPROCESSING_VERSION = 1


@dataclass(frozen=True)
class Track:
    pid: str
    wid: str


@dataclass
class FeatureStore:
    """Preprocessed features for an ordered list of tracks."""

    tracks: list[Track]
    views: dict[str, np.ndarray]  # view name -> (N, 12, T_view) float32
    stats: dict[str, np.ndarray] = field(default_factory=dict)  # per-track diagnostics

    def __post_init__(self) -> None:
        self._index = {track.pid: i for i, track in enumerate(self.tracks)}

    @property
    def pids(self) -> list[str]:
        return [t.pid for t in self.tracks]

    @property
    def wids(self) -> list[str]:
        return [t.wid for t in self.tracks]

    def index(self, pids: Sequence[str]) -> np.ndarray:
        return np.array([self._index[pid] for pid in pids], dtype=np.int64)

    def get(self, view: str, pids: Sequence[str]) -> np.ndarray:
        return self.views[view][self.index(pids)]

    def subset(self, pids: Sequence[str]) -> FeatureStore:
        idx = self.index(pids)
        return FeatureStore(
            [self.tracks[i] for i in idx],
            {k: v[idx] for k, v in self.views.items()},
            {k: v[idx] for k, v in self.stats.items()},
        )


def preprocessing_settings(config: Mapping[str, Any], nonfinite_policy: str | None = None) -> dict[str, Any]:
    return {
        "version": PREPROCESSING_VERSION,
        "classical_frames": int(config["features"]["n_frames"]),
        "encoder_frames": int(config["encoder"]["cache_frames"]),
        "layout": config["features"]["layout"],
        "nonfinite_policy": nonfinite_policy or config["data"]["nonfinite_policy"],
        "min_frames": int(config["data"]["min_frames"]),
    }


def _fingerprint(tracks: Sequence[Track], settings: Mapping[str, Any]) -> str:
    payload = json.dumps({"tracks": [(t.pid, t.wid) for t in tracks], "settings": dict(settings)})
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _process_one(root: Path, track: Track, settings: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict]:
    raw, result = load_validated_hpcp(
        feature_path(root, track.wid, track.pid),
        track.wid,
        track.pid,
        min_frames=settings["min_frames"],
        nonfinite_policy=settings["nonfinite_policy"],
        layout=settings["layout"],
    )
    normalized = l2_normalize_frames(raw)
    views = {
        "classical": l2_normalize_frames(uniform_resample(normalized, settings["classical_frames"])),
        "encoder": l2_normalize_frames(uniform_resample(normalized, settings["encoder_frames"])),
    }
    stats = {
        "raw_frames": result.n_frames,
        "n_nonfinite": result.n_nonfinite,
        "silent_fraction": result.silent_frame_fraction,
    }
    return views, stats


def build_feature_store(
    tracks: Sequence[Track],
    features_root: str | Path,
    config: Mapping[str, Any],
    *,
    cache_name: str,
    nonfinite_policy: str | None = None,
    threads: int = 8,
    verbose: bool = True,
) -> FeatureStore:
    """Return preprocessed features for ``tracks`` (loading from cache when valid)."""
    tracks = list(tracks)
    settings = preprocessing_settings(config, nonfinite_policy)
    fingerprint = _fingerprint(tracks, settings)
    cache_dir = Path(config["paths"]["cache_dir"])
    cache_path = cache_dir / f"{cache_name}_{fingerprint}.npz"
    if cache_path.exists():
        with np.load(cache_path) as data:
            views = {k[5:]: data[k] for k in data.files if k.startswith("view_")}
            stats = {k[6:]: data[k] for k in data.files if k.startswith("stats_")}
        return FeatureStore(tracks, views, stats)

    root = Path(features_root)
    if verbose:
        print(f"[features] preprocessing {len(tracks)} tracks -> {cache_path}", flush=True)
    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(lambda t: _process_one(root, t, settings), tracks))
    views = {
        name: np.stack([r[0][name] for r in results]).astype(np.float32) for name in ("classical", "encoder")
    }
    stats = {
        key: np.array([r[1][key] for r in results])
        for key in ("raw_frames", "n_nonfinite", "silent_fraction")
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        **{f"view_{k}": v for k, v in views.items()},
        **{f"stats_{k}": v for k, v in stats.items()},
    )
    return FeatureStore(tracks, views, stats)
