"""Da-TACOS metadata and HPCP feature loading with explicit validation.

Only identifiers (WID = work/clique, PID = performance/recording) are used from the
metadata. Titles, artists, years, tags, key, tempo and instrumental flags are never
read into the retrieval pipeline; ``work_index`` deliberately returns IDs only.

File layout (identical for the Zenodo archives and for selective member fetches)::

    <root>/<features_dir>/<WID>_hpcp/<PID>_hpcp.h5

Each H5 file (written by ``deepdish``) holds a ``hpcp`` dataset of shape (T, 12)
and ``label`` (WID) / ``track_id`` (PID) attributes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from cover_retrieval.features.hpcp import to_pitch_by_time

Subset = Literal["coveranalysis", "benchmark"]
NonFinitePolicy = Literal["reject", "zero"]

METADATA_FILES: dict[str, str] = {
    "coveranalysis": "da-tacos_coveranalysis_subset_metadata.json",
    "benchmark": "da-tacos_benchmark_subset_metadata.json",
}


class DataValidationError(ValueError):
    """Raised when metadata or feature files violate the expected format."""


# --------------------------------------------------------------------------- metadata
def load_metadata(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load a Da-TACOS metadata JSON and check its WID -> PID -> record structure."""
    path = Path(path)
    try:
        metadata = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise DataValidationError(f"{path}: malformed JSON ({error})") from error
    if not isinstance(metadata, dict) or not metadata:
        raise DataValidationError(f"{path}: expected a non-empty WID -> PID mapping")
    for wid, performances in metadata.items():
        if not wid.startswith("W_") or not isinstance(performances, dict) or not performances:
            raise DataValidationError(f"{path}: bad work entry {wid!r}")
        for pid, record in performances.items():
            if not pid.startswith("P_") or not isinstance(record, dict):
                raise DataValidationError(f"{path}: bad performance entry {wid}/{pid}")
            if record.get("work_id", wid) != wid or record.get("perf_id", pid) != pid:
                raise DataValidationError(f"{path}: ID mismatch inside {wid}/{pid}")
    return metadata


def work_index(metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    """WID -> lexicographically sorted PIDs. Identifiers only; no descriptive metadata."""
    return {wid: sorted(metadata[wid]) for wid in sorted(metadata)}


def metadata_path(config: Mapping[str, Any], subset: Subset) -> Path:
    root = Path(config["paths"]["datacos_root"])
    return root / config["data"]["metadata_dir"] / METADATA_FILES[subset]


def features_root(config: Mapping[str, Any], subset: Subset) -> Path:
    root = Path(config["paths"]["datacos_root"])
    return root / config["data"][f"{subset}_features_dir"]


def feature_path(root: str | Path, wid: str, pid: str, feature: str = "hpcp") -> Path:
    return Path(root) / f"{wid}_{feature}" / f"{pid}_{feature}.h5"


def archive_member_name(features_dir: str, wid: str, pid: str, feature: str = "hpcp") -> str:
    """Name of the corresponding member inside the Zenodo ZIP archive."""
    return f"{features_dir}/{wid}_{feature}/{pid}_{feature}.h5"


# --------------------------------------------------------------------------- features
@dataclass
class HPCPRecord:
    hpcp: np.ndarray  # raw array exactly as stored, (T, 12) for Da-TACOS
    label: str | None
    track_id: str | None


def _decode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.tolist()
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8")
    return str(value)


def load_hpcp(path: str | Path) -> HPCPRecord:
    """Read the ``hpcp`` array and WID/PID labels from a Da-TACOS H5 file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        with h5py.File(path, "r") as handle:
            if "hpcp" not in handle:
                raise DataValidationError(f"{path}: no 'hpcp' dataset")
            hpcp = np.asarray(handle["hpcp"][()])
            label = handle.attrs.get("label", handle["label"][()] if "label" in handle else None)
            track = handle.attrs.get("track_id", handle["track_id"][()] if "track_id" in handle else None)
    except OSError as error:
        raise DataValidationError(f"{path}: unreadable H5 ({error})") from error
    return HPCPRecord(hpcp=hpcp, label=_decode(label), track_id=_decode(track))


@dataclass
class ValidationResult:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    n_frames: int = 0
    raw_shape: tuple[int, ...] = ()
    n_nonfinite: int = 0
    silent_frame_fraction: float = 0.0


def validate_hpcp(
    raw: np.ndarray,
    *,
    min_frames: int = 1,
    nonfinite_policy: NonFinitePolicy = "reject",
    layout: str = "auto",
) -> ValidationResult:
    """Check one raw HPCP array. Never modifies ``raw``."""
    result = ValidationResult(valid=True, raw_shape=tuple(raw.shape))
    if raw.ndim != 2 or 12 not in raw.shape:
        return ValidationResult(False, [f"bad shape {raw.shape}"], raw_shape=tuple(raw.shape))
    if raw.size == 0:
        return ValidationResult(False, ["empty array"], raw_shape=tuple(raw.shape))
    try:
        pitch_by_time = to_pitch_by_time(raw, layout=layout)
    except ValueError as error:
        return ValidationResult(False, [str(error)], raw_shape=tuple(raw.shape))
    result.n_frames = int(pitch_by_time.shape[1])
    finite = np.isfinite(pitch_by_time)
    result.n_nonfinite = int((~finite).sum())
    if result.n_nonfinite and nonfinite_policy == "reject":
        result.valid = False
        result.reasons.append(f"{result.n_nonfinite} non-finite values")
    energy = np.where(finite, np.abs(pitch_by_time), 0.0).sum(axis=0)
    result.silent_frame_fraction = float((energy == 0).mean())
    if not energy.any():
        result.valid = False
        result.reasons.append("all frames have zero energy")
    if result.n_frames < min_frames:
        result.valid = False
        result.reasons.append(f"only {result.n_frames} frames (< {min_frames})")
    return result


def load_validated_hpcp(
    path: str | Path,
    wid: str,
    pid: str,
    *,
    min_frames: int = 1,
    nonfinite_policy: NonFinitePolicy = "reject",
    layout: str = "auto",
) -> tuple[np.ndarray, ValidationResult]:
    """Load, check labels against the manifest, validate, and return ``(12, T)`` HPCP.

    Raises ``DataValidationError`` for invalid records. Under ``nonfinite_policy="zero"``
    non-finite entries are replaced by 0 and counted in the returned result.
    """
    record = load_hpcp(path)
    if record.label is not None and record.label != wid:
        raise DataValidationError(f"{path}: label {record.label} != manifest WID {wid}")
    if record.track_id is not None and record.track_id != pid:
        raise DataValidationError(f"{path}: track_id {record.track_id} != manifest PID {pid}")
    result = validate_hpcp(
        record.hpcp, min_frames=min_frames, nonfinite_policy=nonfinite_policy, layout=layout
    )
    if not result.valid:
        raise DataValidationError(f"{path}: " + "; ".join(result.reasons))
    hpcp = to_pitch_by_time(record.hpcp, layout=layout).astype(np.float32)
    if result.n_nonfinite:
        hpcp = np.where(np.isfinite(hpcp), hpcp, 0.0).astype(np.float32)
    return hpcp, result
