"""Committed, ID-only manifests that freeze every experimental protocol.

Files written to ``data/manifests/`` (identifiers only, no titles/artists/tags):

* ``coveranalysis_splits.csv``   wid, pid, role    (every assigned Cover Analysis track)
* ``coveranalysis_excluded.csv`` wid, reason       (WIDs rejected by validation)
* ``dev_queries.csv``            pid, wid          (20 fixed development queries)
* ``dev_candidates.csv``         pid, wid, role    (120 = query + distractor tracks)
* ``calibration_queries.csv``    pid, wid          (all 20 calibration tracks)
* ``calibration_candidates.csv`` pid, wid, role    (calibration + validation tracks)
* ``benchmark_tracks.csv``       pid, wid, role    (15,000 = 13,000 clique + 2,000 noise)
* ``manifest_info.json``         seed, sizes, counts, settings

Relevance is *never* stored as a feature: it is recomputed as WID equality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cover_retrieval.data.splits import ROLE_ORDER, SplitAssignment
from cover_retrieval.features.preprocessing import Track
from cover_retrieval.utils.io import read_csv, write_csv, write_json


@dataclass
class Protocol:
    """A retrieval protocol: queries ranked against a fixed candidate pool."""

    name: str
    queries: list[Track]
    candidates: list[Track]

    def relevance_matrix(self) -> np.ndarray:
        """``(Q, C)`` bool: same WID and not the query itself."""
        q_wid = np.array([t.wid for t in self.queries])
        c_wid = np.array([t.wid for t in self.candidates])
        return (q_wid[:, None] == c_wid[None, :]) & ~self.self_mask()

    def self_mask(self) -> np.ndarray:
        """``(Q, C)`` bool: candidate is the query recording itself (to be excluded)."""
        q_pid = np.array([t.pid for t in self.queries])
        c_pid = np.array([t.pid for t in self.candidates])
        return q_pid[:, None] == c_pid[None, :]

    def check(self) -> None:
        pids = [t.pid for t in self.candidates]
        if len(set(pids)) != len(pids):
            raise ValueError(f"{self.name}: duplicate candidate PIDs")
        if pids != sorted(pids):
            raise ValueError(f"{self.name}: candidates must be sorted by PID (deterministic ties)")
        relevance = self.relevance_matrix()
        if not relevance.any(axis=1).all():
            raise ValueError(f"{self.name}: some query has no relevant candidate")


def dev_protocol_tracks(assignment: SplitAssignment, works: Mapping[str, Sequence[str]]):
    """Build the fixed development queries/candidates from a role assignment."""
    queries = [Track(sorted(works[wid])[0], wid) for wid in sorted(assignment.roles["query"])]
    candidates = sorted(
        (
            Track(pid, wid)
            for role in ("query", "distractor")
            for wid in assignment.roles[role]
            for pid in works[wid]
        ),
        key=lambda t: t.pid,
    )
    return queries, candidates


def write_coveranalysis_manifests(
    out_dir: str | Path,
    assignment: SplitAssignment,
    works: Mapping[str, Sequence[str]],
    info: Mapping[str, Any],
) -> None:
    out = Path(out_dir)
    role_of = assignment.role_of()
    split_rows = [
        {"wid": wid, "pid": pid, "role": role_of[wid]}
        for role in ROLE_ORDER
        for wid in sorted(assignment.roles[role])
        for pid in works[wid]
    ]
    write_csv(out / "coveranalysis_splits.csv", split_rows, ["wid", "pid", "role"])
    write_csv(
        out / "coveranalysis_excluded.csv",
        [{"wid": w, "reason": r} for w, r in sorted(assignment.excluded.items())],
        ["wid", "reason"],
    )
    queries, candidates = dev_protocol_tracks(assignment, works)
    write_csv(out / "dev_queries.csv", [{"pid": t.pid, "wid": t.wid} for t in queries], ["pid", "wid"])
    write_csv(
        out / "dev_candidates.csv",
        [{"pid": t.pid, "wid": t.wid, "role": role_of[t.wid]} for t in candidates],
        ["pid", "wid", "role"],
    )
    calibration = sorted(
        (Track(pid, wid) for wid in assignment.roles["calibration"] for pid in works[wid]),
        key=lambda t: t.pid,
    )
    write_csv(
        out / "calibration_queries.csv", [{"pid": t.pid, "wid": t.wid} for t in calibration], ["pid", "wid"]
    )
    calib_candidates = sorted(
        (
            Track(pid, wid)
            for role in ("calibration", "validation")
            for wid in assignment.roles[role]
            for pid in works[wid]
        ),
        key=lambda t: t.pid,
    )
    write_csv(
        out / "calibration_candidates.csv",
        [{"pid": t.pid, "wid": t.wid, "role": role_of[t.wid]} for t in calib_candidates],
        ["pid", "wid", "role"],
    )
    counts = {
        role: {"wids": len(w), "tracks": sum(len(works[x]) for x in w)}
        for role, w in assignment.roles.items()
    }
    write_json(
        out / "manifest_info.json",
        {
            **info,
            "role_counts": counts,
            "n_excluded_wids": len(assignment.excluded),
            "n_dev_queries": len(queries),
            "n_dev_candidates": len(candidates),
            "n_calibration_queries": len(calibration),
            "n_calibration_candidates": len(calib_candidates),
        },
    )


def write_benchmark_manifest(out_dir: str | Path, works: Mapping[str, Sequence[str]]) -> None:
    rows = sorted(
        (
            {"pid": pid, "wid": wid, "role": "clique" if len(pids) > 1 else "noise"}
            for wid, pids in works.items()
            for pid in pids
        ),
        key=lambda r: r["pid"],
    )
    write_csv(Path(out_dir) / "benchmark_tracks.csv", rows, ["pid", "wid", "role"])


def _tracks(rows: list[dict[str, str]]) -> list[Track]:
    return [Track(r["pid"], r["wid"]) for r in rows]


def load_protocol(manifests_dir: str | Path, name: str) -> Protocol:
    """Load ``dev``, ``calibration`` or ``benchmark`` from committed manifests."""
    d = Path(manifests_dir)
    if name in ("dev", "calibration"):
        protocol = Protocol(
            name,
            _tracks(read_csv(d / f"{name}_queries.csv")),
            _tracks(read_csv(d / f"{name}_candidates.csv")),
        )
    elif name == "benchmark":
        rows = read_csv(d / "benchmark_tracks.csv")
        candidates = _tracks(rows)
        queries = [Track(r["pid"], r["wid"]) for r in rows if r["role"] == "clique"]
        protocol = Protocol(name, queries, candidates)
    else:
        raise ValueError(f"unknown protocol {name!r}")
    protocol.check()
    return protocol


def load_split_tracks(manifests_dir: str | Path, role: str) -> list[Track]:
    rows = read_csv(Path(manifests_dir) / "coveranalysis_splits.csv")
    return sorted((Track(r["pid"], r["wid"]) for r in rows if r["role"] == role), key=lambda t: t.pid)
