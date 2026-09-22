"""Profile the selected Da-TACOS tracks and write reports/dataset_profile.md.

    python scripts/profile_data.py --config configs/base.yaml

Loads every manifest track through the validated loader (building the feature cache
as a side effect), and reports counts, frame-length statistics, silent-frame and
non-finite statistics per role. Uses identifiers only; no descriptive metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cover_retrieval.data.datacos import (
    feature_path,
    features_root,
    load_hpcp,
    load_metadata,
    metadata_path,
    work_index,
)
from cover_retrieval.data.manifests import load_protocol, load_split_tracks
from cover_retrieval.data.splits import ROLE_ORDER
from cover_retrieval.evaluation.analysis import plot_feature_matrices
from cover_retrieval.features.hpcp import to_pitch_by_time
from cover_retrieval.pipeline import figures_dir, markdown_table, protocol_store, results_dir, split_store
from cover_retrieval.utils.io import load_config, read_csv, read_json, write_csv, write_json


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--include-benchmark", action="store_true", help="also profile the benchmark tracks")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    manifests = Path(config["paths"]["manifests_dir"])
    info = read_json(manifests / "manifest_info.json")
    works = work_index(load_metadata(metadata_path(config, "coveranalysis")))

    role_rows, track_rows = [], []
    for role in ROLE_ORDER:
        tracks = load_split_tracks(manifests, role)
        store = split_store(config, tracks, role)
        frames = store.stats["raw_frames"].astype(float)
        role_rows.append(
            {
                "subset/role": f"coveranalysis/{role}",
                "WIDs": len({t.wid for t in tracks}),
                "tracks": len(tracks),
                "min frames": int(frames.min()),
                "median frames": int(np.median(frames)),
                "max frames": int(frames.max()),
                "non-finite values": int(store.stats["n_nonfinite"].sum()),
                "mean silent-frame %": 100 * float(store.stats["silent_fraction"].mean()),
            }
        )
        for i, t in enumerate(tracks):
            track_rows.append(
                {
                    "subset": "coveranalysis",
                    "role": role,
                    "wid": t.wid,
                    "pid": t.pid,
                    "raw_frames": int(store.stats["raw_frames"][i]),
                    "silent_fraction": round(float(store.stats["silent_fraction"][i]), 5),
                    "n_nonfinite": int(store.stats["n_nonfinite"][i]),
                }
            )

    bench_summary = None
    if args.include_benchmark:
        protocol = load_protocol(manifests, "benchmark")
        store = protocol_store(config, protocol, "benchmark", nonfinite_policy="zero")
        frames = store.stats["raw_frames"].astype(float)
        roles = dict((r["pid"], r["role"]) for r in read_csv(manifests / "benchmark_tracks.csv"))
        role_rows.append(
            {
                "subset/role": "benchmark/all",
                "WIDs": len(set(store.wids)),
                "tracks": len(store.tracks),
                "min frames": int(frames.min()),
                "median frames": int(np.median(frames)),
                "max frames": int(frames.max()),
                "non-finite values": int(store.stats["n_nonfinite"].sum()),
                "mean silent-frame %": 100 * float(store.stats["silent_fraction"].mean()),
            }
        )
        bench_summary = {
            "tracks": len(store.tracks),
            "clique_tracks": sum(1 for r in roles.values() if r == "clique"),
            "noise_tracks": sum(1 for r in roles.values() if r == "noise"),
            "raw_frames": describe(frames),
        }

    all_frames = np.array([r["raw_frames"] for r in track_rows], dtype=float)
    excluded = read_csv(manifests / "coveranalysis_excluded.csv")
    summary = {
        "manifest_info": info,
        "coveranalysis_selected_tracks": len(track_rows),
        "coveranalysis_selected_raw_frames": describe(all_frames),
        "n_excluded_wids": len(excluded),
        "excluded": excluded,
        "benchmark": bench_summary,
        "coveranalysis_metadata_wids": len(works),
    }
    out = results_dir(config)
    write_json(out / "dataset_profile.json", summary)
    write_csv(out / "dataset_profile_tracks.csv", track_rows, list(track_rows[0]))
    (out / "dataset_profile_table.md").write_text(
        markdown_table(role_rows, list(role_rows[0]), digits=2) + "\n"
    )

    # Figure: raw vs preprocessed HPCP for the first calibration work (both recordings).
    calib = load_split_tracks(manifests, "calibration")
    wid = calib[0].wid
    pair = [t for t in calib if t.wid == wid]
    store = split_store(config, calib, "calibration")
    root = features_root(config, "coveranalysis")
    raws = [to_pitch_by_time(load_hpcp(feature_path(root, t.wid, t.pid)).hpcp) for t in pair]
    mats = [
        raws[0],
        store.get("classical", [pair[0].pid])[0],
        raws[1],
        store.get("classical", [pair[1].pid])[0],
    ]
    titles = [
        f"{pair[0].pid} raw ({raws[0].shape[1]} frames)",
        f"{pair[0].pid} preprocessed (96)",
        f"{pair[1].pid} raw ({raws[1].shape[1]} frames)",
        f"{pair[1].pid} preprocessed (96)",
    ]
    plot_feature_matrices(mats, titles, figures_dir(config) / "profile_raw_vs_resampled.png")
    print(markdown_table(role_rows, list(role_rows[0]), digits=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
