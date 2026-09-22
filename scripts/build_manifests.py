"""Build deterministic, WID-disjoint manifests for Cover Analysis and the Benchmark.

    python scripts/build_manifests.py --config configs/base.yaml

Reads only WID/PID identifiers from the metadata. Each WID along the seeded order
is checked on disk: exactly two PIDs, both HPCP files present, readable, labelled
consistently, finite, non-silent and at least ``data.min_frames`` long.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cover_retrieval.data.datacos import (
    DataValidationError,
    feature_path,
    features_root,
    load_metadata,
    load_validated_hpcp,
    metadata_path,
    work_index,
)
from cover_retrieval.data.manifests import write_benchmark_manifest, write_coveranalysis_manifests
from cover_retrieval.data.splits import assign_roles, check_disjoint, role_sizes, seeded_wid_order
from cover_retrieval.utils.io import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out_dir = Path(config["paths"]["manifests_dir"])
    data_cfg = config["data"]

    works = work_index(load_metadata(metadata_path(config, "coveranalysis")))
    root = features_root(config, "coveranalysis")
    seed = int(config["splits"]["seed"])

    def check_wid(wid: str) -> str | None:
        pids = works[wid]
        if len(pids) != 2:
            return f"has {len(pids)} PIDs (need exactly 2)"
        if not all(feature_path(root, wid, pid).exists() for pid in pids):
            return "unavailable: HPCP file not downloaded"
        for pid in pids:
            try:
                load_validated_hpcp(
                    feature_path(root, wid, pid),
                    wid,
                    pid,
                    min_frames=int(data_cfg["min_frames"]),
                    nonfinite_policy="reject",
                    layout=config["features"]["layout"],
                )
            except (DataValidationError, FileNotFoundError) as error:
                return f"{pid}: {error}"
        return None

    order = seeded_wid_order(list(works), seed)
    assignment = assign_roles(order, role_sizes(config["splits"]), check_wid)
    check_disjoint(assignment)
    info = {
        "seed": seed,
        "subset": "coveranalysis",
        "n_metadata_wids": len(works),
        "split_sizes": {k: v for k, v in role_sizes(config["splits"]).items()},
        "validation_rules": {
            "pids_per_wid": 2,
            "nonfinite_policy": "reject",
            "min_frames": int(data_cfg["min_frames"]),
            "labels_must_match_manifest": True,
        },
        "selection": "lexicographic sort -> numpy default_rng(seed).permutation -> skip invalid -> "
        "fill calibration, query, distractor, validation, train",
        "config": str(args.config),
    }
    write_coveranalysis_manifests(out_dir, assignment, works, info)

    bench_meta = metadata_path(config, "benchmark")
    if bench_meta.exists():
        write_benchmark_manifest(out_dir, work_index(load_metadata(bench_meta)))

    for role, wids in assignment.roles.items():
        print(f"{role:12s} {len(wids):5d} WIDs")
    print(f"excluded     {len(assignment.excluded):5d} WIDs")
    print(f"manifests written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
