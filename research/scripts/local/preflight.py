"""Preflight for the LOCAL experiments: is everything needed here, and what will it cost?

    python research/scripts/local/preflight.py

Checks (no computation, a few seconds):
  * the benchmark HPCP features (or the preprocessed caches that make them unnecessary)
  * the three encoder checkpoints
  * the locked calibration / hubness files the runners read
  * the cached hubness probe reference (saves ~37 min at 384 frames if present)
  * free disk space
and prints the registry's runtime estimates. Exit 1 if a tier-1 input is missing.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OK, MISSING = "ok     ", "MISSING"


def exists(path: Path, label: str, required: bool, problems: list[str]) -> bool:
    found = path.exists()
    print(f"  {OK if found else MISSING}  {label}: {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    if not found and required:
        problems.append(label)
    return found


def main() -> int:
    problems: list[str] = []
    print("Da-TACOS data")
    feat = REPO / "data/external/da-tacos/da-tacos_benchmark_subset_hpcp"
    cache = REPO / "data/cache"
    has_feat = feat.exists() and any(feat.iterdir())
    caches = sorted(cache.glob("benchmark_benchmark_*.npz")) if cache.exists() else []
    print(f"  {OK if has_feat else MISSING}  benchmark HPCP directory ({sum(1 for _ in feat.glob('*/*.h5')) if has_feat else 0} files; 15,000 expected)")
    print(f"  {OK if caches else MISSING}  preprocessed benchmark caches: {len(caches)} (one per resolution / cache setting)")
    if not has_feat and not caches:
        problems.append("benchmark features (neither raw HPCP nor caches)")
    exists(REPO / "data/external/da-tacos/da-tacos_coveranalysis_subset_hpcp", "Cover Analysis HPCP (calibration + probes, F1 s1_hub)", False, problems)

    print("checkpoints")
    exists(REPO / "runs/encoder_full_long/encoder_best.pt", "encoder_full_long (tier 1)", True, problems)
    exists(REPO / "runs/encoder_full_train/encoder_best.pt", "encoder_full_train (F1 training scale)", False, problems)
    exists(REPO / "runs/sweep_base/encoder_best.pt", "sweep_base (F1 training scale)", False, problems)

    print("locked settings (committed)")
    exists(REPO / "reports/results/hybrid_calibration_long384.json", "alpha / K / checkpoint for run 4", True, problems)
    exists(REPO / "reports/results/hubness_correction_n384.json", "hub lambda at 384 frames", True, problems)
    exists(REPO / "reports/results/benchmark_long384.json", "frozen run-4 result (reproduction target)", True, problems)

    print("reusable intermediate artifacts")
    probe = REPO / "runs/probe_reference/probe200_n384_benchmark.npy"
    exists(probe, "probe reference, 384 frames (saves ~37 min if present)", False, problems)

    free_gb = shutil.disk_usage(REPO).free / 1e9
    print(f"disk: {free_gb:.0f} GB free (A1 block cache needs ~0.1 GB; results < 20 MB)")

    print("\nestimated cost (from runtimes measured in benchmark_long384.json; your machine may differ)")
    print("  A1 k-sweep, K_max = 500, 384 frames : ~2.3 h (+ ~37 min if the probe cache is missing)")
    print("  F1 Stage-1 variants                 : ~15-30 min")
    print("  H1 ablation, none + profile         : ~10 min;  + exhaustive_dtw: +~1.9 h")
    if problems:
        print("\nBLOCKED - missing tier-1 inputs:\n  - " + "\n  - ".join(problems))
        return 1
    print("\nREADY for tier 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
