#!/usr/bin/env bash
# End-to-end smoke pipeline on SYNTHETIC data (outputs under runs/smoke/, git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
rm -rf runs/smoke
"$PY" scripts/make_synthetic_datacos.py --config configs/smoke.yaml
"$PY" scripts/build_manifests.py --config configs/smoke.yaml
"$PY" scripts/profile_data.py --config configs/smoke.yaml --include-benchmark
"$PY" scripts/run_classical_baseline.py --config configs/smoke.yaml
"$PY" scripts/train_encoder.py --config configs/smoke.yaml
"$PY" scripts/run_hybrid_retrieval.py --config configs/smoke.yaml
"$PY" scripts/run_hubness_correction.py --config configs/smoke.yaml --n-probes 8
"$PY" scripts/run_benchmark.py --config configs/smoke.yaml --with-alignment-only \
  --hub-correction runs/smoke/reports/results/hubness_correction.json --n-probes 8
echo "SMOKE PIPELINE OK (synthetic data; numbers are not Da-TACOS results)"
