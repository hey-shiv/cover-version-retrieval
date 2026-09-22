"""End-to-end smoke test on SYNTHETIC data: every script, in order, in a temp directory."""

import json
from pathlib import Path

import pytest
from scripts_helpers import run_script

from cover_retrieval.data.synthetic import make_synthetic_datacos


@pytest.mark.slow
def test_full_pipeline_on_synthetic_fixture(smoke_config):
    paths = smoke_config["paths"]
    make_synthetic_datacos(paths["datacos_root"], n_coveranalysis_works=100, seed=1)
    smoke_config["training"]["epochs"] = 2

    assert run_script("build_manifests", smoke_config) == 0
    assert run_script("profile_data", smoke_config, "--include-benchmark") == 0
    assert run_script("run_classical_baseline", smoke_config) == 0
    assert run_script("train_encoder", smoke_config) == 0
    assert run_script("run_hybrid_retrieval", smoke_config) == 0
    assert run_script("run_hubness_correction", smoke_config, "--n-probes", "8") == 0
    results_dir = Path(paths["reports_dir"]) / "results"
    assert (
        run_script(
            "run_benchmark",
            smoke_config,
            "--with-alignment-only",
            "--hub-correction",
            str(results_dir / "hubness_correction.json"),
            "--n-probes",
            "8",
        )
        == 0
    )

    results = results_dir
    classical = json.loads((results / "classical_dev.json").read_text())
    systems = {row["system"] for row in classical["metrics"]}
    assert {
        "profile_only",
        "classical_none",
        "classical_profile_cosine",
        "classical_exhaustive_dtw",
    } <= systems
    for row in classical["metrics"]:
        assert row["MAP"] == pytest.approx(row["MRR"])  # pair protocol: AP == RR

    hybrid = json.loads((results / "hybrid_dev.json").read_text())
    calibration = json.loads((results / "hybrid_calibration.json").read_text())
    assert hybrid["locked"]["alpha"] == calibration["alpha"]
    assert len(hybrid["metrics"]) == 5
    assert set(hybrid["runtime"]) >= {"global_embedding", "classical_alignment"}

    bench = json.loads((results / "benchmark.json").read_text())
    assert bench["protocol"]["queries"] == 12 * 4
    assert bench["protocol"]["candidates"] == 12 * 4 + 16
    assert {r["system"] for r in bench["metrics"]} >= {
        "global_embedding",
        "classical_alignment",
        "classical_alignment_hubcorr",
    }
    assert bench["locked_from_calibration"]["hub_correction"]["lambda"] >= 0.0
    hub = json.loads((results / "hubness_correction.json").read_text())
    assert hub["locked"]["method"] in {"mean", "topk"}
    for row in bench["metrics"]:
        assert 0.0 <= row["MAP"] <= 1.0
