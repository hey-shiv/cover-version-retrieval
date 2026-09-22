"""Hubness analysis on the development protocol (analysis only; changes no setting).

    python scripts/analyze_hubness.py --config configs/hybrid_dev.yaml

Relates each candidate's tonal dispersion to how often it is a top-10 false positive
in the classical and hybrid rankings produced by the earlier scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cover_retrieval.data.manifests import load_protocol
from cover_retrieval.evaluation.analysis import hub_counts, tonal_dispersion
from cover_retrieval.pipeline import protocol_store, results_dir
from cover_retrieval.utils.io import load_config, read_csv, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_dev.yaml"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = results_dir(config)
    protocol = load_protocol(config["paths"]["manifests_dir"], "dev")
    store = protocol_store(config, protocol, "coveranalysis")
    dispersion = tonal_dispersion(store.views["classical"])
    n_queries = len(protocol.queries)
    expected = n_queries * 10 / (len(protocol.candidates) - 1)
    payload = {
        "n_candidates": len(store.pids),
        "expected_top10_fp_count_if_uniform": expected,
        "dispersion_median": float(np.median(dispersion)),
    }
    for name, file in (
        ("classical", "classical_dev_rankings.csv"),
        ("hybrid", "hybrid_dev_rankings_top10.csv"),
    ):
        counts = hub_counts(read_csv(out / file))
        hubs = np.array([counts.get(pid, 0) for pid in store.pids])
        rho, p_value = spearmanr(dispersion, hubs)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        payload[name] = {
            "spearman_dispersion_vs_top10_fp": float(rho),
            "p_value": float(p_value),
            "max_top10_fp_count": int(hubs.max()),
            "top_hubs": [
                {"pid": pid, "top10_fp_count": c, "dispersion": float(dispersion[store.index([pid])[0]])}
                for pid, c in top
            ],
        }
        print(f"{name}: spearman(dispersion, hub count) = {rho:.3f} (p = {p_value:.2g}); top hubs {top[:3]}")
    write_json(out / "hubness_dev.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
