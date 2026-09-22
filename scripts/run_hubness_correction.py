"""Experiment: hubness correction of the classical alignment score.

    python scripts/run_hubness_correction.py --config configs/hybrid_dev.yaml

Motivation (`reports/error_analysis.md`): tonally static tracks get uniformly low
DTW cost and crowd the head of many rankings. Tonal dispersion predicts top-10
false-positive count at Spearman -0.77 on the development pool.

Correction: ``corrected[q, c] = score[q, c] - lam * reference[c]``, where
``reference[c]`` is candidate ``c``'s general score level, estimated **inductively**
against a fixed probe set of training-split tracks (never the query set, never
labels). The same probe definition is therefore usable at any protocol size.

Protocol discipline: ``lam`` and the reference method are chosen on the
**calibration** protocol only, then applied unchanged to the development protocol.
The benchmark is evaluated separately, once, by ``scripts/run_benchmark.py``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cover_retrieval.data.manifests import load_protocol, load_split_tracks
from cover_retrieval.evaluation.analysis import bootstrap_delta_ci, tonal_dispersion
from cover_retrieval.pipeline import (
    METRIC_COLUMNS,
    aligner_from_config,
    markdown_table,
    metrics_row,
    protocol_store,
    results_dir,
    split_store,
    torch_threads,
)
from cover_retrieval.retrieval.normalization import apply_hub_correction, hub_reference
from cover_retrieval.retrieval.rank import rank_protocol
from cover_retrieval.utils.io import git_commit, load_config, parse_overrides, write_json

LAMBDA_GRID = [round(0.1 * i, 2) for i in range(0, 16)]
METHODS = [("topk", 10), ("topk", 50), ("mean", 0)]


def probe_reference_scores(config: dict, candidate_features: np.ndarray, n_probes: int) -> np.ndarray:
    """``(P, C)`` alignment scores of probe queries against the candidates.

    Probes are the first ``n_probes`` training-split tracks by PID: disjoint from
    calibration, development and benchmark data, and fixed by the manifests.
    """
    probes = load_split_tracks(config["paths"]["manifests_dir"], "train")[:n_probes]
    store = split_store(config, probes, f"probe{n_probes}")
    aligner = aligner_from_config(config)
    scores, _ = aligner.score_matrix(store.views["classical"], candidate_features)
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_dev.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--n-probes", type=int, default=200)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    config = load_config(args.config, parse_overrides(args.set))
    torch_threads(config)
    out = results_dir(config)
    tag = f"_{args.tag}" if args.tag else ""

    results: dict[str, object] = {
        "experiment": "hubness correction of the classical alignment score",
        "reference": f"inductive: {args.n_probes} training-split probe queries",
        "selection": "lambda and method chosen on the calibration protocol only",
        "config": str(args.config),
        "overrides": args.set,
        "n_frames": config["features"]["n_frames"],
        "git_commit": git_commit(),
    }

    # ---------------------------------------------------------------- calibration
    protocol = load_protocol(config["paths"]["manifests_dir"], "calibration")
    store = protocol_store(config, protocol, "coveranalysis")
    aligner = aligner_from_config(config)
    queries = store.get("classical", [t.pid for t in protocol.queries])
    t0 = time.perf_counter()
    scores, _ = aligner.score_matrix(queries, store.views["classical"])
    probe_scores = probe_reference_scores(config, store.views["classical"], args.n_probes)
    results["calibration_seconds"] = time.perf_counter() - t0

    table = []
    for method, k in METHODS:
        reference = hub_reference(probe_scores, method, k)  # type: ignore[arg-type]
        for lam in LAMBDA_GRID:
            summary = metrics_row(
                f"{method}{k or ''}_lam{lam}",
                rank_protocol(apply_hub_correction(scores, reference, lam), protocol),
            )
            table.append(
                {"method": method, "k": k, "lambda": lam, "MAP": summary["MAP"], "Hit@10": summary["Hit@10"]}
            )
    best = max(table, key=lambda r: (round(r["MAP"], 12), -abs(r["lambda"])))
    results["calibration_grid"] = table
    results["locked"] = {"method": best["method"], "k": best["k"], "lambda": best["lambda"]}
    print(
        f"calibration: best {best['method']}{best['k'] or ''} lambda={best['lambda']} MAP={best['MAP']:.4f} "
        f"(uncorrected {table[0]['MAP']:.4f})"
    )

    # ---------------------------------------------------------------- development
    dev = load_protocol(config["paths"]["manifests_dir"], "dev")
    dev_store = protocol_store(config, dev, "coveranalysis")
    dev_queries = dev_store.get("classical", [t.pid for t in dev.queries])
    dev_scores, _ = aligner.score_matrix(dev_queries, dev_store.views["classical"])
    dev_probe = probe_reference_scores(config, dev_store.views["classical"], args.n_probes)
    reference = hub_reference(dev_probe, best["method"], best["k"])  # type: ignore[arg-type]
    corrected = apply_hub_correction(dev_scores, reference, best["lambda"])

    runs = {
        "classical_alignment": rank_protocol(dev_scores, dev),
        f"classical_hub_corrected_lam{best['lambda']}": rank_protocol(corrected, dev),
    }
    rows = [metrics_row(name, ranked) for name, ranked in runs.items()]
    groups = [q.wid for q in dev.queries]
    ci = bootstrap_delta_ci(
        [r.metrics.ap for r in runs["classical_alignment"]],
        [r.metrics.ap for r in runs[f"classical_hub_corrected_lam{best['lambda']}"]],
        groups,
        seed=int(config["seed"]),
    )
    dispersion = tonal_dispersion(dev_store.views["classical"])
    results["dev_metrics"] = rows
    results["dev_bootstrap_corrected_minus_uncorrected"] = ci
    results["dev_reference_vs_dispersion_spearman"] = float(spearmanr(dispersion, reference).statistic)
    write_json(out / f"hubness_correction{tag}.json", results)
    print(markdown_table(rows, METRIC_COLUMNS))
    print(f"dev dAP {ci['delta']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
    print(f"reference vs tonal dispersion: spearman {results['dev_reference_vs_dispersion_spearman']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
