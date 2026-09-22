"""Classical alignment-only baseline on the fixed development protocol.

    python scripts/run_classical_baseline.py --config configs/classical_baseline.yaml

Systems (same 20 queries, same 120-track pool, self excluded):
* ``profile_only``            – best-rotation cosine of time-averaged chroma (no temporal model)
* ``classical_none``          – subsequence DTW without key handling (ablation)
* ``classical_profile_cosine`` – the baseline: profile-selected rotation + subsequence DTW
* ``classical_exhaustive_dtw`` – DTW for all 12 rotations, keep the best (ablation)

Also writes calibration-pair diagnostics (two cover and two non-cover pairs).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cover_retrieval.alignment.transposition import batch_profile_rotation_scores, chroma_profile
from cover_retrieval.data.manifests import load_protocol
from cover_retrieval.evaluation.analysis import (
    bootstrap_delta_ci,
    plot_alignment_grid,
    plot_feature_matrices,
    plot_rotation_scores,
)
from cover_retrieval.pipeline import (
    METRIC_COLUMNS,
    figures_dir,
    markdown_table,
    metrics_row,
    pair_diagnostics,
    protocol_store,
    results_dir,
    run_classical,
    torch_threads,
    write_per_query,
)
from cover_retrieval.retrieval.rank import rank_protocol, ranking_rows
from cover_retrieval.utils.io import deep_merge, git_commit, load_config, write_csv, write_json


def calibration_diagnostics(config: dict, out_fig: Path) -> list[dict]:
    """Two cover pairs and two non-cover pairs from calibration WIDs only."""
    protocol = load_protocol(config["paths"]["manifests_dir"], "calibration")
    calib = [t for t in protocol.queries]
    store = protocol_store(config, protocol, "coveranalysis")
    by_wid: dict[str, list] = {}
    for t in calib:
        by_wid.setdefault(t.wid, []).append(t)
    wids = sorted(by_wid)
    covers = [(by_wid[w][0], by_wid[w][1]) for w in wids[:2]]
    noncovers = [(by_wid[wids[0]][0], by_wid[wids[2]][0]), (by_wid[wids[1]][0], by_wid[wids[3]][0])]
    rows, grid, rot_sets, rot_labels, mats, mat_titles = [], [], [], [], [], []
    for label, pairs in (("cover", covers), ("non-cover", noncovers)):
        for a, b in pairs:
            qa, qb = store.get("classical", [a.pid, b.pid])
            diag = pair_diagnostics(config, qa, qb)
            rows.append(
                {
                    "pair_type": label,
                    "query_pid": a.pid,
                    "candidate_pid": b.pid,
                    "best_shift": diag.shift,
                    "best_profile_cosine": float(diag.rotation_scores.max()),
                    "dtw_normalized_cost": diag.dtw.normalized_cost,
                    "alignment_score": 1.0 - diag.dtw.normalized_cost,
                }
            )
            title = f"{label}: {a.pid} vs {b.pid}\nshift {diag.shift}, cost {diag.dtw.normalized_cost:.3f}"
            grid.append((diag.cost, diag.dtw.path, title))
            rot_sets.append(diag.rotation_scores)
            rot_labels.append(f"{label}: {a.pid}/{b.pid}")
            mats += [qa, qb]
            mat_titles += [f"{label} query {a.pid}", f"{label} candidate {b.pid}"]
    plot_alignment_grid(grid, out_fig / "calibration_alignment_paths.png")
    plot_rotation_scores(rot_sets, rot_labels, out_fig / "calibration_rotation_scores.png")
    plot_feature_matrices(mats[:4], mat_titles[:4], out_fig / "calibration_cover_pairs_hpcp.png")
    plot_feature_matrices(mats[4:], mat_titles[4:], out_fig / "calibration_noncover_pairs_hpcp.png")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/classical_baseline.yaml"))
    parser.add_argument(
        "--resolution-sweep",
        type=int,
        nargs="*",
        default=[],
        help="extra n_frames values to analyse, e.g. 48 192 384",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    torch_threads(config)
    out, figs = results_dir(config), figures_dir(config)

    protocol = load_protocol(config["paths"]["manifests_dir"], "dev")
    store = protocol_store(config, protocol, "coveranalysis")

    runs = {}
    # Profile-only: no temporal alignment at all.
    t0 = time.perf_counter()
    q_profiles = chroma_profile(store.get("classical", [t.pid for t in protocol.queries]))
    c_profiles = chroma_profile(store.views["classical"])
    profile_scores = np.stack([batch_profile_rotation_scores(q, c_profiles).max(axis=1) for q in q_profiles])
    profile_seconds = time.perf_counter() - t0
    runs["profile_only"] = (rank_protocol(profile_scores, protocol), {"seconds": profile_seconds})

    classical = {}
    for selection in ("none", "profile_cosine", "exhaustive_dtw"):
        run = run_classical(config, protocol, store, rotation_selection=selection)
        classical[selection] = run
        runs[run.name] = (run.ranked, run.timings)

    baseline = classical["profile_cosine"]
    rows = [
        metrics_row(name, ranked, **{k: v for k, v in timings.items()})
        for name, (ranked, timings) in runs.items()
    ]
    groups = [q.wid for q in protocol.queries]
    base_ap = [r.metrics.ap for r in baseline.ranked]
    comparisons = {}
    for name, (ranked, _) in runs.items():
        if name != baseline.name:
            comparisons[f"{name} - {baseline.name}"] = bootstrap_delta_ci(
                base_ap, [r.metrics.ap for r in ranked], groups, seed=int(config["seed"])
            )

    diagnostics = calibration_diagnostics(config, figs)

    # Resolution sensitivity (analysis only: the declared default n_frames is not changed).
    sweep = []
    for n_frames in args.resolution_sweep:
        cfg = deep_merge(config, {"features": {"n_frames": n_frames}})
        for name in ("calibration", "dev"):
            proto = load_protocol(cfg["paths"]["manifests_dir"], name)
            run = run_classical(cfg, proto, protocol_store(cfg, proto, "coveranalysis"))
            row = metrics_row(f"{name}_n{n_frames}", run.ranked, ms_per_pair=run.timings["ms_per_pair"])
            sweep.append({"protocol": name, "n_frames": n_frames, **row})
    payload = {
        "protocol": {
            "name": "dev",
            "n_queries": len(protocol.queries),
            "n_candidates": len(protocol.candidates),
            "relevant_per_query": 1,
            "note": "pair protocol: AP == reciprocal rank by construction, so MAP == MRR",
        },
        "config": {k: config[k] for k in ("features", "alignment")},
        "metrics": rows,
        "bootstrap_vs_baseline": comparisons,
        "calibration_pair_diagnostics": diagnostics,
        "resolution_sweep_ANALYSIS_ONLY": sweep,
        "git_commit": git_commit(),
        "config_path": str(args.config),
    }
    write_json(out / "classical_dev.json", payload)
    write_per_query(out / "classical_dev_per_query.csv", protocol, {n: r for n, (r, _) in runs.items()})
    rank_rows = ranking_rows(
        baseline.ranked, protocol, baseline.scores, extra={"shift": baseline.shifts.astype(int)}
    )
    write_csv(out / "classical_dev_rankings.csv", rank_rows, list(rank_rows[0]))
    table = markdown_table(rows, METRIC_COLUMNS + ["ms_per_pair"])
    if sweep:
        table += "\n\nResolution sensitivity (analysis only; default stays at the configured n_frames):\n\n"
        table += markdown_table(
            sweep, ["protocol", "n_frames", "n_queries", "MAP", "Hit@10", "mean_first_rank", "ms_per_pair"]
        )
    (out / "classical_dev_table.md").write_text(table + "\n")
    print(table)
    for name, ci in comparisons.items():
        print(f"{name}: dAP {ci['delta']:+.3f} [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
