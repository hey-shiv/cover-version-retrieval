"""Calibrate and evaluate two-stage hybrid retrieval on the development protocol.

    python scripts/run_hybrid_retrieval.py --config configs/hybrid_dev.yaml

Steps
1. Calibration protocol (20 calibration-WID queries; pool = calibration + validation
   tracks): choose the blend weight alpha for shortlist size K. Locked to
   ``reports/results/hybrid_calibration.json``. No development/test query is used.
2. Development protocol (20 queries x 120 candidates), all with the locked alpha:
   global-only, classical alignment-only, hybrid, rerank-only (alpha = 0),
   a shortlist-size ablation, and test-time-rotation global retrieval.
3. Work-level bootstrap CIs, runtimes, error cases and alignment plots.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cover_retrieval.data.manifests import load_protocol
from cover_retrieval.evaluation.analysis import (
    bootstrap_delta_ci,
    plot_alignment_grid,
    plot_alpha_curve,
    select_error_cases,
)
from cover_retrieval.models.training import load_encoder
from cover_retrieval.pipeline import (
    METRIC_COLUMNS,
    figures_dir,
    markdown_table,
    metrics_row,
    pair_diagnostics,
    protocol_store,
    results_dir,
    run_classical,
    run_dir,
    run_global,
    run_hybrid_shortlists,
    torch_threads,
    write_per_query,
)
from cover_retrieval.retrieval.hybrid import calibrate_alpha
from cover_retrieval.retrieval.rank import ranking_rows
from cover_retrieval.utils.io import (
    deep_merge,
    git_commit,
    load_config,
    parse_overrides,
    write_csv,
    write_json,
)


def calibrate(config: dict, model, k: int) -> dict:
    protocol = load_protocol(config["paths"]["manifests_dir"], "calibration")
    store = protocol_store(config, protocol, "coveranalysis")
    global_run = run_global(config, model, protocol, store)
    hybrid = run_hybrid_shortlists(config, global_run, protocol, store, k)
    calibration = calibrate_alpha(hybrid, float(config["retrieval"]["alpha_grid_step"]))
    return {
        "alpha": calibration.alpha,
        "shortlist_k": k,
        "table": calibration.table,
        "protocol": {
            "queries": len(protocol.queries),
            "candidates": len(protocol.candidates),
            "query_wids": "calibration split only",
            "pool": "calibration + validation tracks (no development or benchmark WIDs)",
        },
        "global_only_map": metrics_row("g", global_run.ranked)["MAP"],
        "shortlist_recall": hybrid.shortlist_recall(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_dev.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value"
    )
    parser.add_argument("--tag", default="", help="suffix for output files, to keep variants side by side")
    args = parser.parse_args(argv)
    config = load_config(args.config, parse_overrides(args.set))
    torch_threads(config)
    out, figs = results_dir(config), figures_dir(config)
    tag = f"_{args.tag}" if args.tag else ""
    checkpoint = (
        args.checkpoint or run_dir(config, config["encoder"].get("run_name", "encoder")) / "encoder_best.pt"
    )
    model = load_encoder(checkpoint)
    k = int(config["retrieval"]["shortlist_k"])
    seed = int(config["seed"])

    # 1. calibration (locks alpha)
    calib = calibrate(config, model, k)
    write_json(
        out / f"hybrid_calibration{tag}.json",
        {**calib, "checkpoint": str(checkpoint), "git_commit": git_commit()},
    )
    plot_alpha_curve(
        calib["table"],
        calib["alpha"],
        figs / f"calibration_alpha_curve{tag}.png",
        f"Calibration MAP vs alpha (K={k})",
    )
    alpha = calib["alpha"]
    print(f"calibrated alpha = {alpha:.2f} (K = {k})")

    # 2. development evaluation
    protocol = load_protocol(config["paths"]["manifests_dir"], "dev")
    store = protocol_store(config, protocol, "coveranalysis")
    global_run = run_global(config, model, protocol, store)
    t0 = time.perf_counter()
    classical = run_classical(config, protocol, store)
    classical_seconds = time.perf_counter() - t0
    hybrid = run_hybrid_shortlists(config, global_run, protocol, store, k)
    hybrid_ranked = hybrid.rank(alpha)
    rerank_only = hybrid.rank(0.0)

    ttr_config = deep_merge(config, {"retrieval": {"test_time_rotations": True}})
    global_ttr = run_global(ttr_config, model, protocol, store)

    systems = {
        "global_embedding": global_run.ranked,
        "classical_alignment": classical.ranked,
        f"hybrid_K{k}_alpha{alpha:.2f}": hybrid_ranked,
        f"rerank_only_K{k}_alpha0.00": rerank_only,
        "global_embedding_test_time_rotations": global_ttr.ranked,
    }
    timing = {
        "global_embedding": {
            "embedding_s": global_run.timings["embedding_seconds"],
            "search_s": global_run.timings["global_search_seconds"],
        },
        "classical_alignment": {
            "alignment_s": classical_seconds,
            "ms_per_pair": classical.timings["ms_per_pair"],
        },
        f"hybrid_K{k}_alpha{alpha:.2f}": {
            "embedding_s": global_run.timings["embedding_seconds"],
            "search_s": global_run.timings["global_search_seconds"],
            "shortlist_s": hybrid.timings["shortlist_seconds"],
            "rerank_s": hybrid.timings["rerank_seconds"],
        },
    }
    rows = [metrics_row(name, ranked) for name, ranked in systems.items()]

    # shortlist-size ablation (analysis only; K and alpha stay locked for the headline result)
    ablation = []
    for k_ab in sorted({10, k, 50, 100, len(protocol.candidates) - 1}):
        run_ab = run_hybrid_shortlists(config, global_run, protocol, store, k_ab)
        for name, a in (("hybrid", alpha), ("rerank_only", 0.0)):
            row = metrics_row(
                f"{name}_K{k_ab}",
                run_ab.rank(a),
                K=k_ab,
                alpha=a,
                shortlist_recall=run_ab.shortlist_recall(),
                rerank_s=run_ab.timings["rerank_seconds"],
            )
            ablation.append(row)
    # post-hoc alpha sensitivity on dev (NOT used for any choice)
    sensitivity = [{"alpha": r["alpha"], "dev_map": hybrid.evaluate(r["alpha"]).map} for r in calib["table"]]

    groups = [q.wid for q in protocol.queries]
    ap = {name: [r.metrics.ap for r in ranked] for name, ranked in systems.items()}
    hybrid_name = f"hybrid_K{k}_alpha{alpha:.2f}"
    comparisons = {
        f"{hybrid_name} - global_embedding": bootstrap_delta_ci(
            ap["global_embedding"], ap[hybrid_name], groups, seed=seed
        ),
        f"{hybrid_name} - classical_alignment": bootstrap_delta_ci(
            ap["classical_alignment"], ap[hybrid_name], groups, seed=seed
        ),
        f"rerank_only_K{k}_alpha0.00 - global_embedding": bootstrap_delta_ci(
            ap["global_embedding"], ap[f"rerank_only_K{k}_alpha0.00"], groups, seed=seed
        ),
        "classical_alignment - global_embedding": bootstrap_delta_ci(
            ap["global_embedding"], ap["classical_alignment"], groups, seed=seed
        ),
    }

    # 3. error analysis on the hybrid system
    first_ranks = [r.metrics.first_rank for r in hybrid_ranked]
    cases = select_error_cases(first_ranks, n=3)
    case_rows = []
    align_scores = {}
    for s in hybrid.shortlists:
        for cand, score, shift in zip(s.shortlist, s.align_scores, s.align_shifts, strict=True):
            align_scores[(s.query_index, int(cand))] = (float(score), int(shift))
    for kind, indices in (
        ("success", cases.successes),
        ("false_positive", cases.false_positives),
        ("false_negative", cases.false_negatives),
    ):
        for qi in indices:
            ranked = hybrid_ranked[qi]
            query = protocol.queries[qi]
            rel_pos = int(np.flatnonzero(ranked.relevance)[0])
            rel_cand = int(ranked.order[rel_pos])
            fp_pos = int(np.flatnonzero(~ranked.relevance)[0])
            fp_cand = int(ranked.order[fp_pos])
            q_feat = store.get("classical", [query.pid])[0]
            rel_diag = pair_diagnostics(config, q_feat, store.views["classical"][rel_cand])
            fp_diag = pair_diagnostics(config, q_feat, store.views["classical"][fp_cand])
            case_rows.append(
                {
                    "case": kind,
                    "query_pid": query.pid,
                    "query_wid": query.wid,
                    "relevant_pid": protocol.candidates[rel_cand].pid,
                    "relevant_rank_hybrid": rel_pos + 1,
                    "relevant_rank_global": systems["global_embedding"][qi].metrics.first_rank,
                    "relevant_rank_classical": systems["classical_alignment"][qi].metrics.first_rank,
                    "relevant_global_cosine": float(global_run.scores[qi, rel_cand]),
                    "relevant_alignment_score": 1.0 - rel_diag.dtw.normalized_cost,
                    "relevant_shift": rel_diag.shift,
                    "relevant_in_shortlist": (qi, rel_cand) in align_scores,
                    "top_false_positive_pid": protocol.candidates[fp_cand].pid,
                    "top_false_positive_wid": protocol.candidates[fp_cand].wid,
                    "top_false_positive_rank": fp_pos + 1,
                    "top_false_positive_global_cosine": float(global_run.scores[qi, fp_cand]),
                    "top_false_positive_alignment_score": 1.0 - fp_diag.dtw.normalized_cost,
                    "top_false_positive_shift": fp_diag.shift,
                    "figure": f"figures/error_{kind}_{query.pid}{tag}.png",
                }
            )
            plot_alignment_grid(
                [
                    (
                        rel_diag.cost,
                        rel_diag.dtw.path,
                        f"query {query.pid} vs relevant {protocol.candidates[rel_cand].pid}\n"
                        f"rank {rel_pos + 1}, shift {rel_diag.shift}, cost {rel_diag.dtw.normalized_cost:.3f}",
                    ),
                    (
                        fp_diag.cost,
                        fp_diag.dtw.path,
                        f"query {query.pid} vs top non-cover {protocol.candidates[fp_cand].pid}\n"
                        f"rank {fp_pos + 1}, shift {fp_diag.shift}, cost {fp_diag.dtw.normalized_cost:.3f}",
                    ),
                ],
                figs / f"error_{kind}_{query.pid}{tag}.png",
            )

    payload = {
        "protocol": {
            "name": "dev",
            "n_queries": len(protocol.queries),
            "n_candidates": len(protocol.candidates),
            "relevant_per_query": 1,
            "note": "pair protocol: AP == RR, so MAP == MRR by construction",
        },
        "locked": {"alpha": alpha, "shortlist_k": k, "source": "hybrid_calibration.json"},
        "metrics": rows,
        "runtime": timing,
        "shortlist_recall": hybrid.shortlist_recall(),
        "shortlist_ablation": ablation,
        "posthoc_alpha_sensitivity_dev_NOT_USED_FOR_SELECTION": sensitivity,
        "bootstrap": comparisons,
        "error_cases": case_rows,
        "checkpoint": str(checkpoint),
        "git_commit": git_commit(),
        "config_path": str(args.config),
    }
    write_json(out / f"hybrid_dev{tag}.json", payload)
    write_per_query(out / f"hybrid_dev_per_query{tag}.csv", protocol, systems)
    rank_rows = ranking_rows(hybrid_ranked, protocol, global_run.scores, top_n=10)
    write_csv(out / f"hybrid_dev_rankings_top10{tag}.csv", rank_rows, list(rank_rows[0]))
    write_csv(out / f"error_cases{tag}.csv", case_rows, list(case_rows[0]))
    table = markdown_table(rows, METRIC_COLUMNS)
    ablation_table = markdown_table(
        ablation, ["system", "K", "alpha", "shortlist_recall", "MAP", "Hit@1", "Hit@10", "rerank_s"]
    )
    (out / f"hybrid_dev_table{tag}.md").write_text(table + "\n\n" + ablation_table + "\n")
    print(table)
    print(ablation_table)
    for name, ci in comparisons.items():
        print(f"{name}: dAP {ci['delta']:+.3f} [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
