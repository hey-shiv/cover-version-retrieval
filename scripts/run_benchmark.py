"""FINAL Da-TACOS benchmark evaluation (13,000 queries x 15,000 candidates).

    python scripts/run_benchmark.py --config configs/benchmark.yaml [--with-alignment-only]

* Every one of the 13,000 clique tracks is a query; all 15,000 tracks (including the
  2,000 noise/distractor tracks) are candidates; the query itself is excluded.
* Nothing is tuned here: the encoder checkpoint, alpha and K are read from files
  produced on training/validation/calibration data before this script runs.
* Tracks are never dropped: non-finite HPCP values would be zeroed and counted.

Alignment-only retrieval over the full benchmark is ~195M DTW alignments; it is
optional and resumable (blocks of 500 queries are saved under runs/benchmark/).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cover_retrieval.data.manifests import load_protocol
from cover_retrieval.evaluation.analysis import bootstrap_delta_ci
from cover_retrieval.models.training import load_encoder
from cover_retrieval.pipeline import (
    METRIC_COLUMNS,
    aligner_from_config,
    markdown_table,
    metrics_row,
    protocol_store,
    results_dir,
    run_dir,
    run_global,
    run_hybrid_shortlists,
    torch_threads,
    write_per_query,
)
from cover_retrieval.retrieval.rank import rank_protocol
from cover_retrieval.utils.io import git_commit, load_config, parse_overrides, read_json, write_json


def alignment_only_scores(config: dict, protocol, store, block: int = 500) -> tuple[np.ndarray, float]:
    aligner = aligner_from_config(config)
    work = run_dir(config, "benchmark_alignment")
    queries = store.get("classical", [t.pid for t in protocol.queries])
    n = queries.shape[0]
    seconds = 0.0
    parts = []
    for start in range(0, n, block):
        path = work / f"scores_{start:05d}.npy"
        timing = work / f"seconds_{start:05d}.txt"
        if not path.exists():
            t0 = time.perf_counter()
            scores, _ = aligner.score_matrix(queries[start : start + block], store.views["classical"])
            np.save(path, scores)
            timing.write_text(str(time.perf_counter() - t0))
            print(f"  alignment-only: {min(start + block, n)}/{n} queries", flush=True)
        seconds += float(timing.read_text())
        parts.append(np.load(path))
    return np.concatenate(parts), seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--with-alignment-only", action="store_true")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value"
    )
    parser.add_argument("--tag", default="", help="suffix for output files, to keep variants side by side")
    args = parser.parse_args(argv)
    config = load_config(args.config, parse_overrides(args.set))
    torch_threads(config)
    out = results_dir(config)
    tag = f"_{args.tag}" if args.tag else ""

    calibration_path = out / f"hybrid_calibration{tag}.json"
    if not calibration_path.exists():
        raise SystemExit(
            "run scripts/run_hybrid_retrieval.py first: alpha/K must be locked before the benchmark"
        )
    locked = read_json(calibration_path)
    alpha, k = float(locked["alpha"]), int(locked["shortlist_k"])
    checkpoint = args.checkpoint or Path(locked["checkpoint"])
    model = load_encoder(checkpoint)

    protocol = load_protocol(config["paths"]["manifests_dir"], "benchmark")
    t0 = time.perf_counter()
    store = protocol_store(config, protocol, "benchmark", nonfinite_policy="zero")
    load_seconds = time.perf_counter() - t0
    print(f"benchmark: {len(protocol.queries)} queries x {len(protocol.candidates)} candidates", flush=True)

    global_run = run_global(config, model, protocol, store)
    print(f"global: MAP {metrics_row('g', global_run.ranked)['MAP']:.4f}", flush=True)
    hybrid = run_hybrid_shortlists(config, global_run, protocol, store, k)
    systems = {
        "global_embedding": global_run.ranked,
        f"hybrid_K{k}_alpha{alpha:.2f}": hybrid.rank(alpha),
        f"rerank_only_K{k}_alpha0.00": hybrid.rank(0.0),
    }
    runtime = {
        "feature_loading_s": load_seconds,
        "embedding_s": global_run.timings["embedding_seconds"],
        "global_search_s": global_run.timings["global_search_seconds"],
        "shortlist_s": hybrid.timings["shortlist_seconds"],
        "rerank_s": hybrid.timings["rerank_seconds"],
        "rerank_ms_per_query": 1e3 * hybrid.timings["rerank_seconds"] / len(protocol.queries),
    }
    if args.with_alignment_only:
        scores, seconds = alignment_only_scores(config, protocol, store)
        systems["classical_alignment"] = rank_protocol(scores, protocol)
        runtime["alignment_only_s"] = seconds
        runtime["alignment_only_ms_per_pair"] = (
            1e3 * seconds / (len(protocol.queries) * (len(protocol.candidates) - 1))
        )

    rows = [metrics_row(name, ranked) for name, ranked in systems.items()]
    groups = [q.wid for q in protocol.queries]
    ap = {name: [r.metrics.ap for r in ranked] for name, ranked in systems.items()}
    comparisons = {
        f"{name} - global_embedding": bootstrap_delta_ci(
            ap["global_embedding"], ap[name], groups, seed=int(config["seed"])
        )
        for name in systems
        if name != "global_embedding"
    }
    payload = {
        "label": "FINAL Da-TACOS benchmark evaluation (no tuning on this set)",
        "protocol": {
            "queries": len(protocol.queries),
            "candidates": len(protocol.candidates),
            "noise_tracks": sum(1 for t in protocol.candidates) - len(protocol.queries),
            "self_excluded": True,
        },
        "locked_from_calibration": {"alpha": alpha, "shortlist_k": k, "checkpoint": str(checkpoint)},
        "nonfinite_values_zeroed": int(store.stats["n_nonfinite"].sum()),
        "metrics": rows,
        "shortlist_recall": hybrid.shortlist_recall(),
        "runtime": runtime,
        "bootstrap_vs_global": comparisons,
        "git_commit": git_commit(),
        "config_path": str(args.config),
    }
    write_json(out / f"benchmark{tag}.json", payload)
    write_per_query(out / f"benchmark_per_query{tag}.csv", protocol, systems)
    table = markdown_table(rows, METRIC_COLUMNS)
    (out / f"benchmark_table{tag}.md").write_text(table + "\n")
    print(table)
    for name, ci in comparisons.items():
        print(f"{name}: dAP {ci['delta']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
