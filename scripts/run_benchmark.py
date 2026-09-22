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
from dataclasses import replace
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
    split_store,  # noqa: F401  (used by probe_reference_scores)
    torch_threads,
    write_per_query,
)
from cover_retrieval.retrieval.hybrid import HybridRun
from cover_retrieval.retrieval.normalization import apply_hub_correction, hub_reference
from cover_retrieval.retrieval.rank import rank_protocol
from cover_retrieval.utils.io import git_commit, load_config, parse_overrides, read_json, write_json


def probe_reference_scores(
    config: dict, candidate_features: np.ndarray, n_probes: int, cache_key: str = ""
) -> np.ndarray:
    """``(P, C)`` alignment scores of fixed training-split probe queries (see D-018).

    The probe set and the candidates are fixed by the manifests, so the result depends
    only on the resolution; it is cached to avoid recomputing 3M alignments per run.
    """
    from cover_retrieval.data.manifests import load_split_tracks

    n_frames = int(config["features"]["n_frames"])
    cache = run_dir(config, "probe_reference") / f"probe{n_probes}_n{n_frames}_{cache_key}.npy"
    if cache.exists():
        cached = np.load(cache)
        if cached.shape == (n_probes, candidate_features.shape[0]):
            print(f"[probe] reusing {cache}", flush=True)
            return cached
    probes = load_split_tracks(config["paths"]["manifests_dir"], "train")[:n_probes]
    probe_store = split_store(config, probes, f"probe{n_probes}")
    scores, _ = aligner_from_config(config).score_matrix(probe_store.views["classical"], candidate_features)
    np.save(cache, scores)
    return scores


def alignment_only_scores(config: dict, protocol, store, block: int = 500) -> tuple[np.ndarray, float]:
    """Full query x candidate alignment, cached per resolution in blocks of 500 queries."""
    aligner = aligner_from_config(config)
    work = run_dir(config, f"benchmark_alignment_n{int(config['features']['n_frames'])}")
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
        "--hub-correction",
        type=Path,
        default=None,
        help="hubness_correction*.json with the method/k/lambda locked on calibration works",
    )
    parser.add_argument("--n-probes", type=int, default=200)
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
    k_shortlist = k
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
    alignment_scores = None
    if args.with_alignment_only:
        alignment_scores, seconds = alignment_only_scores(config, protocol, store)
        systems["classical_alignment"] = rank_protocol(alignment_scores, protocol)
        runtime["alignment_only_s"] = seconds
        runtime["alignment_only_ms_per_pair"] = (
            1e3 * seconds / (len(protocol.queries) * (len(protocol.candidates) - 1))
        )

    # ---- optional hubness correction (settings locked on calibration works) -------
    if args.hub_correction:
        locked = read_json(args.hub_correction)["locked"]
        lam, method, ref_k = float(locked["lambda"]), locked["method"], int(locked["k"])
        t0 = time.perf_counter()
        probe_scores = probe_reference_scores(config, store.views["classical"], args.n_probes, "benchmark")
        reference = hub_reference(probe_scores, method, ref_k)
        runtime["hub_reference_s"] = time.perf_counter() - t0
        runtime["hub_reference_pairs"] = int(probe_scores.size)
        payload_hub = {
            "lambda": lam,
            "method": method,
            "k": ref_k,
            "n_probes": args.n_probes,
            "source": str(args.hub_correction),
        }
        corrected = HybridRun(
            [
                replace(s_, align_scores=s_.align_scores - lam * reference[s_.shortlist])
                for s_ in hybrid.shortlists
            ],
            hybrid.k,
            hybrid.timings,
        )
        systems[f"hybrid_hubcorr_K{k_shortlist}_alpha{alpha:.2f}"] = corrected.rank(alpha)
        if alignment_scores is not None:
            systems["classical_alignment_hubcorr"] = rank_protocol(
                apply_hub_correction(alignment_scores, reference, lam), protocol
            )
    else:
        payload_hub = None

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
        "locked_from_calibration": {
            "alpha": alpha,
            "shortlist_k": k,
            "checkpoint": str(checkpoint),
            "n_frames": config["features"]["n_frames"],
            "hub_correction": payload_hub,
        },
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
