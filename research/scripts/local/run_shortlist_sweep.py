"""Series A/B/C/D(signals)/E(inputs): shortlist-size sweep on the Da-TACOS benchmark.

LOCAL COMPUTE ONLY (needs Da-TACOS HPCP features and the encoder checkpoint).

One pass does everything that depends on the data:

1. Stage 1: embed all candidates with the locked checkpoint, exact cosine scores.
2. For every query, align the first K_max Stage-1 candidates ONCE (the expensive part,
   resumable in blocks under runs/research/, which is git-ignored).
3. For each K in --ks, rerank the first K exactly as HybridRun.rank does, for:
     hyb  alpha locked on calibration works (from hybrid_calibration{tag}.json)
     rr   alpha = 0 (rerank only; needs no tuning)
     hub  alignment - lambda * probe reference, alpha locked (hubness_correction json)
   Nothing is tuned here: alpha, lambda and the checkpoint come from files written
   before this script runs. K itself is the independent variable, not a tuned setting.
4. Label-free Stage-1 difficulty signals per query (for series D/E, analysed in the cloud).
5. Self-check: at the locked K, metrics must reproduce the frozen benchmark result file.

Outputs (small, meant to be committed): metrics.json, per_query.csv, signals.csv,
candidates.csv, runtime.json, config.yaml, env.json, README.md.

Example (run 4 configuration):
    python research/scripts/local/run_shortlist_sweep.py --config configs/full_train.yaml --tag long384 \
        --set features.n_frames=384 --hub-correction reports/results/hubness_correction_n384.json \
        --frozen reports/results/benchmark_long384.json --out research/results/A1_k_sweep_long384
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rrlib  # noqa: E402

from cover_retrieval.data.manifests import load_protocol  # noqa: E402
from cover_retrieval.evaluation.metrics import ranking_from_scores  # noqa: E402
from cover_retrieval.models.training import load_encoder  # noqa: E402
from cover_retrieval.pipeline import aligner_from_config, global_scores, protocol_store, results_dir, run_dir, torch_threads  # noqa: E402
from cover_retrieval.retrieval.normalization import hub_reference  # noqa: E402
from cover_retrieval.utils.io import load_config, parse_overrides, read_json  # noqa: E402

EXPERIMENT_ID = "A1"  # "A2" when --stage1 win_fuse
DEFAULT_KS = [5, 10, 20, 30, 50, 100, 200, 500]


def load_probe_fn():
    """Reuse probe_reference_scores from scripts/run_benchmark.py (and its cache)."""
    path = rrlib.REPO / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.probe_reference_scores


def align_heads(config, store, query_pids, orders, k_max, block, cache_dir):
    """(Q, k_max) alignment scores of each query's first k_max Stage-1 candidates, resumable."""
    aligner = aligner_from_config(config)
    feats = store.views["classical"]
    q_feats = store.get("classical", query_pids)
    n = len(query_pids)
    scores = np.zeros((n, k_max), dtype=np.float64)
    seconds = 0.0
    cache_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, n, block):
        stop = min(start + block, n)
        path = cache_dir / f"align_top{k_max}_{start:05d}.npz"
        heads = orders[start:stop, :k_max]
        if path.exists():
            z = np.load(path)
            if z["heads"].shape == heads.shape and np.array_equal(z["heads"], heads):
                scores[start:stop] = z["scores"]
                seconds += float(z["seconds"])
                continue
        t0 = time.perf_counter()
        for i in range(start, stop):
            scores[i] = aligner.score_pairs(q_feats[i], feats[orders[i, :k_max]]).score
        dt = time.perf_counter() - t0
        np.savez(path, scores=scores[start:stop], heads=heads, seconds=dt)
        seconds += dt
        print(f"  aligned {stop}/{n} queries x {k_max} ({dt:.1f}s this block)", flush=True)
    return scores, seconds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tag", default="", help="suffix of the locked hybrid_calibration{_tag}.json")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--checkpoint", type=Path, default=None, help="default: the one recorded in the calibration file")
    p.add_argument("--hub-correction", type=Path, required=True)
    p.add_argument("--n-probes", type=int, default=200)
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--frozen", type=Path, default=None, help="benchmark result JSON to reproduce at the locked K")
    p.add_argument("--stage1", choices=["global", "win_fuse"], default="global",
                   help="Stage-1 scorer: global embedding (A1) or win_fuse, as defined in run_stage1_variants.py (A2)")
    p.add_argument("--block", type=int, default=500)
    p.add_argument("--max-queries", type=int, default=0, help="sanity check on the first N queries only (0 = all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    config = load_config(args.config, parse_overrides(args.set))
    torch_threads(config)
    label = rrlib.evidence_label(config)
    if args.max_queries:
        label += "-SANITY"
    out = rrlib.prepare_out(args.out, args.force)
    tag = f"_{args.tag}" if args.tag else ""
    locked = read_json(results_dir(config) / f"hybrid_calibration{tag}.json")
    alpha, k_locked = float(locked["alpha"]), int(locked["shortlist_k"])
    hub_locked = read_json(args.hub_correction)["locked"]
    lam = float(hub_locked["lambda"])
    checkpoint = args.checkpoint or Path(locked["checkpoint"])
    ks = sorted(set(args.ks))
    k_max = ks[-1]
    exp_id = "A2" if args.stage1 == "win_fuse" else EXPERIMENT_ID
    rrlib.write_meta(out, exp_id, config, {**vars(args), "alpha": alpha, "lambda": lam, "checkpoint": str(checkpoint)}, label)

    # ---- data + Stage 1 ------------------------------------------------------------
    protocol = load_protocol(config["paths"]["manifests_dir"], "benchmark")
    t0 = time.perf_counter()
    store = protocol_store(config, protocol, "benchmark", nonfinite_policy="zero")
    runtime = {"feature_loading_s": time.perf_counter() - t0}
    model = load_encoder(checkpoint)
    scores, timings = global_scores(config, model, protocol, store)
    runtime.update({"embedding_s": timings["embedding_seconds"], "global_search_s": timings["global_search_seconds"]})
    if args.stage1 == "win_fuse":
        # identical definition to F1 (run_stage1_variants.py): 3 windows of 60% at offsets 0/20/40%,
        # MaxSim over the 3 x 3 window pairs, fused 50/50 with the global cosine
        from run_stage1_variants import windows  # noqa: E402

        from cover_retrieval.models.training import embed  # noqa: E402

        t0 = time.perf_counter()
        wins = [embed(model, w) for w in windows(store.views["encoder"])]
        q_idx = store.index([t.pid for t in protocol.queries])
        for start in range(0, len(q_idx), 500):
            sl = q_idx[start : start + 500]
            best = None
            for wq in wins:
                for wc in wins:
                    s_ = wq[sl] @ wc.T
                    best = s_ if best is None else np.maximum(best, s_)
            scores[start : start + len(sl)] = 0.5 * scores[start : start + len(sl)] + 0.5 * best
        runtime["win_fuse_s"] = time.perf_counter() - t0
        if args.frozen:
            print("[A2] --frozen ignored: the frozen run used the global Stage 1")
            args.frozen = None
    relevance = protocol.relevance_matrix()
    self_mask = protocol.self_mask()
    n_q = len(protocol.queries) if not args.max_queries else min(args.max_queries, len(protocol.queries))
    k_max = min(k_max, len(protocol.candidates) - 1)
    ks = [k for k in ks if k <= k_max]

    # keep only what the metrics need: the first k_max Stage-1 positions and the relevant ranks
    t0 = time.perf_counter()
    orders = np.zeros((n_q, k_max), dtype=np.int64)
    s1_rel_ranks, head_rels = [], []
    for i in range(n_q):
        order, rel = ranking_from_scores(scores[i], relevance[i], exclude=self_mask[i])
        orders[i] = order[:k_max]
        s1_rel_ranks.append(np.flatnonzero(rel) + 1)
        head_rels.append(rel[:k_max].copy())
    runtime["stage1_ranking_s"] = time.perf_counter() - t0
    kocc = rrlib.k_occurrence(scores, self_mask, k=10)

    # ---- hubness reference (cached by run_benchmark.py under runs/probe_reference) ----
    t0 = time.perf_counter()
    probe_scores = load_probe_fn()(config, store.views["classical"], args.n_probes, "benchmark")
    reference = hub_reference(probe_scores, hub_locked["method"], int(hub_locked["k"]))
    runtime["hub_reference_s"] = time.perf_counter() - t0

    # ---- the expensive part: align each query's first k_max candidates once ---------
    q_pids = [t.pid for t in protocol.queries[:n_q]]
    # cache key: resolution + every alignment setting + checkpoint; sanity runs never share the full cache
    import hashlib
    import json as _json

    key = hashlib.sha1(_json.dumps({"alignment": config["alignment"], "n_frames": config["features"]["n_frames"],
                                    "checkpoint": str(checkpoint), "stage1": args.stage1}, sort_keys=True).encode()).hexdigest()[:10]
    cache = run_dir(config, "research") / f"shortlist_sweep{tag}_n{config['features']['n_frames']}_{key}{'_sanity' + str(n_q) if args.max_queries else ''}"
    align, align_s = align_heads(config, store, q_pids, orders, k_max, args.block, cache)
    runtime["align_top_kmax_s"] = align_s
    runtime["align_pairs"] = int(n_q * k_max)
    runtime["ms_per_aligned_pair"] = 1e3 * align_s / max(1, n_q * k_max)

    # ---- per-K reranking (cheap) -----------------------------------------------------
    cand_pids = [t.pid for t in protocol.candidates]
    rows, sig_rows = [], []
    agg = {f"{s}_K{k}": {"first": [], "ap": []} for s in ("hyb", "rr", "hub") for k in ks}
    s1_first, s1_ap, cov = [], [], {k: [] for k in ks}
    n_rel_total = 0
    for i in range(n_q):
        s1_ranks = s1_rel_ranks[i]
        n_rel = int(s1_ranks.size)
        n_rel_total += n_rel
        head_rel = head_rels[i]
        g_head = scores[i, orders[i, :k_max]].astype(np.float64)
        a_head = align[i]
        h_head = a_head - lam * reference[orders[i, :k_max]]
        row = {
            "query_pid": protocol.queries[i].pid,
            "query_wid": protocol.queries[i].wid,
            "n_rel": n_rel,
            "s1_rel_ranks": rrlib.ranks_to_str(s1_ranks),
            "s1_first": int(s1_ranks[0]),
            "s1_ap": rrlib.ap_from_ranks(s1_ranks, n_rel),
            "s1_top1": cand_pids[orders[i, 0]],
        }
        s1_first.append(row["s1_first"])
        s1_ap.append(row["s1_ap"])
        for k in ks:
            c = int(head_rel[:k].sum())
            row[f"cov_K{k}"] = c
            cov[k].append(c)
            for name, a_vals, a in (("hyb", a_head, alpha), ("rr", a_head, 0.0), ("hub", h_head, alpha)):
                ranks, top = rrlib.rerank_ranks(s1_ranks, head_rel, g_head, a_vals, k, a)
                first, ap = int(ranks.min()), rrlib.ap_from_ranks(ranks, n_rel)
                row[f"{name}_first_K{k}"] = first
                row[f"{name}_ap_K{k}"] = round(ap, 8)
                if name != "rr":
                    row[f"{name}_top1_K{k}"] = cand_pids[orders[i, top]]
                agg[f"{name}_K{k}"]["first"].append(first)
                agg[f"{name}_K{k}"]["ap"].append(ap)
        rows.append(row)
        sig = rrlib.stage1_signals(scores[i], int(np.flatnonzero(self_mask[i])[0]) if self_mask[i].any() else None)
        sig.update({"query_pid": row["query_pid"], "query_wid": row["query_wid"], "query_kocc10": int(kocc[store.index([row["query_pid"]])[0]]),
                    "top1_kocc10": int(kocc[orders[i, 0]]), "top1_hub_reference": float(reference[orders[i, 0]])})
        sig_rows.append(sig)

    # ---- aggregate + reproduction check ------------------------------------------------
    per_pair_ms = runtime["ms_per_aligned_pair"]
    metrics = {"experiment_id": exp_id, "stage1_scorer": args.stage1, "evidence": label, "alpha": alpha, "lambda": lam, "k_locked": k_locked,
               "stage1": rrlib.summarize(np.array(s1_first), np.array(s1_ap)), "by_k": []}
    for k in ks:
        entry = {"K": k, "pairs_per_query": k, "est_rerank_ms_per_query": k * per_pair_ms,
                 "coverage": float(np.mean(np.array(cov[k]) > 0)), "shortlist_recall": float(np.sum(cov[k]) / max(1, n_rel_total))}
        for s in ("hyb", "rr", "hub"):
            entry[s] = rrlib.summarize(np.array(agg[f"{s}_K{k}"]["first"]), np.array(agg[f"{s}_K{k}"]["ap"]))
        metrics["by_k"].append(entry)
    if args.frozen and not args.max_queries and k_locked in ks:
        frozen = {m["system"]: m for m in read_json(args.frozen)["metrics"]}
        at = next(e for e in metrics["by_k"] if e["K"] == k_locked)
        pairs = {"global_embedding": metrics["stage1"], f"hybrid_K{k_locked}_alpha{alpha:.2f}": at["hyb"],
                 f"rerank_only_K{k_locked}_alpha0.00": at["rr"], f"hybrid_hubcorr_K{k_locked}_alpha{alpha:.2f}": at["hub"]}
        checks = {name: {"frozen_MAP": frozen[name]["MAP"], "reproduced_MAP": v["MAP"], "abs_diff": abs(frozen[name]["MAP"] - v["MAP"])}
                  for name, v in pairs.items() if name in frozen}
        checks["shortlist_recall"] = {"frozen": read_json(args.frozen)["shortlist_recall"], "reproduced": at["shortlist_recall"]}
        metrics["reproduction"] = {"source": str(args.frozen), "tolerance": 1e-6, "checks": checks,
                                   "pass": all(c.get("abs_diff", 0.0) <= 1e-6 for c in checks.values())
                                   and abs(checks["shortlist_recall"]["frozen"] - at["shortlist_recall"]) <= 1e-9}

    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "signals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sig_rows[0].keys()))
        w.writeheader()
        w.writerows(sig_rows)
    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_pid", "kocc10", "hub_reference"])
        for j, pid in enumerate(cand_pids):
            w.writerow([pid, int(kocc[j]), f"{reference[j]:.6f}"])
    rrlib.write_json(out / "metrics.json", metrics)
    rrlib.write_json(out / "runtime.json", runtime)
    (out / "README.md").write_text(
        f"# {EXPERIMENT_ID} shortlist-size sweep ({label})\n\nGenerated by `research/scripts/local/run_shortlist_sweep.py`. "
        "Do not edit; regenerate. See research/experiments/registry.yaml.\n"
    )
    print(f"[{EXPERIMENT_ID}] wrote {out} ({label})")
    if "reproduction" in metrics:
        print(f"[{EXPERIMENT_ID}] reproduction of {args.frozen}: {'PASS' if metrics['reproduction']['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
