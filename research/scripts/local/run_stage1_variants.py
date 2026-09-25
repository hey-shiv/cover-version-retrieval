"""Series F / G1 / H1 / I: Stage-1 shortlist recall under different Stage-1 scorers.

LOCAL COMPUTE ONLY. Cheap: embeddings and dot products, no alignment.

Question: which change to Stage 1 raises shortlist recall at fixed K (fixed DTW work)?
Every variant is scored by the ranks of ALL relevant items, so coverage and recall at
any K can be derived later without re-running.

Variants (each independently selectable with --variants):
  global:<name>   one per --checkpoint name=path (training-scale series I)
  ttr             12-rotation test-time max over query rotations (primary checkpoint)
  win_max         windowed multi-vector: 3 windows of 60% length (offsets 0, 20, 40%),
                  score = max over the 3x3 window-pair cosines (MaxSim-style)
  win_fuse        (global + win_max) / 2, fixed equal weights (not tuned)
  s1_hub          global - lambda * r(c), r(c) = mean cosine of candidate c to 200 fixed
                  training-split probe tracks (inductive, as D-018 for alignment);
                  lambda chosen on the CALIBRATION protocol by Stage-1 MAP, grid 0..1 step 0.1

The windows reuse the trained encoder unchanged. Training crops cover >= 60% of a track
(training.crop_min_fraction = 0.6), so 60% windows stay inside the training distribution.

Example:
    python research/scripts/local/run_stage1_variants.py --config configs/full_train.yaml \
        --checkpoint full_long=runs/encoder_full_long/encoder_best.pt \
        --checkpoint full_60=runs/encoder_full_train/encoder_best.pt \
        --checkpoint base_1500=runs/sweep_base/encoder_best.pt \
        --primary full_long --out research/results/F1_stage1_variants
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rrlib  # noqa: E402

from cover_retrieval.data.manifests import load_protocol, load_split_tracks  # noqa: E402
from cover_retrieval.evaluation.metrics import ranking_from_scores  # noqa: E402
from cover_retrieval.models.training import embed, load_encoder  # noqa: E402
from cover_retrieval.pipeline import protocol_store, split_store, torch_threads  # noqa: E402
from cover_retrieval.utils.io import load_config, parse_overrides  # noqa: E402

EXPERIMENT_ID = "F1"
KS = [1, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
WINDOW_FRACTION = 0.6
WINDOW_OFFSETS = (0.0, 0.2, 0.4)
LAMBDA_GRID = [round(0.1 * i, 1) for i in range(11)]


def windows(features: np.ndarray) -> list[np.ndarray]:
    t = features.shape[-1]
    w = int(round(WINDOW_FRACTION * t))
    return [features[..., int(round(o * t)) : int(round(o * t)) + w] for o in WINDOW_OFFSETS]


def rel_ranks(scores: np.ndarray, relevance: np.ndarray, self_mask: np.ndarray) -> list[np.ndarray]:
    out = []
    for i in range(scores.shape[0]):
        _, rel = ranking_from_scores(scores[i], relevance[i], exclude=self_mask[i])
        out.append(np.flatnonzero(rel) + 1)
    return out


def chunked(fn, n_q: int, chunk: int = 500) -> np.ndarray:
    return np.concatenate([fn(slice(s, min(s + chunk, n_q))) for s in range(0, n_q, chunk)])


def stage1_map(scores, protocol) -> float:
    rr = rel_ranks(scores, protocol.relevance_matrix(), protocol.self_mask())
    return float(np.mean([rrlib.ap_from_ranks(r) for r in rr]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--primary", required=True, help="checkpoint NAME used for ttr / win / s1_hub")
    p.add_argument("--variants", nargs="+", default=["global", "ttr", "win_max", "win_fuse", "s1_hub"])
    p.add_argument("--n-probes", type=int, default=200)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    config = load_config(args.config, parse_overrides(args.set))
    torch_threads(config)
    label = rrlib.evidence_label(config) + ("-SANITY" if args.max_queries else "")
    out = rrlib.prepare_out(args.out, args.force)
    ckpts = dict(c.split("=", 1) for c in args.checkpoint)
    rrlib.write_meta(out, EXPERIMENT_ID, config, vars(args), label)

    protocol = load_protocol(config["paths"]["manifests_dir"], "benchmark")
    store = protocol_store(config, protocol, "benchmark", nonfinite_policy="zero")
    n_q = len(protocol.queries) if not args.max_queries else min(args.max_queries, len(protocol.queries))
    q_idx = store.index([t.pid for t in protocol.queries[:n_q]])
    relevance = protocol.relevance_matrix()[:n_q]
    self_mask = protocol.self_mask()[:n_q]
    feats = store.views["encoder"]
    runtime: dict[str, float] = {}
    ranks: dict[str, list[np.ndarray]] = {}
    notes: dict[str, object] = {"primary": args.primary}

    emb = {}
    for name, path in ckpts.items():
        if "global" not in args.variants and name != args.primary:
            continue
        t0 = time.perf_counter()
        model = load_encoder(path)
        emb[name] = (model, embed(model, feats))
        runtime[f"embed_{name}_s"] = time.perf_counter() - t0
        if "global" in args.variants:
            ranks[f"global:{name}"] = rel_ranks(emb[name][1][q_idx] @ emb[name][1].T, relevance, self_mask)
    model, cand = emb[args.primary]

    if "ttr" in args.variants:
        t0 = time.perf_counter()
        rot = np.stack([embed(model, feats[q_idx], rotation=r) for r in range(12)])  # (12, Q, D)
        s = chunked(lambda sl: np.max(np.einsum("rqd,cd->rqc", rot[:, sl], cand), axis=0), n_q)
        ranks["ttr"] = rel_ranks(s, relevance, self_mask)
        runtime["ttr_s"] = time.perf_counter() - t0

    if {"win_max", "win_fuse"} & set(args.variants):
        t0 = time.perf_counter()
        wins = [embed(model, w) for w in windows(feats)]  # 3 x (C, D)
        def win_scores(sl):
            best = None
            for wq in wins:
                for wc in wins:
                    s = wq[q_idx[sl]] @ wc.T
                    best = s if best is None else np.maximum(best, s)
            return best
        s_win = chunked(win_scores, n_q)
        runtime["windows_s"] = time.perf_counter() - t0
        if "win_max" in args.variants:
            ranks["win_max"] = rel_ranks(s_win, relevance, self_mask)
        if "win_fuse" in args.variants:
            ranks["win_fuse"] = rel_ranks(0.5 * (cand[q_idx] @ cand.T) + 0.5 * s_win, relevance, self_mask)

    if "s1_hub" in args.variants:
        t0 = time.perf_counter()
        probes = load_split_tracks(config["paths"]["manifests_dir"], "train")[: args.n_probes]
        probe_emb = embed(model, split_store(config, probes, f"probe{args.n_probes}").views["encoder"])
        ref_bench = (probe_emb @ cand.T).mean(axis=0)
        # lambda on the calibration protocol only (queries: calibration works; pool: calibration + validation)
        cal = load_protocol(config["paths"]["manifests_dir"], "calibration")
        cal_store = protocol_store(config, cal, "coveranalysis")
        cal_emb = embed(model, cal_store.views["encoder"])
        cal_q = cal_emb[cal_store.index([t.pid for t in cal.queries])]
        ref_cal = (probe_emb @ cal_emb.T).mean(axis=0)
        grid = [{"lambda": lam, "calibration_MAP": stage1_map(cal_q @ cal_emb.T - lam * ref_cal[None, :], cal)} for lam in LAMBDA_GRID]
        best = max(grid, key=lambda g: (round(g["calibration_MAP"], 12), -g["lambda"]))  # ties -> smaller lambda
        notes["s1_hub"] = {"grid": grid, "locked_lambda": best["lambda"], "selection": "calibration protocol, Stage-1 MAP"}
        ranks["s1_hub"] = rel_ranks(cand[q_idx] @ cand.T - best["lambda"] * ref_bench[None, :], relevance, self_mask)
        runtime["s1_hub_s"] = time.perf_counter() - t0

    # ---- outputs -------------------------------------------------------------------------
    names = list(ranks)
    rows = []
    for i in range(n_q):
        row = {"query_pid": protocol.queries[i].pid, "query_wid": protocol.queries[i].wid, "n_rel": int(ranks[names[0]][i].size)}
        for v in names:
            row[f"{v}_rel_ranks"] = rrlib.ranks_to_str(ranks[v][i])
        rows.append(row)
    metrics = {"experiment_id": EXPERIMENT_ID, "evidence": label, "notes": notes, "variants": {}}
    for v in names:
        rr = ranks[v]
        n_rel_total = sum(r.size for r in rr)
        metrics["variants"][v] = {
            **rrlib.summarize(np.array([r.min() for r in rr]), np.array([rrlib.ap_from_ranks(r) for r in rr])),
            "coverage": {f"@{k}": float(np.mean([np.any(r <= k) for r in rr])) for k in KS},
            "shortlist_recall": {f"@{k}": float(sum(int((r <= k).sum()) for r in rr) / n_rel_total) for k in KS},
        }
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rrlib.write_json(out / "metrics.json", metrics)
    rrlib.write_json(out / "runtime.json", runtime)
    (out / "README.md").write_text(f"# {EXPERIMENT_ID} Stage-1 variants ({label})\n\nGenerated by run_stage1_variants.py; do not edit.\n")
    print(f"[{EXPERIMENT_ID}] wrote {out} ({label}): " + ", ".join(f"{v} cov@30={metrics['variants'][v]['coverage']['@30']:.3f}" for v in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
