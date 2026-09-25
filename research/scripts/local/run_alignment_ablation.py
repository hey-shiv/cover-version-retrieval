"""Series H: key handling x alignment resolution inside a fixed Stage-1 shortlist.

LOCAL COMPUTE ONLY. The shortlist (first K Stage-1 candidates of the primary
checkpoint) is identical for every variant, so differences isolate Stage 2.
Every variant is scored as rerank-only (alpha = 0): no blend weight exists for most
of these combinations, and alpha = 0 needs no tuning.

Variants: rotation in {none, profile_cosine, exhaustive_dtw} x n_frames in {96, 384}.
exhaustive_dtw costs 12x per pair (about 99 min at 384 frames for K = 30 on the
benchmark); drop it with --rotations none profile_cosine if the budget is short.

Example:
    python research/scripts/local/run_alignment_ablation.py --config configs/full_train.yaml \
        --checkpoint runs/encoder_full_long/encoder_best.pt --k 30 --out research/results/H1_alignment_ablation
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

from cover_retrieval.data.manifests import load_protocol  # noqa: E402
from cover_retrieval.evaluation.metrics import ranking_from_scores  # noqa: E402
from cover_retrieval.models.training import load_encoder  # noqa: E402
from cover_retrieval.pipeline import aligner_from_config, global_scores, protocol_store, torch_threads  # noqa: E402
from cover_retrieval.utils.io import load_config, parse_overrides  # noqa: E402

EXPERIMENT_ID = "H1"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--frames", type=int, nargs="+", default=[96, 384])
    p.add_argument("--rotations", nargs="+", default=["none", "profile_cosine", "exhaustive_dtw"])
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    base = load_config(args.config, parse_overrides(args.set))
    torch_threads(base)
    label = rrlib.evidence_label(base) + ("-SANITY" if args.max_queries else "")
    out = rrlib.prepare_out(args.out, args.force)
    rrlib.write_meta(out, EXPERIMENT_ID, base, vars(args), label)

    protocol = load_protocol(base["paths"]["manifests_dir"], "benchmark")
    store0 = protocol_store(base, protocol, "benchmark", nonfinite_policy="zero")
    scores, _ = global_scores(base, load_encoder(args.checkpoint), protocol, store0)
    n_q = len(protocol.queries) if not args.max_queries else min(args.max_queries, len(protocol.queries))
    rel_m, self_m = protocol.relevance_matrix(), protocol.self_mask()
    heads, s1_ranks, head_rel = [], [], []
    for i in range(n_q):
        order, rel = ranking_from_scores(scores[i], rel_m[i], exclude=self_m[i])
        heads.append(order[: args.k])
        s1_ranks.append(np.flatnonzero(rel) + 1)
        head_rel.append(rel[: args.k])

    results: dict[str, dict[str, list]] = {}
    runtime: dict[str, float] = {}
    for n_frames in args.frames:
        config = load_config(args.config, parse_overrides([*args.set, f"features.n_frames={n_frames}"]))
        store = protocol_store(config, protocol, "benchmark", nonfinite_policy="zero")
        q_feats = store.get("classical", [t.pid for t in protocol.queries[:n_q]])
        for rot in args.rotations:
            name = f"{rot}_n{n_frames}"
            aligner = aligner_from_config(config, rotation_selection=rot)
            t0 = time.perf_counter()
            firsts, aps = [], []
            for i in range(n_q):
                a = aligner.score_pairs(q_feats[i], store.views["classical"][heads[i]]).score
                ranks, _ = rrlib.rerank_ranks(s1_ranks[i], head_rel[i], np.zeros(args.k), a, args.k, 0.0)
                firsts.append(int(ranks.min()))
                aps.append(rrlib.ap_from_ranks(ranks, s1_ranks[i].size))
            runtime[f"{name}_s"] = time.perf_counter() - t0
            runtime[f"{name}_ms_per_pair"] = 1e3 * runtime[f"{name}_s"] / (n_q * args.k)
            results[name] = {"first": firsts, "ap": aps}
            print(f"[{EXPERIMENT_ID}] {name}: MAP {np.mean(aps):.4f} ({runtime[f'{name}_s']:.0f}s)", flush=True)

    rows = []
    for i in range(n_q):
        row = {"query_pid": protocol.queries[i].pid, "query_wid": protocol.queries[i].wid,
               "s1_first": int(s1_ranks[i].min()), "s1_ap": rrlib.ap_from_ranks(s1_ranks[i]), "cov": int(head_rel[i].sum())}
        for name, r in results.items():
            row[f"{name}_first"] = r["first"][i]
            row[f"{name}_ap"] = round(r["ap"][i], 8)
        rows.append(row)
    metrics = {"experiment_id": EXPERIMENT_ID, "evidence": label, "K": args.k, "scoring": "rerank only (alpha = 0)",
               "stage1": rrlib.summarize(np.array([r["s1_first"] for r in rows]), np.array([r["s1_ap"] for r in rows])),
               "variants": {n: rrlib.summarize(np.array(r["first"]), np.array(r["ap"])) for n, r in results.items()}}
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rrlib.write_json(out / "metrics.json", metrics)
    rrlib.write_json(out / "runtime.json", runtime)
    (out / "README.md").write_text(f"# {EXPERIMENT_ID} alignment ablation ({label})\n\nGenerated by run_alignment_ablation.py; do not edit.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
