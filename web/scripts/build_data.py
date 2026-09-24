"""Collect the committed research results into one JSON file for the website.

Every value in ``web/src/data/generated/results.json`` is copied or aggregated from a
file under ``reports/results/``; each block records its ``source``. Nothing here is
typed in by hand. Also copies the error-case figures the website shows into
``web/public/figures/`` as WebP.

Run from the repository root (stdlib + Pillow only):  python web/scripts/build_data.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
OUT = ROOT / "web" / "src" / "data" / "generated" / "results.json"
PUBLIC_FIGURES = ROOT / "web" / "public" / "figures"


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def load_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def r(x: float, digits: int = 6) -> float:
    return round(float(x), digits)


METRIC_KEYS = ["MAP", "MRR", "Hit@1", "Hit@10", "Hit@100", "Recall@10", "Recall@100", "mean_first_rank", "median_first_rank"]


def metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {m["system"]: {k: r(m[k]) for k in METRIC_KEYS if k in m} for m in rows}


# ------------------------------------------------------------------ benchmark
BENCHMARK_RUNS = [
    # (run id, results json, per-query csv, human context). Context strings restate the
    # "locked_from_calibration" block of each JSON and the tables in improvement_experiments.md.
    ("run1", "benchmark.json", "benchmark_per_query.csv", "1,500-work encoder, 60 epochs · 96-frame alignment · no hubness correction"),
    ("run2", "benchmark_full96.json", "benchmark_per_query_full96.csv", "full-data encoder (4,780 works), 60 epochs · 96-frame alignment · hubness λ = 0.5"),
    ("run3", "benchmark_full384.json", "benchmark_per_query_full384.csv", "full-data encoder, 60 epochs · 384-frame reranking · hubness λ = 0.6"),
    ("run4", "benchmark_long384.json", "benchmark_per_query_long384.csv", "full-data encoder, 150 epochs · 384-frame reranking · hubness λ = 0.6"),
]

# log-spaced rank grid (1 .. 14,999) for first-correct-rank distributions
RANK_GRID = sorted({max(1, min(14999, round(10 ** (i / 40)))) for i in range(0, 168)})
# integer-aligned log bins [e_i, e_{i+1}); 31 is an edge so ranks 1-30 (the K = 30 shortlist) are exact
RANK_EDGES = sorted({round(10 ** (k / 10)) for k in range(0, 42)} | {31, 15000})


def first_rank_curves(rows: list[dict[str, str]]) -> dict[str, dict]:
    systems = [c[: -len("_first_rank")] for c in rows[0] if c.endswith("_first_rank")]
    out = {}
    n = len(rows)
    for s in systems:
        ranks = sorted(int(row[f"{s}_first_rank"]) for row in rows)
        cdf, k = [], 0
        for g in RANK_GRID:
            while k < n and ranks[k] <= g:
                k += 1
            cdf.append(r(k / n, 5))
        # histogram of first-correct rank over RANK_EDGES
        bins = [0] * (len(RANK_EDGES) - 1)
        b = 0
        for x in ranks:  # ranks are sorted
            while x >= RANK_EDGES[b + 1]:
                b += 1
            bins[b] += 1
        out[s] = {
            "cdf": cdf,
            "log_bins": bins,
            "in_shortlist_30": r(sum(1 for x in ranks if x <= 30) / n, 4),
            "median": statistics.median(ranks),
        }
    return out


def benchmark() -> dict:
    runs = []
    for run_id, js, pq, context in BENCHMARK_RUNS:
        d = load_json(js)
        rows = load_csv(pq)
        runs.append(
            {
                "id": run_id,
                "source": f"reports/results/{js}",
                "per_query_source": f"reports/results/{pq}",
                "context": context,
                "locked": d["locked_from_calibration"],
                "protocol": d["protocol"],
                "metrics": metrics(d["metrics"]),
                "shortlist_recall": r(d["shortlist_recall"]),
                "runtime": {k: r(v, 3) for k, v in d["runtime"].items()},
                "bootstrap_vs_global": {
                    k: {"delta": r(v["delta"]), "lo": r(v["ci_low"]), "hi": r(v["ci_high"])}
                    for k, v in d["bootstrap_vs_global"].items()
                },
                "git_commit": d.get("git_commit"),
                "first_rank": first_rank_curves(rows),
            }
        )
    return {"rank_grid": RANK_GRID, "rank_edges": RANK_EDGES, "runs": runs}


# ------------------------------------------------------------------ development protocol
DEV_RUNS = [
    ("base", "", "1,500-work encoder · 96 frames"),
    ("n384", "_n384", "1,500-work encoder · 384 frames"),
    ("full96", "_full96", "full-data encoder, 60 ep · 96 frames"),
    ("full384", "_full384", "full-data encoder, 60 ep · 384 frames"),
    ("long384", "_long384", "full-data encoder, 150 ep · 384 frames"),
]


def dev() -> dict:
    runs = []
    for run_id, suffix, context in DEV_RUNS:
        d = load_json(f"hybrid_dev{suffix}.json")
        per_query = load_csv(f"hybrid_dev_per_query{suffix}.csv")
        systems = [c[: -len("_first_rank")] for c in per_query[0] if c.endswith("_first_rank")]
        cases = load_csv(f"error_cases{suffix}.csv")
        top10 = load_csv(f"hybrid_dev_rankings_top10{suffix}.csv")
        runs.append(
            {
                "id": run_id,
                "context": context,
                "source": f"reports/results/hybrid_dev{suffix}.json",
                "locked": d["locked"],
                "shortlist_recall": r(d["shortlist_recall"]),
                "metrics": metrics(d["metrics"]),
                "systems": systems,
                "queries": [
                    {"query": q["query_pid"], "wid": q["query_wid"], "ranks": {s: int(q[f"{s}_first_rank"]) for s in systems}}
                    for q in per_query
                ],
                "k_sweep": [
                    {"K": m["K"], "recall": r(m["shortlist_recall"]), "MAP": r(m["MAP"])}
                    for m in d["shortlist_ablation"]
                    if m["system"].startswith("hybrid")
                ],
                "bootstrap": {k: {"delta": r(v["delta"]), "lo": r(v["ci_low"]), "hi": r(v["ci_high"])} for k, v in d["bootstrap"].items()},
                "cases": [
                    {
                        "case": c["case"],
                        "query": c["query_pid"],
                        "partner": c["relevant_pid"],
                        "rank_hybrid": int(c["relevant_rank_hybrid"]),
                        "rank_global": int(c["relevant_rank_global"]),
                        "rank_classical": int(c["relevant_rank_classical"]),
                        "partner_cos": r(c["relevant_global_cosine"]),
                        "partner_align": r(c["relevant_alignment_score"]),
                        "partner_shift": int(c["relevant_shift"]),
                        "in_shortlist": c["relevant_in_shortlist"] == "True",
                        "fp": c["top_false_positive_pid"],
                        "fp_rank": int(c["top_false_positive_rank"]),
                        "fp_cos": r(c["top_false_positive_global_cosine"]),
                        "fp_align": r(c["top_false_positive_alignment_score"]),
                        "fp_shift": int(c["top_false_positive_shift"]),
                        "figure": Path(c["figure"]).stem + ".webp",
                    }
                    for c in cases
                ],
                "top10": [
                    {"query": t["query_pid"], "rank": int(t["rank"]), "candidate": t["candidate_pid"], "cos": r(t["score"]), "relevant": t["is_relevant"] == "True"}
                    for t in top10
                    if t["query_pid"] == CASE_QUERY
                ],
            }
        )
    return {"protocol": load_json("hybrid_dev.json")["protocol"], "runs": runs}


# ------------------------------------------------------------------ classical view
CASE_QUERY = "P_797406"  # reports/error_analysis.md: "the clearest case of reranking fixing Stage 1"

def classical() -> dict:
    d = load_json("classical_dev.json")
    rankings = load_csv("classical_dev_rankings.csv")
    # alignment score + chosen shift of every (query, candidate) pair, for the case study
    pairs: dict[str, dict] = {}
    for row in rankings:
        if row["query_pid"] != CASE_QUERY:  # the site only needs the case-study query
            continue
        pairs.setdefault(row["query_pid"], {})[row["candidate_pid"]] = {
            "rank": int(row["rank"]),
            "score": r(row["score"]),
            "shift": int(row["shift"]),
            "relevant": row["is_relevant"] == "True",
        }
    # distribution of chosen shifts across all pairs (D-004 discussion)
    shifts = [0] * 12
    for row in rankings:
        shifts[int(row["shift"])] += 1
    return {
        "source": "reports/results/classical_dev.json",
        "metrics": metrics(d["metrics"]),
        "resolution_sweep": [
            {"protocol": m["protocol"], "n_frames": m["n_frames"], "MAP": r(m["MAP"]), "ms_per_pair": r(m["ms_per_pair"], 4)}
            for m in d["resolution_sweep_ANALYSIS_ONLY"]
        ],
        "rankings_source": "reports/results/classical_dev_rankings.csv",
        "pairs": pairs,
        "shift_histogram": shifts,
    }


# ------------------------------------------------------------------ hubness
def hubness() -> dict:
    d = load_json("hubness_dev.json")
    classical_rows = load_csv("classical_dev_rankings.csv")
    hybrid_rows = load_csv("hybrid_dev_rankings_top10.csv")

    def edges(rows):
        return [
            [row["query_pid"], row["candidate_pid"]]
            for row in rows
            if int(row["rank"]) <= 10 and row["is_relevant"] in ("False", "false", "0")
        ]

    corrections = []
    for name in ("hubness_correction.json", "hubness_correction_n384.json"):
        h = load_json(name)
        corrections.append(
            {
                "source": f"reports/results/{name}",
                "n_frames": h["n_frames"],
                "locked": h["locked"],
                "grid": [
                    {"method": g["method"], "k": g.get("k", 0), "lambda": r(g["lambda"], 2), "MAP": r(g["MAP"])}
                    for g in h["calibration_grid"]
                ],
                "dev_metrics": metrics(h["dev_metrics"]),
                "dev_bootstrap": {k: r(v) for k, v in h["dev_bootstrap_corrected_minus_uncorrected"].items() if k != "n_groups"},
                "reference_vs_dispersion_spearman": r(h["dev_reference_vs_dispersion_spearman"], 3),
            }
        )
    return {
        "source": "reports/results/hubness_dev.json",
        "summary": d,
        "edges_classical": edges(classical_rows),
        "edges_hybrid": edges(hybrid_rows),
        "edges_source": ["reports/results/classical_dev_rankings.csv", "reports/results/hybrid_dev_rankings_top10.csv"],
        "corrections": corrections,
    }


# ------------------------------------------------------------------ training
def training() -> dict:
    out = {}
    for run in ("sweep_base", "encoder_full_train", "encoder_full_long"):
        hist = load_csv(f"training_history_{run}.csv")
        summary = load_json(f"training_summary_{run}.json")
        out[run] = {
            "source": f"reports/results/training_history_{run}.csv",
            "summary": {k: summary[k] for k in ("best_epoch", "best_val_map", "epochs_run", "n_parameters", "receptive_field_frames", "train_wids", "total_seconds") if k in summary},
            "encoder": summary.get("encoder"),
            "training": summary.get("training"),
            "epochs": [[int(h["epoch"]), r(h["train_loss"]), r(h["val_map"])] for h in hist],
        }
    sweep = load_csv("encoder_sweep.csv")
    return {"runs": out, "sweep": sweep, "sweep_source": "reports/results/encoder_sweep.csv"}


# ------------------------------------------------------------------ calibration
def calibration() -> dict:
    out = {}
    for tag in ("", "_n384", "_full96", "_full384", "_long384"):
        d = load_json(f"hybrid_calibration{tag}.json")
        out[tag.lstrip("_") or "base"] = {
            "source": f"reports/results/hybrid_calibration{tag}.json",
            "alpha": d["alpha"],
            "shortlist_k": d["shortlist_k"],
            "shortlist_recall": r(d["shortlist_recall"]),
            "table": [[r(t["alpha"], 2), r(t["map"])] for t in d["table"]],
        }
    return out


def copy_figures(dev_data: dict) -> list[str]:
    PUBLIC_FIGURES.mkdir(parents=True, exist_ok=True)
    names = sorted({c["figure"] for run in dev_data["runs"] for c in run["cases"]})
    for name in names:
        src = FIGURES / name.replace(".webp", ".png")
        dst = PUBLIC_FIGURES / name
        img = Image.open(src).convert("RGB")
        if img.width > 1400:
            img = img.resize((1400, round(img.height * 1400 / img.width)), Image.LANCZOS)
        img.save(dst, "WEBP", quality=82, method=6)
    return names


def main() -> None:
    dev_data = dev()
    data = {
        "_note": "Generated by web/scripts/build_data.py from reports/results/. Do not edit by hand.",
        "benchmark": benchmark(),
        "dev": dev_data,
        "classical": classical(),
        "hubness": hubness(),
        "training": training(),
        "calibration": calibration(),
        "dataset": {"source": "data/manifests/manifest_info.json", **json.loads((ROOT / "data/manifests/manifest_info.json").read_text())},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    figs = copy_figures(dev_data)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB) and {len(figs)} figures")


if __name__ == "__main__":
    main()
