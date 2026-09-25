"""Paper figures, generated ONLY from research/results/*/metrics.json.

Palette: the dataviz skill's validated categorical slots 1-5 (light mode), assigned
in fixed order and never cycled; contrast warnings for slots 3-5 are relieved by
direct labels plus the tables in research/tables/. One y-axis per panel, hairline
solid grid, 2 px lines, markers r >= 4 with a 2 px surface ring.

Figures whose inputs do not exist yet (local experiments not run) are skipped and
listed in research/figures/MISSING.txt, so a missing experiment is never papered over.

    python research/scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrlib  # noqa: E402

RES = rrlib.REPO / "research" / "results"
FIG = rrlib.REPO / "research" / "figures"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated, fixed order
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round", "font.family": "DejaVu Sans",
})
MISSING: list[str] = []


def load(exp: str) -> dict | None:
    p = RES / exp / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def need(name: str, *exps: str) -> list[dict] | None:
    got = [load(e) for e in exps]
    if any(g is None for g in got):
        MISSING.append(f"{name}: needs {', '.join(e for e, g in zip(exps, got) if g is None)}")
        return None
    return got  # type: ignore[return-value]


def save(fig, name: str, evidence: str) -> None:
    fig.text(0.995, -0.03, evidence, ha="right", va="top", fontsize=6.5, color=MUTED)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def dot(ax, x, y, color, **kw):
    ax.plot(x, y, "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=5, **kw)


# ------------------------------------------------------------------------------ fig 2
def fig_coverage() -> None:
    got = need("fig2_stage1_coverage", "X1_stage1_coverage")
    if not got:
        return
    cov = got[0]["coverage"]
    series = [("run1", "Stage 1 · 1,500 works, 60 ep", SLOT[0], "-"), ("run2", "Stage 1 · 4,780 works, 60 ep", SLOT[0], "--"),
              ("run4", "Stage 1 · 4,780 works, 150 ep", SLOT[0], "-"),
              ("run2/classical_alignment", "Exhaustive alignment (96 fr.)", SLOT[3], "--"),
              ("run2/classical_alignment_hubcorr", "Exhaustive alignment + hub corr.", SLOT[3], "-")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, lab, c, ls in series:
        ks = [int(k) for k in cov[key]["curve"]]
        est = np.array([cov[key]["curve"][str(k)]["estimate"] for k in ks])
        lo = np.array([cov[key]["curve"][str(k)]["ci_low"] for k in ks])
        hi = np.array([cov[key]["curve"][str(k)]["ci_high"] for k in ks])
        alpha = 0.45 if key == "run1" else 1.0
        ax.plot(ks, est, ls, color=c, alpha=alpha, label=lab)
        ax.fill_between(ks, lo, hi, color=c, alpha=0.10, lw=0)
    ax.axvline(30, color=MUTED, lw=1)
    ax.text(31, 0.03, "K = 30 (locked)", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1, 15000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("shortlist size K (candidates passed to alignment)")
    ax.set_ylabel("queries with ≥ 1 cover in top K")
    ax.set_title("Stage-1 coverage: the ceiling on any reranker")
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, "fig2_stage1_coverage", "ARTIFACT-ANALYSIS · Da-TACOS benchmark, 13,000 queries · work-level 95% CI")


# ------------------------------------------------------------------------------ fig 3
def fig_compute() -> None:
    got = need("fig3_quality_vs_compute", "X3_compute_frontier_measured", "X1_stage1_coverage")
    if not got:
        return
    pts, cov = got[0]["points"], got[1]["coverage"]["run4"]["curve"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    colors = {"global_embedding": SLOT[0], "hybrid": SLOT[1], "hubcorr": SLOT[2], "classical": SLOT[3]}
    for ax, metric in zip(axes, ("MAP", "Hit@1")):
        for p in pts:
            s = p["system"]
            if s.startswith("rerank_only"):
                continue
            c = colors["classical"] if s.startswith("classical") else colors["hubcorr"] if "hubcorr" in s and s.startswith("hybrid") else colors["hybrid"] if s.startswith("hybrid") else colors["global_embedding"]
            x = max(p["dtw_pairs_per_query"], 0.6)
            dot(ax, x, p[metric], c, alpha=1.0 if p["run"] == "run4" or s.startswith("classical") else 0.45)
        if metric == "Hit@1":
            ks = [int(k) for k in cov]
            ax.plot(ks, [cov[str(k)]["estimate"] for k in ks], color=MUTED, lw=1.2)
            ax.text(200, cov["200"]["estimate"] - 0.07, "ceiling: Stage-1 coverage\n(run 4 encoder)", color=MUTED, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 20000)
        ax.set_ylim(0, None)
        ax.set_xlabel("DTW alignments per query")
        ax.set_title(metric)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7) for c in (SLOT[0], SLOT[1], SLOT[2], SLOT[3])]
    fig.legend(handles, ["Stage 1 (0 alignments, plotted at 0.6)", "Hybrid K = 30", "Hybrid K = 30 + hub corr.", "Exhaustive alignment"],
               loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Accuracy against alignment work (faded = runs 1–3)", x=0.01, y=0.99, ha="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.34, wspace=0.25)
    save(fig, "fig3_quality_vs_compute", "ARTIFACT-ANALYSIS · measured points from four benchmark runs")


# ------------------------------------------------------------------------------ fig 4
def fig_failure() -> None:
    got = need("fig4_failure_decomposition", "X2_failure_decomposition_K30")
    if not got:
        return
    cls = got[0]["classes"]
    rows = [(k, v) for k, v in cls.items() if "transitions" not in k]
    labels = {"run1": "Run 1", "run2": "Run 2", "run3": "Run 3", "run4": "Run 4"}
    parts = [("A_no_cover_in_shortlist", "A · no cover in top 30", SLOT[0]), ("B_demoted_below_stage1", "B · demoted by reranker", SLOT[1]),
             ("B_not_promoted", "B · not promoted to rank 1", SLOT[3]), ("R_cover_at_rank_1", "R · cover at rank 1", SLOT[2])]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ylab = []
    for i, (k, v) in enumerate(rows):
        run, sysname = k.split("/")
        ylab.append(f"{labels[run]} · {'hub-corrected' if 'hubcorr' in sysname else 'uncorrected'}")
        left = 0.0
        for key, _, c in parts:
            w = v[key]["estimate"]
            ax.barh(i, w - 0.004, left=left + 0.002, height=0.62, color=c)
            if w > 0.07:
                ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7.5, color="white" if c in (SLOT[0], SLOT[1]) else INK)
            left += w
    ax.set_yticks(range(len(rows)), ylab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title("Where each benchmark query ends up at K = 30")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts], [lab for _, lab, _ in parts], loc="lower center",
              ncol=4, fontsize=7.5, bbox_to_anchor=(0.45, -0.28))
    save(fig, "fig4_failure_decomposition", "ARTIFACT-ANALYSIS · 13,000 queries per row · class A is identical within a run by construction")


# ------------------------------------------------------------------------------ fig 5
def fig_oracle() -> None:
    got = need("fig5_oracle_allocation", "X4_oracle_allocation_bound")
    if not got:
        return
    runs = got[0]["runs"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for run, c, lab in [("run4", SLOT[0], "run 4 encoder")]:
        r = runs[run]
        ax.plot([f["mean_pairs"] for f in r["fixed_K"]], [f["coverage"] for f in r["fixed_K"]], color=c, label=f"fixed K ({lab})")
        ax.plot([o["mean_pairs"] for o in r["oracle"]], [o["coverage"] for o in r["oracle"]], "--", color=SLOT[1], label="oracle allocation (upper bound)")
        dot(ax, 30, r["fixed_K30"]["coverage"], c)
        dot(ax, r["oracle_pairs_for_fixed_K30_coverage"], r["fixed_K30"]["coverage"], SLOT[1])
        ax.annotate(f"same coverage at {r['oracle_pairs_for_fixed_K30_coverage']:.1f} vs 30 alignments",
                    (r["oracle_pairs_for_fixed_K30_coverage"], r["fixed_K30"]["coverage"]), xytext=(0.9, 0.72), fontsize=7.5, color=MUTED,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xscale("log")
    ax.set_xlim(0.8, 5000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean DTW alignments per query")
    ax.set_ylabel("queries with ≥ 1 cover aligned")
    ax.set_title("Headroom for adaptive shortlist size (oracle bound)")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "fig5_oracle_allocation", "ARTIFACT-ANALYSIS · UPPER BOUND, not a policy · realisable part = experiments D1/E1")


# ------------------------------------------------------------------------------ fig 7
def fig_factorisation() -> None:
    got = need("fig7_hit1_factorisation", "X2_failure_decomposition_K30")
    if not got:
        return
    cls = got[0]["classes"]
    keys = [k for k in cls if "transitions" not in k]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), sharey=True)
    names = [("A_no_cover_in_shortlist", "coverage C(30)", lambda v: 1 - v["A_no_cover_in_shortlist"]["estimate"]),
             ("reranker_efficiency_R_given_covered", "efficiency: P(R | covered)", lambda v: v["reranker_efficiency_R_given_covered"]["estimate"]),
             ("R_cover_at_rank_1", "Hit@1 = C(30) × efficiency", lambda v: v["R_cover_at_rank_1"]["estimate"])]
    for ax, (_, title, fn) in zip(axes, names):
        for variant, c in (("hybrid_K30", SLOT[1]), ("hybrid_hubcorr", SLOT[2])):
            xs, ys = [], []
            for k in keys:
                if variant in k:
                    xs.append(int(k.split("/")[0][-1]))
                    ys.append(fn(cls[k]))
            ax.plot(xs, ys, color=c)
            for x, y in zip(xs, ys):
                dot(ax, x, y, c)
        ax.set_xticks([1, 2, 3, 4], ["R1", "R2", "R3", "R4"])
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, 1)
    axes[0].legend([plt.Line2D([], [], color=SLOT[1]), plt.Line2D([], [], color=SLOT[2])], ["uncorrected", "hub-corrected"], fontsize=7.5, loc="upper left")
    save(fig, "fig7_hit1_factorisation", "ARTIFACT-ANALYSIS · coverage is shared by both variants within a run")


# ------------------------------------------------------------------------------ local figures (skipped until the data exists)
def fig_k_sweep() -> None:
    got = need("figA_k_sweep", "A1_k_sweep_long384")
    if not got:
        return
    by_k = got[0]["by_k"]
    ks = [e["K"] for e in by_k]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    axes[0].plot(ks, [e["coverage"] for e in by_k], color=SLOT[0], label="coverage")
    axes[0].plot(ks, [e["shortlist_recall"] for e in by_k], "--", color=SLOT[0], label="shortlist recall")
    for ax, m in zip(axes[1:], ("MAP", "Hit@1")):
        for s, c, lab in (("hyb", SLOT[1], "hybrid"), ("hub", SLOT[2], "hybrid + hub"), ("rr", SLOT[3], "rerank only")):
            ax.plot(ks, [e[s][m] for e in by_k], color=c, label=lab)
        ax.set_title(m)
    axes[0].set_title("Stage 1")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("K")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    save(fig, "figA_k_sweep", f"{got[0]['evidence']} · A1 · {got[0]['stage1']['n_queries']:,} queries")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig_coverage, fig_compute, fig_failure, fig_oracle, fig_factorisation, fig_k_sweep):
        fn()
    (FIG / "MISSING.txt").write_text("\n".join(MISSING) + ("\n" if MISSING else ""))
    if MISSING:
        print("skipped (inputs not present yet):\n  " + "\n  ".join(MISSING))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
