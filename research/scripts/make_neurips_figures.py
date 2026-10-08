"""Additional paper figures in a NeurIPS-style look, generated only from committed per-query result files.

    .venv/bin/python research/scripts/make_neurips_figures.py

Writes paper/figures/nx_*.pdf:
  nx_teaser       Figure 1: the coverage ceiling, the bottleneck shift and the failure classes across K (A1)
  nx_rank_ecdf    distribution of the Stage-1 rank of the first cover, with shortlist sizes marked (A1)
  nx_scatter      per-query AP before and after reranking at K=30 (A1)
  nx_gain_by_k    per-K MAP gain over Stage 1, with and without hub correction, with work-level intervals (A1)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
A1 = REPO / "research" / "results" / "A1_k_sweep_long384" / "per_query.csv"
OUT = REPO / "paper" / "figures"

BLUE, ORANGE, GREEN, PINK, GREY, INK = "#2a6fb0", "#e8743b", "#2e9e6f", "#d9558a", "#8a8a85", "#1f1f1e"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "axes.grid": True,
    "grid.color": "#e4e4e0", "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
KS = [5, 10, 20, 30, 50, 100, 200, 500]


def load() -> dict[str, np.ndarray]:
    with open(A1) as fh:
        rows = list(csv.DictReader(fh))
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "s1_rel_ranks"}
    out = {}
    for k, v in cols.items():
        try:
            out[k] = v.astype(float)
        except ValueError:
            out[k] = v
    return out


def work_boot(delta: np.ndarray, wid: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% interval resampling works (cliques), not queries."""
    uniq, inv = np.unique(wid, return_inverse=True)
    sums = np.bincount(inv, weights=delta)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    est = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(delta.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def panel_label(ax, s: str) -> None:
    ax.text(-0.02, 1.08, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def teaser(d: dict) -> None:
    cov = np.array([(d[f"cov_K{k}"] > 0).mean() for k in KS])
    hit = np.array([(d[f"hub_first_K{k}"] == 1).mean() for k in KS])
    mapk = np.array([d[f"hub_ap_K{k}"].mean() for k in KS])
    fa = 1 - cov
    fb = cov - hit
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.15), gridspec_kw={"wspace": 0.42})
    x = np.arange(len(KS))
    a = ax[0]
    a.plot(x, cov * 100, "-o", color=BLUE, ms=3.5, label=r"coverage $C(K)$ (ceiling)")
    a.plot(x, hit * 100, "-s", color=GREEN, ms=3.5, label="Hit@1 achieved")
    a.fill_between(x, hit * 100, cov * 100, color=PINK, alpha=0.18, lw=0, label="lost to misranking")
    a.fill_between(x, cov * 100, 100, color=BLUE, alpha=0.10, lw=0, label="lost to exclusion")
    a.set_xticks(x, [str(k) for k in KS]); a.set_xlabel("shortlist size $K$"); a.set_ylabel("% of 13,000 queries")
    a.set_ylim(0, 100); a.legend(loc="upper left", fontsize=6, handlelength=1.2, borderaxespad=0.2)
    a.set_title("Where Hit@1 is lost"); panel_label(a, "a")
    b = ax[1]
    b.bar(x, hit * 100, color=GREEN, width=0.72, label="cover at rank 1")
    b.bar(x, fb * 100, bottom=hit * 100, color=PINK, width=0.72, label="B: covered, misranked")
    b.bar(x, fa * 100, bottom=cov * 100, color=BLUE, width=0.72, label="A: not covered")
    cross = np.argmin(np.abs(fa - fb))
    b.axvline(x[cross] + 0.5, color=INK, lw=0.7, ls=(0, (3, 2)))
    b.text(x[cross] + 0.5, 101, "A = B", fontsize=6.5, ha="center", va="bottom")
    b.set_xticks(x, [str(k) for k in KS]); b.set_xlabel("shortlist size $K$"); b.set_ylim(0, 100)
    b.set_title("Failure class shifts A$\\to$B"); b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=6, handlelength=1.0)
    b.grid(axis="x", visible=False); panel_label(b, "b")
    c = ax[2]
    c.plot(x, mapk, "-o", color=ORANGE, ms=3.5)
    c.axhline(d["s1_ap"].mean(), color=GREY, lw=0.9, ls=(0, (4, 2)))
    c.text(0.1, d["s1_ap"].mean() + 0.003, "Stage 1 alone", color=GREY, fontsize=6.5)
    c.set_xticks(x, [str(k) for k in KS]); c.set_xlabel("shortlist size $K$"); c.set_ylabel("MAP (hub-corrected)")
    c.set_title("Accuracy has not saturated"); panel_label(c, "c")
    save(fig, "nx_teaser")


def rank_ecdf(d: dict) -> None:
    r = d["s1_first"]
    xs = np.unique(r)
    cdf = np.searchsorted(np.sort(r), xs, side="right") / len(r)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(xs, cdf * 100, color=BLUE, lw=1.6, drawstyle="steps-post")
    ax.set_xscale("log")
    for k, c in zip((5, 30, 100, 500), (GREY, ORANGE, ORANGE, ORANGE)):
        v = (r <= k).mean() * 100
        ax.vlines(k, 0, v, color=c, lw=0.8, ls=(0, (3, 2)))
        ax.plot([k], [v], "o", color=c, ms=3.5)
        ax.annotate(f"{v:.0f}%", (k, v), textcoords="offset points", xytext=(-3, 4), ha="right", fontsize=6.5)
    ax.set_xlabel("Stage-1 rank of the first cover, $r_q$ (log)")
    ax.set_ylabel(r"coverage $C(K)$ (%)")
    ax.set_xlim(1, 15000); ax.set_ylim(0, 100)
    ax.text(0.97, 0.06, f"median $r_q$ = {np.median(r):.0f}\nmean $r_q$ = {r.mean():.0f}", transform=ax.transAxes, ha="right", fontsize=6.5)
    save(fig, "nx_rank_ecdf")


def scatter(d: dict) -> None:
    s1, hb = d["s1_ap"], d["hub_ap_K30"]
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    hb2 = ax.hexbin(np.sqrt(s1), np.sqrt(hb), gridsize=38, bins="log", cmap="Blues", mincnt=1, extent=(0, 1, 0, 1), linewidths=0.1)
    ax.plot([0, 1], [0, 1], color=INK, lw=0.7, ls=(0, (3, 2)))
    up, dn = (hb > s1 + 1e-9).mean() * 100, (hb < s1 - 1e-9).mean() * 100
    ax.text(0.04, 0.93, f"improved {up:.1f}%", fontsize=7, color=GREEN, fontweight="bold")
    ax.text(0.04, 0.86, f"worsened {dn:.1f}%", fontsize=7, color=PINK, fontweight="bold")
    ax.text(0.04, 0.79, f"unchanged {100 - up - dn:.1f}%", fontsize=7, color=GREY)
    ax.set_xlabel(r"Stage-1 AP ($\sqrt{\cdot}$ scale)"); ax.set_ylabel(r"hub-corrected hybrid AP at $K=30$ ($\sqrt{\cdot}$ scale)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cb = plt.colorbar(hb2, ax=ax, fraction=0.045, pad=0.02); cb.set_label("queries (log)", fontsize=7); cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    save(fig, "nx_scatter")


def gain_by_k(d: dict) -> None:
    wid = d["query_wid"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    x = np.arange(len(KS))
    for key, col, lab, off in (("hyb", BLUE, "hybrid", -0.07), ("hub", ORANGE, "hybrid + hub correction", 0.07)):
        m, lo, hi = zip(*[work_boot(d[f"{key}_ap_K{k}"] - d["s1_ap"], wid, seed=k) for k in KS])
        m, lo, hi = map(np.array, (m, lo, hi))
        ax.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt="-o", color=col, ms=3, lw=1.3, capsize=1.8, elinewidth=0.8, label=lab)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x, [str(k) for k in KS]); ax.set_xlabel("shortlist size $K$"); ax.set_ylabel(r"$\Delta$MAP over Stage 1")
    ax.legend(loc="upper left")
    save(fig, "nx_gain_by_k")


if __name__ == "__main__":
    data = load()
    teaser(data); rank_ecdf(data); scatter(data); gain_by_k(data)
