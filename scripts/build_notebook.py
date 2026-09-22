"""Generate notebooks/01_feature_and_alignment_inspection.ipynb (library calls only).

    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_feature_and_alignment_inspection.ipynb

Set ``CONFIG = "configs/smoke.yaml"`` in the first cell (after running
``scripts/run_smoke.sh``) to explore the synthetic fixture instead of Da-TACOS.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = [
    (
        "md",
        "# Feature and alignment inspection\n\n"
        "Visual checks of the classical stage on **calibration WIDs only** (never on development or "
        "benchmark queries). Every computation is a call into `cover_retrieval`; the notebook "
        "contains no pipeline logic of its own.\n\n"
        "1. raw vs. preprocessed HPCP for two cover pairs and two non-cover pairs\n"
        "2. the 12 key-rotation scores\n"
        "3. cross-similarity (cosine-distance) matrices with subsequence-DTW paths",
    ),
    (
        "code",
        "import os\nfrom pathlib import Path\n\n"
        "# run from the repository root regardless of where Jupyter was started\n"
        "if Path.cwd().name == 'notebooks':\n    os.chdir('..')\n\n"
        "CONFIG = 'configs/base.yaml'   # or 'configs/smoke.yaml' for the synthetic fixture\n\n"
        "import matplotlib.pyplot as plt\nimport numpy as np\n\n"
        "from cover_retrieval.data.datacos import feature_path, features_root, load_hpcp\n"
        "from cover_retrieval.data.manifests import load_protocol\n"
        "from cover_retrieval.evaluation.analysis import plot_alignment\n"
        "from cover_retrieval.features.hpcp import to_pitch_by_time\n"
        "from cover_retrieval.pipeline import pair_diagnostics, protocol_store\n"
        "from cover_retrieval.utils.io import load_config\n\n"
        "config = load_config(CONFIG)\n"
        "protocol = load_protocol(config['paths']['manifests_dir'], 'calibration')\n"
        "store = protocol_store(config, protocol, 'coveranalysis')\n"
        "by_wid = {}\nfor t in protocol.queries:\n    by_wid.setdefault(t.wid, []).append(t)\n"
        "wids = sorted(by_wid)\n"
        "covers = [tuple(by_wid[w]) for w in wids[:2]]\n"
        "noncovers = [(by_wid[wids[0]][0], by_wid[wids[2]][0]), (by_wid[wids[1]][0], by_wid[wids[3]][0])]\n"
        "pairs = [('cover', a, b) for a, b in covers] + [('non-cover', a, b) for a, b in noncovers]\n"
        "print(len(protocol.queries), 'calibration tracks;', [(k, a.pid, b.pid) for k, a, b in pairs])",
    ),
    (
        "md",
        "## 1. Raw vs. preprocessed HPCP\n\nRaw Da-TACOS HPCP is `(T, 12)`, max-normalised per frame; the classical view is `(12, 96)`: frame-L2 → area resampling → frame-L2.",
    ),
    (
        "code",
        "root = features_root(config, 'coveranalysis')\n"
        "fig, axes = plt.subplots(len(pairs), 4, figsize=(16, 2.6 * len(pairs)))\n"
        "for row, (kind, a, b) in enumerate(pairs):\n"
        "    for col, t in enumerate((a, b)):\n"
        "        raw = to_pitch_by_time(load_hpcp(feature_path(root, t.wid, t.pid)).hpcp)\n"
        "        axes[row, 2 * col].imshow(raw, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')\n"
        "        axes[row, 2 * col].set_title(f'{kind}: {t.pid} raw ({raw.shape[1]} frames)', fontsize=9)\n"
        "        axes[row, 2 * col + 1].imshow(store.get('classical', [t.pid])[0], aspect='auto', origin='lower', cmap='magma')\n"
        "        axes[row, 2 * col + 1].set_title(f'{t.pid} preprocessed (12 x 96)', fontsize=9)\n"
        "plt.tight_layout()",
    ),
    (
        "md",
        "## 2. Key rotation scores\n\nCosine between time-averaged chroma profiles for each of the 12 candidate rotations; the arg-max is the rotation used for alignment.",
    ),
    (
        "code",
        "diags = [pair_diagnostics(config, *store.get('classical', [a.pid, b.pid])) for _, a, b in pairs]\n"
        "fig, axes = plt.subplots(1, len(pairs), figsize=(16, 2.8))\n"
        "for ax, (kind, a, b), d in zip(axes, pairs, diags):\n"
        "    ax.bar(range(12), d.rotation_scores, color=['tab:red' if k == d.shift else 'tab:gray' for k in range(12)])\n"
        "    ax.set_title(f'{kind}: best shift {d.shift}', fontsize=9)\n"
        "    ax.set_ylim(0, 1)\n"
        "plt.tight_layout()",
    ),
    (
        "md",
        "## 3. Cross-similarity matrices and subsequence-DTW paths\n\nCosine distance between query frames (rows) and rotated candidate frames (columns); red = optimal slope-constrained subsequence path. Lower normalised cost = better match.",
    ),
    (
        "code",
        "fig, axes = plt.subplots(1, len(pairs), figsize=(18, 4.2))\n"
        "for ax, (kind, a, b), d in zip(axes, pairs, diags):\n"
        "    plot_alignment(d.cost, d.dtw.path, f'{kind}: {a.pid} vs {b.pid}\\ncost {d.dtw.normalized_cost:.3f}', None, ax=ax)\n"
        "plt.tight_layout()\n"
        "for (kind, a, b), d in zip(pairs, diags):\n"
        "    print(f'{kind:9s} {a.pid:>10s} vs {b.pid:>10s}  shift={d.shift:2d}  normalised cost={d.dtw.normalized_cost:.4f}')",
    ),
    (
        "md",
        "### Reading the plots\n\n"
        "* Low cost is bright yellow (reversed viridis, per-matrix colour range). A faithful cover shows a low-cost diagonal band that the path follows.\n"
        "* At this resolution many matrices are dominated by horizontal and vertical stripes (frames that are tonally central or atypical for the whole track) rather than diagonals.\n"
        "* At 96 frames per track each column averages a large stretch of raw frames, so the band is coarse; "
        "see the resolution sensitivity analysis in `reports/classical_baseline.md`.\n"
        "* A low cost for a non-cover (shared common progression) is exactly the false-positive mechanism discussed in `reports/error_analysis.md`.",
    ),
]


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = Path("notebooks/01_feature_and_alignment_inspection.ipynb")
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
