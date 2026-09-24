"""Decode committed report figures back into small numeric grids for the website.

The Da-TACOS HPCP features are CC BY-NC-SA and are never committed, so the website
cannot load them. What *is* committed are the matplotlib figures in
``reports/figures/``, rendered by ``src/cover_retrieval/evaluation/analysis.py``:

* HPCP matrices: ``imshow(..., cmap="magma", origin="lower")``, per-panel min/max colour scale;
* cost matrices: ``imshow(..., cmap="viridis_r", vmin=p1, vmax=p99, origin="lower")`` with the
  DTW path drawn on top in pure red;
* rotation scores: 12 bars of profile cosine, y axis from 0 to 1.

This script inverts the colour maps (nearest colour in the 256-entry LUT) and reads
bar heights, so every value it writes is *relative* and quantised: a colour-scale
position in [0, 1], not the original feature value. The website says so wherever it
shows them. Output: ``web/src/data/generated/figures.json``.

Run from the repository root:  python web/scripts/extract_figures.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
OUT = ROOT / "web" / "src" / "data" / "generated" / "figures.json"


def load(name: str) -> np.ndarray:
    return np.asarray(Image.open(FIGURES / name).convert("RGB")).astype(np.int32)


def axes_boxes(img: np.ndarray, min_run: int = 150) -> list[tuple[int, int, int, int]]:
    """Axes rectangles (x0, x1, y_top, y_bottom) found from the black spines, left to right."""
    dark = img.sum(axis=2) < 150
    found: dict[tuple[int, int], list[int]] = {}
    for y in range(dark.shape[0]):
        row = dark[y]
        x = 0
        while x < row.size:
            if row[x]:
                s = x
                while x < row.size and row[x]:
                    x += 1
                if x - s > min_run:
                    found.setdefault((s, x), []).append(y)
            x += 1
    # a real axes box has its left and right spine running between the two horizontal spines
    boxes = []
    for (x0, x1), ys in found.items():
        top, bottom = min(ys), max(ys)
        if bottom - top < 40:
            continue
        lx = max(range(x0, min(x0 + 5, x1)), key=lambda x: dark[top:bottom, x].mean())
        rx = max(range(max(x1 - 5, x0), x1), key=lambda x: dark[top:bottom, x].mean())
        if dark[top:bottom, lx].mean() > 0.95 and dark[top:bottom, rx].mean() > 0.95:
            boxes.append((lx, rx + 1, top, bottom))
    boxes.sort()
    # drop boxes nested in another (dark magma rows can mimic spines)
    return [b for b in boxes if not any(o != b and o[0] <= b[0] and o[1] >= b[1] and o[2] <= b[2] and o[3] >= b[3] for o in boxes)]


def lut(name: str) -> np.ndarray:
    cmap = colormaps[name]
    return np.array([cmap(i / 255)[:3] for i in range(256)]) * 255


def invert(pixels: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Nearest LUT index for each pixel -> value in [0, 1]."""
    flat = pixels.reshape(-1, 3)[:, None, :]
    d = ((flat - table[None, :, :]) ** 2).sum(axis=2)
    return (d.argmin(axis=1) / 255.0).reshape(pixels.shape[:-1])


def is_red(px: np.ndarray) -> np.ndarray:
    return (px[..., 0] > 180) & (px[..., 1] < 110) & (px[..., 2] < 110)


def cell_centres(lo: int, hi: int, n: int) -> np.ndarray:
    """Pixel centres of n cells spanning the interior between two spine pixels."""
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


def sample_matrix(img, box, rows: int, cols: int, cmap: str, reverse: bool = False):
    x0, x1, top, bottom = box
    xs = np.round(cell_centres(x0, x1 - 1, cols)).astype(int)
    ys = np.round(cell_centres(top, bottom, rows)).astype(int)[::-1]  # origin="lower"
    patch = img[np.ix_(ys, xs)]
    values = invert(patch, lut(cmap))
    if reverse:
        values = 1.0 - values
    red = is_red(patch)
    if red.any():  # in-paint cells hidden under the drawn path from their horizontal neighbours
        for i, j in zip(*np.nonzero(red)):
            neigh = [values[i, k] for k in (j - 2, j - 1, j + 1, j + 2) if 0 <= k < cols and not red[i, k]]
            values[i, j] = float(np.mean(neigh)) if neigh else values[i, j]
    return values, red


def red_path(img, box, n: int) -> list[list[float]]:
    """DTW path as drawn: for each query frame i, the mean candidate position of red pixels."""
    x0, x1, top, bottom = box
    region = is_red(img[top + 1 : bottom, x0 + 1 : x1 - 1])
    path = []
    height = bottom - top
    width = x1 - 1 - x0
    for i in range(n):
        yc = bottom - (i + 0.5) * height / n
        y_lo, y_hi = int(yc - height / n / 2) - top - 1, int(yc + height / n / 2) - top - 1
        band = region[max(y_lo, 0) : max(y_hi, 1)]
        cols = np.nonzero(band.any(axis=0))[0]
        if cols.size:
            j = (cols.mean() + 1) * n / width - 0.5
            path.append([i, round(float(j), 2)])
    return path


def bars(img, box) -> list[float]:
    """Heights of 12 bars on a 0..1 y axis (x axis at -0.5..11.5 + matplotlib margins)."""
    x0, x1, top, bottom = box
    # matplotlib bar default: xlim = (-0.5 - 0.6*? ) -> use detected non-white columns instead
    region = img[top + 2 : bottom - 1, x0 + 2 : x1 - 2]
    coloured = (np.abs(region - 255).sum(axis=2) > 60)
    cols = coloured.any(axis=0)
    # group contiguous coloured columns into bars
    groups, start = [], None
    for k, c in enumerate(list(cols) + [False]):
        if c and start is None:
            start = k
        elif not c and start is not None:
            groups.append((start, k))
            start = None
    groups = [g for g in groups if g[1] - g[0] > 5]
    assert len(groups) == 12, f"expected 12 bars, found {len(groups)}"
    height = bottom - top
    out = []
    for a, b in groups:
        mid = (a + b) // 2
        ys = np.nonzero(coloured[:, mid])[0]
        top_px = ys.min() + top + 2
        out.append(round(float((bottom - top_px) / height), 3))
    return out


def packed(m: np.ndarray) -> dict[str, object]:
    """Colour-scale positions are 256-level LUT indices, so uint8 + base64 is lossless."""
    q = np.clip(np.round(m * 255), 0, 255).astype(np.uint8)
    return {"shape": list(q.shape), "u8": base64.b64encode(q.tobytes()).decode("ascii")}


def main() -> None:
    result: dict[str, object] = {
        "_note": "Generated by web/scripts/extract_figures.py from reports/figures/*.png. "
        "Values are colour-scale positions recovered by colour-map inversion (uint8 / 255, row-major, base64), not raw features.",
    }

    # --- HPCP pairs (12 x 96, magma, per-panel scale) --------------------------------
    hpcp = {}
    for fig, titles in [
        ("calibration_cover_pairs_hpcp.png", ["P_510723", "P_77960", "P_242247", "P_476324"]),
        ("calibration_noncover_pairs_hpcp.png", ["P_510723", "P_355091", "P_242247", "P_130947"]),
    ]:
        img = load(fig)
        boxes = axes_boxes(img)
        assert len(boxes) == 4, (fig, boxes)
        for pid, box in zip(titles, boxes):
            values, _ = sample_matrix(img, box, 12, 96, "magma")
            hpcp.setdefault(pid, {"source": f"reports/figures/{fig}", "values": packed(values)})
    result["hpcp"] = hpcp

    # --- raw vs preprocessed (profile figure) ----------------------------------------
    img = load("profile_raw_vs_resampled.png")
    boxes = axes_boxes(img)
    assert len(boxes) == 4, boxes
    raw_w = boxes[0][1] - boxes[0][0] - 2
    raw, _ = sample_matrix(img, boxes[0], 12, raw_w, "magma")
    pre, _ = sample_matrix(img, boxes[1], 12, 96, "magma")
    result["profile"] = {
        "source": "reports/figures/profile_raw_vs_resampled.png",
        "pid": "P_130947",
        "raw_frames": 15164,
        "raw_pixels": packed(raw),
        "preprocessed": packed(pre),
    }

    # --- rotation scores (bars) ------------------------------------------------------
    img = load("calibration_rotation_scores.png")
    boxes = axes_boxes(img)
    assert len(boxes) == 4, boxes
    pairs = [
        ("cover", "P_510723", "P_77960", 11),
        ("cover", "P_242247", "P_476324", 7),
        ("non-cover", "P_510723", "P_355091", 0),
        ("non-cover", "P_242247", "P_130947", 11),
    ]
    rotations = []
    for (kind, q, c, best), box in zip(pairs, boxes):
        scores = bars(img, box)
        assert int(np.argmax(scores)) == best, (q, c, scores)
        rotations.append({"kind": kind, "query": q, "candidate": c, "best_shift": best, "profile_cosine": scores})
    result["rotations"] = {"source": "reports/figures/calibration_rotation_scores.png", "pairs": rotations}

    # --- cost matrices with DTW path -------------------------------------------------
    alignments = {}
    specs = [
        ("error_success_P_797406.png", [
            ("P_797406|P_797408", "cover", 0, 0.021),
            ("P_797406|P_31055", "non-cover", 5, 0.036),
        ]),
        ("calibration_alignment_paths.png", [
            ("P_510723|P_77960", "cover", 11, 0.124),
            ("P_242247|P_476324", "cover", 7, 0.042),
            ("P_510723|P_355091", "non-cover", 0, 0.109),
            ("P_242247|P_130947", "non-cover", 11, 0.109),
        ]),
    ]
    for fig, items in specs:
        img = load(fig)
        boxes = axes_boxes(img)
        assert len(boxes) == len(items), (fig, boxes)
        for (key, kind, shift, cost), box in zip(items, boxes):
            values, _ = sample_matrix(img, box, 96, 96, "viridis", reverse=True)
            alignments[key] = {
                "source": f"reports/figures/{fig}",
                "kind": kind,
                "shift": shift,
                "published_cost": cost,
                "cost": packed(values),
                "published_path": red_path(img, box, 96),
            }
    result["alignments"] = alignments

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
