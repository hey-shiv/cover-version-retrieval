"""Validate local experiment outputs before anything analyses them.

    python research/scripts/validate_results.py                 # research/results/, strict
    python research/scripts/validate_results.py --root runs/smoke/research --synthetic

Checks, per experiment directory present:
  * required files exist and parse; evidence label is what the run type claims
  * per-query tables have one row per benchmark query (13,000 unless --synthetic),
    unique query ids, no missing values
  * ranks >= 1, AP in [0, 1], coverage non-decreasing in K, first rank >= 1
  * A1: the reproduction self-check against the frozen benchmark file PASSED
  * A1: at every K the hybrid first rank equals the Stage-1 first rank whenever no
    cover is in the shortlist (the structural property the whole analysis relies on)
Exit code 1 on any failure, so CI and the analysis scripts can refuse bad inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrlib  # noqa: E402

N_BENCH = 13000
REQUIRED = {
    "A1": ["metrics.json", "per_query.csv", "signals.csv", "candidates.csv", "runtime.json", "config.yaml", "env.json", "README.md"],
    "F1": ["metrics.json", "per_query.csv", "runtime.json", "config.yaml", "env.json", "README.md"],
    "H1": ["metrics.json", "per_query.csv", "runtime.json", "config.yaml", "env.json", "README.md"],
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.ok: list[str] = []

    def check(self, cond: bool, msg: str) -> None:
        (self.ok if cond else self.errors).append(msg)


def rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def common(rep: Report, d: Path, exp: str, synthetic: bool) -> list[dict] | None:
    for f in REQUIRED[exp]:
        rep.check((d / f).exists(), f"{d.name}: has {f}")
    if not (d / "metrics.json").exists() or not (d / "per_query.csv").exists():
        return None
    m = json.loads((d / "metrics.json").read_text())
    want = "SYNTHETIC-MECHANICS" if synthetic else "LOCAL-FULL"
    rep.check(m.get("evidence") == want, f"{d.name}: evidence label is {want} (got {m.get('evidence')})")
    pq = rows(d / "per_query.csv")
    if not synthetic:
        rep.check(len(pq) == N_BENCH, f"{d.name}: {N_BENCH} query rows (got {len(pq)})")
    rep.check(len({r['query_pid'] for r in pq}) == len(pq), f"{d.name}: unique query ids")
    rep.check(all(v not in ("", None) for r in pq for v in r.values()), f"{d.name}: no missing values")
    return pq


def a1(rep: Report, d: Path, synthetic: bool) -> None:
    pq = common(rep, d, "A1", synthetic)
    if pq is None:
        return
    m = json.loads((d / "metrics.json").read_text())
    ks = [e["K"] for e in m["by_k"]]
    rep.check(30 in ks, f"{d.name}: K = 30 (the locked K) is in the sweep")
    rep.check(m.get("reproduction", {}).get("pass") is True, f"{d.name}: reproduces the frozen benchmark at the locked K")
    cov = [e["coverage"] for e in m["by_k"]]
    rep.check(all(b >= a - 1e-12 for a, b in zip(cov, cov[1:])), f"{d.name}: coverage non-decreasing in K")
    bad_struct = bad_range = 0
    for r in pq:
        s1 = int(r["s1_first"])
        ranks = rrlib.str_to_ranks(r["s1_rel_ranks"])
        if ranks.size != int(r["n_rel"]) or ranks.min() < 1:
            bad_range += 1
        for k in ks:
            for s in ("hyb", "rr", "hub"):
                f, ap = int(r[f"{s}_first_K{k}"]), float(r[f"{s}_ap_K{k}"])
                if f < 1 or not (0.0 <= ap <= 1.0) or math.isnan(ap):
                    bad_range += 1
                if int(r[f"cov_K{k}"]) == 0 and f != s1:
                    bad_struct += 1
    rep.check(bad_range == 0, f"{d.name}: ranks >= 1 and AP in [0, 1] everywhere ({bad_range} violations)")
    rep.check(bad_struct == 0, f"{d.name}: no-cover queries keep their Stage-1 first rank at every K ({bad_struct} violations)")
    sig = rows(d / "signals.csv")
    rep.check([r["query_pid"] for r in sig] == [r["query_pid"] for r in pq], f"{d.name}: signals.csv aligned with per_query.csv")
    rep.check(all(math.isfinite(float(v)) for r in sig for k, v in r.items() if k not in ("query_pid", "query_wid")), f"{d.name}: signals finite")


def f1(rep: Report, d: Path, synthetic: bool) -> None:
    pq = common(rep, d, "F1", synthetic)
    if pq is None:
        return
    cols = [c for c in pq[0] if c.endswith("_rel_ranks")]
    rep.check(any(c.startswith("global:") for c in cols), f"{d.name}: at least one global:<checkpoint> variant")
    bad = sum(1 for r in pq for c in cols if rrlib.str_to_ranks(r[c]).size != int(r["n_rel"]))
    rep.check(bad == 0, f"{d.name}: every variant ranks every relevant item ({bad} violations)")


def h1(rep: Report, d: Path, synthetic: bool) -> None:
    pq = common(rep, d, "H1", synthetic)
    if pq is None:
        return
    aps = [c for c in pq[0] if c.endswith("_ap")]
    bad = sum(1 for r in pq for c in aps if not 0.0 <= float(r[c]) <= 1.0)
    rep.check(bad == 0, f"{d.name}: AP in [0, 1] ({bad} violations)")
    bad = sum(1 for r in pq if int(r["cov"]) == 0 and any(int(r[c.replace('_ap', '_first')]) != int(r["s1_first"]) for c in aps if c != "s1_ap"))
    rep.check(bad == 0, f"{d.name}: no-cover queries keep their Stage-1 first rank ({bad} violations)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=rrlib.REPO / "research" / "results")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args(argv)
    rep = Report()
    found = 0
    for d in sorted(args.root.glob("*")):
        prefix = d.name.split("_")[0]
        fn = {"A1": a1, "F1": f1, "H1": h1}.get(prefix)
        if fn and d.is_dir():
            found += 1
            fn(rep, d, args.synthetic)
    for msg in rep.ok:
        print(f"  ok    {msg}")
    for msg in rep.errors:
        print(f"  FAIL  {msg}")
    print(f"{found} local experiment(s) checked: {len(rep.ok)} ok, {len(rep.errors)} failed")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
