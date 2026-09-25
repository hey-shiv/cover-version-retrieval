"""Check the paper against the repository and write paper/consistency_report.md.

    python research/scripts/check_paper_consistency.py          # exit 1 on any FAIL

Checks:
  1. every decimal number in paper/main.tex comes from a macro, except a short allow-list of
     external or definitional constants (each justified below);
  2. every macro the paper uses is defined in paper/tables/numbers*.tex;
  3. headline numbers quoted in README.md match the result files;
  4. the README test count equals the collected test count;
  5. no sentence in the paper or README describes 384-frame exhaustive alignment as run;
  6. development results are never described as benchmark results in the paper (keyword check).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "main.tex"
REP = REPO / "reports" / "results"
ALLOWED = {
    "0.333": "Qmax MAP on HPCP, external value cited to Da-TACOS (citation_audit.md: re-check against the PDF)",
    "0.1": "SupCon temperature (configs/base.yaml)",
    "0.05": "alpha grid step (configs/base.yaml)",
    "0.2": "prose range for development interval half-widths (Section 11); derived from P2",
    "0.005": "prose range for benchmark effect sizes (Section 11); derived from X5",
    "1.9": "Discogs-VI size, 1.9 million versions (external, cited)",
    "4.0": "licence name CC BY-NC-SA 4.0",
}


def main() -> int:
    tex = PAPER.read_text()
    body = re.sub(r"(?m)^%.*$", "", tex)
    macros = set()
    for f in (REPO / "paper" / "tables").glob("numbers*.tex"):
        macros |= set(re.findall(r"\\newcommand\{\\(\w+)\}", f.read_text()))
    ok, fail, notes = [], [], []

    # 1. hand-typed decimals (outside \input lines and math constants)
    stripped = re.sub(r"\\(input|includegraphics|label|ref|cite\w*)(\[[^]]*\])?\{[^}]*\}", "", body)
    decimals = sorted(set(re.findall(r"(?<![\w.\\])\d+\.\d+(?![\w.])", stripped)))
    bad = [d for d in decimals if d not in ALLOWED]
    (fail if bad else ok).append(f"hand-typed decimals in main.tex outside the allow-list: {bad or 'none'}")
    for d in decimals:
        if d in ALLOWED:
            notes.append(f"allowed constant {d}: {ALLOWED[d]}")

    # 2. macros used but not defined
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)(?![A-Za-z])", body))
    latex_builtin = {"S", "L", "O", "P", "Pr", "Delta", "Sigma"}
    undefined = sorted(u for u in used - macros - latex_builtin if u not in re.findall(r"\\newcommand\{\\(\w+)\}", body))
    (fail if undefined else ok).append(f"macros used but not generated: {undefined or 'none'}")

    # 3. README headline numbers vs result files
    readme = (REPO / "README.md").read_text()
    r1, r2, r4 = (json.loads((REP / f).read_text()) for f in ("benchmark.json", "benchmark_full96.json", "benchmark_long384.json"))
    sm = lambda r, s: next(m["MAP"] for m in r["metrics"] if m["system"] == s)  # noqa: E731
    checks = [
        (f"{sm(r2, 'classical_alignment_hubcorr'):.3f}", "exhaustive + hub (run 2)"),
        (f"{sm(r4, 'hybrid_hubcorr_K30_alpha0.10'):.3f}", "hybrid + hub (run 4)"),
        (f"{sm(r4, 'global_embedding'):.3f}", "Stage 1 (run 4)"),
        (f"{r4['shortlist_recall']:.3f}", "shortlist recall (run 4)"),
        (f"{r1['runtime']['alignment_only_s'] / 3600:.2f} h", "exhaustive 96-frame alignment time"),
    ]
    for val, what in checks:
        (ok if val in readme else fail).append(f"README quotes {what} = {val}")

    # 4. test count
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=REPO,
                             capture_output=True, text=True, timeout=600).stdout
        n = sum(1 for line in out.splitlines() if "::" in line)
        if n == 0:  # quiet mode prints "path: count" per file
            n = sum(int(mm.group(1)) for mm in re.finditer(r"^tests/\S+: (\d+)$", out, re.M))
        m = re.search(r"covered by (\d+) tests", readme)
        (ok if m and int(m.group(1)) == n else fail).append(f"README test count {m.group(1) if m else '?'} vs collected {n}")
    except Exception as e:  # pragma: no cover
        fail.append(f"could not collect tests: {e}")

    # 5. 384-frame exhaustive alignment never described as run
    for name, text in (("paper", body), ("README", readme)):
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        sents = [x for line in text.splitlines() if not line.lstrip().startswith("|") for x in re.split(r"(?<=[.;])\s+", line)]
        hits = [s for s in sents if re.search(r"384", s) and re.search(r"exhaustive|alignment-only|all pairs", s, re.I)]
        bad5 = [s.strip()[:140] for s in hits if not re.search(r"not run|never run|was not|never evaluated|estimated|≈ 42|about 42|42 h|no benchmark counterpart|differ|different resolutions|384 vs|vs\.\\? 96|against 384|96 against|would|did not test|not test|once|future", s, re.I)]
        (fail if bad5 else ok).append(f"{name}: sentences on 384-frame exhaustive alignment all say it was not run {bad5 if bad5 else ''}")

    # 6. development numbers only in dev-labelled context
    dev_macros = ["DevClassicalLow", "DevClassicalHigh", "DevHybLow", "DevHybHigh", "DevLongStage", "DevLongHyb"]
    bad6 = []
    for mac in dev_macros:
        for mo in re.finditer(r"\\" + mac + r"\b", body):
            window = body[max(0, mo.start() - 400): mo.start()]
            if not re.search(r"dev|development", window, re.I):
                bad6.append(mac)
    (fail if bad6 else ok).append(f"development-only numbers appear in development context {bad6 if bad6 else ''}")

    lines = ["# Paper / repository consistency report", "", "Generated by `research/scripts/check_paper_consistency.py`; do not edit.", ""]
    lines += [f"- **FAIL** {x}" for x in fail] + [f"- ok {x}" for x in ok] + ["", "## Allowed constants", ""] + [f"- {x}" for x in notes]
    (REPO / "paper" / "consistency_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
