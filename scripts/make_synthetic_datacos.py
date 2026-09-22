"""Write the synthetic Da-TACOS-shaped fixture used by the smoke pipeline.

python scripts/make_synthetic_datacos.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cover_retrieval.data.synthetic import make_synthetic_datacos
from cover_retrieval.utils.io import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    root = Path(config["paths"]["datacos_root"])
    if "smoke" not in str(root) and "synthetic" not in str(root):
        raise SystemExit(f"refusing to write synthetic data into {root}: use a smoke/synthetic path")
    counts = make_synthetic_datacos(
        root,
        features_dirs=(
            config["data"]["coveranalysis_features_dir"],
            config["data"]["benchmark_features_dir"],
        ),
        metadata_dir=config["data"]["metadata_dir"],
        seed=args.seed,
    )
    print(f"synthetic fixture written to {root}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
