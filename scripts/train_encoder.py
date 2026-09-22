"""Train the TCN global encoder on training-split WIDs (validation WIDs select the checkpoint).

    python scripts/train_encoder.py --config configs/hybrid_dev.yaml

Writes the checkpoint to ``runs/<run_name>/encoder_best.pt`` (git-ignored) and the
training history / summary to ``reports/results/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cover_retrieval.data.manifests import load_split_tracks
from cover_retrieval.evaluation.analysis import plot_training_history
from cover_retrieval.models.training import train_encoder
from cover_retrieval.pipeline import figures_dir, results_dir, run_dir, split_store
from cover_retrieval.utils.io import git_commit, load_config, parse_overrides, write_csv, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_dev.yaml"))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set training.lr=0.003 --set encoder.run_name=lr3e-3",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config, parse_overrides(args.set))
    manifests = config["paths"]["manifests_dir"]

    train_tracks = load_split_tracks(manifests, "train")
    val_tracks = load_split_tracks(manifests, "validation")
    held_out = {
        t.wid for role in ("calibration", "query", "distractor") for t in load_split_tracks(manifests, role)
    }
    leaked = held_out & ({t.wid for t in train_tracks} | {t.wid for t in val_tracks})
    if leaked:
        raise SystemExit(f"leakage: evaluation WIDs in train/validation: {sorted(leaked)[:5]}")

    train_store = split_store(config, train_tracks, "train")
    val_store = split_store(config, val_tracks, "validation")
    out_dir = run_dir(config, config["encoder"].get("run_name", "encoder"))
    summary = train_encoder(train_store, val_store, config, out_dir)

    history = summary.pop("history")
    results = results_dir(config)
    run_name = config["encoder"].get("run_name", "encoder")
    write_csv(results / f"training_history_{run_name}.csv", history, list(history[0]))
    write_json(
        results / f"training_summary_{run_name}.json",
        {
            **summary,
            "checkpoint": str(out_dir / "encoder_best.pt"),
            "git_commit": git_commit(),
            "config_path": str(args.config),
            "overrides": args.set,
            "encoder": config["encoder"],
            "training": config["training"],
        },
    )
    plot_training_history(history, figures_dir(config) / f"training_history_{run_name}.png")
    print(f"best validation MAP {summary['best_val_map']:.4f} at epoch {summary['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
