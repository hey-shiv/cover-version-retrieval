"""Deterministic encoder training on training-split WIDs only.

Batches contain ``batch_wids`` works x both of their recordings, so every anchor has
exactly one positive (same WID) and ``2 * (batch_wids - 1)`` negatives (other WIDs).
Validation WIDs are disjoint from training WIDs and are only used to choose the
checkpoint (retrieval MAP on the validation pair protocol) and for early stopping.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cover_retrieval.data.manifests import Protocol
from cover_retrieval.features.preprocessing import FeatureStore
from cover_retrieval.models.losses import batch_hard_triplet_loss, supervised_contrastive_loss
from cover_retrieval.models.tcn_encoder import EncoderConfig, TCNEncoder, count_parameters, prepare_input
from cover_retrieval.retrieval.rank import rank_protocol, summarize_ranked
from cover_retrieval.utils.seed import resolve_device, seed_everything


@dataclass
class TrainSettings:
    device: str = "cpu"
    epochs: int = 60
    batch_wids: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.1
    patience: int = 12
    crop_min_fraction: float = 0.6
    pitch_shift_augment: bool = True
    num_threads: int = 8
    loss: str = "supcon"  # supcon | triplet
    triplet_margin: float = 0.2
    seed: int = 20260817

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TrainSettings:
        known = set(cls.__dataclass_fields__)
        values = {k: v for k, v in config["training"].items() if k in known}
        values.setdefault("seed", int(config["seed"]))
        return cls(**values)


def pair_batches(wids: list[str], batch_wids: int, rng: np.random.Generator) -> list[list[int]]:
    """Shuffle track indices grouped by WID into batches of whole works."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, wid in enumerate(wids):
        groups[wid].append(index)
    usable = sorted(w for w, idx in groups.items() if len(idx) >= 2)
    order = rng.permutation(len(usable))
    batches = []
    for start in range(0, len(order), batch_wids):
        chunk = [usable[i] for i in order[start : start + batch_wids]]
        indices = [i for w in chunk for i in groups[w]]
        if len(chunk) >= 2 or not batches:
            batches.append(indices)
        else:  # a lone trailing work has no negatives: merge it into the previous batch
            batches[-1].extend(indices)
    return [b for b in batches if len({wids[i] for i in b}) >= 2]


def augment(
    x: torch.Tensor, input_frames: int, crop_min: float, pitch_shift: bool, gen: torch.Generator
) -> torch.Tensor:
    """Random time crop (fraction in [crop_min, 1]) + random cyclic pitch rotation per item."""
    out = []
    length = x.shape[-1]
    for item in x:
        fraction = crop_min + (1.0 - crop_min) * torch.rand(1, generator=gen).item()
        crop = max(input_frames // 2, int(round(fraction * length)))
        start = int(torch.randint(0, length - crop + 1, (1,), generator=gen).item())
        segment = item[:, start : start + crop]
        if pitch_shift:
            segment = torch.roll(segment, int(torch.randint(0, 12, (1,), generator=gen).item()), dims=0)
        out.append(F.adaptive_avg_pool1d(segment.unsqueeze(0), input_frames).squeeze(0))
    return F.normalize(torch.stack(out), dim=1, eps=1e-8)


@torch.no_grad()
def embed(
    model: TCNEncoder,
    features: np.ndarray,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
    rotation: int = 0,
) -> np.ndarray:
    """Embed ``(N, 12, T)`` features (optionally pitch-rotated) -> ``(N, D)`` float32."""
    model.eval()
    device = torch.device(device)
    chunks = []
    for start in range(0, features.shape[0], batch_size):
        x = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
        if rotation:
            x = torch.roll(x, rotation, dims=1)
        chunks.append(model(prepare_input(x, model.cfg.input_frames)).cpu().numpy())
    return (
        np.concatenate(chunks).astype(np.float32)
        if chunks
        else np.zeros((0, model.cfg.embedding_dim), np.float32)
    )


def validation_metrics(model: TCNEncoder, store: FeatureStore, device: torch.device) -> dict[str, float]:
    emb = embed(model, store.views["encoder"], device)
    protocol = Protocol("validation", store.tracks, store.tracks)
    summary = summarize_ranked(rank_protocol(emb @ emb.T, protocol))
    return {"val_map": summary.map, "val_mrr": summary.mrr, "val_hit@10": summary.hit[10]}


def train_encoder(
    train_store: FeatureStore,
    val_store: FeatureStore,
    config: dict[str, Any],
    out_dir: str | Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train, select the best checkpoint by validation MAP, and write artefacts to ``out_dir``."""
    settings = TrainSettings.from_config(config)
    enc_cfg = EncoderConfig.from_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(settings.seed, deterministic=True, num_threads=settings.num_threads)
    device = resolve_device(settings.device)

    overlap = set(train_store.wids) & set(val_store.wids)
    if overlap:
        raise ValueError(f"train/validation WID leakage: {sorted(overlap)[:5]}")

    model = TCNEncoder(enc_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    rng = np.random.default_rng(settings.seed)
    gen = torch.Generator().manual_seed(settings.seed)
    labels_all = torch.as_tensor(np.unique(train_store.wids, return_inverse=True)[1])
    features = torch.as_tensor(train_store.views["encoder"])

    history: list[dict[str, float]] = []
    best = {"val_map": -1.0, "epoch": -1}
    stale = 0
    start_time = time.monotonic()
    for epoch in range(1, settings.epochs + 1):
        model.train()
        epoch_start = time.monotonic()
        losses = []
        for batch in pair_batches(train_store.wids, settings.batch_wids, rng):
            idx = torch.as_tensor(batch)
            x = augment(
                features[idx],
                enc_cfg.input_frames,
                settings.crop_min_fraction,
                settings.pitch_shift_augment,
                gen,
            )
            embeddings = model(x.to(device))
            labels = labels_all[idx].to(device)
            if settings.loss == "supcon":
                loss = supervised_contrastive_loss(embeddings, labels, settings.temperature)
            elif settings.loss == "triplet":
                loss = batch_hard_triplet_loss(embeddings, labels, settings.triplet_margin)
            else:
                raise ValueError(f"unknown loss {settings.loss!r}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = validation_metrics(model, val_store, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **metrics,
            "epoch_seconds": time.monotonic() - epoch_start,
        }
        history.append(row)
        if verbose:
            print(
                f"epoch {epoch:3d} loss {row['train_loss']:.4f} val MAP {row['val_map']:.4f} "
                f"hit@10 {row['val_hit@10']:.3f} ({row['epoch_seconds']:.1f}s)",
                flush=True,
            )
        checkpoint = {
            "model_state": model.state_dict(),
            "encoder_config": asdict(enc_cfg),
            "train_settings": asdict(settings),
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(checkpoint, out / "encoder_last.pt")
        if row["val_map"] > best["val_map"]:
            best = {"val_map": row["val_map"], "epoch": epoch}
            torch.save(checkpoint, out / "encoder_best.pt")
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                if verbose:
                    print(f"early stopping after {epoch} epochs (best epoch {best['epoch']})", flush=True)
                break

    with (out / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    return {
        "best_epoch": best["epoch"],
        "best_val_map": best["val_map"],
        "epochs_run": len(history),
        "n_parameters": count_parameters(model),
        "receptive_field_frames": enc_cfg.receptive_field(),
        "device": str(device),
        "train_tracks": len(train_store.tracks),
        "train_wids": len(set(train_store.wids)),
        "val_tracks": len(val_store.tracks),
        "total_seconds": time.monotonic() - start_time,
        "history": history,
    }


def load_encoder(path: str | Path, device: torch.device | str = "cpu") -> TCNEncoder:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TCNEncoder(EncoderConfig(**checkpoint["encoder_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
