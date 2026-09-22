"""Compact temporal convolutional encoder: ``(B, 12, T)`` HPCP -> ``(B, D)`` unit vectors.

Architecture (Bai, Kolter & Koltun 2018, adapted non-causally for whole-track input):

    1x1 conv 12 -> C
    n_blocks x residual block [dilated conv k -> BN -> GELU -> dropout -> dilated conv k -> BN] + skip, GELU
        dilations 1, 2, 4, ... (receptive field ~ 1 + 2 (k - 1) (2^n_blocks - 1) frames)
    global pooling: concat(mean over time, max over time)       -> 2C
    projection: Linear 2C -> C -> GELU -> Linear C -> D, then L2 normalisation

Transposition robustness is learned through random pitch-rotation augmentation during
training; optionally, retrieval can also max over the 12 rotations of the query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class EncoderConfig:
    input_frames: int = 256
    channels: int = 128
    n_blocks: int = 6
    kernel_size: int = 3
    embedding_dim: int = 128
    dropout: float = 0.1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EncoderConfig:
        enc = config["encoder"]
        return cls(
            input_frames=int(enc["input_frames"]),
            channels=int(enc["channels"]),
            n_blocks=int(enc["n_blocks"]),
            kernel_size=int(enc["kernel_size"]),
            embedding_dim=int(enc["embedding_dim"]),
            dropout=float(enc["dropout"]),
        )

    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * (2**self.n_blocks - 1)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        y = self.bn2(self.conv2(y))
        return F.gelu(x + y)


class TCNEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        if cfg.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for length-preserving padding")
        self.cfg = cfg
        self.input = nn.Conv1d(12, cfg.channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(cfg.channels, cfg.kernel_size, 2**b, cfg.dropout) for b in range(cfg.n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(2 * cfg.channels, cfg.channels),
            nn.GELU(),
            nn.Linear(cfg.channels, cfg.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 12:
            raise ValueError(f"expected (B, 12, T), got {tuple(x.shape)}")
        h = self.blocks(self.input(x))
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return F.normalize(self.head(pooled), dim=1)


def prepare_input(x: torch.Tensor, input_frames: int) -> torch.Tensor:
    """Area-pool ``(B, 12, T)`` to ``input_frames`` and re-normalise frames (train == eval)."""
    if x.shape[-1] != input_frames:
        x = F.adaptive_avg_pool1d(x, input_frames)
    return F.normalize(x, dim=1, eps=1e-8)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
