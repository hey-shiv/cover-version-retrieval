"""Reproducibility controls (PyTorch "Reproducibility" notes)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True, num_threads: int | None = None) -> None:
    """Seed Python, NumPy and PyTorch; optionally force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG third-party code may use
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    """``auto`` -> cuda, then mps, then cpu; otherwise the named device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
