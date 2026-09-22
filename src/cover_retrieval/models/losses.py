"""Metric-learning objectives on L2-normalised embeddings, written out explicitly.

* ``supervised_contrastive_loss`` — SupCon "L_out" (Khosla et al., 2020, Eq. 2): for
  anchor i with positives P(i) (same WID, excluding i) and all others A(i),
  ``-1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p / t) / sum_{a in A(i)} exp(z_i.z_a / t) )``.
* ``batch_hard_triplet_loss`` — Hermans et al. (2017): hardest positive vs hardest
  negative per anchor with a margin; provided as an alternative/ablation.

Anchors without any positive in the batch are ignored (the sampler guarantees two
recordings per WID, so this only matters for malformed batches).
"""

from __future__ import annotations

import torch


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    if embeddings.ndim != 2 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("embeddings must be (B, D) with one label per row")
    n = embeddings.shape[0]
    logits = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_pos = positives.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        raise ValueError("batch contains no positive pairs")
    pos_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    return -(pos_log_prob[valid] / n_pos[valid]).mean()


def batch_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2
) -> torch.Tensor:
    distances = torch.cdist(embeddings, embeddings)
    n = embeddings.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    same = labels[:, None] == labels[None, :]
    positives = same & ~self_mask
    negatives = ~same
    valid = positives.any(dim=1) & negatives.any(dim=1)
    if not valid.any():
        raise ValueError("batch needs at least one anchor with a positive and a negative")
    hardest_pos = torch.where(positives, distances, torch.zeros_like(distances)).amax(dim=1)
    hardest_neg = torch.where(negatives, distances, torch.full_like(distances, float("inf"))).amin(dim=1)
    return torch.relu(hardest_pos - hardest_neg + margin)[valid].mean()
