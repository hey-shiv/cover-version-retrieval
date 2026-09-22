"""Encoder shapes, loss correctness, batch construction and deterministic training."""

import numpy as np
import pytest
import torch

from cover_retrieval.features.preprocessing import FeatureStore, Track
from cover_retrieval.models.losses import batch_hard_triplet_loss, supervised_contrastive_loss
from cover_retrieval.models.tcn_encoder import EncoderConfig, TCNEncoder, prepare_input
from cover_retrieval.models.training import augment, embed, pair_batches, train_encoder


def test_encoder_output_is_unit_norm_embedding():
    torch.manual_seed(0)
    cfg = EncoderConfig(input_frames=64, channels=16, n_blocks=3, embedding_dim=128)
    model = TCNEncoder(cfg).eval()
    x = prepare_input(torch.rand(5, 12, 200), cfg.input_frames)
    out = model(x)
    assert out.shape == (5, 128)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.norm(dim=1), torch.ones(5))
    assert cfg.receptive_field() == 1 + 2 * 2 * 7
    with pytest.raises(ValueError):
        model(torch.rand(2, 11, 64))


def test_supcon_matches_manual_formula():
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]]), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    t = 0.5
    loss = supervised_contrastive_loss(z, labels, t)
    sims = (z @ z.T / t).numpy()
    manual = 0.0
    for i in range(4):
        others = [a for a in range(4) if a != i]
        denom = np.log(np.exp(sims[i, others]).sum())
        positives = [p for p in others if labels[p] == labels[i]]
        manual += -np.mean([sims[i, p] - denom for p in positives])
    assert loss.item() == pytest.approx(manual / 4, rel=1e-5)


def test_supcon_gradients_finite_and_ordering():
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [1.0, 0.05], [0.0, 1.0], [0.05, 1.0]]), dim=1
    )
    bad = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.05], [0.05, 1.0]]), dim=1
    )
    assert supervised_contrastive_loss(good, labels) < supervised_contrastive_loss(bad, labels)
    z = torch.randn(8, 4, requires_grad=True)
    loss = supervised_contrastive_loss(torch.nn.functional.normalize(z, dim=1), torch.arange(8) // 2)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_triplet_loss_zero_when_separated():
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    assert batch_hard_triplet_loss(z, torch.tensor([0, 0, 1, 1]), margin=0.2).item() == 0.0
    assert batch_hard_triplet_loss(z, torch.tensor([0, 1, 0, 1]), margin=0.2).item() > 0.0


def test_pair_batches_keep_whole_works():
    wids = [f"W_{i // 2}" for i in range(20)]
    batches = pair_batches(wids, 3, np.random.default_rng(0))
    seen = []
    for batch in batches:
        batch_wids = [wids[i] for i in batch]
        assert all(batch_wids.count(w) == 2 for w in set(batch_wids))  # positive always present
        assert len(set(batch_wids)) >= 2  # negatives always present
        seen += batch
    assert sorted(seen) == list(range(20))
    assert batches == pair_batches(wids, 3, np.random.default_rng(0))


def test_augment_shapes_and_normalisation():
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(4, 12, 128)
    y = augment(x, 64, 0.6, True, gen)
    assert y.shape == (4, 12, 64)
    torch.testing.assert_close(y.norm(dim=1), torch.ones(4, 64))


def _toy_store(n_works, seed):
    rng = np.random.default_rng(seed)
    tracks, arrays = [], []
    for w in range(n_works):
        base = rng.random((12, 64)) ** 3
        for p in range(2):
            tracks.append(Track(f"P_{seed}{w:03d}{p}", f"W_{seed}{w:03d}"))
            arrays.append(np.roll(base, p * 3, axis=0) + 0.05 * rng.random((12, 64)))
    order = np.argsort([t.pid for t in tracks])
    tracks = [tracks[i] for i in order]
    feats = np.stack(arrays)[order].astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return FeatureStore(tracks, {"encoder": feats, "classical": feats[:, :, :32]})


def test_training_is_deterministic_and_rejects_leakage(tmp_path):
    config = {
        "seed": 1,
        "encoder": {
            "input_frames": 32,
            "channels": 8,
            "n_blocks": 2,
            "kernel_size": 3,
            "embedding_dim": 16,
            "dropout": 0.0,
        },
        "training": {
            "device": "cpu",
            "epochs": 2,
            "batch_wids": 4,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "temperature": 0.1,
            "patience": 5,
            "crop_min_fraction": 0.7,
            "pitch_shift_augment": True,
            "num_threads": 1,
        },
    }
    train, val = _toy_store(12, 1), _toy_store(4, 2)
    a = train_encoder(train, val, config, tmp_path / "a", verbose=False)
    b = train_encoder(train, val, config, tmp_path / "b", verbose=False)
    assert [h["train_loss"] for h in a["history"]] == [h["train_loss"] for h in b["history"]]
    sa = torch.load(tmp_path / "a" / "encoder_best.pt", weights_only=False)["model_state"]
    sb = torch.load(tmp_path / "b" / "encoder_best.pt", weights_only=False)["model_state"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    with pytest.raises(ValueError, match="leakage"):
        train_encoder(train, train.subset(train.pids[:4]), config, tmp_path / "c", verbose=False)


def test_embed_rotation_changes_input():
    torch.manual_seed(0)
    model = TCNEncoder(EncoderConfig(input_frames=32, channels=8, n_blocks=2, embedding_dim=8))
    x = np.random.default_rng(0).random((3, 12, 32)).astype(np.float32)
    e0, e3 = embed(model, x), embed(model, x, rotation=3)
    assert e0.shape == (3, 8) and not np.allclose(e0, e3)
    np.testing.assert_allclose(embed(model, np.roll(x, 3, axis=1)), e3, rtol=1e-5, atol=1e-6)
