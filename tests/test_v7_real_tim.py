import torch

from modelsgenerate.v7_real_tim import TIMAutoencoder, TIMTemporalPredictor
from train_v7_real_tim import sample_errors


def test_v7a_reconstructs_tim_shape():
    x = torch.rand(2, 6, 3, 64, 64)
    model = TIMAutoencoder(latent_channels=32)
    pred, target = model(x)
    assert pred.shape == target.shape == (2, 5, 3, 64, 64)


def test_v7b_predicts_final_tim_shape():
    x = torch.rand(2, 6, 3, 64, 64)
    model = TIMTemporalPredictor(latent_channels=32)
    pred, target = model(x)
    assert pred.shape == target.shape == (2, 3, 64, 64)


def test_sample_errors_are_per_clip():
    pred = torch.zeros(3, 4, 3, 8, 8)
    target = torch.ones(3, 4, 3, 8, 8)
    scores = sample_errors(pred, target, "l1")
    assert scores.shape == (3,)
    assert torch.allclose(scores, torch.ones_like(scores))
