import torch

from modelsgenerate.tim_spectral_relation import TIMSpectralRelationBranch


def test_tim_spectral_relation_shape():
    x = torch.rand(2, 5, 3, 64, 64)
    branch = TIMSpectralRelationBranch(out_dim=32)
    y = branch(x)
    assert y.shape == (2, 5, 32)


def test_tim_spectral_relation_rejects_bad_shape():
    branch = TIMSpectralRelationBranch(out_dim=16)
    x = torch.rand(2, 3, 64, 64)
    try:
        branch(x)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-clip input")
