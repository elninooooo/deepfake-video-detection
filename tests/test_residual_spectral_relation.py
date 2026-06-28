import torch

from modelsgenerate.residual_spectral_relation import ResidualSpectralRelationBranch


def test_residual_spectral_relation_shape():
    x = torch.rand(2, 6, 3, 64, 64)
    branch = ResidualSpectralRelationBranch(out_dim=32, encoder_dim=64)
    y = branch(x)
    assert y.shape == (2, 5, 32)


def test_residual_spectral_relation_gradient_mode_shape():
    x = torch.rand(1, 5, 3, 64, 64)
    branch = ResidualSpectralRelationBranch(
        out_dim=16,
        encoder_dim=32,
        residual_mode="gradient",
    )
    y = branch(x)
    assert y.shape == (1, 4, 16)
