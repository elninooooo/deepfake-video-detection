import torch

from modelsgenerate.residual_spectral_relation import (
    RELATION_MODES,
    ResidualSpectralRelationBranch,
)


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


def test_residual_spectral_relation_modes_shape():
    x = torch.rand(1, 4, 3, 64, 64)
    for mode in RELATION_MODES:
        branch = ResidualSpectralRelationBranch(
            out_dim=12,
            encoder_dim=16,
            relation_mode=mode,
        )
        y = branch(x)
        assert y.shape == (1, 3, 12)
