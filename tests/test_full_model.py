"""Smoke test: every variant must accept (B,N,3,H,W) and return (B,1)."""

import torch

from modelsgenerate.full_model import ModelConfig, PhaseTransformerDetector


def _build(variant):
    flags = {
        "v1": dict(use_tim=True,  use_phase=False, use_mask=False, phase_residual=False),
        "v2": dict(use_tim=False, use_phase=True,  use_mask=False, phase_residual=False),
        "v3": dict(use_tim=True,  use_phase=True,  use_mask=False, phase_residual=False),
        "v4": dict(use_tim=True,  use_phase=True,  use_mask=True,  phase_residual=False),
        "v5": dict(use_tim=False, use_phase=True,  use_mask=False, phase_residual=True),
        "v6": dict(use_tim=False, use_phase=True,  use_mask=True,  phase_residual=True),
        "v8": dict(use_tim=True,  use_phase=False, use_mask=False, phase_residual=False,
                   use_tim_spectral_relation=True),
    }[variant]
    return PhaseTransformerDetector(ModelConfig(**flags, max_len=16))


def test_v1_forward():
    m = _build("v1")
    x = torch.rand(1, 8, 3, 64, 64)
    y = m(x)
    assert y.shape == (1, 1)


def test_v2_forward():
    m = _build("v2")
    x = torch.rand(1, 8, 3, 64, 64)
    y = m(x)
    assert y.shape == (1, 1)


def test_v3_forward():
    m = _build("v3")
    x = torch.rand(1, 8, 3, 64, 64)
    y = m(x)
    assert y.shape == (1, 1)


def test_v4_forward_and_mask_changes():
    m = _build("v4")
    x = torch.rand(2, 8, 3, 64, 64)
    m.train()
    torch.manual_seed(0)
    y_train = m(x)
    m.eval()
    y_eval = m(x)
    assert y_train.shape == (2, 1)
    assert y_eval.shape == (2, 1)


def test_v5_phase_residual_forward():
    m = _build("v5")
    x = torch.rand(1, 8, 3, 64, 64)
    y = m(x)
    assert y.shape == (1, 1)


def test_v6_phase_residual_mask_forward():
    m = _build("v6")
    x = torch.rand(2, 8, 3, 64, 64)
    m.train()
    y_train = m(x)
    m.eval()
    y_eval = m(x)
    assert y_train.shape == (2, 1)
    assert y_eval.shape == (2, 1)


def test_v8_tim_spectral_relation_forward():
    m = _build("v8")
    x = torch.rand(1, 8, 3, 64, 64)
    y = m(x)
    assert y.shape == (1, 1)
