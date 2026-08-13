import torch

from modelsgenerate.phase_branch import PhaseBranch


def test_shape_no_mask():
    pb = PhaseBranch(mask_ratio=0.0)
    x = torch.rand(2, 8, 3, 64, 64)
    out = pb(x)
    assert out.shape == (2, 8, 6, 64, 64)


def test_sin_cos_identity():
    """sin² + cos² == 1 for each (channel pair, pixel)."""
    pb = PhaseBranch(mask_ratio=0.0)
    pb.eval()
    x = torch.rand(1, 2, 3, 32, 32)
    out = pb(x)                          # (1,2,6,32,32)
    s, c = out[:, :, :3], out[:, :, 3:]
    norm = s.pow(2) + c.pow(2)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_mask_changes_output():
    torch.manual_seed(0)
    pb = PhaseBranch(mask_ratio=0.5, mask_radius_ratio=0.3)
    pb.train()
    x = torch.rand(1, 2, 3, 64, 64)
    out_masked = pb(x).clone()
    pb.eval()
    out_clean = pb(x)
    assert not torch.allclose(out_masked, out_clean), "Mask should affect training output."


def test_weighted_phase_has_bounded_confidence():
    pb = PhaseBranch(phase_mode="weighted")
    x = torch.rand(1, 2, 3, 32, 32)
    out = pb(x)
    s, c = out[:, :, :3], out[:, :, 3:]
    confidence_sq = s.pow(2) + c.pow(2)
    assert torch.all(confidence_sq >= 0)
    assert torch.all(confidence_sq <= 1.0 + 1e-5)
    assert not torch.allclose(confidence_sq, torch.ones_like(confidence_sq))


def test_mid_weighted_zeros_outside_frequency_band():
    pb = PhaseBranch(
        phase_mode="mid_weighted",
        phase_mid_low=0.2,
        phase_mid_high=0.6,
    )
    x = torch.rand(1, 1, 3, 32, 32)
    out = pb(x)
    s, c = out[:, :, :3], out[:, :, 3:]
    confidence_sq = s.pow(2) + c.pow(2)
    assert torch.allclose(confidence_sq[..., 0, 0], torch.zeros_like(confidence_sq[..., 0, 0]))
    assert torch.allclose(
        confidence_sq[..., 16, 16], torch.zeros_like(confidence_sq[..., 16, 16]))
