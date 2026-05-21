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
