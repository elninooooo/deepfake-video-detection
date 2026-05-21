"""Phase-spectrum branch with optional high-frequency random masking.

Forward pipeline (per-frame, per-channel):

    F(u,v) = FFT2(x)
    F'(u,v) = F(u,v) * (1 - M)            # optional HF mask
    phi(u,v) = angle(F'(u,v))
    out = concat([sin(phi), cos(phi)], dim=C)   # double the channel count

The (sin, cos) encoding sidesteps phase wrap-around discontinuities so the
downstream CNN can treat the output as an ordinary image tensor.

Mask is applied in the *shifted* spectrum: pixels whose normalized radial
distance from the DC component is > `mask_radius_ratio` are considered
high-frequency, and a Bernoulli(`mask_ratio`) random pattern zeros them out.
"""

from typing import Optional

import torch
import torch.nn as nn


def make_radius_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """Normalized radial distance from spectrum center, range [0, 1]."""
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(yy * yy + xx * xx)
    return r.clamp_(0.0, 1.0)


class PhaseBranch(nn.Module):
    def __init__(self,
                 mask_ratio: float = 0.0,
                 mask_radius_ratio: float = 0.5,
                 train_only_mask: bool = True):
        """
        Parameters
        ----------
        mask_ratio
            Probability of zeroing each high-frequency cell. 0 disables masking.
        mask_radius_ratio
            Normalized radial threshold: cells with r > this are 'high freq'.
        train_only_mask
            If True, apply mask only when self.training is True (test-time
            uses the unmasked spectrum, matching the methodology).
        """
        super().__init__()
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio in [0,1); got {mask_ratio}")
        if not 0.0 < mask_radius_ratio <= 1.0:
            raise ValueError(f"mask_radius_ratio in (0,1]; got {mask_radius_ratio}")
        self.mask_ratio = float(mask_ratio)
        self.mask_radius_ratio = float(mask_radius_ratio)
        self.train_only_mask = train_only_mask

    def _hf_random_mask(self, shape, device, dtype):
        """1 where pixel is dropped, 0 elsewhere. Same shape as the spectrum (B*N, C, H, W)."""
        b_n, c, h, w = shape
        r = make_radius_grid(h, w, device, dtype)               # (H, W)
        hf_region = (r > self.mask_radius_ratio).to(dtype)      # (H, W)
        rand = torch.rand((b_n, c, h, w), device=device, dtype=dtype)
        drop = (rand < self.mask_ratio).to(dtype) * hf_region   # (B*N, C, H, W)
        return drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, 3, H, W) RGB clip in [0, 1].

        Returns
        -------
        (B, N, 6, H, W) tensor (3 channels of sin(phi), 3 channels of cos(phi)).
        """
        if x.dim() != 5:
            raise ValueError(f"PhaseBranch expects (B,N,C,H,W); got {tuple(x.shape)}")
        b, n, c, h, w = x.shape
        x_flat = x.reshape(b * n, c, h, w)

        # FFT and shift so that DC sits at the center
        spec = torch.fft.fft2(x_flat, norm="ortho")
        spec = torch.fft.fftshift(spec, dim=(-2, -1))

        apply_mask = (self.mask_ratio > 0.0) and ((not self.train_only_mask) or self.training)
        if apply_mask:
            drop = self._hf_random_mask(spec.shape, spec.device, spec.real.dtype)
            keep = 1.0 - drop
            spec = torch.complex(spec.real * keep, spec.imag * keep)

        phi = torch.angle(spec)                       # (B*N, C, H, W)
        out = torch.cat([torch.sin(phi), torch.cos(phi)], dim=1)  # (B*N, 2C, H, W)
        return out.reshape(b, n, 2 * c, h, w)
