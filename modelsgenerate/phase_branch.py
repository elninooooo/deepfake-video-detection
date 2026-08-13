"""Phase-spectrum branch with optional confidence weighting and frequency bands.

Forward pipeline (per-frame, per-channel):

    F(u,v) = FFT2(x)
    F'(u,v) = F(u,v) * (1 - M)            # optional HF mask
    phi(u,v) = angle(F'(u,v))
    out = concat([w * sin(phi), w * cos(phi)], dim=C)

The (sin, cos) encoding sidesteps phase wrap-around discontinuities so the
downstream CNN can treat the output as an ordinary image tensor.

Phase modes:
  * raw:          w = 1 (legacy behavior)
  * weighted:     w = robustly normalized log-magnitude confidence
  * mid_weighted: w = confidence inside a normalized radial mid-frequency band

Mask is applied in the *shifted* spectrum: pixels whose normalized radial
distance from the DC component is > `mask_radius_ratio` are considered
high-frequency, and a Bernoulli(`mask_ratio`) random pattern zeros them out.
"""

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
                 train_only_mask: bool = True,
                 phase_mode: str = "raw",
                 phase_confidence_quantile: float = 0.95,
                 phase_mid_low: float = 0.15,
                 phase_mid_high: float = 0.65):
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
        phase_mode
            raw, weighted, or mid_weighted.
        phase_confidence_quantile
            Per-frame/channel log-magnitude quantile used to normalize phase
            confidence robustly without letting the DC peak dominate.
        phase_mid_low, phase_mid_high
            Normalized radial bounds used by mid_weighted mode.
        """
        super().__init__()
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio in [0,1); got {mask_ratio}")
        if not 0.0 < mask_radius_ratio <= 1.0:
            raise ValueError(f"mask_radius_ratio in (0,1]; got {mask_radius_ratio}")
        if phase_mode not in {"raw", "weighted", "mid_weighted"}:
            raise ValueError(f"Unknown phase_mode: {phase_mode}")
        if not 0.0 < phase_confidence_quantile <= 1.0:
            raise ValueError(
                "phase_confidence_quantile must be in (0,1]; "
                f"got {phase_confidence_quantile}")
        if not 0.0 <= phase_mid_low < phase_mid_high <= 1.0:
            raise ValueError(
                "Expected 0 <= phase_mid_low < phase_mid_high <= 1; "
                f"got {phase_mid_low}, {phase_mid_high}")
        self.mask_ratio = float(mask_ratio)
        self.mask_radius_ratio = float(mask_radius_ratio)
        self.train_only_mask = train_only_mask
        self.phase_mode = phase_mode
        self.phase_confidence_quantile = float(phase_confidence_quantile)
        self.phase_mid_low = float(phase_mid_low)
        self.phase_mid_high = float(phase_mid_high)

    def _hf_random_mask(self, shape, device, dtype):
        """1 where pixel is dropped, 0 elsewhere. Same shape as the spectrum (B*N, C, H, W)."""
        b_n, c, h, w = shape
        r = make_radius_grid(h, w, device, dtype)               # (H, W)
        hf_region = (r > self.mask_radius_ratio).to(dtype)      # (H, W)
        rand = torch.rand((b_n, c, h, w), device=device, dtype=dtype)
        drop = (rand < self.mask_ratio).to(dtype) * hf_region   # (B*N, C, H, W)
        return drop

    def _phase_weight(self, spec: torch.Tensor) -> torch.Tensor:
        """Return a confidence/band weight with the same shape as spec.real."""
        magnitude = torch.log1p(torch.abs(spec))
        flat = magnitude.flatten(-2)
        scale = torch.quantile(
            flat, self.phase_confidence_quantile, dim=-1, keepdim=True
        ).unsqueeze(-1)
        weight = (magnitude / scale.clamp_min(1e-6)).clamp_(0.0, 1.0)

        if self.phase_mode == "mid_weighted":
            h, w = spec.shape[-2:]
            radius = make_radius_grid(h, w, spec.device, spec.real.dtype)
            band = ((radius >= self.phase_mid_low) &
                    (radius <= self.phase_mid_high)).to(spec.real.dtype)
            weight = weight * band
        return weight

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
        if self.phase_mode == "raw":
            phase_sin = torch.sin(phi)
            phase_cos = torch.cos(phi)
        else:
            weight = self._phase_weight(spec)
            phase_sin = weight * torch.sin(phi)
            phase_cos = weight * torch.cos(phi)
        out = torch.cat([phase_sin, phase_cos], dim=1)  # (B*N, 2C, H, W)
        return out.reshape(b, n, 2 * c, h, w)
