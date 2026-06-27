"""TIM spectral relationship branch for v8.

This branch turns each TIM frame into compact spectral relationship features.
It intentionally avoids feeding raw phase maps to the detector. Instead, it
uses low/mid/high frequency relationships, band energy ratios, and confidence-
weighted mid-band phase statistics.
"""

import math

import torch
import torch.nn as nn


def _band_masks(h: int, w: int, low: float, high: float, device: torch.device):
    yy = torch.linspace(-1.0, 1.0, h, device=device)
    xx = torch.linspace(-1.0, 1.0, w, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square()) / math.sqrt(2.0)
    low_mask = (radius < low).float()
    mid_mask = ((radius >= low) & (radius <= high)).float()
    high_mask = (radius > high).float()
    return low_mask, mid_mask, high_mask


def _channel_stats(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            x.amin(dim=1),
            x.amax(dim=1),
        ],
        dim=1,
    )


def _cosine_channelwise(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a = a.flatten(2)
    b = b.flatten(2)
    return (a * b).sum(dim=2) / (a.norm(dim=2) * b.norm(dim=2) + eps)


class TIMSpectralRelationBranch(nn.Module):
    """Extract compact per-TIM-frame spectral relationship features."""

    raw_dim = 48

    def __init__(
        self,
        out_dim: int = 128,
        mid_low: float = 0.10,
        mid_high: float = 0.70,
        confidence_quantile: float = 0.95,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not 0.0 <= mid_low < mid_high <= 1.0:
            raise ValueError("Expected 0 <= mid_low < mid_high <= 1.")
        self.out_dim = out_dim
        self.mid_low = mid_low
        self.mid_high = mid_high
        self.confidence_quantile = confidence_quantile
        self.proj = nn.Sequential(
            nn.LayerNorm(self.raw_dim),
            nn.Linear(self.raw_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def _raw_features(self, tim_clip: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = tim_clip.shape
        flat = tim_clip.reshape(b * t, c, h, w)
        spec = torch.fft.fftshift(torch.fft.fft2(flat, norm="ortho"), dim=(-2, -1))

        low_mask, mid_mask, high_mask = _band_masks(
            h, w, self.mid_low, self.mid_high, tim_clip.device
        )
        masks = [low_mask, mid_mask, high_mask]
        parts = []
        for mask in masks:
            masked = torch.fft.ifftshift(spec * mask.view(1, 1, h, w), dim=(-2, -1))
            parts.append(torch.fft.ifft2(masked, norm="ortho").real)
        low_part, mid_part, high_part = parts

        cos_feats = [
            _channel_stats(_cosine_channelwise(flat, low_part)),
            _channel_stats(_cosine_channelwise(flat, mid_part)),
            _channel_stats(_cosine_channelwise(flat, high_part)),
            _channel_stats(_cosine_channelwise(low_part, mid_part)),
            _channel_stats(_cosine_channelwise(mid_part, high_part)),
        ]

        energy_total = flat.square().flatten(2).mean(dim=2).clamp_min(1e-8)
        energy_feats = [
            _channel_stats(part.square().flatten(2).mean(dim=2) / energy_total)
            for part in parts
        ]

        amp = spec.abs()
        log_amp = torch.log1p(amp)
        denom = torch.quantile(
            log_amp.flatten(-2),
            self.confidence_quantile,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        confidence = (log_amp.flatten(-2) / denom).clamp(0.0, 1.0).view_as(log_amp)
        phase = torch.angle(spec)
        mid = mid_mask.view(1, 1, h, w)
        sin_mid = confidence * mid * torch.sin(phase)
        cos_mid = confidence * mid * torch.cos(phase)
        phase_feats = [
            _channel_stats(sin_mid.flatten(2).mean(dim=2)),
            _channel_stats(sin_mid.flatten(2).std(dim=2, unbiased=False)),
            _channel_stats(cos_mid.flatten(2).mean(dim=2)),
            _channel_stats(cos_mid.flatten(2).std(dim=2, unbiased=False)),
        ]

        return torch.cat(cos_feats + energy_feats + phase_feats, dim=1)

    def forward(self, tim_clip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tim_clip : (B,T,3,H,W)

        Returns
        -------
        (B,T,out_dim)
        """
        if tim_clip.dim() != 5:
            raise ValueError(
                f"TIMSpectralRelationBranch expects (B,T,C,H,W); got {tuple(tim_clip.shape)}"
            )
        b, t = tim_clip.shape[:2]
        raw = self._raw_features(tim_clip)
        return self.proj(raw).view(b, t, self.out_dim)
