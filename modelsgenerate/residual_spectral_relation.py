"""Trainable residual spectral deep-relation branch for v10."""

import math
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


RELATION_MODES = {
    "full",
    "feat_only",
    "abs_only",
    "cos_only",
    "dist_only",
    "abs_cos",
    "abs_dist",
    "cos_dist",
    "no_original",
    "concat_views",
}


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class ResidualViewEncoder(nn.Module):
    """Small shared encoder for residual frequency views."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, out_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


def _band_masks(h: int, w: int, low: float, high: float, device: torch.device):
    yy = torch.linspace(-1.0, 1.0, h, device=device)
    xx = torch.linspace(-1.0, 1.0, w, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square()) / math.sqrt(2.0)
    low_mask = (radius < low).float()
    mid_mask = ((radius >= low) & (radius <= high)).float()
    high_mask = (radius > high).float()
    return low_mask, mid_mask, high_mask


def _sobel_gray(x: torch.Tensor) -> torch.Tensor:
    gray = 0.2989 * x[:, :, 0:1] + 0.5870 * x[:, :, 1:2] + 0.1140 * x[:, :, 2:3]
    b, n, _, h, w = gray.shape
    flat = gray.reshape(b * n, 1, h, w)
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=flat.dtype,
        device=flat.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=flat.dtype,
        device=flat.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(flat, kx, padding=1)
    gy = F.conv2d(flat, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
    return mag.view(b, n, 1, h, w).repeat(1, 1, 3, 1, 1)


class ResidualSpectralRelationBranch(nn.Module):
    """Learn deep relationships among residual original/low/mid/high views."""

    def __init__(
        self,
        out_dim: int = 128,
        encoder_dim: int = 256,
        mid_low: float = 0.10,
        mid_high: float = 0.70,
        residual_mode: str = "gradient",
        relation_mode: str = "full",
        dropout: float = 0.1,
    ):
        super().__init__()
        if not 0.0 <= mid_low < mid_high <= 1.0:
            raise ValueError("Expected 0 <= mid_low < mid_high <= 1.")
        if residual_mode not in {"abs", "signed", "gradient"}:
            raise ValueError(f"Unsupported residual_mode: {residual_mode}")
        if relation_mode not in RELATION_MODES:
            raise ValueError(
                f"Unsupported relation_mode: {relation_mode}. "
                f"Expected one of {sorted(RELATION_MODES)}"
            )
        self.mid_low = mid_low
        self.mid_high = mid_high
        self.residual_mode = residual_mode
        self.relation_mode = relation_mode
        self.encoder = ResidualViewEncoder(out_dim=encoder_dim)
        self.pairs = list(combinations(range(4), 2))
        relation_dim = self._relation_dim(encoder_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def _relation_dim(self, encoder_dim: int) -> int:
        n_pairs = len(self.pairs)
        if self.relation_mode == "full":
            return encoder_dim * (1 + n_pairs) + 2 * n_pairs
        if self.relation_mode == "feat_only":
            return encoder_dim
        if self.relation_mode == "abs_only":
            return encoder_dim * n_pairs
        if self.relation_mode == "cos_only":
            return n_pairs
        if self.relation_mode == "dist_only":
            return n_pairs
        if self.relation_mode == "abs_cos":
            return encoder_dim * n_pairs + n_pairs
        if self.relation_mode == "abs_dist":
            return encoder_dim * n_pairs + n_pairs
        if self.relation_mode == "cos_dist":
            return 2 * n_pairs
        if self.relation_mode == "no_original":
            return encoder_dim * n_pairs + 2 * n_pairs
        if self.relation_mode == "concat_views":
            return encoder_dim * 4
        raise AssertionError(f"Unhandled relation_mode: {self.relation_mode}")

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_mode == "abs":
            return (x[:, 1:] - x[:, :-1]).abs()
        if self.residual_mode == "signed":
            return x[:, 1:] - x[:, :-1]
        grad = _sobel_gray(x)
        return (grad[:, 1:] - grad[:, :-1]).abs()

    def _frequency_views(self, residual: torch.Tensor):
        b, t, c, h, w = residual.shape
        flat = residual.reshape(b * t, c, h, w)
        spec = torch.fft.fftshift(torch.fft.fft2(flat, norm="ortho"), dim=(-2, -1))
        views = [flat]
        for mask in _band_masks(h, w, self.mid_low, self.mid_high, residual.device):
            masked = torch.fft.ifftshift(spec * mask.view(1, 1, h, w), dim=(-2, -1))
            views.append(torch.fft.ifft2(masked, norm="ortho").real)
        return views

    def _relation_features(self, feats):
        if self.relation_mode == "feat_only":
            return feats[0]
        if self.relation_mode == "concat_views":
            return torch.cat(feats, dim=1)

        abs_parts = []
        cos_parts = []
        dist_parts = []
        for i, j in self.pairs:
            diff = feats[i] - feats[j]
            abs_parts.append(diff.abs())
            cos_parts.append(F.cosine_similarity(feats[i], feats[j], dim=1, eps=1e-8))
            dist_parts.append(diff.norm(dim=1))

        parts = []
        if self.relation_mode == "full":
            parts.append(feats[0])
        if self.relation_mode in {"full", "abs_only", "abs_cos", "abs_dist", "no_original"}:
            parts.extend(abs_parts)
        if self.relation_mode in {"full", "cos_only", "abs_cos", "cos_dist", "no_original"}:
            parts.append(torch.stack(cos_parts, dim=1))
        if self.relation_mode in {"full", "dist_only", "abs_dist", "cos_dist", "no_original"}:
            parts.append(torch.stack(dist_parts, dim=1))
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B,N,3,H,W) RGB clip in [0,1].

        Returns
        -------
        (B,N-1,out_dim)
        """
        if x.dim() != 5:
            raise ValueError(
                f"ResidualSpectralRelationBranch expects (B,N,C,H,W); got {tuple(x.shape)}"
            )
        residual = self._residual(x)
        b, t = residual.shape[:2]
        views = self._frequency_views(residual)
        feats = self.encoder(torch.cat(views, dim=0)).chunk(4, dim=0)

        relation = self._relation_features(feats)
        return self.proj(relation).view(b, t, self.out_dim)
