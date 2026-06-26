"""Real-only TIM reconstruction/prediction models for v7 experiments.

These models are anomaly detectors rather than binary classifiers. They train
only on real clips, then use reconstruction or prediction error as the fake
score at evaluation time.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .tim_extractor import TIMExtractor


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class TIMFrameEncoder(nn.Module):
    """Encode one TIM frame from (3,H,W) to a compact spatial feature map."""

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, latent_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TIMFrameDecoder(nn.Module):
    """Decode a compact spatial feature map back to one TIM frame."""

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _deconv_block(latent_channels, 128),
            _deconv_block(128, 64),
            _deconv_block(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TIMAutoencoder(nn.Module):
    """v7a: reconstruct every TIM frame independently."""

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.tim_extract = TIMExtractor()
        self.encoder = TIMFrameEncoder(latent_channels)
        self.decoder = TIMFrameDecoder(latent_channels)

    def forward(self, x: torch.Tensor):
        tim = self.tim_extract(x)
        b, t, c, h, w = tim.shape
        flat = tim.reshape(b * t, c, h, w)
        recon = self.decoder(self.encoder(flat))
        recon = recon.view(b, t, c, h, w)
        return recon, tim


class TIMTemporalPredictor(nn.Module):
    """v7b: predict the final TIM frame from previous TIM frames."""

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.tim_extract = TIMExtractor()
        self.encoder = TIMFrameEncoder(latent_channels)
        self.temporal = nn.Sequential(
            nn.Conv3d(
                latent_channels,
                latent_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(latent_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                latent_channels,
                latent_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(latent_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = TIMFrameDecoder(latent_channels)

    def forward(self, x: torch.Tensor):
        tim = self.tim_extract(x)
        if tim.size(1) < 2:
            raise ValueError("TIMTemporalPredictor needs at least 3 input RGB frames.")
        context = tim[:, :-1]
        target = tim[:, -1]

        b, t, c, h, w = context.shape
        flat = context.reshape(b * t, c, h, w)
        z = self.encoder(flat).view(b, t, -1, flat.shape[-2] // 16, flat.shape[-1] // 16)
        z = z.permute(0, 2, 1, 3, 4).contiguous()
        z = self.temporal(z)
        z_last = z[:, :, -1]
        pred = self.decoder(z_last)
        return pred, target


@dataclass
class V7TIMConfig:
    variant: str = "v7a"
    latent_channels: int = 128


def build_v7_tim_model(cfg: V7TIMConfig) -> nn.Module:
    if cfg.variant == "v7a":
        return TIMAutoencoder(latent_channels=cfg.latent_channels)
    if cfg.variant == "v7b":
        return TIMTemporalPredictor(latent_channels=cfg.latent_channels)
    raise ValueError(f"Unsupported v7 TIM variant: {cfg.variant}")
