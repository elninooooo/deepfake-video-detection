"""FreqNet-style image-frequency baseline.

The original FreqNet is an image detector. For video-level evaluation in this
repository, each sampled frame is scored by the same frequency-aware CNN and
the frame logits are averaged into one clip logit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyConv2d(nn.Module):
    """Apply learnable convolutions to amplitude and phase spectra."""

    def __init__(self, channels: int):
        super().__init__()
        self.amp_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.phase_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.fft2(x, norm="ortho")
        amp = torch.abs(spec)
        phase = torch.angle(spec)
        amp = F.softplus(self.amp_conv(amp))
        phase = self.phase_conv(phase)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        rec = torch.fft.ifft2(torch.complex(real, imag), norm="ortho").real
        return rec


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


class HighFrequencyModule(nn.Module):
    """Keep a high-frequency residual while preserving RGB context."""

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high = x - low
        return self.fuse(torch.cat([x, high], dim=1))


class FreqNetBaseline(nn.Module):
    def __init__(self, base_channels: int = 32, frame_agg: str = "mean_logit"):
        super().__init__()
        if frame_agg not in {"mean_logit", "max_logit"}:
            raise ValueError("frame_agg must be 'mean_logit' or 'max_logit'.")
        self.frame_agg = frame_agg

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.high = HighFrequencyModule(base_channels)
        self.freq = FrequencyConv2d(base_channels)
        self.stage1 = ResidualBlock2d(base_channels, base_channels)
        self.stage2 = ResidualBlock2d(base_channels, base_channels * 2, stride=2)
        self.stage3 = ResidualBlock2d(base_channels * 2, base_channels * 4, stride=2)
        self.stage4 = ResidualBlock2d(base_channels * 4, base_channels * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, 1)

    def forward_frames(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.high(x)
        x = x + self.freq(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        logits = self.forward_frames(x.reshape(b * t, c, h, w)).view(b, t, 1)
        if self.frame_agg == "max_logit":
            return logits.max(dim=1).values
        return logits.mean(dim=1)
