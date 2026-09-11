"""FTCN-style video baseline.

This module implements a fair, self-contained FTCN baseline for the local
training pipeline. It follows the paper-level design: temporal convolutions
use spatial kernel size 1, and a shallow transformer aggregates the temporal
sequence into a video-level logit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, mid_channels: int, stride_t: int = 1):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(
            mid_channels,
            mid_channels,
            kernel_size=(3, 1, 1),
            stride=(stride_t, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels or stride_t != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride_t, 1, 1),
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(out + identity, inplace=True)


class FTCNBaseline(nn.Module):
    """Fully Temporal Convolution Network with transformer aggregation.

    Input shape is (B, T, 3, H, W), matching the rest of this repository.
    """

    def __init__(
        self,
        n_frames: int = 16,
        base_channels: int = 32,
        d_model: int = 1008,
        n_heads: int = 12,
        transformer_layers: int = 1,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.d_model = d_model

        self.stem = nn.Sequential(
            nn.Conv3d(
                3,
                base_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        c1 = base_channels * TemporalBottleneck.expansion
        c2 = base_channels * 2 * TemporalBottleneck.expansion
        c3 = base_channels * 4 * TemporalBottleneck.expansion
        c4 = base_channels * 8 * TemporalBottleneck.expansion
        self.layer1 = self._make_stage(base_channels, base_channels, blocks=3)
        self.layer2 = self._make_stage(c1, base_channels * 2, blocks=4)
        self.layer3 = self._make_stage(c2, base_channels * 4, blocks=6)
        self.layer4 = self._make_stage(c3, base_channels * 8, blocks=3)
        self.proj = nn.Linear(c4, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_frames + 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        self._init_weights()

    def _make_stage(self, in_channels: int, mid_channels: int, blocks: int) -> nn.Sequential:
        layers = [TemporalBottleneck(in_channels, mid_channels)]
        out_channels = mid_channels * TemporalBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(TemporalBottleneck(out_channels, mid_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)
        x = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.layer1(x)
        x = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.layer2(x)
        x = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.layer3(x)
        x = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.layer4(x)

        x = x.mean(dim=(-1, -2)).transpose(1, 2)  # (B, T, C)
        x = self.proj(x)
        b, t, _ = x.shape
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : t + 1]
        x = self.transformer(x)
        return self.fc(self.norm(x[:, 0]))
