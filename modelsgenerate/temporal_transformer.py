"""Single-layer Transformer encoder over per-frame features.

Input  : (B, T, d_model)
Output : (B, d_model)   # CLS-token output

Following FTCN guidance, we use one layer to limit over-fitting on the
relatively small Celeb-DF training set.
"""

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 4,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 32):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # +1 to account for CLS token
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, d_model)

        Returns
        -------
        (B, d_model) CLS token after encoder.
        """
        b, t, d = x.shape
        if t > self.max_len:
            raise ValueError(f"Clip length {t} exceeds max_len={self.max_len}")
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)             # (B, T+1, D)
        x = x + self.pos_embed[:, : t + 1]
        x = self.encoder(x)
        x = self.norm(x[:, 0])                      # CLS
        return x
