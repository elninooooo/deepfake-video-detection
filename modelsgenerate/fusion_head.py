"""Fusion head that concatenates branch features and feeds the Temporal Transformer."""

import torch
import torch.nn as nn

from .temporal_transformer import TemporalTransformer


class FusionHead(nn.Module):
    def __init__(self,
                 branch_dims,
                 d_model: int = 512,
                 nhead: int = 4,
                 dropout: float = 0.1,
                 max_len: int = 32):
        """
        Parameters
        ----------
        branch_dims : list[int]
            Output dim of each branch's per-frame feature. Sum is concat dim.
        """
        super().__init__()
        in_dim = sum(branch_dims)
        self.proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.transformer = TemporalTransformer(
            d_model=d_model, nhead=nhead, dropout=dropout, max_len=max_len
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, branch_feats):
        """
        Parameters
        ----------
        branch_feats : list[torch.Tensor]
            Each of shape (B, T, d_i). T must match across branches.

        Returns
        -------
        logits : (B, 1)
        """
        # align time dimension by truncating to the shortest
        t_min = min(f.shape[1] for f in branch_feats)
        feats = [f[:, :t_min] for f in branch_feats]
        z = torch.cat(feats, dim=-1)             # (B, T, sum_d)
        z = self.proj(z)
        h = self.transformer(z)                  # (B, d_model)
        return self.classifier(h)                # (B, 1)
