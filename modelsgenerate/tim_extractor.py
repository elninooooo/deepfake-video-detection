"""TIM (Temporal Interpolation Map) extractor — parameter-free directional frame diff.

Definition (per the frozen design decision):

    tim_t = |x_{t+1} - x_t|        # absolute temporal difference

Input  : (B, N,   3, H, W)
Output : (B, N-1, 3, H, W)

The diff highlights interpolation artefacts at face/background boundaries that
deepfake generators tend to introduce.
"""

import torch
import torch.nn as nn


class TIMExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"TIMExtractor expects (B,N,C,H,W); got {tuple(x.shape)}")
        return (x[:, 1:] - x[:, :-1]).abs()
