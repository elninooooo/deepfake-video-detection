"""Assemble the full detector based on which branches are enabled.

Variants used in the experimental matrix:
    V1 Baseline-TIM          : RGB residual
    V2 Phase-only            : phase map
    V3 TIM + Phase           : RGB residual + phase map, late fusion
    V4 TIM + Phase + HF      : RGB residual + masked phase map, late fusion
    V5 Phase-Residual        : phase map -> temporal residual
    V6 Phase-Residual + HF   : masked phase map -> temporal residual
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .fusion_head import FusionHead
from .phase_branch import PhaseBranch
from .spatial_backbone import SpatialBackbone
from .tim_extractor import TIMExtractor


@dataclass
class ModelConfig:
    use_tim: bool = True
    use_phase: bool = False
    use_mask: bool = False
    phase_residual: bool = False
    mask_ratio: float = 0.5
    mask_radius_ratio: float = 0.5
    phase_mode: str = "raw"
    phase_confidence_quantile: float = 0.95
    phase_mid_low: float = 0.15
    phase_mid_high: float = 0.65
    d_model: int = 512
    n_heads: int = 4
    max_len: int = 32       # supports N up to 32 frames

    def variant_name(self) -> str:
        for name, flags in {
            "v1_tim": (True, False, False, False),
            "v2_phase": (False, True, False, False),
            "v3_tim_phase": (True, True, False, False),
            "v4_tim_phase_mask": (True, True, True, False),
            "v5_phase_residual": (False, True, False, True),
            "v6_phase_residual_mask": (False, True, True, True),
        }.items():
            if (self.use_tim, self.use_phase, self.use_mask, self.phase_residual) == flags:
                return name
        return "custom"


class PhaseTransformerDetector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.phase_residual and not cfg.use_phase:
            raise ValueError("phase_residual=True requires use_phase=True.")
        if not (cfg.use_tim or cfg.use_phase):
            raise ValueError("At least one of use_tim / use_phase must be True.")

        branch_dims = []

        if cfg.use_tim:
            self.tim_extract = TIMExtractor()
            self.tim_backbone = SpatialBackbone(in_channels=3)
            branch_dims.append(self.tim_backbone.out_dim)

        if cfg.use_phase:
            mask_ratio = cfg.mask_ratio if cfg.use_mask else 0.0
            self.phase_extract = PhaseBranch(
                mask_ratio=mask_ratio,
                mask_radius_ratio=cfg.mask_radius_ratio,
                train_only_mask=True,
                phase_mode=cfg.phase_mode,
                phase_confidence_quantile=cfg.phase_confidence_quantile,
                phase_mid_low=cfg.phase_mid_low,
                phase_mid_high=cfg.phase_mid_high,
            )
            # Phase branch output channels = 2 * input RGB = 6. In phase-residual
            # variants, the temporal residual keeps the same channel count.
            self.phase_backbone = SpatialBackbone(in_channels=6)
            branch_dims.append(self.phase_backbone.out_dim)

        self.head = FusionHead(
            branch_dims=branch_dims,
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            max_len=cfg.max_len,
        )

    def _run_backbone(self, backbone: SpatialBackbone, x_clip: torch.Tensor) -> torch.Tensor:
        """x_clip : (B, T, C, H, W) → feature (B, T, 512)."""
        b, t, c, h, w = x_clip.shape
        flat = x_clip.reshape(b * t, c, h, w)
        feat = backbone(flat)                          # (B*T, 512)
        return feat.view(b, t, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, 3, H, W) RGB clip in [0, 1].

        Returns
        -------
        logits : (B, 1)
        """
        branch_feats = []
        if self.cfg.use_tim:
            tim_clip = self.tim_extract(x)                  # (B, N-1, 3, H, W)
            branch_feats.append(self._run_backbone(self.tim_backbone, tim_clip))

        if self.cfg.use_phase:
            phase_clip = self.phase_extract(x)              # (B, N, 6, H, W)
            if self.cfg.phase_residual:
                phase_clip = (phase_clip[:, 1:] - phase_clip[:, :-1]).abs()
            branch_feats.append(self._run_backbone(self.phase_backbone, phase_clip))

        return self.head(branch_feats)


def build_model_from_args(args) -> PhaseTransformerDetector:
    """Convenience factory used by train.py."""
    variant_map = {
        "v1": dict(use_tim=True,  use_phase=False, use_mask=False, phase_residual=False),
        "v2": dict(use_tim=False, use_phase=True,  use_mask=False, phase_residual=False),
        "v3": dict(use_tim=True,  use_phase=True,  use_mask=False, phase_residual=False),
        "v4": dict(use_tim=True,  use_phase=True,  use_mask=True,  phase_residual=False),
        "v5": dict(use_tim=False, use_phase=True,  use_mask=False, phase_residual=True),
        "v6": dict(use_tim=False, use_phase=True,  use_mask=True,  phase_residual=True),
    }
    flags = variant_map[args.variant]
    cfg = ModelConfig(
        **flags,
        mask_ratio=args.mask_ratio,
        mask_radius_ratio=args.mask_radius_ratio,
        phase_mode=args.phase_mode,
        phase_confidence_quantile=args.phase_confidence_quantile,
        phase_mid_low=args.phase_mid_low,
        phase_mid_high=args.phase_mid_high,
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_len=max(args.n_frames + 1, 32),
    )
    return PhaseTransformerDetector(cfg)
