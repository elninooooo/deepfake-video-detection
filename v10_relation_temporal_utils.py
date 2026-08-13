"""Utilities for frozen v10 cosine-relation temporal experiments."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_v10_frame_supervised import V10FrameClassifier


RELATION_NAMES = [
    "orig_low",
    "orig_mid",
    "orig_high",
    "low_mid",
    "low_high",
    "mid_high",
]

RELATION_GROUPS = {
    "orig_low": [0],
    "orig_mid": [1],
    "orig_high": [2],
    "low_mid": [3],
    "low_high": [4],
    "mid_high": [5],
    "low_mid_group": [0, 1, 3],
    "high_related": [2, 4, 5],
    "all_relations": [0, 1, 2, 3, 4, 5],
}

STAT_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "slope",
    "smoothness",
    "first_half_mean",
    "second_half_mean",
    "half_diff",
]


def frame_args_from_checkpoint(state, cli_args):
    ckpt_args = dict(state.get("args", {}) if isinstance(state, dict) else {})
    ckpt_args.setdefault("relation_mode", "full")
    ckpt_args["splits"] = cli_args.splits
    ckpt_args["face_cache"] = cli_args.face_cache
    ckpt_args["n_frames"] = cli_args.n_frames
    return SimpleNamespace(**ckpt_args)


def load_frozen_frame_model(frame_ckpt: str, cli_args, device):
    state = torch.load(frame_ckpt, map_location=device)
    frame_args = frame_args_from_checkpoint(state, cli_args)
    frame_model = V10FrameClassifier(frame_args).to(device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    frame_model.load_state_dict(sd)
    frame_model.eval()
    for param in frame_model.parameters():
        param.requires_grad_(False)
    return frame_model, frame_args


@torch.no_grad()
def raw_cosine_relations(branch, x: torch.Tensor):
    """Return raw six cosine relations with shape (B,T,6)."""
    branch.eval()
    residual = branch._residual(x)
    b, t = residual.shape[:2]
    views = branch._frequency_views(residual)
    feats = branch.encoder(torch.cat(views, dim=0)).chunk(4, dim=0)
    cos_parts = []
    for i, j in branch.pairs:
        cos_parts.append(F.cosine_similarity(feats[i], feats[j], dim=1, eps=1e-8))
    return torch.stack(cos_parts, dim=1).view(b, t, len(branch.pairs))


@torch.no_grad()
def pair_scores(frame_model, x: torch.Tensor):
    """Return per-adjacent-pair fake probabilities with shape (B,T)."""
    frame_model.eval()
    scores = []
    for idx in range(x.size(1) - 1):
        logits = frame_model.forward_pair(x[:, idx:idx + 2])
        scores.append(torch.sigmoid(logits))
    return torch.stack(scores, dim=1)


def temporal_stats_np(seq):
    """Compute temporal stats for an array shaped (N,T,D)."""
    import numpy as np

    seq = np.asarray(seq, dtype=np.float32)
    mean = seq.mean(axis=1)
    std = seq.std(axis=1)
    min_v = seq.min(axis=1)
    max_v = seq.max(axis=1)
    range_v = max_v - min_v
    if seq.shape[1] > 1:
        smoothness = np.abs(np.diff(seq, axis=1)).mean(axis=1)
    else:
        smoothness = np.zeros_like(mean)

    t = np.arange(seq.shape[1], dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-6)
    slope = (seq * t.reshape(1, -1, 1)).mean(axis=1)

    split = max(1, seq.shape[1] // 2)
    first = seq[:, :split].mean(axis=1)
    second = seq[:, split:].mean(axis=1) if split < seq.shape[1] else first
    half_diff = second - first
    return np.concatenate(
        [mean, std, min_v, max_v, range_v, slope, smoothness, first, second, half_diff],
        axis=1,
    )
