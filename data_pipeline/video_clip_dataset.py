"""PyTorch Dataset over face-cropped video clips produced by preprocess_faces.py.

Returns
-------
frames : float tensor (N, 3, H, W) in [0, 1]
label  : int   0 = real, 1 = fake
meta   : dict  {"video": str, "crf": str, "identity": str}
"""

import json
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


def _read_jpg(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class CelebDFClipDataset(Dataset):
    SAMPLING_MODES = ("legacy", "local16", "global16", "s2", "4x4")

    def __init__(self,
                 split_json: str,
                 face_cache_dir: str,
                 crf_tag: str = "crf23",
                 split: str = "train",
                 n_frames: int = 16,
                 train: bool = True,
                 horiz_flip: bool = True,
                 sampling_mode: str = "legacy"):
        super().__init__()
        if sampling_mode not in self.SAMPLING_MODES:
            raise ValueError(
                f"Unsupported sampling_mode={sampling_mode!r}. "
                f"Choose from {', '.join(self.SAMPLING_MODES)}."
            )
        if sampling_mode == "4x4" and n_frames % 4 != 0:
            raise ValueError("sampling_mode='4x4' requires n_frames divisible by 4.")
        with open(split_json, "r", encoding="utf-8") as f:
            self.splits = json.load(f)
        self.records_all = self.splits[split]
        self.cache_root = Path(face_cache_dir).resolve() / crf_tag
        self.crf_tag = crf_tag
        self.n_frames = n_frames
        self.train = train
        self.horiz_flip = horiz_flip
        self.sampling_mode = sampling_mode

        index_path = self.cache_root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing {index_path}. Run preprocess_faces.py first.")
        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        # keep only videos that have enough extracted faces for the selected protocol
        min_frames = self._min_total_frames()
        self.records = [r for r in self.records_all
                        if self.index.get(r["path"], 0) >= min_frames]
        if len(self.records) == 0:
            raise RuntimeError(
                f"No records left for split={split} crf={crf_tag}. "
                f"Cache root={self.cache_root}. sampling_mode={sampling_mode} "
                f"requires at least {min_frames} cached frames. "
                f"Check preprocess_faces.py output.")

    def __len__(self):
        return len(self.records)

    def _min_total_frames(self) -> int:
        if self.sampling_mode == "s2":
            return (self.n_frames - 1) * 2 + 1
        return self.n_frames

    def _sample_local(self, total: int, n: int) -> List[int]:
        start = random.randint(0, total - n) if self.train else max(0, (total - n) // 2)
        return list(range(start, start + n))

    def _sample_global(self, total: int, n: int) -> List[int]:
        return np.linspace(0, total - 1, n).round().astype(int).tolist()

    def _sample_stride2(self, total: int, n: int) -> List[int]:
        span = (n - 1) * 2 + 1
        start = random.randint(0, total - span) if self.train else max(0, (total - span) // 2)
        return [start + i * 2 for i in range(n)]

    def _sample_4x4(self, total: int, n: int) -> List[int]:
        block = n // 4
        max_start = max(0, total - block)
        if self.train:
            anchors = np.linspace(0, max_start, 4).round().astype(int).tolist()
            jitter = max(0, total // 32)
            starts = []
            for anchor in anchors:
                lo = max(0, anchor - jitter)
                hi = min(max_start, anchor + jitter)
                starts.append(random.randint(lo, hi) if hi >= lo else anchor)
        else:
            starts = np.linspace(0, max_start, 4).round().astype(int).tolist()

        idxs = []
        for start in starts:
            idxs.extend(range(start, start + block))
        return idxs

    def _sample_indices(self, total: int) -> List[int]:
        n = self.n_frames
        if self.sampling_mode == "legacy":
            if self.train:
                return self._sample_local(total, n)
            return self._sample_global(total, n)
        if self.sampling_mode == "local16":
            return self._sample_local(total, n)
        if self.sampling_mode == "global16":
            return self._sample_global(total, n)
        if self.sampling_mode == "s2":
            return self._sample_stride2(total, n)
        if self.sampling_mode == "4x4":
            return self._sample_4x4(total, n)
        raise ValueError(f"Unsupported sampling_mode={self.sampling_mode!r}")

    def _load_clip(self, rec) -> np.ndarray:
        total = self.index[rec["path"]]
        idxs = self._sample_indices(total)
        stem = Path(rec["path"]).with_suffix("")
        frames = []
        for i in idxs:
            fp = self.cache_root / stem / f"frame_{i:03d}.jpg"
            frames.append(_read_jpg(fp))
        return np.stack(frames, axis=0)  # (N, H, W, 3)

    def __getitem__(self, i):
        rec = self.records[i]
        clip = self._load_clip(rec)  # (N, H, W, 3) uint8

        if self.train and self.horiz_flip and random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()

        # (N, H, W, 3) → (N, 3, H, W) float in [0, 1]
        x = torch.from_numpy(clip).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(rec["label"], dtype=torch.float32)
        meta = {"video": rec["path"], "crf": self.crf_tag, "identity": rec["identity"]}
        return x, y, meta


def _dataset_labels(dataset: Dataset) -> List[int]:
    if isinstance(dataset, CelebDFClipDataset):
        return [r["label"] for r in dataset.records]
    if isinstance(dataset, ConcatDataset):
        labels = []
        for child in dataset.datasets:
            labels.extend(_dataset_labels(child))
        return labels
    raise TypeError(f"Unsupported dataset type for balanced sampling: {type(dataset).__name__}")


def make_class_balanced_sampler(dataset: Dataset):
    """Return a WeightedRandomSampler that draws real and fake clips with equal probability."""
    from torch.utils.data import WeightedRandomSampler
    labels = np.array(_dataset_labels(dataset))
    n_real = max(1, int((labels == 0).sum()))
    n_fake = max(1, int((labels == 1).sum()))
    w = np.where(labels == 0, 1.0 / n_real, 1.0 / n_fake)
    return WeightedRandomSampler(weights=w.tolist(), num_samples=len(dataset), replacement=True)
