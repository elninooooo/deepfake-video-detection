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
from torch.utils.data import Dataset


def _read_jpg(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class CelebDFClipDataset(Dataset):
    def __init__(self,
                 split_json: str,
                 face_cache_dir: str,
                 crf_tag: str = "crf23",
                 split: str = "train",
                 n_frames: int = 16,
                 train: bool = True,
                 horiz_flip: bool = True):
        super().__init__()
        with open(split_json, "r", encoding="utf-8") as f:
            self.splits = json.load(f)
        self.records_all = self.splits[split]
        self.cache_root = Path(face_cache_dir).resolve() / crf_tag
        self.crf_tag = crf_tag
        self.n_frames = n_frames
        self.train = train
        self.horiz_flip = horiz_flip

        index_path = self.cache_root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing {index_path}. Run preprocess_faces.py first.")
        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        # keep only videos that have >= n_frames extracted faces
        self.records = [r for r in self.records_all
                        if self.index.get(r["path"], 0) >= n_frames]
        if len(self.records) == 0:
            raise RuntimeError(
                f"No records left for split={split} crf={crf_tag}. "
                f"Cache root={self.cache_root}.  Check preprocess_faces.py output.")

    def __len__(self):
        return len(self.records)

    def _sample_indices(self, total: int) -> List[int]:
        n = self.n_frames
        if self.train:
            # random contiguous window
            start = random.randint(0, total - n)
            return list(range(start, start + n))
        # uniform sampling
        return np.linspace(0, total - 1, n).round().astype(int).tolist()

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


def make_class_balanced_sampler(dataset: CelebDFClipDataset):
    """Return a WeightedRandomSampler that draws real and fake clips with equal probability."""
    from torch.utils.data import WeightedRandomSampler
    labels = np.array([r["label"] for r in dataset.records])
    n_real = max(1, int((labels == 0).sum()))
    n_fake = max(1, int((labels == 1).sum()))
    w = np.where(labels == 0, 1.0 / n_real, 1.0 / n_fake)
    return WeightedRandomSampler(weights=w.tolist(), num_samples=len(dataset), replacement=True)
