"""Real-only multi-layer ResNet spectral relationship probe.

This script tests a SPAI-like idea without training a new encoder:

1. Build temporal residual maps from RGB clips.
2. Split each residual into original / low / mid / high frequency views.
3. Feed all views through a frozen ImageNet-pretrained ResNet.
4. Compute cosine relationships between frequency views at layer1..layer4.
5. Fit a real-only Gaussian model over these relationship vectors.
6. Score test clips by Mahalanobis distance from the real relationship model.

Higher score means farther from the learned real residual spectral relation.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from utils.metrics import compute_eer, evaluate
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser("Probe real-only ResNet spectral relationships")
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_s2_all")
    p.add_argument("--train_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--test_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--residual_mode", choices=["abs", "signed", "gradient"], default="abs")
    p.add_argument("--mid_low", type=float, default=0.10)
    p.add_argument("--mid_high", type=float, default=0.70)
    p.add_argument("--resnet", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--probe_mode", choices=["oneclass", "supervised"], default="oneclass",
                   help="oneclass fits only real samples; supervised fits real/fake labels.")
    p.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4],
                   choices=[1, 2, 3, 4],
                   help="ResNet layers used for relation features.")
    p.add_argument("--torch_home", default=".cache/torch",
                   help="Torch model cache directory for pretrained ResNet weights.")
    p.add_argument("--max_train_per_crf", type=int, default=160)
    p.add_argument("--max_test_per_crf", type=int, default=0, help="0 means all available clips")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results/resnet_spectral_relation_oneclass")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def collate_drop_meta(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


def keep_real_subset(dataset: CelebDFClipDataset):
    dataset.records = [r for r in dataset.records if int(r["label"]) == 0]
    if not dataset.records:
        raise RuntimeError(f"No real clips for split={dataset.crf_tag}.")
    return dataset


def balanced_subset(dataset: CelebDFClipDataset, max_items: int, seed: int):
    if max_items <= 0 or len(dataset) <= max_items:
        return dataset
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for idx, rec in enumerate(dataset.records):
        by_label[int(rec["label"])].append(idx)
    per_label = max_items // 2
    chosen = []
    for label in (0, 1):
        ids = by_label[label]
        rng.shuffle(ids)
        chosen.extend(ids[: min(per_label, len(ids))])
    if len(chosen) < max_items:
        chosen_set = set(chosen)
        rest = [i for i in range(len(dataset)) if i not in chosen_set]
        rng.shuffle(rest)
        chosen.extend(rest[: max_items - len(chosen)])
    rng.shuffle(chosen)
    return Subset(dataset, chosen)


def build_dataset(args, crfs, split: str, real_only: bool, max_per_crf: int):
    datasets = []
    for i, crf in enumerate(crfs):
        dataset = CelebDFClipDataset(
            args.splits,
            args.face_cache,
            crf_tag=crf,
            split=split,
            n_frames=args.n_frames,
            train=(split == "train"),
            horiz_flip=False,
        )
        if real_only:
            dataset = keep_real_subset(dataset)
        else:
            dataset = balanced_subset(dataset, max_per_crf, args.seed + i)
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


class MultiLayerResNet(nn.Module):
    def __init__(self, name: str, weights_mode: str):
        super().__init__()
        try:
            from torchvision import models
        except Exception as exc:
            raise RuntimeError(
                "torchvision is required for pretrained ResNet features. "
                "Install the PyTorch/torchvision pair for your CUDA version."
            ) from exc

        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if weights_mode == "imagenet" else None
            net = models.resnet18(weights=weights)
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if weights_mode == "imagenet" else None
            net = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet: {name}")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


def radius_masks(h: int, w: int, low: float, high: float, device: torch.device):
    yy = torch.linspace(-1.0, 1.0, h, device=device)
    xx = torch.linspace(-1.0, 1.0, w, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square()) / np.sqrt(2.0)
    low_mask = (radius < low).float()
    mid_mask = ((radius >= low) & (radius <= high)).float()
    high_mask = (radius > high).float()
    return low_mask, mid_mask, high_mask


def sobel_gray(x: torch.Tensor):
    gray = 0.2989 * x[:, :, 0:1] + 0.5870 * x[:, :, 1:2] + 0.1140 * x[:, :, 2:3]
    b, n, _, h, w = gray.shape
    flat = gray.reshape(b * n, 1, h, w)
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=flat.dtype,
        device=flat.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=flat.dtype,
        device=flat.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(flat, kx, padding=1)
    gy = F.conv2d(flat, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
    return mag.view(b, n, 1, h, w).repeat(1, 1, 3, 1, 1)


def temporal_residual(x: torch.Tensor, mode: str):
    if mode == "abs":
        return (x[:, 1:] - x[:, :-1]).abs()
    if mode == "signed":
        return x[:, 1:] - x[:, :-1]
    if mode == "gradient":
        g = sobel_gray(x)
        return (g[:, 1:] - g[:, :-1]).abs()
    raise ValueError(f"Unknown residual mode: {mode}")


def split_frequency_views(residual: torch.Tensor, low: float, high: float):
    b, t, c, h, w = residual.shape
    flat = residual.reshape(b * t, c, h, w)
    spec = torch.fft.fftshift(torch.fft.fft2(flat, norm="ortho"), dim=(-2, -1))
    masks = radius_masks(h, w, low, high, residual.device)
    views = [flat]
    for mask in masks:
        masked = torch.fft.ifftshift(spec * mask.view(1, 1, h, w), dim=(-2, -1))
        views.append(torch.fft.ifft2(masked, norm="ortho").real)
    return views


def normalize_for_resnet(x: torch.Tensor):
    b = x.size(0)
    flat = x.flatten(1)
    lo = flat.amin(dim=1).view(b, 1, 1, 1)
    hi = flat.amax(dim=1).view(b, 1, 1, 1)
    x = (x - lo) / (hi - lo).clamp_min(1e-6)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def temporal_stats(x: torch.Tensor):
    return torch.cat(
        [
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            x.amin(dim=1),
            x.amax(dim=1),
        ],
        dim=1,
    )


@torch.no_grad()
def extract_relation_features(x: torch.Tensor, encoder: MultiLayerResNet, args):
    residual = temporal_residual(x, args.residual_mode)
    b, t = residual.shape[:2]
    views = split_frequency_views(residual, args.mid_low, args.mid_high)
    inputs = torch.cat([normalize_for_resnet(view) for view in views], dim=0)
    layer_feats = encoder(inputs)
    selected = {layer - 1 for layer in args.layers}

    relation_blocks = []
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for layer_idx, feats in enumerate(layer_feats):
        if layer_idx not in selected:
            continue
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        view_feats = pooled.chunk(4, dim=0)
        sims = []
        dists = []
        for i, j in pairs:
            sims.append(F.cosine_similarity(view_feats[i], view_feats[j], dim=1))
            dists.append((view_feats[i] - view_feats[j]).norm(dim=1))
        sims = torch.stack(sims, dim=1).view(b, t, len(pairs))
        dists = torch.stack(dists, dim=1).view(b, t, len(pairs))
        relation_blocks.append(temporal_stats(sims))
        relation_blocks.append(temporal_stats(dists))
    return torch.cat(relation_blocks, dim=1)


@torch.no_grad()
def collect_features(args, dataset, encoder, device, desc):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_drop_meta,
    )
    feats, labels = [], []
    for x, y in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        feats.append(extract_relation_features(x, encoder, args).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0).astype(int)


def fit_real_model(x_real: np.ndarray):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_real)
    cov = LedoitWolf().fit(x_scaled)
    return scaler, cov


def score_real_model(x: np.ndarray, scaler: StandardScaler, cov: LedoitWolf):
    x_scaled = scaler.transform(x)
    return cov.mahalanobis(x_scaled)


def fit_supervised_model(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    clf.fit(x_train, y_train)
    return clf


def score_supervised_model(x: np.ndarray, clf):
    return clf.predict_proba(x)[:, 1]


def report_row(name: str, y_true, y_score):
    report = evaluate(y_true, y_score)
    eer, threshold = compute_eer(np.asarray(y_true), np.asarray(y_score))
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "split": name,
        "acc_at_eer": float((y_pred == y_true).mean()),
        "auc": report.auc,
        "eer": eer,
        "threshold_eer": threshold,
        "n_samples": int(len(y_true)),
        "score_mean_real": float(np.mean(np.asarray(y_score)[np.asarray(y_true) == 0])),
        "score_mean_fake": float(np.mean(np.asarray(y_score)[np.asarray(y_true) == 1])),
    }


def write_outputs(rows, args, feature_dim: int, n_train: int):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_tag = "l" + "".join(str(layer) for layer in args.layers)
    stem = f"{args.resnet}_{args.residual_mode}_{layer_tag}_relation_{args.probe_mode}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"feature_dim": feature_dim, "n_train": n_train}, f, indent=2)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    os.environ.setdefault("TORCH_HOME", str(Path(args.torch_home).resolve()))
    encoder = MultiLayerResNet(args.resnet, args.weights).to(device).eval()

    if args.probe_mode == "oneclass":
        train_set = build_dataset(args, args.train_crfs, "train", real_only=True, max_per_crf=0)
        if args.max_train_per_crf > 0:
            if isinstance(train_set, ConcatDataset):
                capped = []
                for i, ds in enumerate(train_set.datasets):
                    capped.append(balanced_subset(ds, args.max_train_per_crf, args.seed + i))
                train_set = ConcatDataset(capped)
            else:
                train_set = balanced_subset(train_set, args.max_train_per_crf, args.seed)
        x_train, _ = collect_features(args, train_set, encoder, device, "train real relations")
        scorer = fit_real_model(x_train)
    else:
        train_set = build_dataset(
            args, args.train_crfs, "train", real_only=False, max_per_crf=args.max_train_per_crf
        )
        x_train, y_train = collect_features(args, train_set, encoder, device, "train supervised relations")
        scorer = fit_supervised_model(x_train, y_train, args.seed)

    rows = []
    all_scores, all_labels = [], []
    for crf in args.test_crfs:
        test_set = build_dataset(args, [crf], "test", real_only=False, max_per_crf=args.max_test_per_crf)
        x_test, y_test = collect_features(args, test_set, encoder, device, crf)
        if args.probe_mode == "oneclass":
            scores = score_real_model(x_test, *scorer)
        else:
            scores = score_supervised_model(x_test, scorer)
        rows.append(report_row(crf, y_test, scores))
        all_scores.append(scores)
        all_labels.append(y_test)
    rows.append(report_row("mixed", np.concatenate(all_labels), np.concatenate(all_scores)))
    write_outputs(rows, args, feature_dim=x_train.shape[1], n_train=len(x_train))

    for row in rows:
        print(
            f"{row['split']:>7} AUC={row['auc']:.4f} Acc@EER={row['acc_at_eer']:.4f} "
            f"real={row['score_mean_real']:.4f} fake={row['score_mean_fake']:.4f}"
        )


if __name__ == "__main__":
    main()
