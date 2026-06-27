"""Lightweight probe for TIM spectral relationship features.

This script is a small validation step before building a heavier model. It
extracts hand-crafted spectral relationship statistics from TIM maps, trains a
Logistic Regression classifier on mixed CRF clips, and evaluates per CRF.

The goal is not to be a final detector. It is a fast sanity check for whether
TIM frequency relationships and mid-band phase statistics carry useful signal.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
    p = argparse.ArgumentParser("Probe TIM spectral relationship features")
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_s2_all")
    p.add_argument("--train_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--test_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--phase_mid_low", type=float, default=0.10)
    p.add_argument("--phase_mid_high", type=float, default=0.70)
    p.add_argument("--phase_confidence_quantile", type=float, default=0.95)
    p.add_argument("--max_train_per_crf", type=int, default=120)
    p.add_argument("--max_test_per_crf", type=int, default=0, help="0 means use all available test clips")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results/tim_spectral_relation_probe")
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
    metas = [b[2] for b in batch]
    return xs, ys, metas


def balanced_subset(dataset: CelebDFClipDataset, max_items: int, seed: int) -> Subset | CelebDFClipDataset:
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
        rest = [i for i in range(len(dataset)) if i not in set(chosen)]
        rng.shuffle(rest)
        chosen.extend(rest[: max_items - len(chosen)])
    rng.shuffle(chosen)
    return Subset(dataset, chosen)


def build_dataset(args, crfs, split: str, max_per_crf: int):
    datasets = []
    for i, crf in enumerate(crfs):
        ds = CelebDFClipDataset(
            args.splits,
            args.face_cache,
            crf_tag=crf,
            split=split,
            n_frames=args.n_frames,
            train=(split == "train"),
            horiz_flip=False,
        )
        datasets.append(balanced_subset(ds, max_per_crf, args.seed + i))
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def radius_masks(h: int, w: int, low: float, high: float, device: torch.device):
    yy = torch.linspace(-1.0, 1.0, h, device=device)
    xx = torch.linspace(-1.0, 1.0, w, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square()) / np.sqrt(2.0)
    low_mask = (radius < low).float()
    mid_mask = ((radius >= low) & (radius <= high)).float()
    high_mask = (radius > high).float()
    return low_mask, mid_mask, high_mask


def stats(x: torch.Tensor):
    return torch.stack(
        [
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            x.amin(dim=1),
            x.amax(dim=1),
        ],
        dim=1,
    )


def cosine_per_sample(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a = a.flatten(2)
    b = b.flatten(2)
    return (a * b).sum(dim=2) / (a.norm(dim=2) * b.norm(dim=2) + eps)


def extract_tim_spectral_features(
    x: torch.Tensor,
    phase_mid_low: float,
    phase_mid_high: float,
    confidence_quantile: float,
):
    tim = (x[:, 1:] - x[:, :-1]).abs()
    b, t, c, h, w = tim.shape
    flat = tim.reshape(b, t * c, h, w)

    spec = torch.fft.fftshift(torch.fft.fft2(flat, norm="ortho"), dim=(-2, -1))
    low_mask, mid_mask, high_mask = radius_masks(h, w, phase_mid_low, phase_mid_high, x.device)
    masks = [low_mask, mid_mask, high_mask]

    parts = []
    for mask in masks:
        masked = torch.fft.ifftshift(spec * mask.view(1, 1, h, w), dim=(-2, -1))
        parts.append(torch.fft.ifft2(masked, norm="ortho").real)

    orig = flat
    low_part, mid_part, high_part = parts
    cos_ol = cosine_per_sample(orig, low_part)
    cos_om = cosine_per_sample(orig, mid_part)
    cos_oh = cosine_per_sample(orig, high_part)
    cos_lm = cosine_per_sample(low_part, mid_part)
    cos_mh = cosine_per_sample(mid_part, high_part)

    energy_total = orig.square().flatten(2).mean(dim=2) + 1e-8
    energy_feats = []
    for part in parts:
        energy_feats.append(part.square().flatten(2).mean(dim=2) / energy_total)

    amp = spec.abs()
    log_amp = torch.log1p(amp)
    denom = torch.quantile(
        log_amp.flatten(-2),
        confidence_quantile,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)
    confidence = (log_amp.flatten(-2) / denom).clamp(0.0, 1.0).view_as(log_amp)
    phase = torch.angle(spec)
    mid = mid_mask.view(1, 1, h, w)
    sin_mid = confidence * mid * torch.sin(phase)
    cos_mid = confidence * mid * torch.cos(phase)
    phase_energy = torch.stack(
        [
            sin_mid.flatten(2).mean(dim=2),
            sin_mid.flatten(2).std(dim=2, unbiased=False),
            cos_mid.flatten(2).mean(dim=2),
            cos_mid.flatten(2).std(dim=2, unbiased=False),
        ],
        dim=1,
    )

    phase_pair = torch.cat([sin_mid, cos_mid], dim=1).reshape(b, 2, t, c, h, w)
    phase_pair = phase_pair.permute(0, 2, 1, 3, 4, 5).flatten(2)
    if phase_pair.size(1) > 1:
        temporal_phase_cos = (
            phase_pair[:, 1:] * phase_pair[:, :-1]
        ).sum(dim=2) / (phase_pair[:, 1:].norm(dim=2) * phase_pair[:, :-1].norm(dim=2) + 1e-8)
        temporal_stats = stats(temporal_phase_cos)
    else:
        temporal_stats = torch.zeros(b, 4, device=x.device)

    feature_blocks = [
        stats(cos_ol),
        stats(cos_om),
        stats(cos_oh),
        stats(cos_lm),
        stats(cos_mh),
        *[stats(e) for e in energy_feats],
        phase_energy.flatten(1),
        temporal_stats,
    ]
    return torch.cat(feature_blocks, dim=1)


@torch.no_grad()
def collect_features(args, dataset, device, desc):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_drop_meta,
    )
    feats, labels = [], []
    for x, y, _ in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        feats.append(
            extract_tim_spectral_features(
                x,
                args.phase_mid_low,
                args.phase_mid_high,
                args.phase_confidence_quantile,
            )
            .cpu()
            .numpy()
        )
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0).astype(int)


def report_row(name: str, y_true, y_score):
    report = evaluate(y_true, y_score)
    eer, threshold = compute_eer(np.asarray(y_true), np.asarray(y_score))
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "split": name,
        "acc_at_eer": float(accuracy_score(y_true, y_pred)),
        "precision_at_eer": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_eer": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_eer": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": report.auc,
        "eer": eer,
        "threshold_eer": threshold,
        "n_samples": int(len(y_true)),
        "score_mean_real": float(np.mean(np.asarray(y_score)[np.asarray(y_true) == 0])),
        "score_mean_fake": float(np.mean(np.asarray(y_score)[np.asarray(y_true) == 1])),
    }


def write_rows(rows, out_csv: Path, meta: dict):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    train_set = build_dataset(args, args.train_crfs, "train", args.max_train_per_crf)
    x_train, y_train = collect_features(args, train_set, device, "train probe features")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
    )
    clf.fit(x_train, y_train)

    rows = []
    all_scores, all_labels = [], []
    for crf in args.test_crfs:
        test_set = build_dataset(args, [crf], "test", args.max_test_per_crf)
        x_test, y_test = collect_features(args, test_set, device, crf)
        scores = clf.predict_proba(x_test)[:, 1]
        rows.append(report_row(crf, y_test, scores))
        all_scores.append(scores)
        all_labels.append(y_test)
    rows.append(report_row("mixed", np.concatenate(all_labels), np.concatenate(all_scores)))

    out_csv = Path(args.out_dir) / "tim_spectral_relation_probe.csv"
    write_rows(rows, out_csv, vars(args) | {"n_train": int(len(y_train)), "feature_dim": int(x_train.shape[1])})
    print(f"Wrote {out_csv}")
    for row in rows:
        print(
            f"{row['split']:>7} AUC={row['auc']:.4f} "
            f"F1@EER={row['f1_at_eer']:.4f} "
            f"real={row['score_mean_real']:.4f} fake={row['score_mean_fake']:.4f}"
        )


if __name__ == "__main__":
    main()
