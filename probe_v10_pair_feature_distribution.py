"""Analyze GRFR 128D pair-feature distributions.

This script probes the projected pair-level feature produced by the trained
cosine-relation GRFR frame branch. It does not train a classifier; it compares
real/fake separability of simple summaries and PCA components of the 128D pair
feature across compression levels.
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from train import collate_drop_meta
from v10_relation_temporal_utils import load_frozen_frame_model


FEATURE_STATS = ("feature_mean", "feature_std", "feature_l2", "pc1", "pc2", "pc3")


def parse_args():
    p = argparse.ArgumentParser("Analyze projected GRFR 128D pair-feature distributions")
    p.add_argument("--frame_ckpt", required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_uniform16_all")
    p.add_argument("--crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--sampling_mode", choices=sorted(CelebDFClipDataset.SAMPLING_MODES),
                   default="global16")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples_per_class", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results/v10_pair_feature_distribution")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def balanced_subset(dataset, max_per_class: int, seed: int):
    if max_per_class <= 0:
        return dataset
    by_label = {0: [], 1: []}
    for idx, rec in enumerate(dataset.records):
        by_label[int(rec["label"])].append(idx)
    rng = random.Random(seed)
    chosen = []
    for label in (0, 1):
        ids = by_label[label][:]
        rng.shuffle(ids)
        chosen.extend(ids[: min(max_per_class, len(ids))])
    chosen.sort()
    return Subset(dataset, chosen)


def build_dataset(args, crf: str):
    dataset = CelebDFClipDataset(
        args.splits,
        args.face_cache,
        crf_tag=crf,
        split=args.split,
        n_frames=args.n_frames,
        train=False,
        horiz_flip=False,
        sampling_mode=args.sampling_mode,
    )
    return balanced_subset(dataset, args.max_samples_per_class, args.seed)


@torch.no_grad()
def collect_features(args, crf: str, frame_model, device):
    dataset = build_dataset(args, crf)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_drop_meta,
    )
    features = []
    labels = []
    videos = []
    for x, y, metas in tqdm(loader, desc=f"collect/{crf}", leave=False):
        x = x.to(device, non_blocking=True)
        pair_feat = frame_model.branch(x).detach().cpu().numpy()
        features.append(pair_feat)
        labels.append(y.numpy().astype(np.int64))
        videos.extend([m["video"] for m in metas])
    return videos, np.concatenate(labels, axis=0), np.concatenate(features, axis=0)


def fit_pca(all_features, n_components=3):
    flat = all_features.reshape(-1, all_features.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return mean, components


def project_pca(features, mean, components):
    video_feat = features.mean(axis=1).astype(np.float64)
    return (video_feat - mean) @ components.T


def feature_stat_values(features, pca_values):
    video_feat = features.mean(axis=1)
    rows = {
        "feature_mean": video_feat.mean(axis=1),
        "feature_std": video_feat.std(axis=1),
        "feature_l2": np.linalg.norm(video_feat, axis=1),
    }
    for idx in range(pca_values.shape[1]):
        rows[f"pc{idx + 1}"] = pca_values[:, idx]
    return rows


def empirical_overlap(real_values, fake_values, bins=50):
    lo = float(min(real_values.min(), fake_values.min()))
    hi = float(max(real_values.max(), fake_values.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1.0
    hist_real, edges = np.histogram(real_values, bins=bins, range=(lo, hi), density=True)
    hist_fake, _ = np.histogram(fake_values, bins=edges, density=True)
    return float(np.minimum(hist_real, hist_fake).sum() * (edges[1] - edges[0]))


def distribution_metrics(labels, values):
    real = np.asarray(values)[labels == 0].astype(np.float64)
    fake = np.asarray(values)[labels == 1].astype(np.float64)
    n_real = real.size
    n_fake = fake.size
    std_real = real.std(ddof=1) if n_real > 1 else 0.0
    std_fake = fake.std(ddof=1) if n_fake > 1 else 0.0
    denom = max(1, n_real + n_fake - 2)
    pooled = math.sqrt(((n_real - 1) * std_real ** 2 + (n_fake - 1) * std_fake ** 2) / denom)
    signed_gap = float(fake.mean() - real.mean())
    signed_d = signed_gap / (pooled + 1e-12)
    return {
        "n_real": int(n_real),
        "n_fake": int(n_fake),
        "mean_real": float(real.mean()),
        "mean_fake": float(fake.mean()),
        "std_real": float(std_real),
        "std_fake": float(std_fake),
        "signed_gap": signed_gap,
        "abs_gap": abs(signed_gap),
        "signed_cohen_d": signed_d,
        "abs_cohen_d": abs(signed_d),
        "overlap": empirical_overlap(real, fake),
    }


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_model, _ = load_frozen_frame_model(args.frame_ckpt, args, device)

    collected = {}
    all_features = []
    for crf in args.crfs:
        videos, labels, features = collect_features(args, crf, frame_model, device)
        print(
            f"{crf}: clips={len(labels)} real={int((labels == 0).sum())} "
            f"fake={int((labels == 1).sum())}"
        )
        collected[crf] = (videos, labels, features)
        all_features.append(features)

    pca_mean, pca_components = fit_pca(np.concatenate(all_features, axis=0), n_components=3)
    rows = []
    per_video_rows = []
    for crf, (videos, labels, features) in collected.items():
        pca_values = project_pca(features, pca_mean, pca_components)
        stats = feature_stat_values(features, pca_values)
        for stat_name, values in stats.items():
            rows.append({"crf": crf, "feature_stat": stat_name, **distribution_metrics(labels, values)})
        for idx, video in enumerate(videos):
            for stat_name, values in stats.items():
                per_video_rows.append(
                    {
                        "crf": crf,
                        "video": video,
                        "label": int(labels[idx]),
                        "feature_stat": stat_name,
                        "value": float(values[idx]),
                    }
                )

    stability_rows = []
    for stat_name in FEATURE_STATS:
        stat_rows = [row for row in rows if row["feature_stat"] == stat_name]
        ds = np.array([float(row["abs_cohen_d"]) for row in stat_rows], dtype=np.float64)
        by_crf = {row["crf"]: float(row["abs_cohen_d"]) for row in stat_rows}
        stability_rows.append(
            {
                "feature_stat": stat_name,
                "mean_abs_cohen_d": float(ds.mean()),
                "std_abs_cohen_d_across_crfs": float(ds.std(ddof=1)) if ds.size > 1 else 0.0,
                "crf_src_abs_cohen_d": by_crf.get("crf_src", float("nan")),
                "crf40_abs_cohen_d": by_crf.get("crf40", float("nan")),
            }
        )

    metric_fields = [
        "crf",
        "feature_stat",
        "n_real",
        "n_fake",
        "mean_real",
        "mean_fake",
        "std_real",
        "std_fake",
        "signed_gap",
        "abs_gap",
        "signed_cohen_d",
        "abs_cohen_d",
        "overlap",
    ]
    per_video_fields = ["crf", "video", "label", "feature_stat", "value"]
    stability_fields = [
        "feature_stat",
        "mean_abs_cohen_d",
        "std_abs_cohen_d_across_crfs",
        "crf_src_abs_cohen_d",
        "crf40_abs_cohen_d",
    ]
    write_csv(out_dir / "pair_feature_distribution_metrics.csv", rows, metric_fields)
    write_csv(out_dir / "pair_feature_distribution_per_video.csv", per_video_rows, per_video_fields)
    write_csv(out_dir / "pair_feature_compression_stability.csv", stability_rows, stability_fields)

    lines = [
        "# GRFR 128D Pair-Feature Distribution Probe",
        "",
        f"- frame checkpoint: `{args.frame_ckpt}`",
        f"- split: `{args.split}`",
        f"- CRFs: `{', '.join(args.crfs)}`",
        "",
        "| feature statistic | mean abs Cohen d | std across CRFs | crf_src d | crf40 d |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(stability_rows, key=lambda r: -r["mean_abs_cohen_d"]):
        lines.append(
            "| {feature_stat} | {mean_abs_cohen_d:.4f} | "
            "{std_abs_cohen_d_across_crfs:.4f} | {crf_src_abs_cohen_d:.4f} | "
            "{crf40_abs_cohen_d:.4f} |".format(**row)
        )
    (out_dir / "pair_feature_distribution_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. Results written to {out_dir}")


if __name__ == "__main__":
    main()
