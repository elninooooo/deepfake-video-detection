"""Compare projected 128D pair features across real relation-mode checkpoints.

Each checkpoint is loaded as its own real model:

    relation mode -> projection module -> 128D pair-level feature

The script does not train a classifier. It compares the real/fake distribution
separability of projected pair features under the same dataset and CRF protocol.
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
    p = argparse.ArgumentParser("Compare projected 128D pair features across relation modes")
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Relation checkpoints in tag=path format, e.g. cos_only=checkpoints/.../best.pth",
    )
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_uniform16_all")
    p.add_argument("--crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument(
        "--sampling_mode",
        choices=sorted(CelebDFClipDataset.SAMPLING_MODES),
        default="global16",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples_per_class", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results/v10_projected_feature_compare")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def parse_model_specs(specs):
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected tag=path model spec, got: {spec}")
        tag, ckpt = spec.split("=", 1)
        tag = tag.strip()
        ckpt = ckpt.strip()
        if not tag or not ckpt:
            raise ValueError(f"Invalid model spec: {spec}")
        parsed.append((tag, ckpt))
    return parsed


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
        features.append(frame_model.branch(x).detach().cpu().numpy())
        labels.append(y.numpy().astype(np.int64))
        videos.extend([m["video"] for m in metas])
    return videos, np.concatenate(labels, axis=0), np.concatenate(features, axis=0)


def fit_pca(all_features, n_components=3):
    flat = all_features.reshape(-1, all_features.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:n_components]


def feature_stat_values(features, pca_mean, pca_components):
    video_feat = features.mean(axis=1).astype(np.float64)
    pca_values = (video_feat - pca_mean) @ pca_components.T
    values = {
        "feature_mean": video_feat.mean(axis=1),
        "feature_std": video_feat.std(axis=1),
        "feature_l2": np.linalg.norm(video_feat, axis=1),
    }
    for idx in range(pca_values.shape[1]):
        values[f"pc{idx + 1}"] = pca_values[:, idx]
    return values


def empirical_overlap(real_values, fake_values, bins=50):
    real_values = np.asarray(real_values, dtype=np.float64)
    fake_values = np.asarray(fake_values, dtype=np.float64)
    lo = float(min(real_values.min(), fake_values.min()))
    hi = float(max(real_values.max(), fake_values.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1.0
    hist_real, edges = np.histogram(real_values, bins=bins, range=(lo, hi), density=True)
    hist_fake, _ = np.histogram(fake_values, bins=edges, density=True)
    return float(np.minimum(hist_real, hist_fake).sum() * (edges[1] - edges[0]))


def distribution_metrics(labels, values):
    labels = np.asarray(labels).astype(int)
    values = np.asarray(values, dtype=np.float64)
    real = values[labels == 0]
    fake = values[labels == 1]
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


def build_stability_rows(metric_rows):
    grouped = {}
    for row in metric_rows:
        grouped.setdefault((row["model_tag"], row["relation_mode"], row["feature_stat"]), []).append(row)
    stability_rows = []
    for (model_tag, relation_mode, stat_name), rows in sorted(grouped.items()):
        ds = np.array([float(row["abs_cohen_d"]) for row in rows], dtype=np.float64)
        by_crf = {row["crf"]: float(row["abs_cohen_d"]) for row in rows}
        stability_rows.append(
            {
                "model_tag": model_tag,
                "relation_mode": relation_mode,
                "feature_stat": stat_name,
                "mean_abs_cohen_d": float(ds.mean()),
                "std_abs_cohen_d_across_crfs": float(ds.std(ddof=1)) if ds.size > 1 else 0.0,
                "min_abs_cohen_d": float(ds.min()),
                "max_abs_cohen_d": float(ds.max()),
                "crf_src_abs_cohen_d": by_crf.get("crf_src", float("nan")),
                "crf23_abs_cohen_d": by_crf.get("crf23", float("nan")),
                "crf40_abs_cohen_d": by_crf.get("crf40", float("nan")),
            }
        )
    return stability_rows


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out_dir: Path, args, stability_rows, model_specs):
    focus = [row for row in stability_rows if row["feature_stat"] == "pc1"]
    focus = sorted(focus, key=lambda r: -r["mean_abs_cohen_d"])
    lines = [
        "# Projected Pair-Feature Comparison across Relation Modes",
        "",
        "This probe compares real checkpoints after their own relation-to-128D projection.",
        "",
        f"- split: `{args.split}`",
        f"- sampling mode: `{args.sampling_mode}`",
        f"- face cache: `{args.face_cache}`",
        f"- CRFs: `{', '.join(args.crfs)}`",
        "",
        "## Model Checkpoints",
        "",
    ]
    for tag, ckpt in model_specs:
        lines.append(f"- `{tag}`: `{ckpt}`")
    lines.extend(
        [
            "",
            "## Main PC1 Separability",
            "",
            "| model | relation mode | mean abs Cohen d | std across CRFs | crf_src d | crf23 d | crf40 d |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in focus:
        lines.append(
            "| {model_tag} | {relation_mode} | {mean_abs_cohen_d:.4f} | "
            "{std_abs_cohen_d_across_crfs:.4f} | {crf_src_abs_cohen_d:.4f} | "
            "{crf23_abs_cohen_d:.4f} | {crf40_abs_cohen_d:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Use `projected_feature_metrics.csv` for per-CRF statistics and "
            "`projected_feature_stability.csv` for cross-CRF stability.",
            "",
            "PCA is fitted separately for each checkpoint because different relation modes "
            "produce different 128D feature spaces.",
        ]
    )
    (out_dir / "projected_feature_compare.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_specs = parse_model_specs(args.models)

    metric_rows = []
    per_video_rows = []
    model_info_rows = []
    for model_tag, ckpt in model_specs:
        frame_model, frame_args = load_frozen_frame_model(ckpt, args, device)
        relation_mode = getattr(frame_args, "relation_mode", "unknown")
        model_info_rows.append(
            {
                "model_tag": model_tag,
                "checkpoint": ckpt,
                "relation_mode": relation_mode,
                "residual_mode": getattr(frame_args, "residual_mode", "unknown"),
                "feature_dim": getattr(frame_args, "spectral_relation_dim", "unknown"),
            }
        )

        collected = {}
        all_features = []
        for crf in args.crfs:
            videos, labels, features = collect_features(args, crf, frame_model, device)
            print(
                f"{model_tag}/{crf}: clips={len(labels)} real={int((labels == 0).sum())} "
                f"fake={int((labels == 1).sum())}"
            )
            collected[crf] = (videos, labels, features)
            all_features.append(features)

        pca_mean, pca_components = fit_pca(np.concatenate(all_features, axis=0), n_components=3)
        for crf, (videos, labels, features) in collected.items():
            stat_values = feature_stat_values(features, pca_mean, pca_components)
            for stat_name, values in stat_values.items():
                metric_rows.append(
                    {
                        "model_tag": model_tag,
                        "relation_mode": relation_mode,
                        "crf": crf,
                        "feature_stat": stat_name,
                        **distribution_metrics(labels, values),
                    }
                )
            for idx, video in enumerate(videos):
                for stat_name, values in stat_values.items():
                    per_video_rows.append(
                        {
                            "model_tag": model_tag,
                            "relation_mode": relation_mode,
                            "crf": crf,
                            "video": video,
                            "label": int(labels[idx]),
                            "feature_stat": stat_name,
                            "value": float(values[idx]),
                        }
                    )

    stability_rows = build_stability_rows(metric_rows)
    metric_fields = [
        "model_tag",
        "relation_mode",
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
    stability_fields = [
        "model_tag",
        "relation_mode",
        "feature_stat",
        "mean_abs_cohen_d",
        "std_abs_cohen_d_across_crfs",
        "min_abs_cohen_d",
        "max_abs_cohen_d",
        "crf_src_abs_cohen_d",
        "crf23_abs_cohen_d",
        "crf40_abs_cohen_d",
    ]
    per_video_fields = [
        "model_tag",
        "relation_mode",
        "crf",
        "video",
        "label",
        "feature_stat",
        "value",
    ]
    model_info_fields = ["model_tag", "checkpoint", "relation_mode", "residual_mode", "feature_dim"]

    write_csv(out_dir / "projected_feature_metrics.csv", metric_rows, metric_fields)
    write_csv(out_dir / "projected_feature_stability.csv", stability_rows, stability_fields)
    write_csv(out_dir / "projected_feature_per_video.csv", per_video_rows, per_video_fields)
    write_csv(out_dir / "projected_feature_model_info.csv", model_info_rows, model_info_fields)
    write_summary(out_dir, args, stability_rows, model_specs)
    print(f"Done. Results written to {out_dir}")


if __name__ == "__main__":
    main()
