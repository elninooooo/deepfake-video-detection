"""Probe relation separability and compression stability for GRFR.

This script does not train a classifier. It loads a trained GRFR frame-level
encoder, computes abs/cos/L2 relations among original/low/mid/high residual
frequency-band features, and compares real/fake distribution separability
across compression levels.
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from train import collate_drop_meta
from v10_relation_temporal_utils import RELATION_NAMES, load_frozen_frame_model


STAT_NAMES = ("mean", "std", "min", "max", "range", "smoothness")
RELATION_TYPES = ("cos", "abs", "l2")


def parse_args():
    parser = argparse.ArgumentParser(
        "Analyze GRFR relation distributions without training a classifier."
    )
    parser.add_argument("--frame_ckpt", required=True)
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--face_cache", default="face_cache_uniform16_all")
    parser.add_argument("--crfs", nargs="+", default=["crf_src", "crf23", "crf40"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument(
        "--sampling_mode",
        default="global16",
        choices=sorted(CelebDFClipDataset.SAMPLING_MODES),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=0,
        help="Optional balanced cap per class for each CRF. 0 uses all available samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/v10_relation_distribution")
    return parser.parse_args()


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


@torch.no_grad()
def relation_tensors(branch, x: torch.Tensor):
    """Return relation tensors shaped (B,T,6) for cos, abs, and L2."""
    residual = branch._residual(x)
    b, t = residual.shape[:2]
    views = branch._frequency_views(residual)
    feats = branch.encoder(torch.cat(views, dim=0)).chunk(4, dim=0)

    cos_parts = []
    abs_parts = []
    l2_parts = []
    for i, j in branch.pairs:
        diff = feats[i] - feats[j]
        cos_parts.append(F.cosine_similarity(feats[i], feats[j], dim=1, eps=1e-8))
        abs_parts.append(diff.abs().mean(dim=1))
        l2_parts.append(diff.norm(dim=1))

    return {
        "cos": torch.stack(cos_parts, dim=1).view(b, t, len(branch.pairs)),
        "abs": torch.stack(abs_parts, dim=1).view(b, t, len(branch.pairs)),
        "l2": torch.stack(l2_parts, dim=1).view(b, t, len(branch.pairs)),
    }


def temporal_stat_values(values: np.ndarray):
    """Compute per-video temporal statistics for (N,T,6) values."""
    stats = {
        "mean": values.mean(axis=1),
        "std": values.std(axis=1),
        "min": values.min(axis=1),
        "max": values.max(axis=1),
    }
    stats["range"] = stats["max"] - stats["min"]
    if values.shape[1] > 1:
        stats["smoothness"] = np.abs(np.diff(values, axis=1)).mean(axis=1)
    else:
        stats["smoothness"] = np.zeros_like(stats["mean"])
    return stats


def empirical_overlap(real_values: np.ndarray, fake_values: np.ndarray, bins: int = 50):
    real_values = np.asarray(real_values, dtype=np.float64)
    fake_values = np.asarray(fake_values, dtype=np.float64)
    lo = float(min(real_values.min(), fake_values.min()))
    hi = float(max(real_values.max(), fake_values.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1.0
    hist_real, edges = np.histogram(real_values, bins=bins, range=(lo, hi), density=True)
    hist_fake, _ = np.histogram(fake_values, bins=edges, density=True)
    bin_width = edges[1] - edges[0]
    return float(np.minimum(hist_real, hist_fake).sum() * bin_width)


def distribution_metrics(real_values: np.ndarray, fake_values: np.ndarray):
    real_values = np.asarray(real_values, dtype=np.float64)
    fake_values = np.asarray(fake_values, dtype=np.float64)
    n_real = int(real_values.size)
    n_fake = int(fake_values.size)
    mean_real = float(real_values.mean())
    mean_fake = float(fake_values.mean())
    std_real = float(real_values.std(ddof=1)) if n_real > 1 else 0.0
    std_fake = float(fake_values.std(ddof=1)) if n_fake > 1 else 0.0
    denom = max(1, n_real + n_fake - 2)
    pooled = math.sqrt(
        (((n_real - 1) * std_real * std_real) + ((n_fake - 1) * std_fake * std_fake))
        / denom
    )
    signed_gap = mean_fake - mean_real
    signed_d = signed_gap / (pooled + 1e-12)
    return {
        "n_real": n_real,
        "n_fake": n_fake,
        "mean_real": mean_real,
        "mean_fake": mean_fake,
        "std_real": std_real,
        "std_fake": std_fake,
        "signed_gap": signed_gap,
        "abs_gap": abs(signed_gap),
        "signed_cohen_d": signed_d,
        "abs_cohen_d": abs(signed_d),
        "overlap": empirical_overlap(real_values, fake_values),
    }


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_crf(args, crf: str, frame_model, device):
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
    dataset = balanced_subset(dataset, args.max_samples_per_class, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_drop_meta,
    )

    values = {name: [] for name in RELATION_TYPES}
    labels = []
    videos = []
    for x, y, metas in tqdm(loader, desc=f"collect {crf}", leave=False):
        x = x.to(device, non_blocking=True)
        rels = relation_tensors(frame_model.branch, x)
        for rel_name, tensor in rels.items():
            values[rel_name].append(tensor.detach().cpu().numpy())
        labels.append(y.numpy().astype(np.int64))
        videos.extend([m["video"] for m in metas])

    labels = np.concatenate(labels, axis=0)
    values = {name: np.concatenate(chunks, axis=0) for name, chunks in values.items()}
    return videos, labels, values


def build_rows_for_crf(crf: str, videos, labels, values):
    per_video_rows = []
    metric_rows = []

    for rel_type, rel_values in values.items():
        stat_map = temporal_stat_values(rel_values)
        for sample_idx, video in enumerate(videos):
            for stat_name, stat_values in stat_map.items():
                for band_idx, band_name in enumerate(RELATION_NAMES):
                    per_video_rows.append(
                        {
                            "crf": crf,
                            "video": video,
                            "label": int(labels[sample_idx]),
                            "relation_type": rel_type,
                            "band_pair": band_name,
                            "stat": stat_name,
                            "value": float(stat_values[sample_idx, band_idx]),
                        }
                    )
                per_video_rows.append(
                    {
                        "crf": crf,
                        "video": video,
                        "label": int(labels[sample_idx]),
                        "relation_type": rel_type,
                        "band_pair": "mean_over_bands",
                        "stat": stat_name,
                        "value": float(stat_values[sample_idx].mean()),
                    }
                )

        for stat_name, stat_values in stat_map.items():
            for band_idx, band_name in enumerate(RELATION_NAMES):
                real = stat_values[labels == 0, band_idx]
                fake = stat_values[labels == 1, band_idx]
                metric_rows.append(
                    {
                        "crf": crf,
                        "relation_type": rel_type,
                        "band_pair": band_name,
                        "stat": stat_name,
                        **distribution_metrics(real, fake),
                    }
                )
            real = stat_values[labels == 0].mean(axis=1)
            fake = stat_values[labels == 1].mean(axis=1)
            metric_rows.append(
                {
                    "crf": crf,
                    "relation_type": rel_type,
                    "band_pair": "mean_over_bands",
                    "stat": stat_name,
                    **distribution_metrics(real, fake),
                }
            )
    return per_video_rows, metric_rows


def build_stability_rows(metric_rows):
    grouped = {}
    for row in metric_rows:
        key = (row["relation_type"], row["band_pair"], row["stat"])
        grouped.setdefault(key, []).append(row)

    rows = []
    for (rel_type, band_pair, stat_name), items in sorted(grouped.items()):
        ds = np.array([float(item["abs_cohen_d"]) for item in items], dtype=np.float64)
        by_crf = {item["crf"]: float(item["abs_cohen_d"]) for item in items}
        src_d = by_crf.get("crf_src", np.nan)
        crf40_d = by_crf.get("crf40", np.nan)
        if np.isfinite(src_d) and abs(src_d) > 1e-12 and np.isfinite(crf40_d):
            src_to_crf40_drop = (src_d - crf40_d) / abs(src_d)
        else:
            src_to_crf40_drop = np.nan
        rows.append(
            {
                "relation_type": rel_type,
                "band_pair": band_pair,
                "stat": stat_name,
                "mean_abs_cohen_d": float(ds.mean()),
                "std_abs_cohen_d_across_crfs": float(ds.std(ddof=1)) if ds.size > 1 else 0.0,
                "min_abs_cohen_d": float(ds.min()),
                "max_abs_cohen_d": float(ds.max()),
                "crf_src_abs_cohen_d": src_d,
                "crf40_abs_cohen_d": crf40_d,
                "src_to_crf40_relative_drop": src_to_crf40_drop,
                "stability_score": float(ds.mean() / (ds.std(ddof=1) + 1e-6))
                if ds.size > 1
                else float("nan"),
            }
        )
    return rows


def write_markdown_summary(out_dir: Path, args, stability_rows):
    focus = [
        row
        for row in stability_rows
        if row["band_pair"] == "mean_over_bands" and row["stat"] == "mean"
    ]
    focus = sorted(
        focus,
        key=lambda r: (
            -float(r["mean_abs_cohen_d"]),
            float(r["std_abs_cohen_d_across_crfs"]),
        ),
    )
    lines = [
        "# Relation Distribution Separability and Compression Stability",
        "",
        "This probe directly compares relation-value distributions; it does not train a classifier.",
        "",
        f"- frame checkpoint: `{args.frame_ckpt}`",
        f"- split: `{args.split}`",
        f"- sampling mode: `{args.sampling_mode}`",
        f"- CRFs: `{', '.join(args.crfs)}`",
        "",
        "## Mean-over-band Temporal-Mean Summary",
        "",
        "| relation | mean abs Cohen d | std across CRFs | crf_src d | crf40 d | overlap-oriented note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in focus:
        lines.append(
            "| {relation_type} | {mean_abs_cohen_d:.4f} | "
            "{std_abs_cohen_d_across_crfs:.4f} | {crf_src_abs_cohen_d:.4f} | "
            "{crf40_abs_cohen_d:.4f} | higher d and lower std is better |".format(**row)
        )
    lines.extend(
        [
            "",
            "Use `relation_distribution_metrics.csv` for per-CRF real/fake separability, "
            "and `relation_compression_stability.csv` for cross-compression stability.",
            "",
            "Note: abs relation is scalarized as the mean absolute feature difference, "
            "because the training-time abs relation is a feature vector.",
        ]
    )
    (out_dir / "relation_distribution_probe.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    args = parse_args()
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_model, _ = load_frozen_frame_model(args.frame_ckpt, args, device)
    per_video_rows = []
    metric_rows = []
    for crf in args.crfs:
        videos, labels, values = collect_crf(args, crf, frame_model, device)
        print(
            f"{crf}: clips={len(labels)} real={int((labels == 0).sum())} "
            f"fake={int((labels == 1).sum())}"
        )
        crf_video_rows, crf_metric_rows = build_rows_for_crf(crf, videos, labels, values)
        per_video_rows.extend(crf_video_rows)
        metric_rows.extend(crf_metric_rows)

    metric_fields = [
        "crf",
        "relation_type",
        "band_pair",
        "stat",
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
    per_video_fields = [
        "crf",
        "video",
        "label",
        "relation_type",
        "band_pair",
        "stat",
        "value",
    ]
    stability_fields = [
        "relation_type",
        "band_pair",
        "stat",
        "mean_abs_cohen_d",
        "std_abs_cohen_d_across_crfs",
        "min_abs_cohen_d",
        "max_abs_cohen_d",
        "crf_src_abs_cohen_d",
        "crf40_abs_cohen_d",
        "src_to_crf40_relative_drop",
        "stability_score",
    ]

    stability_rows = build_stability_rows(metric_rows)
    write_csv(out_dir / "relation_distribution_per_video.csv", per_video_rows, per_video_fields)
    write_csv(out_dir / "relation_distribution_metrics.csv", metric_rows, metric_fields)
    write_csv(
        out_dir / "relation_compression_stability.csv",
        stability_rows,
        stability_fields,
    )
    write_markdown_summary(out_dir, args, stability_rows)
    print(f"Done. Results written to {out_dir}")


if __name__ == "__main__":
    main()
