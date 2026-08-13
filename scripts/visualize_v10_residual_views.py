"""Visualize the residual views used by V10.

The script loads one cached video clip, builds the Sobel-gradient residual for
one adjacent frame pair, splits it into original/low/mid/high residual views,
and saves publication-friendly PNG images. It can also compare RGB residual
views with Sobel-gradient residual views.

Example:
    python scripts/visualize_v10_residual_views.py ^
        --splits splits.json ^
        --face_cache face_cache_uniform16_all ^
        --crf crf_src ^
        --split test ^
        --sampling_mode global16 ^
        --sample_index 0 ^
        --pair_index 7 ^
        --out_dir output/v10_residual_views
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from modelsgenerate.residual_spectral_relation import ResidualSpectralRelationBranch


VIEW_NAMES = ("original", "low", "mid", "high")


def normalize_map(arr: np.ndarray, percentile: float = 99.0, use_abs: bool = True) -> np.ndarray:
    """Convert a residual map to uint8 for visualization."""
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if use_abs:
        arr = np.abs(arr)
        lo = 0.0
        hi = float(np.percentile(arr, percentile))
    else:
        lo = float(np.percentile(arr, 100.0 - percentile))
        hi = float(np.percentile(arr, percentile))
    if hi <= lo + 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    vis = (arr - lo) / (hi - lo)
    vis = np.clip(vis, 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def colorize(gray: np.ndarray, cmap: int) -> Image.Image:
    bgr = cv2.applyColorMap(gray, cmap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def tensor_view_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().permute(1, 2, 0).numpy()


def save_labeled_grid(images, labels, out_path: Path, pad: int = 18, label_h: int = 34):
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    w = sum(widths) + pad * (len(images) + 1)
    h = max(heights) + label_h + pad * 2
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    x = pad
    for img, label in zip(images, labels):
        canvas.paste(img, (x, pad + label_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (img.width - text_w) // 2, pad), label, fill=(20, 20, 20), font=font)
        x += img.width + pad
    canvas.save(out_path)


def save_two_row_grid(rows, row_labels, col_labels, out_path: Path,
                      pad: int = 18, label_h: int = 36, row_label_w: int = 150):
    panel_w = max(img.width for row in rows for img in row)
    panel_h = max(img.height for row in rows for img in row)
    w = row_label_w + pad * (len(col_labels) + 1) + panel_w * len(col_labels)
    h = label_h + pad * (len(rows) + 1) + panel_h * len(rows)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arial.ttf", 18)
        row_font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        row_font = ImageFont.load_default()

    for col, label in enumerate(col_labels):
        x = row_label_w + pad + col * (panel_w + pad)
        bbox = draw.textbbox((0, 0), label, font=title_font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (panel_w - text_w) // 2, pad), label, fill=(20, 20, 20), font=title_font)

    for row_idx, (images, row_label) in enumerate(zip(rows, row_labels)):
        y = label_h + pad + row_idx * (panel_h + pad)
        bbox = draw.textbbox((0, 0), row_label, font=row_font)
        text_h = bbox[3] - bbox[1]
        draw.text((pad, y + (panel_h - text_h) // 2), row_label, fill=(20, 20, 20), font=row_font)
        for col, img in enumerate(images):
            x = row_label_w + pad + col * (panel_w + pad)
            canvas.paste(img, (x, y))

    canvas.save(out_path)


def render_view_panels(views, pair_index: int, args, cmap_lookup):
    panels = []
    for view in views:
        arr = tensor_view_to_numpy(view[pair_index])
        gray = normalize_map(arr, percentile=args.percentile, use_abs=True)
        if args.colormap == "gray":
            img = Image.fromarray(gray).convert("RGB")
        else:
            img = colorize(gray, cmap_lookup[args.colormap])
        if args.image_size > 0:
            img = img.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        panels.append(img)
    return panels


def main():
    parser = argparse.ArgumentParser("Visualize V10 original/low/mid/high residual views")
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--face_cache", default="face_cache_uniform16_all")
    parser.add_argument("--crf", default="crf_src")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--sampling_mode",
                        choices=sorted(CelebDFClipDataset.SAMPLING_MODES),
                        default="global16")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Index after Dataset filtering.")
    parser.add_argument("--video_path", default=None,
                        help="Optional relative video path in splits.json, e.g. Celeb-synthesis/xxx.mp4.")
    parser.add_argument("--pair_index", type=int, default=7,
                        help="Adjacent pair index in [0, n_frames-2].")
    parser.add_argument("--mid_low", type=float, default=0.10)
    parser.add_argument("--mid_high", type=float, default=0.70)
    parser.add_argument("--image_size", type=int, default=224,
                        help="Resize output panels for display only.")
    parser.add_argument("--out_dir", default="output/v10_residual_views")
    parser.add_argument("--colormap", choices=["gray", "magma", "viridis", "turbo"], default="magma")
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="Robust upper percentile used for visualization scaling.")
    parser.add_argument("--compare_rgb_residual", action="store_true",
                        help="Also save a two-row RGB residual vs gradient residual comparison.")
    args = parser.parse_args()

    if not 0 <= args.pair_index <= args.n_frames - 2:
        raise ValueError(f"--pair_index must be in [0, {args.n_frames - 2}]")

    dataset = CelebDFClipDataset(
        split_json=args.splits,
        face_cache_dir=args.face_cache,
        crf_tag=args.crf,
        split=args.split,
        n_frames=args.n_frames,
        train=False,
        horiz_flip=False,
        sampling_mode=args.sampling_mode,
    )

    if args.video_path:
        sample_index = None
        for i, rec in enumerate(dataset.records):
            if rec["path"].replace("\\", "/") == args.video_path.replace("\\", "/"):
                sample_index = i
                break
        if sample_index is None:
            raise ValueError(f"Video path not found after filtering: {args.video_path}")
    else:
        sample_index = args.sample_index

    x, y, meta = dataset[sample_index]
    clip = x.unsqueeze(0)

    gradient_branch = ResidualSpectralRelationBranch(
        mid_low=args.mid_low,
        mid_high=args.mid_high,
        residual_mode="gradient",
        relation_mode="cos_only",
    )
    with torch.no_grad():
        gradient_residual = gradient_branch._residual(clip)
        gradient_views = gradient_branch._frequency_views(gradient_residual)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap_lookup = {
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
    }

    panels = render_view_panels(gradient_views, args.pair_index, args, cmap_lookup)
    for name, img in zip(VIEW_NAMES, panels):
        img.save(out_dir / f"{name}_residual_pair{args.pair_index:02d}.png")

    save_labeled_grid(
        panels,
        ["Original residual", "Low band", "Mid band", "High band"],
        out_dir / f"v10_residual_views_pair{args.pair_index:02d}.png",
    )

    if args.compare_rgb_residual:
        rgb_branch = ResidualSpectralRelationBranch(
            mid_low=args.mid_low,
            mid_high=args.mid_high,
            residual_mode="abs",
            relation_mode="cos_only",
        )
        with torch.no_grad():
            rgb_residual = rgb_branch._residual(clip)
            rgb_views = rgb_branch._frequency_views(rgb_residual)
        rgb_panels = render_view_panels(rgb_views, args.pair_index, args, cmap_lookup)

        for name, img in zip(VIEW_NAMES, rgb_panels):
            img.save(out_dir / f"rgb_{name}_residual_pair{args.pair_index:02d}.png")
        for name, img in zip(VIEW_NAMES, panels):
            img.save(out_dir / f"gradient_{name}_residual_pair{args.pair_index:02d}.png")

        save_two_row_grid(
            [rgb_panels, panels],
            ["RGB residual", "Gradient residual"],
            ["Original", "Low band", "Mid band", "High band"],
            out_dir / f"rgb_vs_gradient_residual_views_pair{args.pair_index:02d}.png",
        )

    info = {
        "video": meta["video"],
        "label": int(y.item()),
        "label_name": "fake" if int(y.item()) == 1 else "real",
        "crf": args.crf,
        "split": args.split,
        "sample_index": sample_index,
        "pair_index": args.pair_index,
        "sampling_mode": args.sampling_mode,
        "mid_low": args.mid_low,
        "mid_high": args.mid_high,
    }
    (out_dir / "metadata.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in info.items()),
        encoding="utf-8",
    )

    print("Saved residual view images to:")
    print(out_dir.resolve())
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
