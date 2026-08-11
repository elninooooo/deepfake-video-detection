"""Visualize RGB adjacent-frame residual frequency-band views.

This script uses the same cached clip source and frequency-band split as the
V10 gradient-residual visualization, but the residual is computed directly in
RGB space:

    R_t = |x_{t+1} - x_t|

It saves the original residual view and the low/mid/high band views.
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from modelsgenerate.residual_spectral_relation import ResidualSpectralRelationBranch
from scripts.visualize_v10_residual_views import (
    VIEW_NAMES,
    render_view_panels,
    save_labeled_grid,
)


def main():
    parser = argparse.ArgumentParser(
        "Visualize RGB residual original/low/mid/high frequency-band views"
    )
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--face_cache", default="face_cache_uniform16_all")
    parser.add_argument("--crf", default="crf_src")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument(
        "--sampling_mode",
        choices=sorted(CelebDFClipDataset.SAMPLING_MODES),
        default="global16",
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--video_path",
        default=None,
        help="Optional relative video path in splits.json, e.g. Celeb-synthesis/xxx.mp4.",
    )
    parser.add_argument(
        "--pair_index",
        type=int,
        default=7,
        help="Adjacent pair index in [0, n_frames-2].",
    )
    parser.add_argument("--mid_low", type=float, default=0.10)
    parser.add_argument("--mid_high", type=float, default=0.70)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--out_dir", default="output/v10_rgb_residual_views")
    parser.add_argument(
        "--colormap",
        choices=["gray", "magma", "viridis", "turbo"],
        default="magma",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Robust upper percentile used for visualization scaling.",
    )
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
        target = args.video_path.replace("\\", "/")
        for i, rec in enumerate(dataset.records):
            if rec["path"].replace("\\", "/") == target:
                sample_index = i
                break
        if sample_index is None:
            raise ValueError(f"Video path not found after filtering: {args.video_path}")
    else:
        sample_index = args.sample_index

    x, y, meta = dataset[sample_index]
    clip = x.unsqueeze(0)

    branch = ResidualSpectralRelationBranch(
        mid_low=args.mid_low,
        mid_high=args.mid_high,
        residual_mode="abs",
        relation_mode="cos_only",
    )
    with torch.no_grad():
        rgb_residual = branch._residual(clip)
        rgb_views = branch._frequency_views(rgb_residual)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap_lookup = {
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
    }
    panels = render_view_panels(rgb_views, args.pair_index, args, cmap_lookup)

    for name, img in zip(VIEW_NAMES, panels):
        img.save(out_dir / f"rgb_{name}_residual_pair{args.pair_index:02d}.png")

    save_labeled_grid(
        panels,
        ["RGB residual", "Low band", "Mid band", "High band"],
        out_dir / f"rgb_residual_views_pair{args.pair_index:02d}.png",
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
        "residual_mode": "rgb_abs",
        "mid_low": args.mid_low,
        "mid_high": args.mid_high,
    }
    (out_dir / "metadata.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in info.items()),
        encoding="utf-8",
    )

    print("Saved RGB residual view images to:")
    print(out_dir.resolve())
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
