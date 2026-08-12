"""Build a small image library for V10 residual visualization.

The library contains one global16 clip with:
  * 16 RGB frames
  * 16 Sobel gradient magnitude maps
  * 15 adjacent-frame gradient residual maps

All images come from the same cached clip selected by split/sample_index or
video_path, so they can be used together in methodology figures.
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


def to_uint8_rgb(frames):
    arr = frames.detach().cpu().permute(0, 2, 3, 1).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def sobel_gradient_maps(rgb_frames):
    maps = []
    for frame in rgb_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        maps.append(np.sqrt(gx * gx + gy * gy + 1e-8))
    return np.stack(maps, axis=0)


def normalize_stack(stack, percentile=99.0):
    hi = float(np.percentile(np.abs(stack), percentile))
    if hi <= 1e-12:
        return np.zeros(stack.shape, dtype=np.uint8)
    vis = np.clip(np.abs(stack) / hi, 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def colorize(gray, cmap):
    if cmap == "gray":
        return Image.fromarray(gray).convert("RGB")
    lookup = {
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
    }
    bgr = cv2.applyColorMap(gray, lookup[cmap])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def tensor_view_to_gray(view, pair_index, percentile):
    arr = view[pair_index].detach().cpu().permute(1, 2, 0).numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return normalize_stack(np.abs(arr)[None], percentile=percentile)[0]


def frequency_pipeline_images(residual, mid_low, mid_high, percentile, cmap):
    h, w = residual.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    radius = np.sqrt(grid_x * grid_x + grid_y * grid_y) / np.sqrt(2.0)
    masks = {
        "low": (radius < mid_low).astype(np.float32),
        "mid": ((radius >= mid_low) & (radius <= mid_high)).astype(np.float32),
        "high": (radius > mid_high).astype(np.float32),
    }

    spec = np.fft.fftshift(np.fft.fft2(residual, norm="ortho"))
    full_ifft = np.fft.ifft2(np.fft.ifftshift(spec), norm="ortho").real

    def spectrum_to_gray(spectrum):
        mag = np.log1p(np.abs(spectrum))
        return normalize_stack(mag[None], percentile=percentile)[0]

    def spatial_to_img(arr):
        gray = normalize_stack(np.abs(arr)[None], percentile=percentile)[0]
        return colorize(gray, cmap)

    outputs = [
        ("01_gradient_residual", "Gradient residual", spatial_to_img(residual)),
        ("02_fft_spectrum", "FFT log spectrum", colorize(spectrum_to_gray(spec), cmap)),
        ("03_full_ifft", "Full-spectrum iFFT", spatial_to_img(full_ifft)),
    ]

    for name, mask in masks.items():
        mask_gray = (mask * 255).astype(np.uint8)
        outputs.append((f"04_{name}_mask", f"{name.capitalize()} mask", Image.fromarray(mask_gray).convert("RGB")))

    for name, mask in masks.items():
        masked_spec = spec * mask
        outputs.append(
            (
                f"05_{name}_masked_spectrum",
                f"{name.capitalize()} masked spectrum",
                colorize(spectrum_to_gray(masked_spec), cmap),
            )
        )

    for name, mask in masks.items():
        masked_spec = spec * mask
        ifft_view = np.fft.ifft2(np.fft.ifftshift(masked_spec), norm="ortho").real
        outputs.append(
            (
                f"06_{name}_ifft",
                f"{name.capitalize()}-band iFFT",
                spatial_to_img(ifft_view),
            )
        )

    return outputs


def labeled_grid(images, labels, cols, out_path, cell_size=160, pad=14, label_h=28):
    rows = int(np.ceil(len(images) / cols))
    w = cols * cell_size + (cols + 1) * pad
    h = rows * (cell_size + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = pad + col * (cell_size + pad)
        y = pad + row * (cell_size + label_h + pad)
        label = labels[idx]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (cell_size - text_w) // 2, y), label, fill=(20, 20, 20), font=font)
        img = img.resize((cell_size, cell_size), Image.Resampling.BICUBIC)
        canvas.paste(img, (x, y + label_h))
    canvas.save(out_path)


def select_dataset_item(dataset, video_path, sample_index):
    if video_path:
        target = video_path.replace("\\", "/")
        for i, rec in enumerate(dataset.records):
            if rec["path"].replace("\\", "/") == target:
                return i
        raise ValueError(f"Video path not found after filtering: {video_path}")
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index={sample_index} outside dataset size {len(dataset)}")
    return sample_index


def main():
    parser = argparse.ArgumentParser("Build a global16 RGB/gradient/residual image library")
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
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--out_dir", default="output/v10_frame_gradient_library")
    parser.add_argument("--colormap", choices=["gray", "magma", "viridis", "turbo"], default="magma")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--mid_low", type=float, default=0.10)
    parser.add_argument("--mid_high", type=float, default=0.70)
    parser.add_argument(
        "--frequency_pair_index",
        type=int,
        default=0,
        help="Adjacent pair used for original/low/mid/high gradient residual views.",
    )
    parser.add_argument("--grid_cell_size", type=int, default=160)
    args = parser.parse_args()

    if args.n_frames != 16:
        raise ValueError("This library script is intended for n_frames=16.")
    if not 0 <= args.frequency_pair_index <= args.n_frames - 2:
        raise ValueError(f"--frequency_pair_index must be in [0, {args.n_frames - 2}]")

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
    sample_index = select_dataset_item(dataset, args.video_path, args.sample_index)
    x, y, meta = dataset[sample_index]
    rec = dataset.records[sample_index]
    source_indices = dataset._sample_indices(dataset.index[rec["path"]])

    rgb_frames = to_uint8_rgb(x)
    gradient_maps = sobel_gradient_maps(rgb_frames)
    gradient_residuals = np.abs(gradient_maps[1:] - gradient_maps[:-1])

    grad_vis = normalize_stack(gradient_maps, percentile=args.percentile)
    residual_vis = normalize_stack(gradient_residuals, percentile=args.percentile)

    out_dir = Path(args.out_dir)
    rgb_dir = out_dir / "rgb_frames"
    grad_dir = out_dir / "gradient_frames"
    residual_dir = out_dir / "gradient_residuals"
    freq_dir = out_dir / f"gradient_frequency_views_pair{args.frequency_pair_index:02d}"
    pipeline_dir = out_dir / f"frequency_pipeline_pair{args.frequency_pair_index:02d}"
    sheet_dir = out_dir / "contact_sheets"
    for d in (rgb_dir, grad_dir, residual_dir, freq_dir, pipeline_dir, sheet_dir):
        d.mkdir(parents=True, exist_ok=True)

    rgb_images = []
    grad_images = []
    residual_images = []
    frequency_images = []

    for i, frame in enumerate(rgb_frames):
        img = Image.fromarray(frame)
        rgb_images.append(img)
        img.save(rgb_dir / f"rgb_frame_{i:02d}_src{source_indices[i]:03d}.png")

        grad_img = colorize(grad_vis[i], args.colormap)
        grad_images.append(grad_img)
        grad_img.save(grad_dir / f"gradient_frame_{i:02d}_src{source_indices[i]:03d}.png")

    for i, residual in enumerate(residual_vis):
        res_img = colorize(residual, args.colormap)
        residual_images.append(res_img)
        residual_name = (
            f"gradient_residual_{i:02d}_{i + 1:02d}_"
            f"src{source_indices[i]:03d}_src{source_indices[i + 1]:03d}.png"
        )
        res_img.save(residual_dir / residual_name)

    branch = ResidualSpectralRelationBranch(
        mid_low=args.mid_low,
        mid_high=args.mid_high,
        residual_mode="gradient",
        relation_mode="cos_only",
    )
    with torch.no_grad():
        residual_tensor = branch._residual(x.unsqueeze(0))
        frequency_views = branch._frequency_views(residual_tensor)

    frequency_names = ("original", "low", "mid", "high")
    frequency_labels = (
        "Original residual",
        "Low-band iFFT",
        "Mid-band iFFT",
        "High-band iFFT",
    )
    for name, view in zip(frequency_names, frequency_views):
        gray = tensor_view_to_gray(view, args.frequency_pair_index, args.percentile)
        img = colorize(gray, args.colormap)
        frequency_images.append(img)
        img.save(
            freq_dir
            / f"gradient_{name}_residual_pair{args.frequency_pair_index:02d}_"
              f"src{source_indices[args.frequency_pair_index]:03d}_"
              f"src{source_indices[args.frequency_pair_index + 1]:03d}.png"
        )

    pipeline_outputs = frequency_pipeline_images(
        residual=gradient_residuals[args.frequency_pair_index],
        mid_low=args.mid_low,
        mid_high=args.mid_high,
        percentile=args.percentile,
        cmap=args.colormap,
    )
    pipeline_images = []
    pipeline_labels = []
    pair_name = (
        f"pair{args.frequency_pair_index:02d}_"
        f"src{source_indices[args.frequency_pair_index]:03d}_"
        f"src{source_indices[args.frequency_pair_index + 1]:03d}"
    )
    for file_stem, label, img in pipeline_outputs:
        pipeline_images.append(img)
        pipeline_labels.append(label)
        img.save(pipeline_dir / f"{file_stem}_{pair_name}.png")

    rgb_labels = [f"F{i:02d} / src {source_indices[i]:03d}" for i in range(16)]
    grad_labels = [f"G{i:02d} / src {source_indices[i]:03d}" for i in range(16)]
    res_labels = [
        f"R{i:02d}-{i + 1:02d}" for i in range(15)
    ]
    labeled_grid(
        rgb_images,
        rgb_labels,
        cols=4,
        out_path=sheet_dir / "global16_rgb_frames.png",
        cell_size=args.grid_cell_size,
    )
    labeled_grid(
        grad_images,
        grad_labels,
        cols=4,
        out_path=sheet_dir / "global16_gradient_frames.png",
        cell_size=args.grid_cell_size,
    )
    labeled_grid(
        residual_images,
        res_labels,
        cols=5,
        out_path=sheet_dir / "global16_gradient_residuals.png",
        cell_size=args.grid_cell_size,
    )
    labeled_grid(
        frequency_images,
        list(frequency_labels),
        cols=4,
        out_path=sheet_dir / f"gradient_frequency_views_pair{args.frequency_pair_index:02d}.png",
        cell_size=args.grid_cell_size,
    )
    labeled_grid(
        pipeline_images,
        pipeline_labels,
        cols=3,
        out_path=sheet_dir / f"gradient_fft_mask_ifft_pipeline_pair{args.frequency_pair_index:02d}.png",
        cell_size=args.grid_cell_size,
    )

    info = {
        "video": meta["video"],
        "label": int(y.item()),
        "label_name": "fake" if int(y.item()) == 1 else "real",
        "crf": args.crf,
        "split": args.split,
        "sample_index": sample_index,
        "sampling_mode": args.sampling_mode,
        "source_frame_indices": ",".join(str(i) for i in source_indices),
        "gradient": "Sobel magnitude",
        "gradient_residual": "|G_{i+1} - G_i|",
        "frequency_pair_index": args.frequency_pair_index,
        "frequency_pair_source_indices": (
            f"{source_indices[args.frequency_pair_index]},"
            f"{source_indices[args.frequency_pair_index + 1]}"
        ),
        "frequency_views": "original residual, low-band iFFT, mid-band iFFT, high-band iFFT",
        "frequency_pipeline": (
            "gradient residual -> FFT log spectrum -> low/mid/high masks -> "
            "masked spectra -> low/mid/high iFFT spatial views"
        ),
        "mid_low": args.mid_low,
        "mid_high": args.mid_high,
        "colormap": args.colormap,
        "percentile": args.percentile,
    }
    (out_dir / "metadata.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in info.items()),
        encoding="utf-8",
    )

    print("Saved image library to:")
    print(out_dir.resolve())
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
