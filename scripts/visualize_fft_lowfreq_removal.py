"""Visualize FFT frequency-region removal for one image.

Outputs four images:
  1. original image
  2. FFT log-magnitude spectrum
  3. FFT spectrum after covering selected centered frequency area
  4. inverse-FFT reconstruction after frequency removal

Unlike the V10 band split, this script does not use radial low/mid/high masks.
It simply masks a centered square region in the shifted FFT spectrum. Use
--mask_mode low to cover only the low-frequency center, or --mask_mode low_mid
to cover a larger centered area representing low- and mid-frequency removal.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize_uint8(arr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.percentile(arr, 100.0 - percentile))
    hi = float(np.percentile(arr, percentile))
    if hi <= lo + 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def spectrum_image(shifted_fft: np.ndarray, percentile: float) -> Image.Image:
    mag = np.log1p(np.abs(shifted_fft))
    if mag.ndim == 3:
        mag = mag.mean(axis=2)
    gray = normalize_uint8(mag, percentile=percentile)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def centered_square_mask(shape, mask_ratio: float) -> np.ndarray:
    h, w = shape[:2]
    side = int(round(min(h, w) * mask_ratio))
    side = max(1, min(side, min(h, w)))
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - side // 2)
    y1 = min(h, y0 + side)
    x0 = max(0, cx - side // 2)
    x1 = min(w, x0 + side)
    mask = np.ones((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 0.0
    return mask


def remove_frequency_region(rgb: np.ndarray, mask_ratio: float, percentile: float):
    img = rgb.astype(np.float32) / 255.0
    shifted = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1), norm="ortho"), axes=(0, 1))
    mask = centered_square_mask(img.shape, mask_ratio)
    masked = shifted * mask[:, :, None]
    recon = np.fft.ifft2(np.fft.ifftshift(masked, axes=(0, 1)), axes=(0, 1), norm="ortho").real

    # Frequency removal produces signed residuals. Shift around
    # zero for visualization rather than clipping negative values away.
    recon_vis = normalize_uint8(recon, percentile=percentile)
    return shifted, masked, recon_vis


def save_contact_sheet(images, labels, out_path: Path, cell_size: int = 256):
    pad = 18
    label_h = 34
    w = len(images) * cell_size + (len(images) + 1) * pad
    h = cell_size + label_h + pad * 2
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    x = pad
    for img, label in zip(images, labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (cell_size - text_w) // 2, pad), label, fill=(20, 20, 20), font=font)
        panel = img.resize((cell_size, cell_size), Image.Resampling.BICUBIC)
        canvas.paste(panel, (x, pad + label_h))
        x += cell_size + pad
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser("Show FFT frequency masking and iFFT reconstruction")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out_dir", default="output/fft_lowfreq_removal")
    parser.add_argument(
        "--mask_mode",
        choices=["low", "low_mid"],
        default="low",
        help="low masks the frequency center; low_mid masks a larger center area.",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.16,
        help="Centered square side ratio for --mask_mode low.",
    )
    parser.add_argument(
        "--low_mid_mask_ratio",
        type=float,
        default=0.52,
        help="Centered square side ratio for --mask_mode low_mid.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Robust percentile for visual normalization.",
    )
    parser.add_argument("--contact_size", type=int, default=256)
    args = parser.parse_args()

    selected_mask_ratio = args.mask_ratio if args.mask_mode == "low" else args.low_mid_mask_ratio
    if not 0.0 < selected_mask_ratio < 1.0:
        raise ValueError("Selected mask ratio must be in (0, 1).")

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = read_rgb(image_path)
    shifted, masked, recon_vis = remove_frequency_region(
        rgb,
        mask_ratio=selected_mask_ratio,
        percentile=args.percentile,
    )

    original_img = Image.fromarray(rgb)
    fft_img = spectrum_image(shifted, percentile=args.percentile)
    masked_fft_img = spectrum_image(masked, percentile=args.percentile)
    ifft_img = Image.fromarray(recon_vis).convert("RGB")

    original_img.save(out_dir / "01_original.png")
    fft_img.save(out_dir / "02_fft_spectrum.png")
    prefix = "lowfreq" if args.mask_mode == "low" else "low_mid_freq"
    masked_fft_img.save(out_dir / f"03_{prefix}_removed_fft_spectrum.png")
    ifft_img.save(out_dir / f"04_ifft_{prefix}_removed_image.png")

    save_contact_sheet(
        [original_img, fft_img, masked_fft_img, ifft_img],
        [
            "Original",
            "FFT spectrum",
            "Low-frequency covered" if args.mask_mode == "low" else "Low+mid covered",
            "iFFT result",
        ],
        out_dir / f"fft_{prefix}_removal_overview.png",
        cell_size=args.contact_size,
    )

    info = {
        "image": str(image_path),
        "mask_shape": "centered square",
        "mask_mode": args.mask_mode,
        "mask_ratio": selected_mask_ratio,
        "percentile": args.percentile,
        "outputs": "original, fft spectrum, masked fft spectrum, ifft frequency-removed image",
    }
    (out_dir / "metadata.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in info.items()),
        encoding="utf-8",
    )

    print("Saved FFT frequency removal visualization to:")
    print(out_dir.resolve())
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
