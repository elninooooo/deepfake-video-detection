"""Offline face extraction with MTCNN.

For every video listed in splits.json (and for every CRF folder under --src),
sample `--num_frames` frames uniformly, detect the largest face on each,
resize to 224x224 and save as JPEG to:

    <out>/<crf_tag>/<sub>/<video_stem>/frame_000.jpg
                                       frame_001.jpg
                                       ...
    <out>/<crf_tag>/index.json   # {video_rel_path: frame_count}

`--src` can either be:
  * The original Celeb-DF tree (no crf sub-folders)  → produces <out>/crf_src/
  * A recompressed tree from ffmpeg_recompress.py    → produces <out>/crf0/, crf23/, ...
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
except ImportError as e:
    raise SystemExit("facenet-pytorch is required. pip install facenet-pytorch") from e


def is_recompressed_tree(root: Path) -> bool:
    return any(p.is_dir() and p.name.startswith("crf") for p in root.iterdir())


def list_crf_dirs(root: Path):
    if is_recompressed_tree(root):
        return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("crf")])
    return [root]


def crf_tag(crf_dir: Path, src_root: Path) -> str:
    if crf_dir == src_root:
        return "crf_src"
    return crf_dir.name


def sample_frame_indices(total: int, n: int):
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n).round().astype(int).tolist()


def extract_video_faces(video_path: Path, n_frames: int, mtcnn: MTCNN, image_size: int,
                        margin: float = 0.3):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(total, n_frames)
    if not idxs:
        cap.release()
        return []
    crops = []
    target = 0
    cur = 0
    while target < len(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[target])
        ok, frame = cap.read()
        if not ok:
            target += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(Image.fromarray(rgb))
        if boxes is None or len(boxes) == 0:
            target += 1
            continue
        # largest face
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        b = boxes[int(np.argmax(areas))]
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        side = max(w, h) * (1 + margin)
        x1 = int(max(0, cx - side / 2))
        y1 = int(max(0, cy - side / 2))
        x2 = int(min(rgb.shape[1], cx + side / 2))
        y2 = int(min(rgb.shape[0], cy + side / 2))
        if x2 - x1 < 16 or y2 - y1 < 16:
            target += 1
            continue
        crop = rgb[y1:y2, x1:x2]
        crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
        crops.append(crop)
        target += 1
        cur += 1
    cap.release()
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Either the original Celeb-DF root or the output of ffmpeg_recompress.py")
    ap.add_argument("--splits", required=True, help="splits.json from celeb_df_split.py")
    ap.add_argument("--out", required=True, help="Output face_cache directory")
    ap.add_argument("--num_frames", type=int, default=32,
                    help="Frames sampled per video (must be >= N used for clip).")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--min_success_ratio", type=float, default=0.5,
                    help="Drop video if fewer than this fraction of frames had a detected face.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process this many videos per split (for quick testing).")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    all_recs = []
    for s in ("train", "val", "test"):
        recs = splits[s]
        if args.limit > 0:
            recs = recs[: args.limit]
        all_recs.extend(recs)

    mtcnn = MTCNN(image_size=args.image_size, margin=0, post_process=False,
                  select_largest=True, device=args.device)
    mtcnn.eval()

    for crf_dir in list_crf_dirs(src):
        tag = crf_tag(crf_dir, src)
        tag_out = out / tag
        tag_out.mkdir(parents=True, exist_ok=True)
        index_path = tag_out / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}

        print(f"\n=== Extracting faces for {tag} ({crf_dir}) ===")
        for rec in tqdm(all_recs):
            rel = rec["path"]
            if rel in index:
                continue
            vid = crf_dir / rel
            if not vid.is_file():
                # fall back to original tree if the user pointed --src at the un-recompressed root
                continue
            stem = Path(rel).with_suffix("")
            target_dir = tag_out / stem
            target_dir.mkdir(parents=True, exist_ok=True)
            crops = extract_video_faces(vid, args.num_frames, mtcnn, args.image_size)
            success_ratio = len(crops) / max(1, args.num_frames)
            if success_ratio < args.min_success_ratio:
                # cleanup partial
                for p in target_dir.glob("*.jpg"):
                    p.unlink(missing_ok=True)
                continue
            for i, img in enumerate(crops):
                cv2.imwrite(str(target_dir / f"frame_{i:03d}.jpg"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            index[rel] = len(crops)
            # incremental save
            if len(index) % 50 == 0:
                index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"Saved index: {index_path}  ({len(index)} videos)")

    print("All done.")


if __name__ == "__main__":
    main()
