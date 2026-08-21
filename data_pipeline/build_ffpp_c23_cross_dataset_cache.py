"""Build FF++ C23 cross-dataset splits and a face cache.

The script is intentionally evaluation-focused: it assigns original videos as
real and the selected manipulation folders as fake, samples balanced subsets,
extracts face crops with MTCNN, and writes a cache compatible with
CelebDFClipDataset. It can either create one combined split or one split per
fake method.

Expected local layout:

    FF++/FaceForensics++_C23/
      original/
      Deepfakes/
      FaceSwap/
      FaceShifter/
      Face2Face/
      NeuralTextures/

Output layout:

    splits_ffpp_c23_by_method/
      splits_ffpp_c23_Deepfakes_300.json
      splits_ffpp_c23_FaceSwap_300.json
      ...
    face_cache_ffpp_c23_300/
      crf23/
        original/<video_stem>/frame_000.jpg
        Deepfakes/<video_stem>/frame_000.jpg
        ...
        index.json
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from preprocess_faces import extract_video_faces


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_FAKE_DIRS = [
    "Deepfakes",
    "FaceSwap",
    "FaceShifter",
    "Face2Face",
    "NeuralTextures",
]


def parse_identity(rel_path: str) -> str:
    stem = Path(rel_path).stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    m = re.match(r"(\d+)", stem)
    if m:
        return m.group(1)
    return stem


def list_method_videos(root: Path, method: str, label: int):
    base = root / method
    if not base.is_dir():
        raise FileNotFoundError(f"Missing FF++ method folder: {base}")
    records = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            rel = path.relative_to(root).as_posix()
            records.append({
                "path": rel,
                "label": label,
                "identity": parse_identity(rel),
                "method": method,
            })
    return records


def sample_records(records, cap: int, seed: int):
    if cap <= 0 or len(records) <= cap:
        return list(records)
    rng = random.Random(seed)
    picked = list(records)
    rng.shuffle(picked)
    return sorted(picked[:cap], key=lambda r: r["path"])


def build_split(root: Path, max_real: int, max_fake_per_method: int,
                fake_dirs, seed: int):
    real_all = list_method_videos(root, "original", label=0)
    real = sample_records(real_all, max_real, seed)

    fake = []
    method_counts = {}
    for offset, method in enumerate(fake_dirs):
        all_method = list_method_videos(root, method, label=1)
        picked = sample_records(all_method, max_fake_per_method, seed + 100 + offset)
        fake.extend(picked)
        method_counts[method] = {
            "available": len(all_method),
            "selected": len(picked),
        }

    test = real + fake
    random.Random(seed).shuffle(test)
    meta = {
        "dataset": "FaceForensics++ C23",
        "source_root": str(root),
        "real_method": "original",
        "fake_methods": list(fake_dirs),
        "available_real": len(real_all),
        "selected_real": len(real),
        "fake_method_counts": method_counts,
        "selected_fake_total": len(fake),
        "seed": seed,
    }
    return {"train": [], "val": [], "test": test, "meta": meta}


def build_per_method_splits(root: Path, max_real: int, max_fake_per_method: int,
                            fake_dirs, seed: int):
    real_all = list_method_videos(root, "original", label=0)
    real = sample_records(real_all, max_real, seed)

    splits = {}
    fake_method_counts = {}
    for offset, method in enumerate(fake_dirs):
        all_method = list_method_videos(root, method, label=1)
        fake = sample_records(all_method, max_fake_per_method, seed + 100 + offset)
        test = real + fake
        random.Random(seed + 1000 + offset).shuffle(test)
        fake_method_counts[method] = {
            "available": len(all_method),
            "selected": len(fake),
        }
        splits[method] = {
            "train": [],
            "val": [],
            "test": test,
            "meta": {
                "dataset": "FaceForensics++ C23",
                "source_root": str(root),
                "real_method": "original",
                "fake_methods": [method],
                "available_real": len(real_all),
                "selected_real": len(real),
                "fake_method_counts": {method: fake_method_counts[method]},
                "selected_fake_total": len(fake),
                "seed": seed,
            },
        }

    cache_records = {}
    for rec in real:
        cache_records[rec["path"]] = rec
    for split in splits.values():
        for rec in split["test"]:
            cache_records[rec["path"]] = rec

    cache_split = {
        "train": [],
        "val": [],
        "test": sorted(cache_records.values(), key=lambda r: r["path"]),
        "meta": {
            "dataset": "FaceForensics++ C23",
            "source_root": str(root),
            "real_method": "original",
            "fake_methods": list(fake_dirs),
            "available_real": len(real_all),
            "selected_real": len(real),
            "fake_method_counts": fake_method_counts,
            "selected_fake_total": sum(v["selected"] for v in fake_method_counts.values()),
            "seed": seed,
            "purpose": "cache union for per-method FF++ C23 evaluation",
        },
    }
    return splits, cache_split


def load_existing_index(index_path: Path):
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return {}


def extract_cache(root: Path, split, out_dir: Path, crf_tag: str, num_frames: int,
                  sampling: str, frame_stride: int, num_segments: int,
                  image_size: int, min_success_ratio: float, device: str):
    tag_out = out_dir / crf_tag
    tag_out.mkdir(parents=True, exist_ok=True)
    index_path = tag_out / "index.json"
    index = load_existing_index(index_path)

    mtcnn = MTCNN(
        image_size=image_size,
        margin=0,
        post_process=False,
        select_largest=True,
        device=device,
    )
    mtcnn.eval()

    records = split["test"]
    print(
        f"Extracting FF++ C23 faces: records={len(records)} "
        f"frames={num_frames} sampling={sampling} out={tag_out}"
    )
    for rec in tqdm(records):
        rel = rec["path"]
        if index.get(rel, 0) >= num_frames:
            continue

        video_path = root / rel
        if not video_path.is_file():
            print(f"[WARN] Missing video: {video_path}")
            continue

        stem = Path(rel).with_suffix("")
        target_dir = tag_out / stem
        target_dir.mkdir(parents=True, exist_ok=True)

        crops = extract_video_faces(
            video_path,
            num_frames,
            mtcnn,
            image_size,
            sampling=sampling,
            frame_stride=frame_stride,
            num_segments=num_segments,
        )
        success_ratio = len(crops) / max(1, num_frames)
        if success_ratio < min_success_ratio:
            for p in target_dir.glob("*.jpg"):
                p.unlink(missing_ok=True)
            print(f"[WARN] Drop {rel}: face success {len(crops)}/{num_frames}")
            continue

        for i, img in enumerate(crops):
            cv2.imwrite(
                str(target_dir / f"frame_{i:03d}.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
        index[rel] = len(crops)
        if len(index) % 50 == 0:
            index_path.write_text(
                json.dumps(index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved index: {index_path} ({len(index)} videos)")


def main():
    parser = argparse.ArgumentParser(
        "Build FF++ C23 split and MTCNN face cache for cross-dataset testing."
    )
    parser.add_argument("--src", default="FF++/FaceForensics++_C23",
                        help="Root containing original and manipulation folders.")
    parser.add_argument("--split_out", default="splits_ffpp_c23_300.json")
    parser.add_argument("--per_method_splits", action="store_true",
                        help=("Write one balanced split per fake method. Each split "
                              "contains --max_real original videos and "
                              "--max_fake_per_method fake videos from that method."))
    parser.add_argument("--cache_out", default="face_cache_ffpp_c23_300")
    parser.add_argument("--crf_tag", default="crf23")
    parser.add_argument("--max_real", type=int, default=300)
    parser.add_argument("--max_fake_per_method", type=int, default=300,
                        help="Fake videos selected per manipulation method.")
    parser.add_argument("--fake_dirs", nargs="+", default=DEFAULT_FAKE_DIRS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling", choices=["uniform", "clip", "segments"],
                        default="uniform")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--num_segments", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--min_success_ratio", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split_only", action="store_true",
                        help="Only write split JSON; skip face extraction.")
    args = parser.parse_args()

    root = Path(args.src).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing FF++ C23 root: {root}")
    if args.sampling != "clip" and args.frame_stride != 1:
        parser.error("--frame_stride only applies when --sampling clip")
    if args.sampling == "segments" and args.num_frames % args.num_segments != 0:
        parser.error("--num_frames must be divisible by --num_segments for segments")

    if args.per_method_splits:
        method_splits, cache_split = build_per_method_splits(
            root=root,
            max_real=args.max_real,
            max_fake_per_method=args.max_fake_per_method,
            fake_dirs=args.fake_dirs,
            seed=args.seed,
        )
        split_dir = Path(args.split_out).resolve()
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saved per-method splits under: {split_dir}")
        for method, split in method_splits.items():
            out = split_dir / f"splits_ffpp_c23_{method}_{args.max_real}.json"
            out.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
            meta = split["meta"]
            print(
                f"  {method}: {out.name} "
                f"real={meta['selected_real']} fake={meta['selected_fake_total']} "
                f"total={len(split['test'])}"
            )
        cache_split_out = split_dir / f"splits_ffpp_c23_cache_union_{args.max_real}.json"
        cache_split_out.write_text(
            json.dumps(cache_split, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Saved cache-union split: {cache_split_out.name} "
            f"records={len(cache_split['test'])}"
        )
        split = cache_split
    else:
        split = build_split(
            root=root,
            max_real=args.max_real,
            max_fake_per_method=args.max_fake_per_method,
            fake_dirs=args.fake_dirs,
            seed=args.seed,
        )
        split_out = Path(args.split_out).resolve()
        split_out.parent.mkdir(parents=True, exist_ok=True)
        split_out.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

        meta = split["meta"]
        print(f"Saved split: {split_out}")
        print(
            f"Selected videos: real={meta['selected_real']} "
            f"fake={meta['selected_fake_total']} total={len(split['test'])}"
        )
        for method, counts in meta["fake_method_counts"].items():
            print(f"  {method}: {counts['selected']}/{counts['available']}")

    if args.split_only:
        return

    extract_cache(
        root=root,
        split=split,
        out_dir=Path(args.cache_out).resolve(),
        crf_tag=args.crf_tag,
        num_frames=args.num_frames,
        sampling=args.sampling,
        frame_stride=args.frame_stride,
        num_segments=args.num_segments,
        image_size=args.image_size,
        min_success_ratio=args.min_success_ratio,
        device=args.device,
    )


if __name__ == "__main__":
    main()
