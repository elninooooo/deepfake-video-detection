"""Build a balanced Celeb-DF-v2 evaluation split.

The output follows the same schema used by CelebDFClipDataset:

    {
      "train": [],
      "val": [],
      "test": [
        {"path": "Celeb-real/id0_0001.mp4", "label": 0, "identity": "id0"},
        {"path": "Celeb-synthesis/id0_id1_0001.mp4", "label": 1, "identity": "id0_id1"}
      ]
    }

Labels are 0 = real and 1 = fake. The script is intentionally test-focused:
for zero-shot cross-dataset evaluation, Celeb-DF-v2 should not be used for
training.
"""

import argparse
import json
import random
import re
from pathlib import Path


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def parse_identity(rel_path: str) -> str:
    name = Path(rel_path).stem
    if rel_path.startswith("YouTube-real"):
        return f"youtube_{name}"
    m = re.match(r"(id\d+)", name)
    if m:
        return m.group(1)
    return name


def list_videos(root: Path, subdirs, label: int):
    records = []
    for sub in subdirs:
        base = root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                rel = path.relative_to(root).as_posix()
                records.append({
                    "path": rel,
                    "label": label,
                    "identity": parse_identity(rel),
                })
    return records


def capped_sample(records, cap: int, seed: int):
    if cap <= 0 or len(records) <= cap:
        return list(records)
    rng = random.Random(seed)
    picked = list(records)
    rng.shuffle(picked)
    return sorted(picked[:cap], key=lambda r: r["path"])


def main():
    parser = argparse.ArgumentParser("Build balanced Celeb-DF-v2 test split")
    parser.add_argument("--src", required=True,
                        help="Celeb-DF-v2 root or recompressed tree source root.")
    parser.add_argument("--out", default="splits_celebdfv2_300.json")
    parser.add_argument("--max_real", type=int, default=300)
    parser.add_argument("--max_fake", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real_dirs", nargs="+",
                        default=["Celeb-real", "YouTube-real"],
                        help="Real-video subdirectories under --src.")
    parser.add_argument("--fake_dirs", nargs="+",
                        default=["Celeb-synthesis"],
                        help="Fake-video subdirectories under --src.")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {src}")

    real = list_videos(src, args.real_dirs, label=0)
    fake = list_videos(src, args.fake_dirs, label=1)
    if not real:
        raise RuntimeError(f"No real videos found under {args.real_dirs} in {src}")
    if not fake:
        raise RuntimeError(f"No fake videos found under {args.fake_dirs} in {src}")

    real = capped_sample(real, args.max_real, args.seed)
    fake = capped_sample(fake, args.max_fake, args.seed + 1)
    test = real + fake
    random.Random(args.seed).shuffle(test)

    splits = {"train": [], "val": [], "test": test}
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Found real={len(list_videos(src, args.real_dirs, label=0))}, "
          f"fake={len(list_videos(src, args.fake_dirs, label=1))}")
    print(f"Saved test split: {out}")
    print(f"test: {len(test)} videos ({len(real)} real / {len(fake)} fake)")


if __name__ == "__main__":
    main()
