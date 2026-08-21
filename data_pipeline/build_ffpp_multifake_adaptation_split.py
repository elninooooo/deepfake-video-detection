"""Build an FF++ C23 multi-manipulation adaptation split.

The split uses a balanced total volume of original and fake videos:

    real: 300 original videos
    fake: 60 videos per manipulation method, 5 methods, 300 fake videos total

The selected records are divided into train and validation subsets while
preserving class and fake-method balance. It is intended for adapting GRFR/V10
to FF++ C23 with multiple manipulation families before testing each method
separately.
"""

import argparse
import json
import random
from pathlib import Path


DEFAULT_FAKE_METHODS = [
    "Deepfakes",
    "FaceSwap",
    "FaceShifter",
    "Face2Face",
    "NeuralTextures",
]


def load_records(split_dir: Path, method: str):
    path = split_dir / f"splits_ffpp_c23_{method}_300.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing method split: {path}")
    with open(path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return split["test"]


def select_original(records, count: int, seed: int):
    original = [r for r in records if r.get("method") == "original" and int(r["label"]) == 0]
    if len(original) < count:
        raise RuntimeError(f"Need {count} original videos, found {len(original)}")
    rng = random.Random(seed)
    original = list(original)
    rng.shuffle(original)
    return sorted(original[:count], key=lambda r: r["path"])


def select_fake(records, method: str, count: int, seed: int):
    fake = [r for r in records if r.get("method") == method and int(r["label"]) == 1]
    if len(fake) < count:
        raise RuntimeError(f"Need {count} {method} videos, found {len(fake)}")
    rng = random.Random(seed)
    fake = list(fake)
    rng.shuffle(fake)
    return sorted(fake[:count], key=lambda r: r["path"])


def split_list(items, train_count: int, seed: int):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    return (
        sorted(items[:train_count], key=lambda r: r["path"]),
        sorted(items[train_count:], key=lambda r: r["path"]),
    )


def main():
    parser = argparse.ArgumentParser("Build FF++ C23 multi-fake adaptation split.")
    parser.add_argument("--split_dir", default="splits_ffpp_c23_by_method")
    parser.add_argument("--out", default="splits_ffpp_c23_multifake_adapt_300.json")
    parser.add_argument("--fake_methods", nargs="+", default=DEFAULT_FAKE_METHODS)
    parser.add_argument("--original_count", type=int, default=300)
    parser.add_argument("--fake_per_method", type=int, default=60)
    parser.add_argument("--train_real", type=int, default=240)
    parser.add_argument("--train_fake_per_method", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dir = Path(args.split_dir).resolve()
    first_records = load_records(split_dir, args.fake_methods[0])
    original = select_original(first_records, args.original_count, args.seed)
    train_real, val_real = split_list(original, args.train_real, args.seed + 10)

    train, val = list(train_real), list(val_real)
    fake_counts = {}
    for offset, method in enumerate(args.fake_methods):
        records = load_records(split_dir, method)
        fake = select_fake(records, method, args.fake_per_method, args.seed + 100 + offset)
        train_fake, val_fake = split_list(fake, args.train_fake_per_method, args.seed + 200 + offset)
        train.extend(train_fake)
        val.extend(val_fake)
        fake_counts[method] = {
            "selected": len(fake),
            "train": len(train_fake),
            "val": len(val_fake),
        }

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)
    split = {
        "train": train,
        "val": val,
        "test": [],
        "meta": {
            "dataset": "FaceForensics++ C23",
            "purpose": "multi-manipulation adaptation training",
            "real_method": "original",
            "fake_methods": list(args.fake_methods),
            "original_count": args.original_count,
            "fake_per_method": args.fake_per_method,
            "train_real": len(train_real),
            "val_real": len(val_real),
            "fake_method_counts": fake_counts,
            "seed": args.seed,
        },
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out}")
    for name in ("train", "val", "test"):
        recs = split[name]
        real = sum(1 for r in recs if int(r["label"]) == 0)
        fake = sum(1 for r in recs if int(r["label"]) == 1)
        print(f"{name}: total={len(recs)} real={real} fake={fake}")
    print("fake method counts:")
    for method, counts in fake_counts.items():
        print(f"  {method}: selected={counts['selected']} train={counts['train']} val={counts['val']}")


if __name__ == "__main__":
    main()
