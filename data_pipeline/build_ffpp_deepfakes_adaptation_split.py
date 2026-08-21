"""Build a small FF++ C23 original-vs-Deepfakes adaptation split.

This script takes an existing balanced FF++ by-method split, normally
`splits_ffpp_c23_by_method/splits_ffpp_c23_Deepfakes_300.json`, and converts
its 300 original + 300 Deepfakes test records into train/val subsets.

The output is meant for adapting GRFR/V10 on FF++ C23 Deepfakes first, then
testing the trained model on other FF++ manipulation methods.
"""

import argparse
import json
import random
from pathlib import Path


def split_records(records, train_per_class: int, val_per_class: int, seed: int):
    by_label = {0: [], 1: []}
    for rec in records:
        label = int(rec["label"])
        if label not in by_label:
            continue
        by_label[label].append(rec)

    rng = random.Random(seed)
    result = {"train": [], "val": []}
    for label in (0, 1):
        items = list(by_label[label])
        if len(items) < train_per_class + val_per_class:
            raise RuntimeError(
                f"Need {train_per_class + val_per_class} samples for label={label}, "
                f"but found {len(items)}."
            )
        rng.shuffle(items)
        train = sorted(items[:train_per_class], key=lambda r: r["path"])
        val = sorted(items[train_per_class:train_per_class + val_per_class],
                     key=lambda r: r["path"])
        result["train"].extend(train)
        result["val"].extend(val)

    rng.shuffle(result["train"])
    rng.shuffle(result["val"])
    return result


def main():
    parser = argparse.ArgumentParser(
        "Build FF++ C23 original-vs-Deepfakes train/val adaptation split."
    )
    parser.add_argument(
        "--source_split",
        default="splits_ffpp_c23_by_method/splits_ffpp_c23_Deepfakes_300.json",
        help="Existing balanced original-vs-Deepfakes split.",
    )
    parser.add_argument(
        "--out",
        default="splits_ffpp_c23_deepfakes_adapt_300.json",
    )
    parser.add_argument("--train_per_class", type=int, default=240)
    parser.add_argument("--val_per_class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source_split).resolve()
    with open(source, "r", encoding="utf-8") as f:
        source_split = json.load(f)

    records = source_split.get("test", [])
    split = split_records(
        records,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        seed=args.seed,
    )
    split["test"] = []
    split["meta"] = {
        "dataset": "FaceForensics++ C23",
        "source_split": str(source),
        "purpose": "original-vs-Deepfakes adaptation training",
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "seed": args.seed,
        "note": (
            "The total selected data volume is 300 original + 300 Deepfakes; "
            "records are divided into train and validation subsets."
        ),
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


if __name__ == "__main__":
    main()
