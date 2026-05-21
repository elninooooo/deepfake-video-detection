"""Build identity-aware train/val/test split for Celeb-DF.

Celeb-DF layout (under --src):
    Celeb-real/idX_YYYY.mp4
    Celeb-synthesis/idA_idB_YYYY.mp4
    YouTube-real/NNNNN.mp4
    List_of_testing_videos.txt        # lines: "label path"  (label==1 => real)

Output:
    splits.json with keys train / val / test, each a list of:
        {"path": "Celeb-real/id0_0001.mp4", "label": 0 or 1, "identity": "id0"}

Labels follow the methodology: 0 = real, 1 = fake.
NB: the official testing-list file uses 1=real and 0=fake — we invert it here.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def parse_identity(rel_path: str) -> str:
    """Extract a stable identity tag from a Celeb-DF video filename."""
    name = Path(rel_path).stem
    if rel_path.startswith("YouTube-real"):
        return f"youtube_{name}"
    m = re.match(r"(id\d+)", name)
    if m:
        return m.group(1)
    return name


def list_videos(src: Path):
    """Return list of (relative_path, label) for every Celeb-DF mp4."""
    items = []
    for sub, lbl in [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]:
        d = src / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.mp4")):
            items.append((f"{sub}/{p.name}", lbl))
    return items


def load_test_list(src: Path):
    """Read List_of_testing_videos.txt → set of relative paths."""
    f = src / "List_of_testing_videos.txt"
    if not f.is_file():
        raise FileNotFoundError(f"Missing {f}")
    test_paths = set()
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # format: "1 YouTube-real/00170.mp4"
            _lbl, rel = line.split(maxsplit=1)
            test_paths.add(rel.replace("\\", "/"))
    return test_paths


def make_split(src: Path, val_ratio: float = 0.1, seed: int = 42):
    random.seed(seed)
    all_items = list_videos(src)
    test_set = load_test_list(src)

    test, remaining = [], []
    for rel, lbl in all_items:
        rec = {"path": rel, "label": lbl, "identity": parse_identity(rel)}
        if rel in test_set:
            test.append(rec)
        else:
            remaining.append(rec)

    by_id = defaultdict(list)
    for r in remaining:
        by_id[r["identity"]].append(r)
    ids = sorted(by_id.keys())
    random.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])

    train, val = [], []
    for i in ids:
        bucket = val if i in val_ids else train
        bucket.extend(by_id[i])

    return {"train": train, "val": val, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True,
                    help="Path to Celeb-DF root containing Celeb-real/, Celeb-synthesis/, "
                         "YouTube-real/ and List_of_testing_videos.txt")
    ap.add_argument("--out", type=str, default="splits.json")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    splits = make_split(src, args.val_ratio, args.seed)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    for k in ("train", "val", "test"):
        n_real = sum(1 for r in splits[k] if r["label"] == 0)
        n_fake = sum(1 for r in splits[k] if r["label"] == 1)
        print(f"{k:>5}: {len(splits[k]):4d} videos  ({n_real} real / {n_fake} fake)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
