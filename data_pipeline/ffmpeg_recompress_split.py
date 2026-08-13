"""Re-encode only videos listed in a split JSON.

This is useful for cross-dataset evaluation subsets, for example a balanced
Celeb-DF-v2 split capped at 300 real and 300 fake videos. The output tree is
compatible with preprocess_faces.py:

    <dst>/crf0/Celeb-real/...
    <dst>/crf23/Celeb-synthesis/...
    <dst>/crf40/YouTube-real/...
"""

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def encode_one(src_path: str, dst_path: str, crf: int) -> str:
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        return f"SKIP  {dst}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src_path,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-an",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return f"OK    {dst}"
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="ignore")[:200]
        return f"FAIL  {dst}  err={err}"


def load_records(split_json: Path, splits):
    with split_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    records = []
    for split in splits:
        records.extend(data.get(split, []))
    # Deduplicate paths while preserving order.
    seen = set()
    unique = []
    for rec in records:
        rel = rec["path"].replace("\\", "/")
        if rel not in seen:
            seen.add(rel)
            unique.append(rel)
    return unique


def main():
    parser = argparse.ArgumentParser("Recompress videos listed in split JSON")
    parser.add_argument("--src", required=True, help="Original dataset root")
    parser.add_argument("--splits", required=True, help="Split JSON with path records")
    parser.add_argument("--dst", required=True, help="Output recompressed root")
    parser.add_argument("--split_names", nargs="+", default=["test"],
                        help="Which split keys to recompress.")
    parser.add_argument("--crfs", type=int, nargs="+", default=[0, 23, 40])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH.", file=sys.stderr)
        sys.exit(1)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    rel_paths = load_records(Path(args.splits), args.split_names)

    jobs = []
    missing = []
    for rel in rel_paths:
        src_path = src / rel
        if not src_path.is_file():
            missing.append(rel)
            continue
        for crf in args.crfs:
            jobs.append((str(src_path), str(dst / f"crf{crf}" / rel), crf))

    if missing:
        print(f"[WARN] Missing {len(missing)} source videos. First few:")
        for rel in missing[:10]:
            print(f"  {rel}")

    print(f"Videos: {len(rel_paths) - len(missing)}  jobs: {len(jobs)}  workers={args.workers}")
    dst.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(encode_one, *job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            msg = fut.result()
            if msg.startswith("FAIL"):
                tqdm.write(msg)
    print("Done.")


if __name__ == "__main__":
    main()
