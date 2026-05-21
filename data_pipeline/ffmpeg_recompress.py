"""Re-encode every video in a Celeb-DF tree at several CRF levels using ffmpeg.

Produces a mirrored directory tree under --dst with sub-folders crf{value}/.
Skips files that already exist (safe to re-run).

Usage example:
    python dataset/ffmpeg_recompress.py \
        --src "../Celeb-DF" \
        --dst "../Celeb-DF-recompressed" \
        --crfs 0 23 40 \
        --workers 4

ffmpeg must be on PATH. On Windows: install via `winget install Gyan.FFmpeg`
or `conda install -c conda-forge ffmpeg`.
"""

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


SUBDIRS = ("Celeb-real", "Celeb-synthesis", "YouTube-real")


def encode_one(src_path: str, dst_path: str, crf: int) -> str:
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        return f"SKIP  {dst}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src_path,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-an",  # drop audio (not needed for visual detection)
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return f"OK    {dst}"
    except subprocess.CalledProcessError as e:
        return f"FAIL  {dst}  err={e.stderr.decode(errors='ignore')[:200]}"


def collect_jobs(src: Path, dst: Path, crfs):
    jobs = []
    for sub in SUBDIRS:
        in_dir = src / sub
        if not in_dir.is_dir():
            continue
        for vid in sorted(in_dir.glob("*.mp4")):
            for crf in crfs:
                out_path = dst / f"crf{crf}" / sub / vid.name
                jobs.append((str(vid), str(out_path), crf))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Original Celeb-DF root")
    ap.add_argument("--dst", required=True, help="Output root (will contain crf0/, crf23/, ...)")
    ap.add_argument("--crfs", type=int, nargs="+", default=[0, 23, 40])
    ap.add_argument("--workers", type=int, default=2,
                    help="Parallel ffmpeg processes. Each uses ~1 CPU core.")
    ap.add_argument("--copy_metadata", action="store_true",
                    help="Also copy List_of_testing_videos.txt to dst/")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH.", file=sys.stderr)
        sys.exit(1)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    if args.copy_metadata:
        meta = src / "List_of_testing_videos.txt"
        if meta.is_file():
            shutil.copy2(meta, dst / "List_of_testing_videos.txt")

    jobs = collect_jobs(src, dst, args.crfs)
    print(f"Total jobs: {len(jobs)}  (workers={args.workers})")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(encode_one, *j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs)):
            msg = f.result()
            if msg.startswith("FAIL"):
                tqdm.write(msg)

    print("Done.")


if __name__ == "__main__":
    main()
