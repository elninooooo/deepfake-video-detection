"""Evaluate v7 real-only TIM anomaly models.

The score is reconstruction/prediction error, where larger means more fake-like.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from modelsgenerate.v7_real_tim import V7TIMConfig, build_v7_tim_model
from train_v7_real_tim import collate_drop_meta, sample_errors
from utils.metrics import evaluate


def parse_args():
    p = argparse.ArgumentParser("Evaluate v7 real-only TIM anomaly model")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--variant", choices=["v7a", "v7b"], required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_s2_all")
    p.add_argument("--train_crf", default="mixed")
    p.add_argument("--crfs", nargs="+", required=True)
    p.add_argument("--include_mixed", action="store_true")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--latent_channels", type=int, default=128)
    p.add_argument("--loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results/v7_real_tim")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def build_dataset(args, crfs):
    datasets = [
        CelebDFClipDataset(
            args.splits,
            args.face_cache,
            crf_tag=crf,
            split=args.split,
            n_frames=args.n_frames,
            train=False,
        )
        for crf in crfs
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


@torch.no_grad()
def collect_scores(model, args, crfs, device, desc):
    dataset = build_dataset(args, crfs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_drop_meta,
    )
    labels, scores = [], []
    model.eval()
    for x, y, _ in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        pred, target = model(x)
        scores.append(sample_errors(pred, target, args.loss).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def write_report(rows, out_csv: Path, out_md: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crf",
        "acc_at_eer",
        "precision_at_eer",
        "recall_at_eer",
        "f1_at_eer",
        "auc",
        "eer",
        "threshold_eer",
        "n_samples",
        "score_mean_real",
        "score_mean_fake",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| crf | Acc@EER | F1@EER | AUC | EER | thr | real score | fake score | n |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['crf']} | {r['acc_at_eer']:.4f} | {r['f1_at_eer']:.4f} | "
                f"{r['auc']:.4f} | {r['eer']:.4f} | {r['threshold_eer']:.6f} | "
                f"{r['score_mean_real']:.6f} | {r['score_mean_fake']:.6f} | "
                f"{r['n_samples']} |\n"
            )


def main():
    args = parse_args()
    device = pick_device(args.device)
    model = build_v7_tim_model(
        V7TIMConfig(variant=args.variant, latent_channels=args.latent_channels)
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])

    rows = []
    for crf in args.crfs:
        y, s = collect_scores(model, args, [crf], device, crf)
        report = evaluate(y, s, threshold=evaluate(y, s).threshold_eer)
        rows.append(
            {
                "crf": crf,
                "acc_at_eer": report.acc,
                "precision_at_eer": report.precision,
                "recall_at_eer": report.recall,
                "f1_at_eer": report.f1,
                "auc": report.auc,
                "eer": report.eer,
                "threshold_eer": report.threshold_eer,
                "n_samples": report.n_samples,
                "score_mean_real": float(np.mean(s[y == 0])),
                "score_mean_fake": float(np.mean(s[y == 1])),
            }
        )
    if args.include_mixed:
        y, s = collect_scores(model, args, args.crfs, device, "mixed")
        report = evaluate(y, s, threshold=evaluate(y, s).threshold_eer)
        rows.append(
            {
                "crf": "mixed",
                "acc_at_eer": report.acc,
                "precision_at_eer": report.precision,
                "recall_at_eer": report.recall,
                "f1_at_eer": report.f1,
                "auc": report.auc,
                "eer": report.eer,
                "threshold_eer": report.threshold_eer,
                "n_samples": report.n_samples,
                "score_mean_real": float(np.mean(s[y == 0])),
                "score_mean_fake": float(np.mean(s[y == 1])),
            }
        )

    out_dir = Path(args.out_dir)
    stem = f"{args.variant}_real_tim_{args.split}_trainedOn_{args.train_crf}"
    out_csv = out_dir / f"{stem}.csv"
    out_md = out_dir / f"{stem}.md"
    write_report(rows, out_csv, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
