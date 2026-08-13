"""Validation-set threshold calibration for same-CRF and cross-CRF evaluation.

This script keeps the model fixed. It first chooses a decision threshold from
validation scores, then applies that threshold to the corresponding test split.
It reports both the original fixed-threshold metrics (@0.5) and calibrated
metrics (@val-threshold), so AUC/ranking and thresholded classification remain
easy to compare.

Examples
--------
# Domain-aware 3x3 cross-CRF calibration for a crf0-trained V1 model.
python eval_threshold_calibration.py \
    --ckpt checkpoints/v1_tim_crf0/best.pth \
    --variant v1 \
    --face_cache face_cache \
    --train_crf crf0 \
    --crfs crf0 crf23 crf40 \
    --threshold_method f1

# Mixed model with one pooled validation threshold.
python eval_threshold_calibration.py \
    --ckpt checkpoints/v1_tim_mixed_all/best.pth \
    --variant v1 \
    --face_cache face_cache \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --threshold_scope mixed
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from modelsgenerate.full_model import build_model_from_args
from utils.metrics import compute_eer, evaluate


def collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


def build_eval_dataset(args, crfs, split):
    datasets = [
        CelebDFClipDataset(args.splits, args.face_cache, crf_tag=crf,
                           split=split, n_frames=args.n_frames, train=False)
        for crf in crfs
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


@torch.no_grad()
def collect_scores(model, args, crfs, split, device, desc):
    ds = build_eval_dataset(args, crfs, split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate)
    scores, labels = [], []
    for x, y in tqdm(loader, desc=f"{split}/{desc}", leave=False):
        x = x.to(device, non_blocking=True)
        logit = model(x).squeeze(-1)
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
    if not scores:
        return np.array([]), np.array([])
    return np.concatenate(labels).astype(int), np.concatenate(scores).astype(float)


def metrics_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def choose_threshold(y_true, y_score, method, steps, target_recall):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    if y_true.size == 0:
        return 0.5, {}

    if method == "eer":
        _, threshold = compute_eer(y_true, y_score)
        return float(threshold), metrics_at_threshold(y_true, y_score, threshold)

    thresholds = np.linspace(0.0, 1.0, int(steps))
    best_threshold = 0.5
    best_metrics = None
    best_key = None

    for threshold in thresholds:
        cur = metrics_at_threshold(y_true, y_score, threshold)
        if method == "f1":
            key = (cur["f1"], cur["balanced_acc"], cur["acc"])
        elif method == "balanced_acc":
            key = (cur["balanced_acc"], cur["f1"], cur["acc"])
        elif method == "target_recall":
            if cur["recall"] < target_recall:
                continue
            key = (cur["precision"], cur["f1"], cur["balanced_acc"])
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = cur

    if best_metrics is None:
        # Fallback when no threshold can satisfy target_recall.
        for threshold in thresholds:
            cur = metrics_at_threshold(y_true, y_score, threshold)
            key = (cur["recall"], cur["f1"], cur["balanced_acc"])
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)
                best_metrics = cur

    return best_threshold, best_metrics


def add_prefixed(row, prefix, metrics):
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def main():
    p = argparse.ArgumentParser(
        "Evaluate checkpoints with validation-set threshold calibration.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--variant", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v8", "v10"],
                   required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache")
    p.add_argument("--train_crf", required=True,
                   help="Training CRF/domain label used in output filenames.")
    p.add_argument("--crfs", nargs="+", required=True,
                   help="Test domains, e.g. crf0 crf23 crf40.")
    p.add_argument("--include_mixed", action="store_true",
                   help="Also evaluate one pooled row over all --crfs.")
    p.add_argument("--threshold_scope", choices=["domain", "mixed"], default="domain",
                   help=("domain: each test domain uses its matching validation domain; "
                         "mixed: all rows share one threshold from pooled validation domains."))
    p.add_argument("--threshold_method",
                   choices=["f1", "balanced_acc", "eer", "target_recall"],
                   default="f1")
    p.add_argument("--target_recall", type=float, default=0.90)
    p.add_argument("--threshold_steps", type=int, default=1001)
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--mask_radius_ratio", type=float, default=0.5)
    p.add_argument("--phase_mode", choices=["raw", "weighted", "mid_weighted"],
                   default="raw")
    p.add_argument("--phase_confidence_quantile", type=float, default=0.95)
    p.add_argument("--phase_mid_low", type=float, default=0.10)
    p.add_argument("--phase_mid_high", type=float, default=0.70)
    p.add_argument("--spectral_relation_dim", type=int, default=128)
    p.add_argument("--residual_encoder_dim", type=int, default=256)
    p.add_argument("--residual_mode", choices=["abs", "signed", "gradient"],
                   default="gradient")
    p.add_argument("--relation_mode",
                   choices=[
                       "abs_cos", "abs_dist", "abs_only", "concat_views", "cos_dist",
                       "cos_only", "dist_only", "feat_only", "full", "no_original",
                   ],
                   default="full")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out_dir", default="results/threshold_calibration")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = build_model_from_args(args).to(device)
    state = torch.load(args.ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    model.eval()

    rows = []
    global_threshold = None
    global_val_metrics = None
    global_val_desc = None
    if args.threshold_scope == "mixed":
        y_val, s_val = collect_scores(model, args, args.crfs, "val", device, "mixed")
        global_threshold, global_val_metrics = choose_threshold(
            y_val, s_val, args.threshold_method, args.threshold_steps, args.target_recall)
        global_val_desc = "+".join(args.crfs)

    eval_items = [(crf, [crf]) for crf in args.crfs]
    if args.include_mixed:
        eval_items.append(("mixed", args.crfs))

    for row_name, test_crfs in eval_items:
        if args.threshold_scope == "mixed":
            threshold = global_threshold
            val_metrics = global_val_metrics
            val_source = global_val_desc
        else:
            val_crfs = test_crfs
            y_val, s_val = collect_scores(model, args, val_crfs, "val", device, row_name)
            threshold, val_metrics = choose_threshold(
                y_val, s_val, args.threshold_method, args.threshold_steps, args.target_recall)
            val_source = "+".join(val_crfs)

        y_test, s_test = collect_scores(model, args, test_crfs, "test", device, row_name)
        report_05 = evaluate(y_test, s_test, threshold=0.5)
        report_cal = evaluate(y_test, s_test, threshold=threshold)

        row = {
            "test_crf": row_name,
            "val_threshold_source": val_source,
            "threshold_method": args.threshold_method,
            "threshold": float(threshold),
            "auc": report_05.auc,
            "eer": report_05.eer,
            "eer_threshold_test_only": report_05.threshold_eer,
            "n_samples": report_05.n_samples,
        }
        add_prefixed(row, "val_at_thr", val_metrics)
        add_prefixed(row, "test_at_0p5", report_05.to_dict())
        add_prefixed(row, "test_at_val_thr", report_cal.to_dict())
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{args.variant}_threshold_{args.threshold_method}_{args.threshold_scope}_trainedOn_{args.train_crf}"
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nSaved {csv_path}\nSaved {md_path}")


if __name__ == "__main__":
    main()
