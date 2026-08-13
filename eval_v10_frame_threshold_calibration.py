"""Validation-threshold calibration for supervised frame-level v10 models."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from eval_v10_frame_supervised import aggregate_pair_scores
from train_v10_frame_supervised import V10FrameClassifier, collate_drop_meta
from utils.metrics import compute_eer, evaluate


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
        best_metrics = metrics_at_threshold(y_true, y_score, best_threshold)
    return best_threshold, best_metrics


def add_prefixed(row, prefix, metrics):
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def build_args_from_checkpoint(cli_args, ckpt_args):
    merged = dict(ckpt_args or {})
    merged.setdefault("relation_mode", "full")
    merged.setdefault("sampling_mode", "legacy")
    for key in [
        "splits",
        "face_cache",
        "n_frames",
        "sampling_mode",
        "residual_mode",
        "relation_mode",
        "phase_mid_low",
        "phase_mid_high",
        "spectral_relation_dim",
        "residual_encoder_dim",
    ]:
        value = getattr(cli_args, key, None)
        if value is not None:
            merged[key] = value
    return SimpleNamespace(**merged)


def build_eval_dataset(args, crfs, split):
    datasets = [
        CelebDFClipDataset(
            args.splits,
            args.face_cache,
            crf_tag=crf,
            split=split,
            n_frames=args.n_frames,
            train=False,
            sampling_mode=args.sampling_mode,
        )
        for crf in crfs
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


@torch.no_grad()
def collect_scores(model, model_args, cli_args, crfs, split, device, desc):
    ds = build_eval_dataset(model_args, crfs, split)
    loader = DataLoader(
        ds,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
        pin_memory=True,
        collate_fn=collate_drop_meta,
    )
    scores, labels = [], []
    model.eval()
    for x, y in tqdm(loader, desc=f"{split}/{desc}", leave=False):
        x = x.to(device, non_blocking=True)
        if cli_args.eval_pair_mode == "all":
            pair_scores = []
            for i in range(x.size(1) - 1):
                logits = model.forward_pair(x[:, i:i + 2])
                pair_scores.append(torch.sigmoid(logits))
            score = aggregate_pair_scores(
                torch.stack(pair_scores, dim=1),
                cli_args.score_agg,
                cli_args.mean_max_alpha,
            )
        else:
            logits = model(x)
            score = torch.sigmoid(logits)
        scores.append(score.cpu().numpy())
        labels.append(y.numpy())
    if not scores:
        return np.array([]), np.array([])
    return np.concatenate(labels).astype(int), np.concatenate(scores).astype(float)


def main():
    p = argparse.ArgumentParser("Evaluate frame-level v10 with validation threshold calibration")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--splits", default=None)
    p.add_argument("--face_cache", default=None)
    p.add_argument("--train_crf", required=True)
    p.add_argument("--crfs", nargs="+", required=True)
    p.add_argument("--include_mixed", action="store_true")
    p.add_argument("--threshold_scope", choices=["domain", "mixed"], default="mixed")
    p.add_argument("--threshold_method",
                   choices=["f1", "balanced_acc", "eer", "target_recall"],
                   default="balanced_acc")
    p.add_argument("--target_recall", type=float, default=0.90)
    p.add_argument("--threshold_steps", type=int, default=1001)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_frames", type=int, default=None)
    p.add_argument("--sampling_mode",
                   choices=sorted(CelebDFClipDataset.SAMPLING_MODES),
                   default=None)
    p.add_argument("--residual_mode", choices=["abs", "signed", "gradient"], default=None)
    p.add_argument("--relation_mode", default=None)
    p.add_argument("--phase_mid_low", type=float, default=None)
    p.add_argument("--phase_mid_high", type=float, default=None)
    p.add_argument("--spectral_relation_dim", type=int, default=None)
    p.add_argument("--residual_encoder_dim", type=int, default=None)
    p.add_argument("--eval_pair_mode", choices=["center", "all"], default="all")
    p.add_argument("--score_agg",
                   choices=["mean", "max", "top3_mean", "top5_mean", "q75", "q90", "mean_max"],
                   default="mean")
    p.add_argument("--mean_max_alpha", type=float, default=0.7)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results/v10_frame_threshold_calibration")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device)
    ckpt_args = state.get("args", {}) if isinstance(state, dict) else {}
    model_args = build_args_from_checkpoint(args, ckpt_args)
    model = V10FrameClassifier(model_args).to(device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)

    global_threshold = None
    global_val_metrics = None
    global_val_desc = None
    if args.threshold_scope == "mixed":
        y_val, s_val = collect_scores(model, model_args, args, args.crfs, "val", device, "mixed")
        global_threshold, global_val_metrics = choose_threshold(
            y_val, s_val, args.threshold_method, args.threshold_steps, args.target_recall)
        global_val_desc = "+".join(args.crfs)

    rows = []
    eval_items = [(crf, [crf]) for crf in args.crfs]
    if args.include_mixed:
        eval_items.append(("mixed", args.crfs))

    for row_name, test_crfs in eval_items:
        if args.threshold_scope == "mixed":
            threshold = global_threshold
            val_metrics = global_val_metrics
            val_source = global_val_desc
        else:
            y_val, s_val = collect_scores(model, model_args, args, test_crfs, "val", device, row_name)
            threshold, val_metrics = choose_threshold(
                y_val, s_val, args.threshold_method, args.threshold_steps, args.target_recall)
            val_source = "+".join(test_crfs)

        y_test, s_test = collect_scores(model, model_args, args, test_crfs, args.split, device, row_name)
        report_05 = evaluate(y_test, s_test, threshold=0.5)
        report_cal = evaluate(y_test, s_test, threshold=threshold)

        row = {
            "test_crf": row_name,
            "val_threshold_source": val_source,
            "threshold_method": args.threshold_method,
            "threshold_scope": args.threshold_scope,
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
    agg_tag = args.score_agg if args.eval_pair_mode == "all" else "center"
    base = (
        f"v10_frame_threshold_{args.threshold_method}_{args.threshold_scope}_"
        f"{agg_tag}_{args.split}_trainedOn_{args.train_crf}"
    )
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nSaved {csv_path}\nSaved {md_path}")


if __name__ == "__main__":
    main()
