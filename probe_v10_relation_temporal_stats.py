"""Probe temporal statistics of six frozen v10 cosine relations.

This script freezes a trained cos_only frame model, extracts the raw six
cross-frequency cosine relations over all adjacent frame pairs, summarizes
their temporal behavior, and trains a supervised logistic-regression probe.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from train_v10_frame_supervised import collate_drop_meta, pick_device
from utils.metrics import evaluate
from utils.seed import set_seed
from v10_relation_temporal_utils import (
    RELATION_GROUPS,
    RELATION_NAMES,
    STAT_NAMES,
    load_frozen_frame_model,
    pair_scores,
    raw_cosine_relations,
    temporal_stats_np,
)


def parse_args():
    p = argparse.ArgumentParser("Probe temporal stats of v10 cosine relations")
    p.add_argument("--frame_ckpt", required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_s2_all")
    p.add_argument("--train_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--test_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--groups", nargs="+", default=list(RELATION_GROUPS.keys()) + ["score_stats", "score_plus_all"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results/v10_relation_temporal_stats")
    return p.parse_args()


def build_dataset(args, crfs, split):
    datasets = [
        CelebDFClipDataset(
            args.splits,
            args.face_cache,
            crf_tag=crf,
            split=split,
            n_frames=args.n_frames,
            train=False,
        )
        for crf in crfs
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


@torch.no_grad()
def collect_sequences(frame_model, args, crfs, split, device, desc):
    ds = build_dataset(args, crfs, split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_drop_meta,
    )
    rels, scores, labels = [], [], []
    for x, y in tqdm(loader, desc=f"collect/{split}/{desc}", leave=False):
        x = x.to(device, non_blocking=True)
        rels.append(raw_cosine_relations(frame_model.branch, x).cpu().numpy())
        scores.append(pair_scores(frame_model, x).cpu().numpy()[..., None])
        labels.append(y.numpy())
    return (
        np.concatenate(rels, axis=0),
        np.concatenate(scores, axis=0),
        np.concatenate(labels, axis=0).astype(int),
    )


def feature_matrix(rel_seq, score_seq, group):
    if group == "score_stats":
        return temporal_stats_np(score_seq)
    if group == "score_plus_all":
        return np.concatenate([temporal_stats_np(score_seq), temporal_stats_np(rel_seq)], axis=1)
    idx = RELATION_GROUPS[group]
    return temporal_stats_np(rel_seq[:, :, idx])


def feature_columns(group):
    if group == "score_stats":
        return [f"score_{stat}" for stat in STAT_NAMES]
    if group == "score_plus_all":
        return feature_columns("score_stats") + [
            f"{name}_{stat}" for stat in STAT_NAMES for name in RELATION_NAMES
        ]
    return [f"{RELATION_NAMES[i]}_{stat}" for stat in STAT_NAMES for i in RELATION_GROUPS[group]]


def fit_probe(x_train, y_train, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    ).fit(x_train, y_train)


def summarize_relation_shift(rel_seq, labels):
    rows = []
    for i, name in enumerate(RELATION_NAMES):
        mean_over_time = rel_seq[:, :, i].mean(axis=1)
        rows.append(
            {
                "relation": name,
                "real_mean": float(mean_over_time[labels == 0].mean()),
                "fake_mean": float(mean_over_time[labels == 1].mean()),
                "fake_minus_real": float(mean_over_time[labels == 1].mean() - mean_over_time[labels == 0].mean()),
                "real_std": float(mean_over_time[labels == 0].std()),
                "fake_std": float(mean_over_time[labels == 1].std()),
            }
        )
    return rows


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_model, frame_args = load_frozen_frame_model(args.frame_ckpt, args, device)
    print(f"loaded frame relation mode: {getattr(frame_args, 'relation_mode', 'unknown')}")

    rel_train, score_train, y_train = collect_sequences(
        frame_model, args, args.train_crfs, "train", device, "mixed_train"
    )
    print(f"train relation seq: {rel_train.shape}  score seq: {score_train.shape}")

    rows = []
    probes = {}
    for group in args.groups:
        if group not in RELATION_GROUPS and group not in {"score_stats", "score_plus_all"}:
            raise ValueError(f"Unknown group: {group}")
        x_train = feature_matrix(rel_train, score_train, group)
        probes[group] = fit_probe(x_train, y_train, args.seed)

    eval_items = [(crf, [crf]) for crf in args.test_crfs]
    eval_items.append(("mixed", args.test_crfs))
    shift_rows = []
    for split_name, crfs in eval_items:
        rel_test, score_test, y_test = collect_sequences(
            frame_model, args, crfs, "test", device, split_name
        )
        for row in summarize_relation_shift(rel_test, y_test):
            shift_rows.append({"split": split_name, **row})
        for group, probe in probes.items():
            x_test = feature_matrix(rel_test, score_test, group)
            y_score = probe.predict_proba(x_test)[:, 1]
            rep = evaluate(y_test, y_score)
            rows.append({"group": group, "split": split_name, **rep.to_dict()})

    df = pd.DataFrame(rows)
    csv_path = out_dir / "relation_temporal_stats_probe.csv"
    md_path = out_dir / "relation_temporal_stats_probe.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")

    shift_df = pd.DataFrame(shift_rows)
    shift_csv = out_dir / "relation_mean_shift.csv"
    shift_df.to_csv(shift_csv, index=False)

    print(df.sort_values(["split", "auc"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved {csv_path}\nSaved {md_path}\nSaved {shift_csv}")


if __name__ == "__main__":
    main()
