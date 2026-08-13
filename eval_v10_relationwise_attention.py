"""Cross-CRF evaluation for relation-wise temporal attention."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from train_v10_frame_supervised import collate_drop_meta
from train_v10_relationwise_attention import RelationWiseAttention
from utils.metrics import evaluate
from v10_relation_temporal_utils import load_frozen_frame_model


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
def eval_on_crfs(model, args, crfs, device, desc):
    ds = build_dataset(args, crfs, args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_drop_meta,
    )
    scores, labels, rel_weights = [], [], []
    model.eval()
    for x, y in tqdm(loader, desc=f"eval/{desc}", leave=False):
        x = x.to(device, non_blocking=True)
        logits, _, rw = model(x, return_attention=True)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
        rel_weights.append(rw.cpu().numpy())
    if not scores:
        return None, None
    return (
        evaluate(np.concatenate(labels), np.concatenate(scores)),
        np.concatenate(rel_weights, axis=0).mean(axis=0),
    )


def main():
    p = argparse.ArgumentParser("Evaluate relation-wise temporal attention")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--splits", default=None)
    p.add_argument("--face_cache", default=None)
    p.add_argument("--train_crf", required=True)
    p.add_argument("--crfs", nargs="+", required=True)
    p.add_argument("--include_mixed", action="store_true")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_frames", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device)
    train_args = dict(state.get("args", {}))
    eval_args = SimpleNamespace(**train_args)
    eval_args.splits = args.splits or train_args.get("splits", "splits.json")
    eval_args.face_cache = args.face_cache or train_args.get("face_cache", "face_cache_s2_all")
    eval_args.n_frames = args.n_frames or int(train_args.get("n_frames", 16))
    eval_args.split = args.split
    eval_args.batch_size = args.batch_size
    eval_args.num_workers = args.num_workers
    eval_args.frame_ckpt = train_args["frame_ckpt"]

    frame_model, _ = load_frozen_frame_model(eval_args.frame_ckpt, eval_args, device)
    indices = list(state.get("relation_indices", [0, 1, 2, 3, 4, 5]))
    relation_names = list(state.get("relation_names", [str(i) for i in indices]))
    model = RelationWiseAttention(
        frozen_branch=frame_model.branch,
        indices=indices,
        max_pairs=eval_args.n_frames - 1,
        hidden_dim=int(train_args.get("hidden_dim", 32)),
        dropout=float(train_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(state["model"])

    rows, weight_rows = [], []
    eval_items = [(crf, [crf]) for crf in args.crfs]
    if args.include_mixed:
        eval_items.append(("mixed", args.crfs))
    for name, crfs in eval_items:
        rep, weights = eval_on_crfs(model, eval_args, crfs, device, name)
        if rep is None:
            continue
        rows.append({"crf": name, **rep.to_dict()})
        for rel_name, weight in zip(relation_names, weights):
            weight_rows.append({"crf": name, "relation": rel_name, "mean_attention": float(weight)})

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"v10_relationwise_attention_cross_crf_{args.split}_trainedOn_{args.train_crf}"
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"
    weight_path = out_dir / f"{base}_relation_weights.csv"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    pd.DataFrame(weight_rows).to_csv(weight_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {csv_path}\nSaved {md_path}\nSaved {weight_path}")


if __name__ == "__main__":
    main()
