"""Cross-CRF evaluation matrix.

Given one checkpoint that was trained at a single CRF, evaluate it on every
available CRF folder under face_cache/. Outputs a markdown + CSV table of
AUC / accuracy and the Performance Drop relative to the training CRF.

Example
-------
python eval_cross_compression.py \
    --ckpt checkpoints/v4/best.pth \
    --splits splits.json \
    --face_cache face_cache \
    --train_crf crf23 \
    --variant v4 \
    --crfs crf0 crf23 crf40
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from modelsgenerate.full_model import build_model_from_args
from utils.metrics import evaluate


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
def eval_on_crfs(model, args, crfs, split, device, desc):
    ds = build_eval_dataset(args, crfs, split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate)
    scores, labels = [], []
    for x, y in tqdm(loader, desc=f"eval/{desc}", leave=False):
        x = x.to(device, non_blocking=True)
        logit = model(x).squeeze(-1)
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
    if not scores:
        return None
    return evaluate(np.concatenate(labels), np.concatenate(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--variant", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v8"],
                   required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache")
    p.add_argument("--train_crf", required=True,
                   help="The CRF the model was trained on (used as reference for drop)")
    p.add_argument("--crfs", nargs="+", required=True, help="e.g. crf0 crf23 crf40")
    p.add_argument("--include_mixed", action="store_true",
                   help="Also report one pooled metric over all folders in --crfs.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
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
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_args(args).to(device)
    state = torch.load(args.ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    model.eval()

    rows = []
    for crf in args.crfs:
        rep = eval_on_crfs(model, args, [crf], args.split, device, crf)
        if rep is None:
            print(f"[WARN] no data for {crf}; skipping")
            continue
        row = {"crf": crf, **rep.to_dict()}
        rows.append(row)
    if args.include_mixed:
        rep = eval_on_crfs(model, args, args.crfs, args.split, device, "mixed")
        if rep is not None:
            rows.append({"crf": "mixed", **rep.to_dict()})

    df = pd.DataFrame(rows)
    ref_row = df[df["crf"] == args.train_crf]
    if args.train_crf != "mixed" and not ref_row.empty:
        df["auc_drop"] = ref_row["auc"].values[0] - df["auc"]
        df["acc_drop"] = ref_row["acc"].values[0] - df["acc"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{args.variant}_cross_crf_{args.split}_trainedOn_{args.train_crf}"
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nSaved {csv_path}\nSaved {md_path}")


if __name__ == "__main__":
    main()
