"""Cross-CRF evaluation for frozen-frame CLS temporal aggregator."""

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
from train_v10_frame_cls_aggregator import CLSAggregator, load_frozen_branch
from train_v10_frame_supervised import collate_drop_meta
from utils.metrics import evaluate


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
def eval_on_crfs(model, args, crfs, device, desc):
    ds = build_eval_dataset(args, crfs, args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_drop_meta,
    )
    scores, labels = [], []
    model.eval()
    for x, y in tqdm(loader, desc=f"eval/{desc}", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    if not scores:
        return None
    return evaluate(np.concatenate(labels), np.concatenate(scores))


def main():
    p = argparse.ArgumentParser("Evaluate frozen-frame CLS temporal aggregator")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--splits", default=None)
    p.add_argument("--face_cache", default=None)
    p.add_argument("--train_crf", required=True)
    p.add_argument("--crfs", nargs="+", required=True)
    p.add_argument("--include_mixed", action="store_true")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_frames", type=int, default=None)
    p.add_argument("--sampling_mode",
                   choices=sorted(CelebDFClipDataset.SAMPLING_MODES),
                   default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device)
    train_args = dict(state.get("args", {}))
    frame_args = dict(state.get("frame_args", {}))

    eval_args = SimpleNamespace(**train_args)
    eval_args.splits = args.splits or train_args.get("splits", "splits.json")
    eval_args.face_cache = args.face_cache or train_args.get("face_cache", "face_cache_s2_all")
    eval_args.n_frames = args.n_frames or int(train_args.get("n_frames", 16))
    eval_args.sampling_mode = args.sampling_mode or train_args.get("sampling_mode", "legacy")
    eval_args.split = args.split
    eval_args.batch_size = args.batch_size
    eval_args.num_workers = args.num_workers

    # Reuse the original frame checkpoint path stored in train args.
    eval_args.frame_ckpt = train_args["frame_ckpt"]
    frozen_branch, loaded_frame_args = load_frozen_branch(eval_args.frame_ckpt, eval_args, device)
    feature_dim = int(frame_args.get("spectral_relation_dim", loaded_frame_args.spectral_relation_dim))
    model = CLSAggregator(
        frozen_branch=frozen_branch,
        feature_dim=feature_dim,
        d_model=int(train_args.get("d_model", 128)),
        n_heads=int(train_args.get("n_heads", 4)),
        num_layers=int(train_args.get("num_layers", 1)),
        dropout=float(train_args.get("dropout", 0.1)),
        max_pairs=eval_args.n_frames - 1,
    ).to(device)
    model.load_state_dict(state["model"])

    rows = []
    for crf in args.crfs:
        rep = eval_on_crfs(model, eval_args, [crf], device, crf)
        if rep is None:
            print(f"[WARN] no data for {crf}; skipping")
            continue
        rows.append({"crf": crf, **rep.to_dict()})
    if args.include_mixed:
        rep = eval_on_crfs(model, eval_args, args.crfs, device, "mixed")
        if rep is not None:
            rows.append({"crf": "mixed", **rep.to_dict()})

    df = pd.DataFrame(rows)
    ref_row = df[df["crf"] == args.train_crf]
    if args.train_crf != "mixed" and not ref_row.empty:
        df["auc_drop"] = ref_row["auc"].values[0] - df["auc"]
        df["acc_drop"] = ref_row["acc"].values[0] - df["acc"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"v10_frame_cls_cross_crf_{args.split}_trainedOn_{args.train_crf}"
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nSaved {csv_path}\nSaved {md_path}")


if __name__ == "__main__":
    main()
