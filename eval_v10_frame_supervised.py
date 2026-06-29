"""Cross-CRF evaluation for supervised frame-level v10 classifier."""

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
from train_v10_frame_supervised import V10FrameClassifier, collate_drop_meta
from utils.metrics import evaluate


def build_args_from_checkpoint(cli_args, ckpt_args):
    merged = dict(ckpt_args or {})
    for key in [
        "splits",
        "face_cache",
        "n_frames",
        "residual_mode",
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
        )
        for crf in crfs
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


@torch.no_grad()
def eval_on_crfs(model, model_args, cli_args, crfs, device, desc):
    ds = build_eval_dataset(model_args, crfs, cli_args.split)
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
    for x, y in tqdm(loader, desc=f"eval/{desc}", leave=False):
        x = x.to(device, non_blocking=True)
        if cli_args.eval_pair_mode == "all":
            pair_scores = []
            for i in range(x.size(1) - 1):
                logits = model.forward_pair(x[:, i:i + 2])
                pair_scores.append(torch.sigmoid(logits))
            score = torch.stack(pair_scores, dim=1).mean(dim=1)
        else:
            logits = model(x)
            score = torch.sigmoid(logits)
        scores.append(score.cpu().numpy())
        labels.append(y.numpy())
    if not scores:
        return None
    return evaluate(np.concatenate(labels), np.concatenate(scores))


def main():
    p = argparse.ArgumentParser("Evaluate supervised frame-level v10 classifier")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--splits", default=None)
    p.add_argument("--face_cache", default=None)
    p.add_argument("--train_crf", required=True)
    p.add_argument("--crfs", nargs="+", required=True)
    p.add_argument("--include_mixed", action="store_true")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_frames", type=int, default=None)
    p.add_argument("--residual_mode", choices=["abs", "signed", "gradient"], default=None)
    p.add_argument("--phase_mid_low", type=float, default=None)
    p.add_argument("--phase_mid_high", type=float, default=None)
    p.add_argument("--spectral_relation_dim", type=int, default=None)
    p.add_argument("--residual_encoder_dim", type=int, default=None)
    p.add_argument("--eval_pair_mode", choices=["center", "all"], default="all")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device)
    ckpt_args = state.get("args", {}) if isinstance(state, dict) else {}
    model_args = build_args_from_checkpoint(args, ckpt_args)
    model = V10FrameClassifier(model_args).to(device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)

    rows = []
    for crf in args.crfs:
        rep = eval_on_crfs(model, model_args, args, [crf], device, crf)
        if rep is None:
            print(f"[WARN] no data for {crf}; skipping")
            continue
        rows.append({"crf": crf, **rep.to_dict()})
    if args.include_mixed:
        rep = eval_on_crfs(model, model_args, args, args.crfs, device, "mixed")
        if rep is not None:
            rows.append({"crf": "mixed", **rep.to_dict()})

    df = pd.DataFrame(rows)
    ref_row = df[df["crf"] == args.train_crf]
    if args.train_crf != "mixed" and not ref_row.empty:
        df["auc_drop"] = ref_row["auc"].values[0] - df["auc"]
        df["acc_drop"] = ref_row["acc"].values[0] - df["acc"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"v10_frame_supervised_cross_crf_{args.split}_trainedOn_{args.train_crf}"
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nSaved {csv_path}\nSaved {md_path}")


if __name__ == "__main__":
    main()
