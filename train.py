"""Train one variant (v1/v2/v3/v4) of the phase-transformer detector on Celeb-DF.

PyCharm usage
-------------
1. Mark `phase-transformer-detector/` as the project root.
2. Run config:  Script = train.py
   Working dir = phase-transformer-detector/
   Parameters  = --variant v1 --train_crf crf23
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

# Ensure we can run the script from anywhere
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset, make_class_balanced_sampler
from modelsgenerate.full_model import build_model_from_args
from options.train_opts import parse_train_args
from utils.metrics import evaluate
from utils.seed import set_seed


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def collate_drop_meta(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return xs, ys, metas


@torch.no_grad()
def run_validation(model, loader, device):
    model.eval()
    scores, labels = [], []
    for x, y, _ in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        logit = model(x).squeeze(-1)
        scores.append(torch.sigmoid(logit).detach().cpu().numpy())
        labels.append(y.numpy())
    if not scores:
        return None
    return evaluate(np.concatenate(labels), np.concatenate(scores))


def adjust_lr(optimizer, factor: float, min_lr: float) -> bool:
    """Multiply lr by `factor` for every param group. Returns False if hit min_lr."""
    new_lrs = []
    for g in optimizer.param_groups:
        new_lr = max(min_lr, g["lr"] * factor)
        g["lr"] = new_lr
        new_lrs.append(new_lr)
    return min(new_lrs) > min_lr


def build_split_dataset(args, crfs, split: str, train: bool):
    datasets = [
        CelebDFClipDataset(args.splits, args.face_cache, crf_tag=crf,
                           split=split, n_frames=args.n_frames, train=train)
        for crf in crfs
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def main():
    args = parse_train_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    run_name = args.name or args.variant
    ckpt_dir = Path(args.checkpoints) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    # ---- dataset ---------------------------------------------------------
    train_crfs = args.train_crfs or [args.train_crf]
    val_crfs = args.val_crfs or (train_crfs if args.train_crfs else [args.val_crf or args.train_crf])
    train_set = build_split_dataset(args, train_crfs, split="train", train=True)
    val_set = build_split_dataset(args, val_crfs, split="val", train=False)
    print(f"train crfs: {train_crfs}  val crfs: {val_crfs}")
    print(f"train clips: {len(train_set)}  val clips: {len(val_set)}")

    sampler = make_class_balanced_sampler(train_set) if args.use_balanced_sampler else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=sampler, shuffle=sampler is None,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_drop_meta, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_drop_meta)

    # ---- model --------------------------------------------------------
    model = build_model_from_args(args).to(device)
    print(model.cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.beta1, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_auc = -1.0
    plateau = 0
    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0) + 1
        best_auc = ck.get("best_auc", -1.0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}, best_auc={best_auc:.4f}")

    # ---- training loop ------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logit = model(x).squeeze(-1)
            loss = loss_fn(logit, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            loss_sum += float(loss) * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{loss_sum / max(1, n_seen):.4f}")

        report = run_validation(model, val_loader, device)
        if report is None:
            print(f"[epoch {epoch+1}] val empty; skipping")
            continue
        train_loss = loss_sum / max(1, n_seen)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1}] train_loss={train_loss:.4f}  lr={cur_lr:.2e}  {report}")

        log_entry = {"epoch": epoch + 1, "train_loss": train_loss, "lr": cur_lr,
                     **report.to_dict()}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # checkpoint
        last_ckpt = {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "best_auc": best_auc,
                     "args": vars(args)}
        torch.save(last_ckpt, ckpt_dir / "last.pth")

        improved = (report.auc != report.auc) is False and report.auc > best_auc
        if improved:
            best_auc = report.auc
            plateau = 0
            torch.save(last_ckpt | {"best_auc": best_auc}, ckpt_dir / "best.pth")
            print(f"  ↳ new best AUC = {best_auc:.4f} (saved to best.pth)")
        else:
            plateau += 1
            if plateau >= args.lr_patience:
                ok = adjust_lr(optimizer, args.lr_factor, args.min_lr)
                print(f"  ↳ plateau {plateau} → decayed lr to {optimizer.param_groups[0]['lr']:.2e}")
                plateau = 0
                if not ok:
                    print("  ↳ reached min_lr; stopping.")
                    break
            if plateau >= args.early_stop_patience:
                print("  ↳ early stop.")
                break

    print(f"\nDone. Best val AUC = {best_auc:.4f}.  Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
