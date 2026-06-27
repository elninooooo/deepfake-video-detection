"""Train v7 real-only TIM anomaly models.

v7a reconstructs TIM maps from real clips.
v7b predicts the final TIM map from previous TIM maps in real clips.

At validation time, real and fake clips are both scored by reconstruction or
prediction error; higher error means more fake-like.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from modelsgenerate.v7_real_tim import V7TIMConfig, build_v7_tim_model
from utils.metrics import evaluate
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser("Train v7 real-only TIM anomaly model")
    p.add_argument("--variant", choices=["v7a", "v7b"], required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_s2_all")
    p.add_argument("--train_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--val_crfs", nargs="+", default=None)
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--latent_channels", type=int, default=128)
    p.add_argument("--loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def keep_real_only(dataset):
    if isinstance(dataset, CelebDFClipDataset):
        dataset.records = [r for r in dataset.records if int(r["label"]) == 0]
        if not dataset.records:
            raise RuntimeError(f"No real clips left for split={dataset.crf_tag}.")
        return dataset
    raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")


def build_split_dataset(args, crfs, split: str, train: bool, real_only: bool):
    datasets = [
        CelebDFClipDataset(
            args.splits,
            args.face_cache,
            crf_tag=crf,
            split=split,
            n_frames=args.n_frames,
            train=train,
        )
        for crf in crfs
    ]
    if real_only:
        datasets = [keep_real_only(dataset) for dataset in datasets]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def collate_drop_meta(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return xs, ys, metas


def reconstruction_loss(pred, target, loss_name: str, reduction: str = "mean"):
    if loss_name == "l1":
        return F.l1_loss(pred, target, reduction=reduction)
    return F.mse_loss(pred, target, reduction=reduction)


def sample_errors(pred, target, loss_name: str) -> torch.Tensor:
    if loss_name == "l1":
        err = (pred - target).abs()
    else:
        err = (pred - target).pow(2)
    return err.flatten(1).mean(dim=1)


@torch.no_grad()
def run_validation(model, loader, device, loss_name: str):
    model.eval()
    scores, labels = [], []
    for x, y, _ in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        pred, target = model(x)
        scores.append(sample_errors(pred, target, loss_name).cpu().numpy())
        labels.append(y.numpy())
    if not scores:
        return None
    return evaluate(np.concatenate(labels), np.concatenate(scores))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    run_name = args.name or f"{args.variant}_tim_real_only"
    ckpt_dir = Path(args.checkpoints) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    val_crfs = args.val_crfs or args.train_crfs
    train_set = build_split_dataset(args, args.train_crfs, "train", train=True, real_only=True)
    val_set = build_split_dataset(args, val_crfs, "val", train=False, real_only=False)
    print(f"variant: {args.variant}")
    print(f"train crfs: {args.train_crfs}  val crfs: {val_crfs}")
    print(f"train real clips: {len(train_set)}  val clips: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_drop_meta,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_drop_meta,
    )

    model = build_v7_tim_model(
        V7TIMConfig(variant=args.variant, latent_channels=args.latent_channels)
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    best_auc = -1.0
    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", -1) + 1
        best_auc = ck.get("best_auc", -1.0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}, best_auc={best_auc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for x, _, _ in pbar:
            x = x.to(device, non_blocking=True)
            pred, target = model(x)
            loss = reconstruction_loss(pred, target, args.loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            loss_sum += float(loss.detach()) * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{loss_sum / max(1, n_seen):.6f}")

        train_loss = loss_sum / max(1, n_seen)
        report = run_validation(model, val_loader, device, args.loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        if report is None:
            print(f"[epoch {epoch + 1}] train_loss={train_loss:.6f} val empty")
            continue
        print(f"[epoch {epoch + 1}] train_loss={train_loss:.6f} lr={cur_lr:.2e} {report}")

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "lr": cur_lr,
            **report.to_dict(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_auc": best_auc,
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_dir / "last.pth")

        improved = (report.auc == report.auc) and report.auc > best_auc
        if improved:
            best_auc = report.auc
            torch.save(ckpt | {"best_auc": best_auc}, ckpt_dir / "best.pth")
            print(f"  -> new best val AUC = {best_auc:.4f}")

    print(f"\nDone. Best val AUC = {best_auc:.4f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
