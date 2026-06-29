"""Frame-level v10 residual spectral relation probe.

This script removes the temporal transformer from v10 and trains only the
spatial residual spectral relation path on one adjacent frame pair at a time.
It is meant for sanity checks:

- Can the trainable residual spectral encoder overfit 32/128/512 samples?
- Does frame/pair-level residual spectral relation carry real/fake signal?
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from modelsgenerate.residual_spectral_relation import ResidualSpectralRelationBranch
from utils.metrics import evaluate
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser("Train frame-level v10 residual spectral relation probe")
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_s2_all")
    p.add_argument("--train_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--val_crfs", nargs="+", default=None)
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--residual_mode", choices=["abs", "signed", "gradient"], default="gradient")
    p.add_argument("--phase_mid_low", type=float, default=0.10)
    p.add_argument("--phase_mid_high", type=float, default=0.70)
    p.add_argument("--spectral_relation_dim", type=int, default=128)
    p.add_argument("--residual_encoder_dim", type=int, default=256)
    p.add_argument("--max_train_samples", type=int, default=128,
                   help="Balanced train subset size. Use 0 for all train clips.")
    p.add_argument("--max_val_samples", type=int, default=0,
                   help="Balanced val subset size. Use 0 for all val clips.")
    p.add_argument("--eval_all_pairs", action="store_true",
                   help="Average scores over all adjacent frame pairs at validation.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--name", default=None)
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def collate_drop_meta(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


def labels_for_dataset(dataset: Dataset):
    if isinstance(dataset, CelebDFClipDataset):
        return [int(r["label"]) for r in dataset.records]
    if isinstance(dataset, ConcatDataset):
        labels = []
        for child in dataset.datasets:
            labels.extend(labels_for_dataset(child))
        return labels
    if isinstance(dataset, Subset):
        base = labels_for_dataset(dataset.dataset)
        return [base[i] for i in dataset.indices]
    raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")


def balanced_subset(dataset: Dataset, max_items: int, seed: int):
    if max_items <= 0 or len(dataset) <= max_items:
        return dataset
    labels = labels_for_dataset(dataset)
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)
    per_label = max_items // 2
    chosen = []
    for label in (0, 1):
        ids = by_label[label]
        rng.shuffle(ids)
        chosen.extend(ids[: min(per_label, len(ids))])
    if len(chosen) < max_items:
        chosen_set = set(chosen)
        rest = [i for i in range(len(labels)) if i not in chosen_set]
        rng.shuffle(rest)
        chosen.extend(rest[: max_items - len(chosen)])
    rng.shuffle(chosen)
    return Subset(dataset, chosen)


def build_split_dataset(args, crfs, split: str, train: bool, max_items: int):
    datasets = [
        CelebDFClipDataset(
            args.splits,
            args.face_cache,
            crf_tag=crf,
            split=split,
            n_frames=args.n_frames,
            train=train,
            horiz_flip=train,
        )
        for crf in crfs
    ]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return balanced_subset(dataset, max_items, args.seed)


def select_adjacent_pair(x: torch.Tensor, train: bool):
    b, n = x.shape[:2]
    if n < 2:
        raise ValueError("Need at least two frames to form a residual pair.")
    if train:
        idx = torch.randint(0, n - 1, (b,), device=x.device)
    else:
        idx = torch.full((b,), (n - 2) // 2, device=x.device, dtype=torch.long)
    batch = torch.arange(b, device=x.device)
    return torch.stack([x[batch, idx], x[batch, idx + 1]], dim=1)


class V10FrameClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.branch = ResidualSpectralRelationBranch(
            out_dim=args.spectral_relation_dim,
            encoder_dim=args.residual_encoder_dim,
            mid_low=args.phase_mid_low,
            mid_high=args.phase_mid_high,
            residual_mode=args.residual_mode,
            dropout=0.1,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(args.spectral_relation_dim),
            nn.Linear(args.spectral_relation_dim, args.spectral_relation_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(args.spectral_relation_dim, 1),
        )

    def forward_pair(self, x_pair: torch.Tensor):
        feat = self.branch(x_pair).squeeze(1)
        return self.classifier(feat).squeeze(-1)

    def forward(self, x: torch.Tensor):
        return self.forward_pair(select_adjacent_pair(x, self.training))


@torch.no_grad()
def run_eval(model, loader, device, eval_all_pairs: bool):
    model.eval()
    scores, labels = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        if eval_all_pairs:
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
    return evaluate(np.concatenate(labels), np.concatenate(scores))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    run_name = args.name or f"v10_frame_{args.residual_mode}_{args.max_train_samples}"
    ckpt_dir = Path(args.checkpoints) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    val_crfs = args.val_crfs or args.train_crfs
    train_set = build_split_dataset(args, args.train_crfs, "train", True, args.max_train_samples)
    val_set = build_split_dataset(args, val_crfs, "val", False, args.max_val_samples)
    print(f"train crfs: {args.train_crfs}  val crfs: {val_crfs}")
    print(f"train clips/pairs sampled per epoch: {len(train_set)}  val clips: {len(val_set)}")
    print(f"train labels: real={labels_for_dataset(train_set).count(0)} fake={labels_for_dataset(train_set).count(1)}")

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

    model = V10FrameClassifier(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e6:.2f} M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        loss_sum, n_seen = 0.0, 0
        train_scores, train_labels = [], []
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            loss_sum += float(loss.detach()) * bs
            n_seen += bs
            train_scores.append(torch.sigmoid(logits.detach()).cpu().numpy())
            train_labels.append(y.detach().cpu().numpy())
            pbar.set_postfix(loss=f"{loss_sum / max(1, n_seen):.4f}")

        train_loss = loss_sum / max(1, n_seen)
        train_report = evaluate(np.concatenate(train_labels), np.concatenate(train_scores))
        val_report = run_eval(model, val_loader, device, args.eval_all_pairs)
        print(
            f"[epoch {epoch + 1}] loss={train_loss:.4f} "
            f"train_auc={train_report.auc:.4f} train_acc={train_report.acc:.4f} "
            f"val_auc={val_report.auc:.4f} val_acc={val_report.acc:.4f}"
        )
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_report.to_dict().items()},
            **{f"val_{k}": v for k, v in val_report.to_dict().items()},
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
        if val_report.auc == val_report.auc and val_report.auc > best_auc:
            best_auc = val_report.auc
            torch.save(ckpt | {"best_auc": best_auc}, ckpt_dir / "best.pth")

    print(f"\nDone. Best val AUC = {best_auc:.4f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
