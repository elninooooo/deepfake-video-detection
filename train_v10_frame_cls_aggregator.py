"""Train a frozen-frame CLS temporal aggregator for v10 frame features.

The frame-level residual spectral relation branch is loaded from a trained
frame checkpoint and frozen. Only a lightweight CLS Transformer aggregator and
video classifier are trained on the sequence of adjacent-pair features.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_v10_frame_supervised import (
    V10FrameClassifier,
    build_split_dataset,
    collate_drop_meta,
    labels_for_dataset,
    pick_device,
)
from utils.metrics import evaluate
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser("Train frozen-frame CLS temporal aggregator")
    p.add_argument("--frame_ckpt", required=True,
                   help="Trained frame-level checkpoint, preferably cos_only.")
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_s2_all")
    p.add_argument("--train_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--val_crfs", nargs="+", default=None)
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--name", default="v10_frame_cos_cls_aggregator_s2_mixed")
    return p.parse_args()


def frame_args_from_checkpoint(state, cli_args):
    ckpt_args = dict(state.get("args", {}) if isinstance(state, dict) else {})
    ckpt_args.setdefault("relation_mode", "full")
    ckpt_args["splits"] = cli_args.splits
    ckpt_args["face_cache"] = cli_args.face_cache
    ckpt_args["n_frames"] = cli_args.n_frames
    return SimpleNamespace(**ckpt_args)


def load_frozen_branch(frame_ckpt: str, cli_args, device):
    state = torch.load(frame_ckpt, map_location=device)
    frame_args = frame_args_from_checkpoint(state, cli_args)
    frame_model = V10FrameClassifier(frame_args).to(device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    frame_model.load_state_dict(sd)
    branch = frame_model.branch
    branch.eval()
    for param in branch.parameters():
        param.requires_grad_(False)
    return branch, frame_args


class CLSAggregator(nn.Module):
    def __init__(
        self,
        frozen_branch: nn.Module,
        feature_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        max_pairs: int,
    ):
        super().__init__()
        self.branch = frozen_branch
        self.input_proj = nn.Identity() if feature_dim == d_model else nn.Linear(feature_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_pairs + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def pair_features(self, x: torch.Tensor):
        with torch.no_grad():
            return self.branch(x).detach()

    def forward(self, x: torch.Tensor):
        feat = self.input_proj(self.pair_features(x))
        b, t, _ = feat.shape
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        tokens = tokens + self.pos_embed[:, : t + 1]
        out = self.encoder(tokens)
        return self.classifier(out[:, 0]).squeeze(-1)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    scores, labels = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return evaluate(np.concatenate(labels), np.concatenate(scores))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    ckpt_dir = Path(args.checkpoints) / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    val_crfs = args.val_crfs or args.train_crfs
    train_set = build_split_dataset(args, args.train_crfs, "train", True, args.max_train_samples)
    val_set = build_split_dataset(args, val_crfs, "val", False, args.max_val_samples)
    train_labels = labels_for_dataset(train_set)
    print(f"train crfs: {args.train_crfs}  val crfs: {val_crfs}")
    print(f"train clips: {len(train_set)}  val clips: {len(val_set)}")
    print(f"train labels: real={train_labels.count(0)} fake={train_labels.count(1)}")

    frozen_branch, frame_args = load_frozen_branch(args.frame_ckpt, args, device)
    feature_dim = int(frame_args.spectral_relation_dim)
    print(f"frozen frame relation mode: {getattr(frame_args, 'relation_mode', 'full')}")
    print(f"pair feature dim: {feature_dim}")

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

    model = CLSAggregator(
        frozen_branch=frozen_branch,
        feature_dim=feature_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_pairs=args.n_frames - 1,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable params: {n_trainable / 1e6:.3f} M  frozen params: {n_frozen / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        model.branch.eval()
        loss_sum, n_seen = 0.0, 0
        train_scores, train_targets = [], []
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
            train_targets.append(y.detach().cpu().numpy())
            pbar.set_postfix(loss=f"{loss_sum / max(1, n_seen):.4f}")

        train_loss = loss_sum / max(1, n_seen)
        train_report = evaluate(np.concatenate(train_targets), np.concatenate(train_scores))
        val_report = run_eval(model, val_loader, device)
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
            "frame_args": vars(frame_args),
        }
        torch.save(ckpt, ckpt_dir / "last.pth")
        if val_report.auc == val_report.auc and val_report.auc > best_auc:
            best_auc = val_report.auc
            torch.save(ckpt | {"best_auc": best_auc}, ckpt_dir / "best.pth")

    print(f"\nDone. Best val AUC = {best_auc:.4f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
