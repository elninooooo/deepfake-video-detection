"""Train relation-wise temporal attention over frozen v10 cosine relations."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_v10_frame_supervised import build_split_dataset, collate_drop_meta, labels_for_dataset, pick_device
from utils.metrics import evaluate
from utils.seed import set_seed
from v10_relation_temporal_utils import RELATION_NAMES, load_frozen_frame_model, raw_cosine_relations


def parse_args():
    p = argparse.ArgumentParser("Train relation-wise attention over frozen v10 cosine relations")
    p.add_argument("--frame_ckpt", required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_s2_all")
    p.add_argument("--train_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--val_crfs", nargs="+", default=None)
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--relation_group", choices=["low_mid", "high_related", "all"], default="all")
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--name", default="v10_relationwise_attention_s2_mixed")
    return p.parse_args()


def group_indices(group):
    if group == "low_mid":
        return [0, 1, 3]
    if group == "high_related":
        return [2, 4, 5]
    return [0, 1, 2, 3, 4, 5]


class RelationWiseAttention(nn.Module):
    def __init__(self, frozen_branch, indices, max_pairs, hidden_dim, dropout):
        super().__init__()
        self.branch = frozen_branch
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        n_rel = len(indices)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.relation_embed = nn.Parameter(torch.zeros(1, 1, n_rel, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_pairs, 1, hidden_dim))
        self.temporal_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.relation_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.trunc_normal_(self.relation_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def relation_sequence(self, x):
        with torch.no_grad():
            rel = raw_cosine_relations(self.branch, x)
        return rel.index_select(dim=2, index=self.indices)

    def forward(self, x, return_attention=False):
        rel = self.relation_sequence(x)
        b, t, r = rel.shape
        tokens = self.value_proj(rel.unsqueeze(-1))
        tokens = tokens + self.pos_embed[:, :t] + self.relation_embed
        temporal_logits = self.temporal_attn(tokens).squeeze(-1)
        temporal_weights = torch.softmax(temporal_logits, dim=1)
        relation_summary = (temporal_weights.unsqueeze(-1) * tokens).sum(dim=1)
        relation_logits = self.relation_attn(relation_summary).squeeze(-1)
        relation_weights = torch.softmax(relation_logits, dim=1)
        video_feat = (relation_weights.unsqueeze(-1) * relation_summary).sum(dim=1)
        logit = self.classifier(video_feat).squeeze(-1)
        if return_attention:
            return logit, temporal_weights, relation_weights
        return logit


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

    frame_model, frame_args = load_frozen_frame_model(args.frame_ckpt, args, device)
    indices = group_indices(args.relation_group)
    print(f"frozen frame relation mode: {getattr(frame_args, 'relation_mode', 'unknown')}")
    print(f"relation group: {args.relation_group} -> {[RELATION_NAMES[i] for i in indices]}")

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

    model = RelationWiseAttention(
        frozen_branch=frame_model.branch,
        indices=indices,
        max_pairs=args.n_frames - 1,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
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
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_report.to_dict().items()},
                **{f"val_{k}": v for k, v in val_report.to_dict().items()},
            }) + "\n")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_auc": best_auc,
            "args": vars(args),
            "frame_args": vars(frame_args),
            "relation_indices": indices,
            "relation_names": [RELATION_NAMES[i] for i in indices],
        }
        torch.save(ckpt, ckpt_dir / "last.pth")
        if val_report.auc == val_report.auc and val_report.auc > best_auc:
            best_auc = val_report.auc
            torch.save(ckpt | {"best_auc": best_auc}, ckpt_dir / "best.pth")

    print(f"\nDone. Best val AUC = {best_auc:.4f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
