"""Train lightweight 1D-logit probes for GRFR relation types.

The frozen GRFR frame-level encoder is used only to extract relation vectors.
For each relation type, a small classifier maps six band-pair relation values
to a single logit. This makes cos/abs/L2 comparable under the same low-capacity
decision head.
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from probe_v10_relation_distribution import relation_tensors
from train import collate_drop_meta
from utils.metrics import evaluate
from utils.seed import set_seed
from v10_relation_temporal_utils import load_frozen_frame_model


RELATION_TYPES = ("cos", "abs", "l2")


def parse_args():
    p = argparse.ArgumentParser("Train 1D-logit probes over GRFR relation vectors")
    p.add_argument("--frame_ckpt", required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--face_cache", default="face_cache_uniform16_all")
    p.add_argument("--train_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--val_crfs", nargs="+", default=None)
    p.add_argument("--test_crfs", nargs="+", default=["crf_src", "crf0", "crf23", "crf40"])
    p.add_argument("--include_mixed", action="store_true")
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--sampling_mode", choices=sorted(CelebDFClipDataset.SAMPLING_MODES),
                   default="global16")
    p.add_argument("--pair_agg", choices=["mean", "max"], default="mean")
    p.add_argument("--probe_type", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--probe_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_train_samples_per_class", type=int, default=0)
    p.add_argument("--max_val_samples_per_class", type=int, default=0)
    p.add_argument("--max_test_samples_per_class", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--name", default="v10_relation_1d_probe")
    p.add_argument("--out_dir", default="results/v10_relation_1d_probe")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def balanced_subset(dataset, max_per_class: int, seed: int):
    if max_per_class <= 0:
        return dataset
    by_label = {0: [], 1: []}
    for idx, rec in enumerate(dataset.records):
        by_label[int(rec["label"])].append(idx)
    rng = random.Random(seed)
    chosen = []
    for label in (0, 1):
        ids = by_label[label][:]
        rng.shuffle(ids)
        chosen.extend(ids[: min(max_per_class, len(ids))])
    chosen.sort()
    from torch.utils.data import Subset

    return Subset(dataset, chosen)


def build_dataset(args, crfs, split: str, max_per_class: int):
    datasets = []
    for crf in crfs:
        ds = CelebDFClipDataset(
            args.splits,
            args.face_cache,
            crf_tag=crf,
            split=split,
            n_frames=args.n_frames,
            train=False,
            horiz_flip=False,
            sampling_mode=args.sampling_mode,
        )
        datasets.append(balanced_subset(ds, max_per_class, args.seed))
    if len(datasets) == 1:
        return datasets[0]
    from torch.utils.data import ConcatDataset

    return ConcatDataset(datasets)


@torch.no_grad()
def extract_relation_features(args, frame_model, crfs, split: str, max_per_class: int, device):
    dataset = build_dataset(args, crfs, split, max_per_class)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_drop_meta,
    )
    feats = {name: [] for name in RELATION_TYPES}
    labels = []
    for x, y, _ in tqdm(loader, desc=f"extract/{split}", leave=False):
        x = x.to(device, non_blocking=True)
        rel = relation_tensors(frame_model.branch, x)
        for name in RELATION_TYPES:
            value = rel[name].mean(dim=1) if args.pair_agg == "mean" else rel[name].max(dim=1).values
            feats[name].append(value.cpu())
        labels.append(y.float())
    feats = {name: torch.cat(chunks, dim=0).float() for name, chunks in feats.items()}
    labels = torch.cat(labels, dim=0).float()
    return feats, labels


class RelationProbe(nn.Module):
    def __init__(self, probe_type: str, hidden_dim: int):
        super().__init__()
        if probe_type == "linear":
            self.net = nn.Linear(6, 1)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cohen_d_from_scores(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    real = scores[labels == 0]
    fake = scores[labels == 1]
    if real.size == 0 or fake.size == 0:
        return float("nan")
    std_real = real.std(ddof=1) if real.size > 1 else 0.0
    std_fake = fake.std(ddof=1) if fake.size > 1 else 0.0
    denom = max(1, real.size + fake.size - 2)
    pooled = math.sqrt(((real.size - 1) * std_real ** 2 + (fake.size - 1) * std_fake ** 2) / denom)
    return float((fake.mean() - real.mean()) / (pooled + 1e-12))


def train_one_probe(args, relation_type, train_x, train_y, val_x, val_y, device):
    model = RelationProbe(args.probe_type, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.probe_batch_size,
        shuffle=True,
        drop_last=False,
    )
    ckpt_dir = Path(args.checkpoints) / f"{args.name}_{relation_type}_{args.probe_type}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * yb.numel()
            n_seen += yb.numel()

        rep = evaluate_probe(model, val_x, val_y, device)
        if rep["auc"] == rep["auc"] and rep["auc"] > best_auc:
            best_auc = rep["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
            print(
                f"[{relation_type} epoch {epoch + 1}/{args.epochs}] "
                f"loss={loss_sum / max(1, n_seen):.4f} val_auc={rep['auc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "relation_type": relation_type,
            "best_val_auc": best_auc,
        },
        ckpt_dir / "best.pth",
    )
    return model, ckpt_dir


@torch.no_grad()
def evaluate_probe(model, x, y, device):
    model.eval()
    logits = model(x.to(device)).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = y.numpy()
    rep = evaluate(labels, probs).to_dict()
    rep["logit_cohen_d"] = cohen_d_from_scores(labels, logits)
    rep["prob_cohen_d"] = cohen_d_from_scores(labels, probs)
    return rep


def write_rows(path: Path, rows):
    fields = [
        "relation_type",
        "eval_set",
        "crf",
        "probe_type",
        "pair_agg",
        "acc",
        "precision",
        "recall",
        "f1",
        "auc",
        "eer",
        "threshold_eer",
        "n_samples",
        "logit_cohen_d",
        "prob_cohen_d",
        "ckpt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    val_crfs = args.val_crfs or args.train_crfs
    frame_model, _ = load_frozen_frame_model(args.frame_ckpt, args, device)

    train_feats, train_y = extract_relation_features(
        args, frame_model, args.train_crfs, "train", args.max_train_samples_per_class, device
    )
    val_feats, val_y = extract_relation_features(
        args, frame_model, val_crfs, "val", args.max_val_samples_per_class, device
    )

    rows = []
    trained = {}
    for relation_type in RELATION_TYPES:
        model, ckpt_dir = train_one_probe(
            args,
            relation_type,
            train_feats[relation_type],
            train_y,
            val_feats[relation_type],
            val_y,
            device,
        )
        trained[relation_type] = (model, str(ckpt_dir / "best.pth"))
        val_rep = evaluate_probe(model, val_feats[relation_type], val_y, device)
        rows.append(
            {
                "relation_type": relation_type,
                "eval_set": "val",
                "crf": "mixed_val",
                "probe_type": args.probe_type,
                "pair_agg": args.pair_agg,
                **val_rep,
                "ckpt": str(ckpt_dir / "best.pth"),
            }
        )

    test_cache = {}
    for crf in args.test_crfs:
        test_cache[crf] = extract_relation_features(
            args, frame_model, [crf], "test", args.max_test_samples_per_class, device
        )
    if args.include_mixed:
        test_cache["mixed"] = extract_relation_features(
            args, frame_model, args.test_crfs, "test", args.max_test_samples_per_class, device
        )

    for relation_type, (model, ckpt_path) in trained.items():
        for crf, (test_feats, test_y) in test_cache.items():
            rep = evaluate_probe(model, test_feats[relation_type], test_y, device)
            rows.append(
                {
                    "relation_type": relation_type,
                    "eval_set": "test",
                    "crf": crf,
                    "probe_type": args.probe_type,
                    "pair_agg": args.pair_agg,
                    **rep,
                    "ckpt": ckpt_path,
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name}_{args.probe_type}_{args.pair_agg}.csv"
    write_rows(csv_path, rows)
    md_path = out_dir / f"{args.name}_{args.probe_type}_{args.pair_agg}.md"
    lines = ["# GRFR Relation 1D-Logit Probe", "", f"- frame checkpoint: `{args.frame_ckpt}`", ""]
    lines.append(Path(csv_path).read_text(encoding="utf-8").splitlines()[0])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {csv_path}")
    for row in rows:
        if row["eval_set"] == "test":
            print(
                f"{row['relation_type']:>3} {row['crf']:>7} "
                f"AUC={row['auc']:.4f} ACC={row['acc']:.4f} "
                f"logit_d={row['logit_cohen_d']:.4f}"
            )


if __name__ == "__main__":
    main()
