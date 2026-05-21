"""argparse options shared by train.py and validate.py."""

import argparse


def add_common_args(p: argparse.ArgumentParser):
    # dataset
    p.add_argument("--splits", default="splits.json",
                   help="Path to splits.json produced by dataset/celeb_df_split.py")
    p.add_argument("--face_cache", default="face_cache",
                   help="Root of face_cache directory produced by dataset/preprocess_faces.py")
    p.add_argument("--train_crf", default="crf23",
                   help="Which crf{N} sub-folder under face_cache to train on")
    p.add_argument("--val_crf", default=None,
                   help="crf folder used for validation (defaults to --train_crf)")
    p.add_argument("--n_frames", type=int, default=16)

    # model
    p.add_argument("--variant", choices=["v1", "v2", "v3", "v4"], default="v1")
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--mask_radius_ratio", type=float, default=0.5)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=4)

    # training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--lr_patience", type=int, default=3,
                   help="Epochs with no val-AUC improvement before lr is decayed.")
    p.add_argument("--lr_factor", type=float, default=0.8)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda",
                   choices=["cuda", "cpu"], help="cuda will fall back to CPU if unavailable")
    p.add_argument("--use_balanced_sampler", action="store_true", default=True)
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--name", default=None, help="Run name; defaults to --variant")
    p.add_argument("--resume", default=None, help="Path to .pth to resume from")


def parse_train_args(argv=None):
    p = argparse.ArgumentParser("Train phase-transformer detector on Celeb-DF")
    add_common_args(p)
    return p.parse_args(argv)


def parse_validate_args(argv=None):
    p = argparse.ArgumentParser("Validate phase-transformer detector on Celeb-DF")
    add_common_args(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--test_crf", default=None,
                   help="crf folder to evaluate on (defaults to --train_crf)")
    return p.parse_args(argv)
