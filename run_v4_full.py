"""One-click launcher for V4 (TIM + Phase + HF mask, full method)."""

import sys

from train import main as train_main


def main():
    overrides = [
        "--variant", "v4",
        "--train_crf", "crf23",
        "--val_crf", "crf23",
        "--n_frames", "16",
        "--batch_size", "4",
        "--epochs", "50",
        "--lr", "1e-4",
        "--mask_ratio", "0.5",
        "--mask_radius_ratio", "0.5",
        "--name", "v4_tim_phase_mask",
    ]
    sys.argv = [sys.argv[0]] + overrides + sys.argv[1:]
    train_main()


if __name__ == "__main__":
    main()
