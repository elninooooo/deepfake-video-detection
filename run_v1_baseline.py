"""One-click launcher for V1 (Baseline TIM-only).

Just press Run in PyCharm. Edit the defaults below or pass CLI overrides.
"""

import sys

from train import main as train_main


def main():
    overrides = [
        "--variant", "v1",
        "--train_crf", "crf23",
        "--val_crf", "crf23",
        "--n_frames", "16",
        "--batch_size", "4",
        "--epochs", "50",
        "--lr", "1e-4",
        "--name", "v1_tim_baseline",
    ]
    # Allow the user to append extra CLI args from PyCharm "Parameters" field
    sys.argv = [sys.argv[0]] + overrides + sys.argv[1:]
    train_main()


if __name__ == "__main__":
    main()
