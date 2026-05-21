"""Evaluate a saved checkpoint on a chosen split / CRF level.

Example
-------
python validate.py --ckpt checkpoints/v1/best.pth --split test --test_crf crf23
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_pipeline.video_clip_dataset import CelebDFClipDataset
from modelsgenerate.full_model import build_model_from_args
from options.train_opts import parse_validate_args
from utils.metrics import evaluate


def collate_with_meta(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return xs, ys, metas


@torch.no_grad()
def evaluate_ckpt(args, ckpt_path: Path, crf_tag: str, split: str):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    ds = CelebDFClipDataset(args.splits, args.face_cache,
                            crf_tag=crf_tag, split=split,
                            n_frames=args.n_frames, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate_with_meta)

    model = build_model_from_args(args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    model.eval()

    scores, labels, vids = [], [], []
    for x, y, m in tqdm(loader, desc=f"{split}/{crf_tag}", leave=False):
        x = x.to(device, non_blocking=True)
        logit = model(x).squeeze(-1)
        s = torch.sigmoid(logit).cpu().numpy()
        scores.append(s)
        labels.append(y.numpy())
        for mm, sc in zip(m, s):
            vids.append({"video": mm["video"], "crf": mm["crf"], "score": float(sc)})

    if not scores:
        return None, []
    return evaluate(np.concatenate(labels), np.concatenate(scores)), vids


def main():
    args = parse_validate_args()
    crf_tag = args.test_crf or args.train_crf
    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    report, per_video = evaluate_ckpt(args, ckpt, crf_tag, args.split)
    print(report)
    out = Path("results") / f"{ckpt.parent.name}_{args.split}_{crf_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report.to_dict(), "per_video": per_video},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
