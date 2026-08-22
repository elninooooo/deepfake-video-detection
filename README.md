# GRFR: Gradient Residual Frequency-Band Relation Learning

This repository contains the implementation and experimental protocols for
**Gradient Residual Frequency-Band Relation (GRFR)**, a deepfake video detection
framework designed for compression-aware evaluation on Celeb-DF.

GRFR models adjacent-frame changes as gradient residuals, decomposes them into
frequency-band residual views, learns cosine-based relations among these views,
and aggregates frame-pair evidence with a lightweight CLS video-level
aggregator.

## Main Method

The final GRFR pipeline is:

```text
global 16-frame face clip
-> adjacent frame pairs
-> Sobel gradient maps
-> adjacent gradient residuals
-> original / low / mid / high residual frequency-band views
-> shared residual-view encoder
-> six cosine frequency-band relations
-> random8 frame-pair training
-> trainable CLS video-level aggregation with frozen frame-level branch
```

The TIM model is used as the main baseline. Earlier variants are retained for
reproducibility of the preliminary study and ablation experiments, but GRFR is
the final proposed method.

## Repository Layout

```text
data_pipeline/                  Dataset split, recompression, face extraction
modelsgenerate/                 GRFR, TIM, phase, and legacy model modules
scripts/                        Batch runners and visualization utilities
docs/                           Reproduction, experiment map, and result notes
results/selected/               Curated results used by the paper
tests/                          Unit and smoke tests

train_v10_frame_supervised.py   GRFR frame-level relation training
train_v10_frame_cls_aggregator.py
                                CLS video-level aggregator training
eval_v10_frame_cls_aggregator.py
                                GRFR video-level evaluation
eval_cross_compression.py       TIM and legacy cross-compression evaluation
eval_threshold_calibration.py   Threshold calibration utilities
train.py                        Legacy baseline training entry
```

## Data

This project uses Celeb-DF-v1 as the source-domain training, validation, and
testing dataset, and Celeb-DF-v2 as an external cross-dataset evaluation set.
Raw videos are not redistributed. After obtaining the original datasets, the
face-cache files can be reproduced with the preprocessing scripts in
`data_pipeline/`.

Public split files included in this repository:

```text
splits.json
splits_celebdfv2_300.json
```

## Quick Start

Create the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Train the GRFR frame-level branch:

```powershell
.\.venv\Scripts\python.exe train_v10_frame_supervised.py `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crfs crf_src crf0 crf23 crf40 `
  --val_crfs crf_src crf0 crf23 crf40 `
  --n_frames 16 `
  --sampling_mode global16 `
  --residual_mode gradient `
  --relation_mode cos_only `
  --train_pair_mode random_k `
  --train_pair_k 8 `
  --eval_pair_mode all `
  --batch_size 2 `
  --num_workers 4 `
  --epochs 80 `
  --lr 0.0003 `
  --seed 42 `
  --device cuda `
  --name v10_frame_cos_only_pair_random8_seed42_mixed
```

Train the CLS video-level aggregator:

```powershell
.\.venv\Scripts\python.exe train_v10_frame_cls_aggregator.py `
  --frame_ckpt checkpoints\v10_frame_cos_only_pair_random8_seed42_mixed\best.pth `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crfs crf_src crf0 crf23 crf40 `
  --val_crfs crf_src crf0 crf23 crf40 `
  --n_frames 16 `
  --sampling_mode global16 `
  --batch_size 2 `
  --num_workers 4 `
  --epochs 50 `
  --lr 0.0003 `
  --seed 42 `
  --device cuda `
  --name v10_frame_cos_only_pair_random8_seed42_frozen_cls_mixed
```

Evaluate the GRFR video-level model:

```powershell
.\.venv\Scripts\python.exe eval_v10_frame_cls_aggregator.py `
  --ckpt checkpoints\v10_frame_cos_only_pair_random8_seed42_frozen_cls_mixed\best.pth `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crf mixed `
  --crfs crf_src crf0 crf23 crf40 `
  --include_mixed `
  --split test `
  --n_frames 16 `
  --sampling_mode global16 `
  --batch_size 8 `
  --num_workers 4 `
  --device cuda `
  --out_dir results\v10_frozen_cls\random8_seed42
```

## Documentation

Detailed documentation is provided in:

```text
docs/reproduction.md
docs/experiment_map.md
docs/data_availability.md
docs/results_summary.md
scripts/README.md
```

## Availability

The repository includes the final GRFR implementation, TIM baseline code,
preprocessing scripts, public split files, preliminary-study utilities,
ablation scripts, and evaluation protocols. Raw datasets, derived face caches,
model checkpoints, and local manuscript drafts are not redistributed.
