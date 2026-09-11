# Reproduction Guide

This document lists the main commands needed to reproduce the GRFR experiments.
All commands assume Windows PowerShell and the repository root as the working
directory.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build if GPU training is required. `ffmpeg`
must be available on `PATH` for recompression.

## Face Cache

Raw videos are not redistributed. After obtaining Celeb-DF-v1, create fixed
splits and face caches:

```powershell
.\.venv\Scripts\python.exe data_pipeline\celeb_df_split.py `
  --src Celeb-real `
  --out splits.json
```

```powershell
.\.venv\Scripts\python.exe data_pipeline\ffmpeg_recompress.py `
  --src Celeb-real `
  --dst Celeb-DF-recompressed `
  --crfs 0 23 40 `
  --workers 4
```

```powershell
.\.venv\Scripts\python.exe data_pipeline\preprocess_faces.py `
  --src Celeb-DF-recompressed `
  --splits splits.json `
  --out face_cache_uniform16_all `
  --num_frames 16 `
  --sampling uniform `
  --image_size 224 `
  --device cuda
```

The final experiments expect:

```text
face_cache_uniform16_all/
  crf_src/
  crf0/
  crf23/
  crf40/
```

## Train GRFR

Stage 1 trains the frame-level residual frequency-band relation model:

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
  --weight_decay 0.00001 `
  --seed 42 `
  --device cuda `
  --name v10_frame_cos_only_pair_random8_seed42_mixed
```

Stage 2 freezes the frame-level branch and trains the CLS aggregator:

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
  --weight_decay 0.0001 `
  --seed 42 `
  --device cuda `
  --name v10_frame_cos_only_pair_random8_seed42_frozen_cls_mixed
```

## Evaluate GRFR

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

## Baselines

All baselines use the same source split, mixed-compression training set, and
global 16-frame face sampling protocol as GRFR. This keeps the comparison
focused on the model design rather than data or sampling differences.

### TIM Baseline

```powershell
.\.venv\Scripts\python.exe train.py `
  --variant v1 `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crfs crf_src crf0 crf23 crf40 `
  --val_crfs crf_src crf0 crf23 crf40 `
  --n_frames 16 `
  --sampling_mode global16 `
  --batch_size 2 `
  --num_workers 4 `
  --epochs 50 `
  --lr 0.0001 `
  --seed 42 `
  --device cuda `
  --name v1_tim_global16_mixed_fair_seed42
```

```powershell
.\.venv\Scripts\python.exe eval_cross_compression.py `
  --ckpt checkpoints\v1_tim_global16_mixed_fair_seed42\best.pth `
  --variant v1 `
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
  --out_dir results\v1_tim_global16_mixed_fair_seed42
```

### FTCN Baseline

The FTCN baseline follows the temporal-convolution idea of using spatial
kernel size 1 in the video convolution branch and a lightweight transformer for
video-level temporal aggregation.

```powershell
.\.venv\Scripts\python.exe train.py `
  --variant ftcn `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crfs crf_src crf0 crf23 crf40 `
  --val_crfs crf_src crf0 crf23 crf40 `
  --n_frames 16 `
  --sampling_mode global16 `
  --batch_size 2 `
  --num_workers 4 `
  --epochs 50 `
  --lr 0.0001 `
  --seed 42 `
  --device cuda `
  --name baseline_ftcn_global16_mixed_seed42
```

```powershell
.\.venv\Scripts\python.exe eval_cross_compression.py `
  --ckpt checkpoints\baseline_ftcn_global16_mixed_seed42\best.pth `
  --variant ftcn `
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
  --out_dir results\baseline_ftcn_global16_mixed_seed42
```

### FreqNet Baseline

The FreqNet baseline is implemented as an image-level frequency-aware detector.
For video-level evaluation, each sampled frame in the global16 clip is scored
by the same FreqNet branch and frame logits are averaged into one clip logit.

```powershell
.\.venv\Scripts\python.exe train.py `
  --variant freqnet `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crfs crf_src crf0 crf23 crf40 `
  --val_crfs crf_src crf0 crf23 crf40 `
  --n_frames 16 `
  --sampling_mode global16 `
  --batch_size 2 `
  --num_workers 4 `
  --epochs 50 `
  --lr 0.0001 `
  --seed 42 `
  --device cuda `
  --name baseline_freqnet_global16_mixed_seed42
```

```powershell
.\.venv\Scripts\python.exe eval_cross_compression.py `
  --ckpt checkpoints\baseline_freqnet_global16_mixed_seed42\best.pth `
  --variant freqnet `
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
  --out_dir results\baseline_freqnet_global16_mixed_seed42
```
