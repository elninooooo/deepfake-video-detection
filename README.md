# Residual Frequency-Band Relation Learning for Compressed Deepfake Detection

This repository contains a Celeb-DF deepfake video detection project focused on
compression-robust residual relation learning. The current main branch of the
project is **V10**, which moves away from direct phase-spectrum modeling and
uses supervised frequency-band relation learning on adjacent-frame residuals.

The older V1-V8 experiments are kept in the repository because they form the
preliminary study: they evaluate TIM baselines, phase representations,
frequency masking, real-only reconstruction/prediction, and early residual
relation variants. They are useful for ablation and thesis analysis, but they
are no longer the primary method described by this README.

## Current Best Method

The strongest evaluated configuration is:

```text
uniform/global 16-frame input
-> adjacent frame pairs
-> Sobel-gradient residuals
-> original / low / mid / high frequency-band residual views
-> shared residual encoder
-> cos_only frequency-band relation features
-> random8 supervised frame-pair training
-> frozen frame-level branch
-> lightweight CLS Transformer video aggregation
```

In the current multi-seed comparison, this setting gives the best overall
stability among the main evaluated V10 settings:

| Method | Test protocol | AUC | ACC |
| --- | --- | ---: | ---: |
| V1 TIM fair baseline | mixed CRF, 3 seeds | 0.8342 +/- 0.0173 | 0.7208 +/- 0.0505 |
| V10 random8 + mean aggregation | mixed CRF, 3 seeds | 0.8443 +/- 0.0114 | 0.7608 +/- 0.0159 |
| V10 random8 + frozen CLS aggregation | mixed CRF, 3 seeds | 0.8541 +/- 0.0125 | 0.7800 +/- 0.0066 |

The CLS model is used as the current best video-level scorer because it keeps
the trained spatial/frame-pair branch frozen and only learns a lightweight
temporal aggregation module over the 15 adjacent-pair features.

## Repository Layout

```text
data_pipeline/
  celeb_df_split.py              Identity-aware split creation
  ffmpeg_recompress.py           CRF recompression for cross-compression tests
  preprocess_faces.py            Offline face crop/cache generation
  video_clip_dataset.py          PyTorch video clip dataset

modelsgenerate/
  residual_spectral_relation.py  V10 residual frequency-band relation modules
  full_model.py                  Legacy V1-V6 model assembly
  temporal_transformer.py        Transformer/CLS components
  tim_extractor.py               Legacy TIM branch
  phase_branch.py                Legacy phase branch

train_v10_frame_supervised.py    V10 supervised frame-pair training
eval_v10_frame_supervised.py     V10 frame model evaluation with score pooling
train_v10_frame_cls_aggregator.py
                                  Frozen frame branch + CLS aggregator training
eval_v10_frame_cls_aggregator.py Frozen CLS video-level evaluation
eval_v10_frame_threshold_calibration.py
                                  Validation-threshold calibration for frame V10

train.py                         Legacy V1-V6 training entry
eval_cross_compression.py        Legacy cross-compression evaluation
scripts/                         Reproducible PowerShell runners
tests/                           Smoke/unit tests
```

## Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build if GPU training is needed. `ffmpeg`
must also be available on `PATH` for recompression and preprocessing.

## Data Preparation

The project expects face caches organized by CRF folders, for example:

```text
face_cache_uniform16_all/
  crf_src/index.json
  crf0/index.json
  crf23/index.json
  crf40/index.json
```

The current V10 experiments use 16-frame cached clips. The main mixed-compression
setting trains and evaluates on:

```text
crf_src crf0 crf23 crf40
```

Typical preprocessing steps are:

```powershell
.\.venv\Scripts\python.exe data_pipeline\celeb_df_split.py `
  --src Celeb-real `
  --out splits.json

.\.venv\Scripts\python.exe data_pipeline\ffmpeg_recompress.py `
  --src Celeb-real `
  --dst Celeb-real-recompressed `
  --crfs 0 23 40 `
  --workers 4

.\.venv\Scripts\python.exe data_pipeline\preprocess_faces.py `
  --src Celeb-real-recompressed `
  --splits splits.json `
  --out face_cache_uniform16_all `
  --num_frames 16 `
  --image_size 224
```

If an existing cache already contains `crf_src`, `crf0`, `crf23`, and `crf40`,
use that cache directly.

## Quick Start: Current V10 Model

### 1. Train the frame-level spatial relation branch

This trains the supervised V10 frame-pair classifier with the current best
relation and pair-sampling setting:

```powershell
.\.venv\Scripts\python.exe train_v10_frame_supervised.py `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crfs crf_src crf0 crf23 crf40 `
  --val_crfs crf_src crf0 crf23 crf40 `
  --n_frames 16 `
  --sampling_mode legacy `
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

### 2. Evaluate the frame-level model with mean pooling

```powershell
.\.venv\Scripts\python.exe eval_v10_frame_supervised.py `
  --ckpt checkpoints\v10_frame_cos_only_pair_random8_seed42_mixed\best.pth `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crf mixed `
  --crfs crf_src crf0 crf23 crf40 `
  --include_mixed `
  --split test `
  --n_frames 16 `
  --sampling_mode legacy `
  --eval_pair_mode all `
  --score_agg mean `
  --batch_size 8 `
  --num_workers 4 `
  --device cuda `
  --out_dir results\v10_pair_training\random8_seed42
```

### 3. Train the frozen CLS video aggregator

This freezes the trained frame-level branch and trains only the lightweight CLS
Transformer aggregator:

```powershell
.\.venv\Scripts\python.exe train_v10_frame_cls_aggregator.py `
  --frame_ckpt checkpoints\v10_frame_cos_only_pair_random8_seed42_mixed\best.pth `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crfs crf_src crf0 crf23 crf40 `
  --val_crfs crf_src crf0 crf23 crf40 `
  --n_frames 16 `
  --sampling_mode legacy `
  --d_model 128 `
  --n_heads 4 `
  --num_layers 1 `
  --dropout 0.1 `
  --batch_size 2 `
  --num_workers 4 `
  --epochs 50 `
  --lr 0.0003 `
  --seed 42 `
  --device cuda `
  --name v10_frame_cos_only_pair_random8_seed42_frozen_cls_mixed
```

### 4. Evaluate the frozen CLS model

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
  --sampling_mode legacy `
  --batch_size 8 `
  --num_workers 4 `
  --device cuda `
  --out_dir results\v10_frozen_cls\random8_seed42
```

## Batch Runners

Run the pair-training ablation:

```powershell
.\scripts\run_v10_pair_training_ablation.ps1 `
  -FaceCache face_cache_uniform16_all `
  -SamplingMode legacy `
  -Experiments random2,random4,random8,all_independent,all_weighted `
  -BatchSize 2 `
  -Epochs 80 `
  -Device cuda `
  -ResultsRoot results\v10_pair_training_ablation
```

Run frozen CLS aggregation after the corresponding frame checkpoints exist:

The script below expects frame-level checkpoints named like
`checkpoints\v10_frame_cos_only_pair_random8_seed42_mixed\best.pth`. Create
those checkpoints by repeating the single-model training command above with
different `--seed` and `--name` values, or by using your own seed loop.

```powershell
.\scripts\run_v10_frozen_cls_ablation.ps1 `
  -FaceCache face_cache_uniform16_all `
  -SamplingMode legacy `
  -Experiments random8 `
  -Seeds 42,43,44 `
  -BatchSize 2 `
  -EvalBatchSize 8 `
  -Epochs 50 `
  -Device cuda `
  -ResultsRoot results\v10_frozen_cls_multiseed
```

Run the fair V1 TIM baseline:

```powershell
.\scripts\run_v1_fair_multiseed.ps1 `
  -FaceCache face_cache_s2_all `
  -Seeds 42,43,44 `
  -BatchSize 2 `
  -Epochs 50 `
  -Device cuda `
  -ResultsRoot results\v1_tim_s2_mixed_fair_multiseed
```

## V10 Ablation Axes

The main V10 ablations are organized around these questions:

| Axis | Options | Purpose |
| --- | --- | --- |
| Relation mode | `full`, `feat_only`, `abs_only`, `cos_only`, `dist_only`, `abs_cos`, `abs_dist`, `cos_dist`, `no_original`, `concat_views` | Tests which frequency-band relation is useful |
| Pair training mode | `random_k`, `all_independent`, `random_k_bag`, `all_weighted` | Tests how many adjacent pairs should supervise the spatial branch |
| Pair count | `random1`, `random2`, `random4`, `random8` | Tests whether more independent frame pairs stabilize training |
| Score aggregation | `mean`, `max`, frozen `CLS` | Tests how pair-level evidence becomes a video-level score |
| Sampling protocol | `s2`, `local16`, `global16`, `4x4` style caches/settings | Tests whether short-term or long-term frame spacing changes residual evidence |

Current experimental evidence supports `cos_only` as the most reliable relation
mode and `random8` as a stronger pair-training setting than using only one
random adjacent pair.

## Threshold Calibration

Threshold calibration is an evaluation strategy, not a separate model. AUC is
threshold-independent, while ACC can change when the decision threshold is
selected from validation data.

For frame-level V10 models:

```powershell
.\.venv\Scripts\python.exe eval_v10_frame_threshold_calibration.py `
  --ckpt checkpoints\v10_frame_cos_only_pair_random8_seed42_mixed\best.pth `
  --splits splits.json `
  --face_cache face_cache_uniform16_all `
  --train_crf mixed `
  --crfs crf_src crf0 crf23 crf40 `
  --include_mixed `
  --n_frames 16 `
  --sampling_mode legacy `
  --eval_pair_mode all `
  --score_agg mean `
  --threshold_scope mixed `
  --threshold_method balanced_acc `
  --batch_size 8 `
  --num_workers 4 `
  --device cuda `
  --out_dir results\v10_frame_threshold_calibration
```

Use fixed-threshold and calibrated-threshold results together in reports:
fixed threshold shows raw score calibration, while validation calibration shows
the best deployable threshold chosen without test labels.

## Legacy Variants

The earlier model variants remain available for comparison:

| Variant | Main idea | Status |
| --- | --- | --- |
| V1 | TIM baseline on RGB temporal residuals | Fair baseline |
| V2/V3 | Phase and TIM-phase combinations | Preliminary study |
| V4-V6 | Phase residual and high-frequency masking | Preliminary study |
| V7 | Real-only reconstruction/prediction attempts | Negative/diagnostic study |
| V8 | Frequency-band relation with temporal modeling | Precursor to V10 |
| V10 | Supervised residual frequency-band relation learning | Current main method |

The phase-related experiments are important negative evidence: phase maps can
encode image structure, but in this project they were sensitive to shift,
compression, and unstable training behavior. The current V10 method therefore
uses frequency-band residual relation learning instead of direct phase-map
classification.

## Testing

Run smoke tests with:

```powershell
pytest -q tests
```

## Outputs

Training checkpoints are saved under `checkpoints/`. Evaluation reports are
saved under `results/`. Large generated artifacts should stay local unless a
specific result table, summary CSV/JSON, or figure is needed for documentation.
