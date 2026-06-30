# Phase + Transformer + High-Freq Mask Deepfake Detector (Celeb-DF)

End-to-end implementation of the methodology described in
`phase+transformer+Mask+Methodology(baha).docx` and
`phase-transformer-mask-methodology-baha-buzzing-blossom.docx`.

It implements six variants in a single codebase:

| Variant | TIM | Phase | HF Mask |
| ------- | --- | ----- | ------- |
| V1 (Baseline-TIM)        | ✓ |   |   |
| V2 (Phase-only)          |   | ✓ |   |
| V3 (TIM + Phase)         | ✓ | ✓ |   |
| V4 (TIM + Phase + Mask, full) | ✓ | ✓ | ✓ |

Additional phase-first residual variants:

| Variant | Strategy | HF Mask |
| ------- | -------- | ------- |
| V5 (Phase-Residual) | Phase map, then temporal residual | |
| V6 (Phase-Residual + Mask) | Masked phase map, then temporal residual | yes |

## Repository layout

```
phase-transformer-detector/
├── data/
│   ├── celeb_df_split.py          identity-aware train/val/test split
│   ├── ffmpeg_recompress.py       re-encode at CRF 0/23/40 for cross-compression
│   ├── preprocess_faces.py        MTCNN offline face crops
│   └── video_clip_dataset.py      PyTorch Dataset over the cached crops
├── models/
│   ├── tim_extractor.py           |x_{t+1} - x_t| directional frame diff
│   ├── phase_branch.py            FFT → sin/cos(phi) → HF random mask
│   ├── spatial_backbone.py        slim ResNet (Bottleneck × 7)
│   ├── temporal_transformer.py    single-layer Transformer + learnable PE + CLS
│   ├── fusion_head.py             branch concat → Transformer → FC
│   └── full_model.py              assemble V1–V4 from flags
├── options/train_opts.py          argparse
├── utils/
│   ├── metrics.py                 Acc / P / R / F1 / AUC / EER
│   └── seed.py
├── train.py
├── validate.py
├── eval_cross_compression.py      3×3 CRF matrix
├── run_v1_baseline.py             one-click V1
├── run_v4_full.py                 one-click V4
├── tests/                         pytest smoke tests
└── requirements.txt
```

## PyCharm setup

1. **Open the project**: File → Open → select `phase-transformer-detector/`.
   Right-click the folder in the Project tab → *Mark Directory as → Sources Root*.
2. **Interpreter**: File → Settings → Project → Python Interpreter → Add → Virtualenv
   (Python ≥ 3.10).
3. **PyTorch with CUDA** (your card has CUDA, pick the right wheel):
   ```
   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
   ```
4. **Other dependencies**:
   ```
   pip install -r requirements.txt
   ```
5. **ffmpeg** must be on PATH (`ffmpeg -version` should print a version).
   Easiest on Windows: `winget install Gyan.FFmpeg` or `conda install -c conda-forge ffmpeg`.

## Data preparation

Assuming the Celeb-DF root is `../Celeb-DF/` relative to this folder.

```bash
# 1. Re-encode each video at CRF 0 / 23 / 40 (only needed for cross-compression).
python dataset/ffmpeg_recompress.py \
    --src ../Celeb-DF \
    --dst ../Celeb-DF-recompressed \
    --crfs 0 23 40 \
    --workers 4 \
    --copy_metadata

# 2. Build identity-aware split JSON.
python dataset/celeb_df_split.py --src ../Celeb-DF --out splits.json

# 3. Detect & cache faces (one face_cache/<crf> tree per CRF).
python dataset/preprocess_faces.py \
    --src ../Celeb-DF-recompressed \
    --splits splits.json \
    --out face_cache \
    --num_frames 32 \
    --image_size 224
```

To skip cross-compression entirely, point `--src` in step 3 at the original
`../Celeb-DF` — the output sub-folder will be `face_cache/crf_src/`, and you
should then pass `--train_crf crf_src` to all training/eval scripts.

## V1 temporal sampling ablation on original videos

The V1 TIM baseline can be compared under different temporal sampling
strategies while keeping the source video quality unchanged. These experiments
use the original Celeb-DF videos only, so every generated cache contains a
`crf_src/` sub-folder. Keep each sampling strategy in its own `--out`
directory; an existing `index.json` causes previously processed videos to be
skipped.

Sampling definitions:

| Experiment | Sampling method | Cached dataset path | V1 input |
| ---------- | --------------- | ------------------- | -------- |
| V1-uniform16 | 16 frames spread over the full video | `face_cache_uniform16/crf_src/` | 16 frames |
| V1-uniform32 | 32 frames spread over the full video | `face_cache_uniform32/crf_src/` | 32 frames |
| V1-clip16-S1 | Center clip of 16 consecutive source frames | `face_cache_clip16_s1/crf_src/` | 16 frames |
| V1-clip16-S2 | Center clip of 16 frames, source-frame stride 2 | `face_cache_clip16_s2/crf_src/` | 16 frames |
| V1-clip16-S4 | Center clip of 16 frames, source-frame stride 4 | `face_cache_clip16_s4/crf_src/` | 16 frames |

From this repository root, where the original dataset folder is `Celeb-real/`,
generate each face-crop cache as follows:

```bash
# Existing baseline: frames distributed across the whole video.
python data_pipeline/preprocess_faces.py --src Celeb-real --splits splits.json --out face_cache_uniform16 --num_frames 16 --sampling uniform --image_size 224

# More uniformly distributed frames; changes both sampled duration and V1 input length.
python data_pipeline/preprocess_faces.py --src Celeb-real --splits splits.json --out face_cache_uniform32 --num_frames 32 --sampling uniform --image_size 224

# Fixed-length TIM continuity ablations: only source-frame spacing changes.
python data_pipeline/preprocess_faces.py --src Celeb-real --splits splits.json --out face_cache_clip16_s1 --num_frames 16 --sampling clip --frame_stride 1 --image_size 224
python data_pipeline/preprocess_faces.py --src Celeb-real --splits splits.json --out face_cache_clip16_s2 --num_frames 16 --sampling clip --frame_stride 2 --image_size 224
python data_pipeline/preprocess_faces.py --src Celeb-real --splits splits.json --out face_cache_clip16_s4 --num_frames 16 --sampling clip --frame_stride 4 --image_size 224
```

Train one independently named V1 model for each cache:

```bash
python run_v1_baseline.py --face_cache face_cache_uniform16 --train_crf crf_src --val_crf crf_src --n_frames 16 --name v1_uniform16 --device cuda
python run_v1_baseline.py --face_cache face_cache_uniform32 --train_crf crf_src --val_crf crf_src --n_frames 32 --name v1_uniform32 --device cuda
python run_v1_baseline.py --face_cache face_cache_clip16_s1 --train_crf crf_src --val_crf crf_src --n_frames 16 --name v1_clip16_s1 --device cuda
python run_v1_baseline.py --face_cache face_cache_clip16_s2 --train_crf crf_src --val_crf crf_src --n_frames 16 --name v1_clip16_s2 --device cuda
python run_v1_baseline.py --face_cache face_cache_clip16_s4 --train_crf crf_src --val_crf crf_src --n_frames 16 --name v1_clip16_s4 --device cuda
```

The corresponding trained models are saved under:

```text
checkpoints/v1_uniform16/best.pth
checkpoints/v1_uniform32/best.pth
checkpoints/v1_clip16_s1/best.pth
checkpoints/v1_clip16_s2/best.pth
checkpoints/v1_clip16_s4/best.pth
```

For the continuity ablation, compare `V1-clip16-S1`, `S2`, and `S4` first:
all three use 16 input frames and differ only in temporal spacing. Compare
the uniform variants separately because `V1-uniform32` also increases input
sequence length and computational cost.

## Training

```bash
# V1 baseline TIM
python run_v1_baseline.py

# V4 full method
python run_v4_full.py

# Or fully custom (any flag is overridable):
python train.py --variant v3 --train_crf crf23 --epochs 30 --batch_size 4
```

Best checkpoints land in `checkpoints/<run_name>/best.pth`, training logs in
`checkpoints/<run_name>/train_log.jsonl`.

### Phase representation ablation

Keep the V2 phase-only architecture fixed and change only the phase
representation:

```bash
# V2a: raw sin/cos phase (legacy V2)
python train.py --variant v2 --phase_mode raw \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 --name v2a_phase_raw_s2_mixed

# V2b: log-magnitude confidence-weighted phase
python train.py --variant v2 --phase_mode weighted \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 --name v2b_phase_weighted_s2_mixed

# V2c: confidence-weighted phase restricted to normalized radius [0.15, 0.65]
python train.py --variant v2 --phase_mode mid_weighted \
    --phase_mid_low 0.15 --phase_mid_high 0.65 \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 --name v2c_phase_mid_weighted_s2_mixed
```

## Evaluation

```bash
# Single-CRF eval
python validate.py --ckpt checkpoints/v1_tim_baseline/best.pth \
                   --variant v1 --split test --test_crf crf23

# Full 3×3 cross-compression matrix
python eval_cross_compression.py \
    --ckpt checkpoints/v4_tim_phase_mask/best.pth \
    --variant v4 \
    --train_crf crf23 \
    --crfs crf0 crf23 crf40
```

### Mixed-CRF training

Train one detector with the original domain and the three recompressed
domains combined:

```bash
python train.py \
    --variant v1 \
    --face_cache face_cache \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 \
    --name v1_tim_mixed_crf \
    --device cuda

python eval_cross_compression.py \
    --ckpt checkpoints/v1_tim_mixed_crf/best.pth \
    --variant v1 \
    --face_cache face_cache \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --n_frames 16 \
    --out_dir results/cross_crf_uniform16
```

### Validation-threshold calibration

To separate score-distribution shift from ranking performance, calibrate the
decision threshold on the validation split and apply it unchanged to test:

```bash
python eval_threshold_calibration.py \
    --ckpt checkpoints/v1_tim_crf0/best.pth \
    --variant v1 \
    --face_cache face_cache \
    --train_crf crf0 \
    --crfs crf0 crf23 crf40 \
    --threshold_scope domain \
    --threshold_method f1 \
    --n_frames 16 \
    --out_dir results/threshold_calibration

python eval_threshold_calibration.py \
    --ckpt checkpoints/v1_tim_mixed_all/best.pth \
    --variant v1 \
    --face_cache face_cache \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --threshold_scope mixed \
    --threshold_method f1 \
    --n_frames 16 \
    --out_dir results/threshold_calibration
```

## Verifying installation

```bash
pytest -q tests/
```

All tests should pass on CPU and run in < 1 minute. They cover:

- TIM extractor output shape & non-negativity.
- Phase branch — sin² + cos² ≡ 1 invariant, mask changes output in train mode.
- Full model — all four variants forward-pass at small resolution.
- Metrics — perfect/random smoke checks.

### V8 TIM spectral relationship fusion

V8 keeps TIM as the main branch and adds a compact TIM spectral relationship
branch. The auxiliary branch computes low/mid/high frequency relationships and
confidence-weighted mid-band phase statistics from TIM maps, then fuses them
late with TIM features.

```bash
python train.py \
    --variant v8 \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 \
    --phase_confidence_quantile 0.95 \
    --phase_mid_low 0.10 \
    --phase_mid_high 0.70 \
    --batch_size 2 \
    --epochs 50 \
    --name v8_tim_spectral_relation_s2_mixed \
    --device cuda

python eval_cross_compression.py \
    --ckpt checkpoints/v8_tim_spectral_relation_s2_mixed/best.pth \
    --variant v8 \
    --face_cache face_cache_s2_all \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --n_frames 16 \
    --phase_confidence_quantile 0.95 \
    --phase_mid_low 0.10 \
    --phase_mid_high 0.70 \
    --batch_size 2 \
    --out_dir results/v8_tim_spectral_relation_cross
```

For calibrated-threshold evaluation:

```bash
python eval_threshold_calibration.py \
    --ckpt checkpoints/v8_tim_spectral_relation_s2_mixed/best.pth \
    --variant v8 \
    --face_cache face_cache_s2_all \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --threshold_scope mixed \
    --threshold_method balanced_acc \
    --n_frames 16 \
    --phase_confidence_quantile 0.95 \
    --phase_mid_low 0.10 \
    --phase_mid_high 0.70 \
    --batch_size 2 \
    --device cuda \
    --out_dir results/v8_tim_spectral_relation_threshold
```

### V10 residual spectral deep relationship

V10 trains a shared residual frequency-view encoder from scratch. It builds
temporal residuals, splits them into original/low/mid/high views, extracts
deep features with the shared encoder, and classifies the temporal sequence of
frequency-view relationships.

```bash
python train.py \
    --variant v10 \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 \
    --residual_mode gradient \
    --phase_mid_low 0.10 \
    --phase_mid_high 0.70 \
    --spectral_relation_dim 128 \
    --residual_encoder_dim 256 \
    --batch_size 2 \
    --epochs 50 \
    --name v10_residual_spectral_relation_s2_mixed \
    --device cuda

python eval_cross_compression.py \
    --ckpt checkpoints/v10_residual_spectral_relation_s2_mixed/best.pth \
    --variant v10 \
    --face_cache face_cache_s2_all \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --n_frames 16 \
    --residual_mode gradient \
    --phase_mid_low 0.10 \
    --phase_mid_high 0.70 \
    --batch_size 2 \
    --out_dir results/v10_residual_spectral_relation_cross
```

### V10 frame-level spatial overfit probe

This probe removes the temporal transformer and trains only the spatial
residual spectral relationship encoder on one adjacent frame pair. Use it to
check whether the custom encoder can overfit small balanced real/fake subsets
before spending time on full video-level training.

Start with 32 samples:

```bash
python train_v10_frame_probe.py \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 \
    --residual_mode gradient \
    --phase_mid_low 0.10 \
    --phase_mid_high 0.70 \
    --max_train_samples 32 \
    --max_val_samples 128 \
    --eval_all_pairs \
    --batch_size 8 \
    --epochs 300 \
    --lr 1e-3 \
    --name v10_frame_overfit32_gradient \
    --device cuda
```

Then repeat with `--max_train_samples 128` and `512`. For this sanity check,
the key metric is train AUC/ACC: if 32 samples cannot approach perfect training
scores, the spatial encoder itself is likely not learning the residual
frequency relationship cleanly.

### V10 frame-level supervised spatial classifier

This is the formal spatial-only experiment. It still removes the temporal
transformer, but unlike the overfit probe it trains on the full supervised
real/fake train split and evaluates whether adjacent-frame residual spectral
relationships generalize by themselves.

```bash
python train_v10_frame_supervised.py \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 \
    --residual_mode gradient \
    --phase_mid_low 0.10 \
    --phase_mid_high 0.70 \
    --eval_pair_mode all \
    --batch_size 8 \
    --epochs 80 \
    --lr 3e-4 \
    --name v10_frame_supervised_s2_mixed \
    --device cuda
```

Evaluate it across compression levels:

```bash
python eval_v10_frame_supervised.py \
    --ckpt checkpoints/v10_frame_supervised_s2_mixed/best.pth \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --split test \
    --eval_pair_mode all \
    --batch_size 8 \
    --device cuda \
    --out_dir results/v10_frame_supervised_cross
```

### V10 frame-level relation ablation

Relation ablation retrains the frame-level supervised model while changing how
the original/low/mid/high residual features are related. Keep score
aggregation fixed, usually `--score_agg mean`, so the comparison isolates the
deep relationship representation.

Available modes:

- `full`: original feature + all pairwise abs differences + cosine + L2.
- `feat_only`: original residual feature only.
- `abs_only`: pairwise absolute feature differences only.
- `cos_only`: pairwise cosine similarities only.
- `dist_only`: pairwise L2 distances only.
- `abs_cos`, `abs_dist`, `cos_dist`: two-relation combinations.
- `no_original`: all pairwise relationships without the original feature.
- `concat_views`: concatenate original/low/mid/high features without explicit relationships.

Train one mode:

```bash
python train_v10_frame_supervised.py \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 \
    --residual_mode gradient \
    --relation_mode abs_only \
    --phase_mid_low 0.10 \
    --phase_mid_high 0.70 \
    --eval_pair_mode all \
    --batch_size 8 \
    --epochs 80 \
    --lr 3e-4 \
    --name v10_frame_relation_abs_only_s2_mixed \
    --device cuda
```

Evaluate that mode:

```bash
python eval_v10_frame_supervised.py \
    --ckpt checkpoints/v10_frame_relation_abs_only_s2_mixed/best.pth \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --split test \
    --eval_pair_mode all \
    --score_agg mean \
    --batch_size 8 \
    --device cuda \
    --out_dir results/v10_frame_relation_ablation/abs_only
```

Run all relation modes on Windows PowerShell:

```powershell
.\scripts\run_v10_frame_relation_ablation.ps1 -Epochs 80 -BatchSize 8 -Device cuda
```

### V7 real-only TIM anomaly experiments

V7 uses only real clips for training and scores test clips by TIM
reconstruction or prediction error. Higher error is treated as more fake-like.

V7a reconstructs all TIM maps:

```bash
python train_v7_real_tim.py \
    --variant v7a \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 \
    --batch_size 2 \
    --epochs 30 \
    --name v7a_tim_recon_real_s2_mixed \
    --device cuda

python eval_v7_real_tim.py \
    --ckpt checkpoints/v7a_tim_recon_real_s2_mixed/best.pth \
    --variant v7a \
    --face_cache face_cache_s2_all \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --n_frames 16 \
    --batch_size 2 \
    --out_dir results/v7a_tim_recon_real_s2_mixed
```

V7b predicts the final TIM map from previous TIM maps:

```bash
python train_v7_real_tim.py \
    --variant v7b \
    --face_cache face_cache_s2_all \
    --train_crfs crf_src crf0 crf23 crf40 \
    --val_crfs crf_src crf0 crf23 crf40 \
    --n_frames 16 \
    --batch_size 2 \
    --epochs 30 \
    --name v7b_tim_predict_real_s2_mixed \
    --device cuda

python eval_v7_real_tim.py \
    --ckpt checkpoints/v7b_tim_predict_real_s2_mixed/best.pth \
    --variant v7b \
    --face_cache face_cache_s2_all \
    --train_crf mixed \
    --crfs crf_src crf0 crf23 crf40 \
    --include_mixed \
    --n_frames 16 \
    --batch_size 2 \
    --out_dir results/v7b_tim_predict_real_s2_mixed
```

## Mapping back to the methodology RQs

| Research question | Compare these variants |
| ----------------- | ---------------------- |
| RQ1 — can phase alone detect fakes? | V1 vs V2 |
| RQ2 — is the phase spectrum stable under compression? | run V2 cross-CRF, plot AUC drop |
| RQ3 — does HF masking improve cross-compression generalization? | V1 vs (V1 + mask), V3 vs V4 |
| RQ4 — does HF masking hurt or help phase detection? | V3 vs V4 |

To answer RQ3 fully, also train one extra variant V1+Mask (set
`--variant v1 --mask_ratio 0.5`) — it requires a tiny tweak to `full_model.py`
to expose the mask on the TIM branch; left as second-round work per the plan.
