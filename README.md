# Phase + Transformer + High-Freq Mask Deepfake Detector (Celeb-DF)

End-to-end implementation of the methodology described in
`phase+transformer+Mask+Methodology(baha).docx` and
`phase-transformer-mask-methodology-baha-buzzing-blossom.docx`.

It implements four variants in a single codebase:

| Variant | TIM | Phase | HF Mask |
| ------- | --- | ----- | ------- |
| V1 (Baseline-TIM)        | ✓ |   |   |
| V2 (Phase-only)          |   | ✓ |   |
| V3 (TIM + Phase)         | ✓ | ✓ |   |
| V4 (TIM + Phase + Mask, full) | ✓ | ✓ | ✓ |

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

## Verifying installation

```bash
pytest -q tests/
```

All tests should pass on CPU and run in < 1 minute. They cover:

- TIM extractor output shape & non-negativity.
- Phase branch — sin² + cos² ≡ 1 invariant, mask changes output in train mode.
- Full model — all four variants forward-pass at small resolution.
- Metrics — perfect/random smoke checks.

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
