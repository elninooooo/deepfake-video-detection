# Script Guide

This folder contains batch runners and visualization utilities used for
reproduction, ablation, and figure generation.

## Batch Runners

| Script | Purpose |
| --- | --- |
| `run_v10_frame_relation_ablation.ps1` | Compare relation modes such as `full`, `feat_only`, `abs_only`, and `cos_only`. |
| `run_v10_pair_training_ablation.ps1` | Compare frame-pair training strategies such as `random2`, `random4`, `random8`, and all-pair variants. |
| `run_v10_sampling_ablation.ps1` | Compare temporal sampling protocols. |

## Visualization Utilities

| Script | Purpose |
| --- | --- |
| `visualize_v10_residual_views.py` | Visualize gradient residual frequency-band views. |
| `visualize_v10_rgb_residual_views.py` | Visualize RGB residual frequency-band views. |
| `visualize_fft_lowfreq_removal.py` | Visualize FFT masking and inverse FFT effects. |
| `build_v10_frame_gradient_library.py` | Build a small image library of RGB frames, gradient maps, and gradient residual views. |

These scripts are not required for basic model training, but they support the
paper's preliminary study, ablation analysis, and visual explanations.
