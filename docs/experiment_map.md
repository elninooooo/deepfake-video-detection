# Experiment Map

This repository keeps several historical variants because they support the
preliminary study and ablation analysis. The final method is GRFR.

## Final Method

| Component | File |
| --- | --- |
| GRFR residual frequency-band relation branch | `modelsgenerate/residual_spectral_relation.py` |
| Frame-level GRFR training | `train_v10_frame_supervised.py` |
| CLS video-level aggregation | `train_v10_frame_cls_aggregator.py` |
| GRFR video-level evaluation | `eval_v10_frame_cls_aggregator.py` |

## Main Baseline

| Variant | Purpose | Entry |
| --- | --- | --- |
| V1 TIM | Temporal inconsistency baseline | `train.py --variant v1` |

## Preliminary Study

| Topic | Representative files |
| --- | --- |
| Compression and TIM baseline behavior | `train.py`, `eval_cross_compression.py` |
| Phase representation reliability | `modelsgenerate/phase_branch.py`, `eval_threshold_calibration.py` |
| Real-only reconstruction / prediction | `modelsgenerate/v7_real_tim.py`, `train_v7_real_tim.py` |
| Early frequency relation probes | `probe_tim_spectral_relation.py`, `probe_resnet_spectral_relation_oneclass.py` |

## Ablation Experiments

| Ablation | Representative files |
| --- | --- |
| Relation mode | `scripts/run_v10_frame_relation_ablation.ps1` |
| Pair sampling | `scripts/run_v10_pair_training_ablation.ps1` |
| Sampling protocol | `scripts/run_v10_sampling_ablation.ps1` |
| Residual representation | `scripts/visualize_v10_rgb_residual_views.py`, `scripts/visualize_v10_residual_views.py` |
| Temporal statistics / relation probes | `probe_v10_relation_temporal_stats.py`, `v10_relation_temporal_utils.py` |

Historical variants are retained for reproducibility but are not required for
running the final GRFR framework.
