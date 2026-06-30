param(
    [string[]]$Modes = @(
        "full",
        "feat_only",
        "abs_only",
        "cos_only",
        "dist_only",
        "abs_cos",
        "abs_dist",
        "cos_dist",
        "no_original",
        "concat_views"
    ),
    [int]$Epochs = 80,
    [int]$BatchSize = 8,
    [string]$Device = "cuda",
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

foreach ($mode in $Modes) {
    $name = "v10_frame_relation_${mode}_s2_mixed"
    Write-Host "=== Training relation_mode=$mode -> $name ==="
    & $Python train_v10_frame_supervised.py `
        --face_cache face_cache_s2_all `
        --train_crfs crf_src crf0 crf23 crf40 `
        --val_crfs crf_src crf0 crf23 crf40 `
        --n_frames 16 `
        --residual_mode gradient `
        --relation_mode $mode `
        --phase_mid_low 0.10 `
        --phase_mid_high 0.70 `
        --eval_pair_mode all `
        --batch_size $BatchSize `
        --epochs $Epochs `
        --lr 3e-4 `
        --name $name `
        --device $Device

    Write-Host "=== Evaluating relation_mode=$mode ==="
    & $Python eval_v10_frame_supervised.py `
        --ckpt "checkpoints/$name/best.pth" `
        --train_crf mixed `
        --crfs crf_src crf0 crf23 crf40 `
        --include_mixed `
        --split test `
        --eval_pair_mode all `
        --score_agg mean `
        --batch_size $BatchSize `
        --device $Device `
        --out_dir "results/v10_frame_relation_ablation/$mode"
}
