param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Splits = "splits.json",
    [string]$FaceCache = "face_cache_uniform16_all",
    [string]$SamplingMode = "legacy",
    [int]$NFrames = 16,
    [int]$BatchSize = 8,
    [int]$NumWorkers = 4,
    [int]$Epochs = 80,
    [string]$Device = "cuda",
    [string]$Checkpoints = "checkpoints",
    [string]$ResultsRoot = "results\v10_pair_training_ablation",
    [string[]]$Experiments = @(
        "random1",
        "random2",
        "random4",
        "random8",
        "all_independent",
        "random4_bag",
        "all_weighted"
    )
)

$ErrorActionPreference = "Stop"

function Get-PairConfig {
    param([string]$Experiment)
    switch ($Experiment) {
        "random1" {
            return @{
                Mode = "random_k"; K = 1; BagWeight = 0.5; PairWeight = "uniform"
            }
        }
        "random2" {
            return @{
                Mode = "random_k"; K = 2; BagWeight = 0.5; PairWeight = "uniform"
            }
        }
        "random4" {
            return @{
                Mode = "random_k"; K = 4; BagWeight = 0.5; PairWeight = "uniform"
            }
        }
        "random8" {
            return @{
                Mode = "random_k"; K = 8; BagWeight = 0.5; PairWeight = "uniform"
            }
        }
        "all_independent" {
            return @{
                Mode = "all_independent"; K = 1; BagWeight = 0.5; PairWeight = "uniform"
            }
        }
        "random4_bag" {
            return @{
                Mode = "random_k_bag"; K = 4; BagWeight = 0.5; PairWeight = "uniform"
            }
        }
        "all_weighted" {
            return @{
                Mode = "all_weighted"; K = 1; BagWeight = 0.5; PairWeight = "mid_residual_energy"
            }
        }
        default {
            throw "Unsupported experiment '$Experiment'"
        }
    }
}

foreach ($crf in @("crf_src", "crf0", "crf23", "crf40")) {
    $indexPath = Join-Path $FaceCache "$crf\index.json"
    if (-not (Test-Path $indexPath)) {
        throw "Missing $indexPath. Build or choose a face cache that contains all mixed CRF indexes."
    }
}

foreach ($exp in $Experiments) {
    $cfg = Get-PairConfig $exp
    $name = "v10_frame_cos_only_pair_${exp}_mixed"
    $outDir = Join-Path $ResultsRoot $exp
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    Write-Host ""
    Write-Host "=== Training pair experiment: $exp ==="
    & $Python train_v10_frame_supervised.py `
        --splits $Splits `
        --face_cache $FaceCache `
        --train_crfs crf_src crf0 crf23 crf40 `
        --val_crfs crf_src crf0 crf23 crf40 `
        --n_frames $NFrames `
        --sampling_mode $SamplingMode `
        --residual_mode gradient `
        --relation_mode cos_only `
        --train_pair_mode $cfg.Mode `
        --train_pair_k $cfg.K `
        --bag_loss_weight $cfg.BagWeight `
        --pair_weight_mode $cfg.PairWeight `
        --eval_pair_mode all `
        --batch_size $BatchSize `
        --num_workers $NumWorkers `
        --epochs $Epochs `
        --device $Device `
        --checkpoints $Checkpoints `
        --name $name

    Write-Host ""
    Write-Host "=== Evaluating pair experiment: $exp ==="
    & $Python eval_v10_frame_supervised.py `
        --ckpt "$Checkpoints\$name\best.pth" `
        --splits $Splits `
        --face_cache $FaceCache `
        --train_crf mixed `
        --crfs crf_src crf0 crf23 crf40 `
        --include_mixed `
        --split test `
        --n_frames $NFrames `
        --sampling_mode $SamplingMode `
        --eval_pair_mode all `
        --score_agg mean `
        --batch_size $BatchSize `
        --num_workers $NumWorkers `
        --device $Device `
        --out_dir $outDir
}
