param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Splits = "splits.json",
    [int]$NFrames = 16,
    [int]$BatchSize = 8,
    [int]$NumWorkers = 4,
    [int]$Epochs = 80,
    [string]$Device = "cuda",
    [string]$Checkpoints = "checkpoints",
    [string]$ResultsRoot = "results\v10_sampling_ablation",
    [string]$S2Cache = "face_cache_clip16_s2_all",
    [string]$Local16Cache = "face_cache_clip16_s1_all",
    [string]$Global16Cache = "face_cache_uniform16_all",
    [string]$FourByFourCache = "face_cache_segments4x4_all",
    [string[]]$Modes = @("s2", "local16", "global16", "4x4")
)

$ErrorActionPreference = "Stop"

$cacheByMode = @{
    "s2" = $S2Cache
    "local16" = $Local16Cache
    "global16" = $Global16Cache
    "4x4" = $FourByFourCache
}

foreach ($mode in $Modes) {
    if (-not $cacheByMode.ContainsKey($mode)) {
        throw "Unsupported mode '$mode'. Use one of: $($cacheByMode.Keys -join ', ')"
    }
    $faceCache = $cacheByMode[$mode]
    $missingIndex = @()
    foreach ($crf in @("crf_src", "crf0", "crf23", "crf40")) {
        $indexPath = Join-Path $faceCache "$crf\index.json"
        if (-not (Test-Path $indexPath)) {
            $missingIndex += $indexPath
        }
    }
    if ($missingIndex.Count -gt 0) {
        throw "Missing index files for mode '$mode' under $faceCache. Build the protocol cache first. Missing: $($missingIndex -join ', ')"
    }
}

foreach ($mode in $Modes) {
    $safeMode = $mode -replace "[^A-Za-z0-9_]+", "_"
    $name = "v10_frame_cos_only_sampling_${safeMode}_mixed"
    $outDir = Join-Path $ResultsRoot $safeMode
    $faceCache = $cacheByMode[$mode]
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    Write-Host ""
    Write-Host "=== Training sampling mode: $mode  cache: $faceCache ==="
    & $Python train_v10_frame_supervised.py `
        --splits $Splits `
        --face_cache $faceCache `
        --train_crfs crf_src crf0 crf23 crf40 `
        --val_crfs crf_src crf0 crf23 crf40 `
        --n_frames $NFrames `
        --sampling_mode legacy `
        --residual_mode gradient `
        --relation_mode cos_only `
        --eval_pair_mode all `
        --batch_size $BatchSize `
        --num_workers $NumWorkers `
        --epochs $Epochs `
        --device $Device `
        --checkpoints $Checkpoints `
        --name $name

    Write-Host ""
    Write-Host "=== Evaluating sampling mode: $mode ==="
    & $Python eval_v10_frame_supervised.py `
        --ckpt "$Checkpoints\$name\best.pth" `
        --splits $Splits `
        --face_cache $faceCache `
        --train_crf mixed `
        --crfs crf_src crf0 crf23 crf40 `
        --include_mixed `
        --split test `
        --n_frames $NFrames `
        --sampling_mode legacy `
        --eval_pair_mode all `
        --score_agg mean `
        --batch_size $BatchSize `
        --num_workers $NumWorkers `
        --device $Device `
        --out_dir $outDir
}
