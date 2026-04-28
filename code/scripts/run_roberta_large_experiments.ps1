# Re-run all GLUE experiments using the FacebookAI/roberta-large preset.
# Mirrors run_roberta_experiments.sh but with the 355M-param model.
#
# Output dirs: results/glue_<task>_roberta_large_<method>_r8/   (or _full for full FT)
# Existing roberta-base results are NOT overwritten.
#
# Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File run_roberta_large_experiments.ps1

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeDir   = Resolve-Path (Join-Path $ScriptDir '..')
Set-Location $CodeDir
Write-Host "Working dir: $CodeDir"

function Run-Train {
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $TrainArgs)
    $line = ($TrainArgs -join ' ')
    Write-Host '========================================='
    Write-Host "START: $line"
    Write-Host '========================================='
    & uv run scripts/train_glue.py @TrainArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED ($LASTEXITCODE): $line"
        exit $LASTEXITCODE
    }
    Write-Host "DONE: $line"
    Write-Host ''
}

# RTE
Run-Train --model roberta_large --task rte  --method dora --bf16
Run-Train --model roberta_large --task rte  --method lora --bf16
Run-Train --model roberta_large --task rte  --method full --bf16

# MRPC
Run-Train --model roberta_large --task mrpc --method dora --bf16
Run-Train --model roberta_large --task mrpc --method lora --bf16
Run-Train --model roberta_large --task mrpc --method full --bf16

# SST-2
Run-Train --model roberta_large --task sst2 --method dora --bf16
Run-Train --model roberta_large --task sst2 --method lora --bf16
Run-Train --model roberta_large --task sst2 --method full --bf16

Write-Host 'All 9 RoBERTa-large GLUE experiments complete.'
Write-Host 'Results in results/glue_{rte,mrpc,sst2}_roberta_large_{dora_r8,lora_r8,full}/'
