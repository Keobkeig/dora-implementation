# Run all 3 SmolVLM Push-T VLA experiments sequentially.
# DoRA vs LoRA vs full fine-tuning on lerobot/pusht (LeRobot dataset).
#
# Output dirs: results/vla_pusht_{dora_r8, lora_r8, full}/
#
# Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File run_vla_experiments.ps1

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
    & uv run scripts/train_vla.py @TrainArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED ($LASTEXITCODE): $line"
        exit $LASTEXITCODE
    }
    Write-Host "DONE: $line"
    Write-Host ''
}

Run-Train --method dora --bf16
Run-Train --method lora --bf16
Run-Train --method full --bf16

Write-Host 'All 3 SmolVLM Push-T experiments complete.'
Write-Host 'Results in results/vla_pusht_{dora_r8, lora_r8, full}/'
