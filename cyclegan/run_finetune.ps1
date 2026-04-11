"""
================================================================================
FINETUNE HELPER SCRIPT — run_finetune.sh / run_finetune.ps1
================================================================================
Convenience script for running the fine-tuning pipeline with various configurations.
Choose the appropriate script for your OS:
  - Linux/macOS: bash run_finetune.sh
  - Windows: powershell -ExecutionPolicy Bypass -File run_finetune.ps1
================================================================================
"""

# Windows PowerShell version: run_finetune.ps1

param(
    [string]$Mode = "full",           # "eval", "full", "resume"
    [string]$DataDir = "./dataset",
    [string]$CheckpointDir = "./checkpoints",
    [int]$Epochs = 50,
    [string]$ResumeCheckpoint = $null,
    [switch]$Verbose,
    [switch]$Reproducible
)

$ErrorActionPreference = "Stop"

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║        CycleGAN Fine-Tuning Pipeline (Windows)                ║"
Write_Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

# Validate Python
$PythonCmd = python
try {
    & $PythonCmd --version | Out-Null
}
catch {
    Write-Host '✗ Python not found. Please install Python and try again.' -ForegroundColor Red
    exit 1
}

# Run fine-tuning based on mode
$Cmd = @(
    "python", "finetune_cyclegan.py",
    "--data_dir", $DataDir,
    "--checkpoint_dir", $CheckpointDir,
    "--epochs", $Epochs
)

if ($Mode -eq "eval") {
    Write-Host "Mode: EVALUATE ONLY (no training)"
    $Cmd += "--eval_only"
}
elseif ($Mode -eq "full") {
    Write-Host "Mode: FULL PIPELINE (evaluate + fine-tune)"
}
elseif ($Mode -eq "resume") {
    Write-Host "Mode: RESUME FROM CHECKPOINT"
    if ($ResumeCheckpoint) {
        $Cmd += "--resume", $ResumeCheckpoint
    }
}

if ($Verbose) {
    $Cmd += "--verbose"
}

if ($Reproducible) {
    $Cmd += "--reproducible"
}

Write-Host ""
Write-Host "Options:"
Write-Host "  Data directory: $DataDir"
Write-Host "  Checkpoints: $CheckpointDir"
Write-Host "  Epochs: $Epochs"
Write-Host "  Verbose: $Verbose"
Write-Host "  Reproducible: $Reproducible"
Write-Host ""
Write-Host "Running: $($Cmd -join ' ')"
Write-Host ""

& $Cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Script failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
else {
    Write-Host ""
    Write-Host "✓ Fine-tuning pipeline complete!" -ForegroundColor Green
}
