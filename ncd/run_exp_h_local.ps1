param(
    [string]$Python = 'python',
    [string]$OutputDir = 'ncd/results/exp_h/section17/local_v1',
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$arguments = @('-I', 'ncd/run_exp_h.py', '--config', 'ncd/configs/exp_h_section17.json',
    '--output-dir', $OutputDir, '--settings', 'S3', 'S2', 'S1')
if ($Smoke) { $arguments += '--smoke' }
& $Python @arguments
$experimentExit = $LASTEXITCODE
Write-Output "EXP_H_PROCESS_EXIT code=$experimentExit"
exit $experimentExit
