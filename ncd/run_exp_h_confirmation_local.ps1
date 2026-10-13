param(
    [string]$Python = $env:NCD_PYTHON,
    [string]$CleanedDataRoot = $env:NCD_CLEANED_DATA_ROOT,
    [string]$Config = 'ncd/configs/exp_h_section19.json',
    [string]$OutputDir = '',
    [switch]$Smoke,
    [switch]$EvaluateOnly
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = 'python' }
if ([string]::IsNullOrWhiteSpace($CleanedDataRoot)) {
    throw 'Set NCD_CLEANED_DATA_ROOT or -CleanedDataRoot to the approved prepared input directory.'
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = if ($Smoke -or $env:NCD_LOCAL_SMOKE -eq '1') {
        'ncd/results/exp_h/section19/smoke_v1'
    } else { 'ncd/results/exp_h/section19/local_v1' }
}
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$Arguments = @('-I', 'ncd/run_exp_h_confirmation.py', '--config', $Config,
               '--cleaned-data-root', $CleanedDataRoot, '--output-dir', $OutputDir)
if ($Smoke) { $Arguments += '--smoke' }
if ($EvaluateOnly) { $Arguments += '--evaluate-only' }
Push-Location $Repo
try {
    & $Python @Arguments
    $Code = $LASTEXITCODE
} finally { Pop-Location }
exit $Code
