param(
    [string]$Python = "python",
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$DataRoot,
    [string]$ArtifactsRoot = (Join-Path $PSScriptRoot ".artifacts\malaya-network-gt"),
    [string]$CacheRoot = "",
    [string]$ResultsRoot = "",
    [string]$OverlapWork = "",
    [int[]]$Seeds = @(1, 2, 3, 4, 42),
    [switch]$SkipCacheBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$DataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
$csvRoot = Join-Path $DataRoot "malaya-network-gt\csv_output"
if (-not (Test-Path -LiteralPath $csvRoot -PathType Container)) {
    throw "Expected the pinned Malaya CSV checkout at: $csvRoot"
}
if ($Seeds.Count -eq 0) {
    throw "At least one seed is required."
}

if (-not $CacheRoot) {
    $CacheRoot = Join-Path $ArtifactsRoot "cache"
}
if (-not $ResultsRoot) {
    $ResultsRoot = Join-Path $ArtifactsRoot "results"
}
if (-not $OverlapWork) {
    $OverlapWork = Join-Path $ArtifactsRoot "overlap-work"
}

function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
    }
}

$dataset = "malaya-network-gt"
$datasetCache = Join-Path $CacheRoot $dataset
$manifest = Join-Path $datasetCache "streaming_manifest.json"
$viewRoot = Join-Path $datasetCache "evaluation_views\duplicate_excluded"
$viewManifest = Join-Path $viewRoot "evaluation_view_manifest.json"

New-Item -ItemType Directory -Force -Path $CacheRoot, $ResultsRoot, $OverlapWork | Out-Null
Push-Location $PSScriptRoot
try {
    if (-not $SkipCacheBuild -and -not (Test-Path $manifest)) {
        Invoke-Python -m fullcache `
            --data-root $DataRoot `
            --output-root $CacheRoot `
            --dataset $dataset `
            --overlap-work-directory $OverlapWork
    }

    if (-not (Test-Path $manifest)) {
        throw "Missing completed streaming manifest: $manifest"
    }

    Invoke-Python -m fullcache.verify `
        --cache-root $CacheRoot `
        --dataset $dataset `
        --recompute-overlap `
        --overlap-work-directory $OverlapWork

    if (-not (Test-Path $viewManifest)) {
        Invoke-Python -m sensitivity_overlap `
            --manifest $manifest `
            --output-dir $viewRoot `
            --view-name duplicate_excluded `
            --work-directory $OverlapWork
    }

    $runs = @(
        @{ Name = "malaya_mlp_128x2"; Config = (Join-Path $PSScriptRoot "configs\malaya_mlp_128x2.json") },
        @{ Name = "malaya_mlp_256x4"; Config = (Join-Path $PSScriptRoot "configs\malaya_mlp_256x4.json") }
    )
    foreach ($run in $runs) {
        $output = Join-Path $ResultsRoot $run.Name
        $arguments = @(
            "-m", "streaming_full",
            "--manifest", $manifest,
            "--evaluation-view", "duplicate_excluded=$viewManifest",
            "--config-json", $run.Config,
            "--seeds"
        ) + @($Seeds | ForEach-Object { [string]$_ }) + @(
            "--output-dir", $output
        )
        Invoke-Python @arguments
    }
}
finally {
    Pop-Location
}
