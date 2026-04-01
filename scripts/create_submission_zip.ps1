# Builds a submission .zip of the project (modality folders + integration + docs).
# From repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/create_submission_zip.ps1
# Omit large video clips (smaller zip; add video/data locally before demo):
#   powershell -ExecutionPolicy Bypass -File scripts/create_submission_zip.ps1 -ExcludeVideoData

param(
    [switch]$ExcludeVideoData
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$zipName = "Multimodal-crime-incident-analyzer-submission-$stamp.zip"
$outZip = Join-Path $repoRoot $zipName
$staging = Join-Path $env:TEMP ("mca-submit-" + [guid]::NewGuid().ToString())

New-Item -ItemType Directory -Path $staging | Out-Null

$xdDirs = @(
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode"
)
if ($ExcludeVideoData) {
    $xdDirs += "video\data"
}

$robArgs = @(
    $repoRoot,
    $staging,
    "/E",
    "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS", "/NP",
    "/XD"
) + $xdDirs

& robocopy @robArgs | Out-Null
$rc = $LASTEXITCODE
if ($rc -ge 8) {
    throw "robocopy failed with exit code $rc"
}

Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $outZip -Force
Remove-Item $staging -Recurse -Force

Write-Host "Created: $outZip"
if ($ExcludeVideoData) {
    Write-Host "Note: video/data was excluded. Copy .mpg or .mp4 files into video/data (see video/README.md) before running the pipeline."
}
