$ErrorActionPreference = "Stop"

$viewerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $viewerDir
# Kept outside OneDrive sync -- see install.ps1.
$pythonPath = Join-Path $env:LOCALAPPDATA "MicrobleedViewer\venv\Scripts\python.exe"
$appPath = Join-Path $viewerDir "desktop_app.py"
# Where this installation's dataset lives.  Kept in an untracked file so the
# tracked launcher carries no study's filenames: copy local.example.ps1 to
# local.ps1 and edit it, or set the three MICROBLEED_* variables yourself.
$localSettings = Join-Path $viewerDir "local.ps1"
if (Test-Path -LiteralPath $localSettings) {
    . $localSettings
}
if (-not $env:MICROBLEED_SOURCE_XLSX) {
    throw "No findings workbook configured. Copy viewer\local.example.ps1 to viewer\local.ps1 and set the paths in it."
}
$sourcePath = $env:MICROBLEED_SOURCE_XLSX
$sourceSnapshot = Join-Path $viewerDir ".source_snapshot.xlsx"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment not found at $pythonPath. Create it and install viewer\requirements.txt first."
}
if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Source workbook not found at $sourcePath."
}

# Keep the original workbook untouched. The app reads this local snapshot so
# a workbook lock or OneDrive read policy does not interrupt the viewer.
Copy-Item -LiteralPath $sourcePath -Destination $sourceSnapshot -Force
$env:MICROBLEED_SOURCE_XLSX = $sourceSnapshot
if (-not $env:MICROBLEED_DATA_ROOT) { $env:MICROBLEED_DATA_ROOT = Join-Path $projectDir "Data" }
if (-not $env:MICROBLEED_REVIEW_DB) { $env:MICROBLEED_REVIEW_DB = Join-Path $viewerDir "microbleed_review_data.sqlite" }

Push-Location $projectDir
try {
    & $pythonPath $appPath
}
finally {
    Pop-Location
}
