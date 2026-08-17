# Copy this file to local.ps1 (untracked) and point it at your dataset.
#
#   Source workbook : the findings sheet described in examples/README.md
#   Data root       : the folder of per-case MRI folders
#   Review database : where this reader's reviews are stored
#
# Keep the review database on a local disk, not in a synced folder.

$env:MICROBLEED_SOURCE_XLSX = Join-Path $projectDir "microbleeds.xlsx"
$env:MICROBLEED_DATA_ROOT = Join-Path $projectDir "Data"
$env:MICROBLEED_REVIEW_DB = Join-Path $viewerDir "microbleed_review_data.sqlite"
