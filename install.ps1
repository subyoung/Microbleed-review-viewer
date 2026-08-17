$ErrorActionPreference = "Stop"

$viewerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $viewerDir
# The venv is machine-specific (pins a Python install path and compiled
# extension ABI), so it lives under %LOCALAPPDATA% instead of the
# OneDrive-synced project folder -- syncing it across machines breaks it.
$venvPath = Join-Path $env:LOCALAPPDATA "MicrobleedViewer\venv"
$requirements = Join-Path $viewerDir "requirements.txt"

$pythonCommand = Get-Command py -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
}
if (-not $pythonCommand) {
    throw "Python 3 was not found. Install Python 3.11 or newer and run this script again."
}

if (-not (Test-Path -LiteralPath (Join-Path $venvPath "Scripts\python.exe"))) {
    & $pythonCommand.Source -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
& $venvPython -m pip install --disable-pip-version-check -r $requirements

# A Start menu entry is where Windows shows an application icon, so the
# install puts one there. It launches through pythonw.exe, which runs the
# viewer without a console window behind it.
$iconPath = Join-Path $viewerDir "microbleed_review_icon.ico"
$launcher = Join-Path $venvPath "Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $launcher)) {
    $launcher = $venvPython
}
if (Test-Path -LiteralPath $iconPath) {
    try {
        $startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
        $shortcut = Join-Path $startMenu "Microbleed Review.lnk"
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut($shortcut)
        $link.TargetPath = $launcher
        $link.Arguments = '"' + (Join-Path $viewerDir "desktop_app.py") + '"'
        $link.WorkingDirectory = $viewerDir
        $link.IconLocation = $iconPath
        $link.Description = "Microbleed Review viewer"
        $link.Save()
        Write-Output "Start menu shortcut created: $shortcut"
    } catch {
        Write-Output "Could not create the Start menu shortcut: $($_.Exception.Message)"
    }
}

Write-Output "Viewer environment is ready. Run .\viewer\run_app.ps1 for the desktop viewer."
Write-Output "The layout used before the redesign is available via .\viewer\run_app_classic.ps1"
