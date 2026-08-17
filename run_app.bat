@echo off
setlocal
set "VIEWER_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%VIEWER_DIR%run_app.ps1"
endlocal
