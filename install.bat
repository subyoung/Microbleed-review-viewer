@echo off
setlocal
set "VIEWER_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%VIEWER_DIR%install.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
