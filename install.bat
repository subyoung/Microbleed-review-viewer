@echo off
setlocal
set "VIEWER_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%VIEWER_DIR%install.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
rem Keep the window open either way: the result is the point of running it.
pause
endlocal & exit /b %EXIT_CODE%
