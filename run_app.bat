@echo off
setlocal
set "VIEWER_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%VIEWER_DIR%run_app.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
rem A double-clicked window closes at once, taking the reason with it.
if not "%EXIT_CODE%"=="0" (
    echo.
    echo The viewer could not start. The message above says why.
    echo On a PC where it has never run, run install.bat in this folder first.
    pause
)
endlocal & exit /b %EXIT_CODE%
