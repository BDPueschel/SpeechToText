@echo off
:: setup.bat - One-click setup for Whisper STT
:: Run this once on a new machine, then use launch_whisper.bat going forward.

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"

echo.
echo  ====================================
echo    Whisper STT - Setup
echo  ====================================
echo.

:: -- Check Python -------------------------
echo [1/2] Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python is not installed or not in PATH.
    echo  Please install Python 3.10+ from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   Found %%v

:: -- Install dependencies ------------------
echo.
echo [2/2] Installing dependencies (this may take a minute)...
echo.

set RETRIES=3
set ATTEMPT=0

:pip_retry
set /a ATTEMPT+=1
echo   Attempt %ATTEMPT% of %RETRIES%...
pip install --timeout 120 -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    if %ATTEMPT% lss %RETRIES% (
        echo.
        echo   Install hit an error, retrying...
        echo.
        goto pip_retry
    )
    echo.
    echo  ERROR: pip install failed after %RETRIES% attempts.
    echo  Check the output above for details.
    pause
    exit /b 1
)

:: -- Done ---------------------------------
echo.
echo  ====================================
echo    Setup complete!
echo.
echo    To start: double-click launch_whisper.bat
echo.
echo    First launch will download the Whisper
echo    model (~150MB) -- needs internet once.
echo  ====================================
echo.
pause
