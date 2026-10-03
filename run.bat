@echo off
title StickerDeskew Studio
cd /d "%~dp0"
echo ========================================================
echo         StickerDeskew Studio Launcher
echo ========================================================
echo.

:: Detect Python executable
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
    goto :found_python
)

for /d %%d in ("%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python*") do (
    if exist "%%d\python.exe" (
        set "PY_CMD=%%d\python.exe"
        goto :found_python
    )
)

if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
    goto :found_python
)

where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :found_python
)

where py >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py"
    goto :found_python
)

echo [Error] Python was not found in your system PATH.
echo Please install Python from https://www.python.org/ and check "Add python.exe to PATH".
echo.
pause
exit /b 1

:found_python
:: Kill any stale background process listening on port 8080
if "%~1"=="" (
    for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8080 " ^| findstr "LISTENING"') do (
        taskkill /PID %%a /F >nul 2>&1
    )
)

%PY_CMD% run.py %*
if errorlevel 1 (
    echo.
    echo [Error] Execution stopped with an error.
    pause
)
