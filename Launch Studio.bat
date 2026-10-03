@echo off
title Sticker Deskew Studio
cd /d "%~dp0"
echo.
echo  ====================================
echo   Sticker Deskew Studio - Starting
echo  ====================================
echo.

:: Detect Python executable with required dependencies
set "PY_CMD="

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
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" -c "import onnxruntime" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
        goto :found_python
    )
)

where python >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import onnxruntime" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=python"
        goto :found_python
    )
)

where py >nul 2>&1
if %errorlevel% equ 0 (
    py -c "import onnxruntime" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py"
        goto :found_python
    )
    set "PY_CMD=py"
    goto :found_python
)

where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :found_python
)

echo [Error] Python with required packages was not found in your system.
echo Please ensure Python with onnxruntime and torch is installed.
echo.
pause
exit /b 1

:found_python
:: Kill any old instance on port 8080
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8080 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: Start server via run.py which auto-handles port binding and browser launching
%PY_CMD% run.py
if errorlevel 1 (
    echo.
    echo [Error] Studio server stopped with an error.
    pause
)
