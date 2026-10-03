# StickerDeskew Studio PowerShell Launcher
Set-Location -Path $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "        StickerDeskew Studio Launcher" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Detect Python with required libraries
$pythonCmd = $null
$psfPython = "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
if (Test-Path $psfPython) {
    $pythonCmd = $psfPython
} else {
    $psfDirs = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WindowsApps" -Filter "PythonSoftwareFoundation.Python*" -Directory -ErrorAction SilentlyContinue
    foreach ($dir in $psfDirs) {
        $candidate = Join-Path $dir.FullName "python.exe"
        if (Test-Path $candidate) {
            $pythonCmd = $candidate
            break
        }
    }
}

if (-not $pythonCmd) {
    $winAppPython = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
    if (Test-Path $winAppPython) {
        $pythonCmd = $winAppPython
    }
}

if (-not $pythonCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $test = python -c "import onnxruntime" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonCmd = "python"
    }
}

if (-not $pythonCmd -and (Get-Command py -ErrorAction SilentlyContinue)) {
    $pythonCmd = "py"
} elseif (-not $pythonCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $pythonCmd = "python"
}

if (-not $pythonCmd) {
    Write-Host "[Error] Python was not found in your system PATH." -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/ or ensure it is in your PATH." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# If launching studio (no arguments), clean up any stale zombie process on port 8080
if ($args.Count -eq 0) {
    Write-Host "Checking for stale processes on port 8080..." -ForegroundColor Gray
    try {
        $portListeners = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
        if ($portListeners) {
            foreach ($conn in $portListeners) {
                $p = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
                if ($p -and ($p.ProcessName -like "*python*" -or $p.ProcessName -like "*py*")) {
                    Write-Host "Stopping previous background studio instance (PID $($p.Id))..." -ForegroundColor Yellow
                    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                }
            }
        }
    } catch {
        # Ignore permission/network querying errors
    }
}

try {
    & $pythonCmd run.py @args
} catch {
    Write-Host "[Error] Failed to execute run.py: $_" -ForegroundColor Red
}

if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne $null) {
    Write-Host ""
    Write-Host "[Notice] Process finished with exit code $LASTEXITCODE." -ForegroundColor Yellow
    Read-Host "Press Enter to close this window"
}
