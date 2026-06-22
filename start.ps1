# start.ps1 - one-shot setup + run script for the Skin Cancer Detection app (Windows).
#             Tip: 'python run.py' works everywhere and has more options.
#
# What it does:
#   1. Verifies Python is installed (tries winget install if missing).
#   2. Creates a virtualenv at backend\.venv if it doesn't exist.
#   3. Installs backend requirements only when they've changed (idempotent).
#   4. Starts the Flask app at http://localhost:8000.
#
# No Node.js needed - the UI is served by Flask as plain HTML/CSS.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File .\start.ps1
#   .\start.ps1 -SetupOnly      # setup only, don't start the server
#
# Tip: if you get an execution-policy error, double-click start.bat instead.

param(
    [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ScriptDir "backend"
$VenvDir    = Join-Path $BackendDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ReqFile    = Join-Path $BackendDir "requirements.txt"
$ReqStamp   = Join-Path $VenvDir ".requirements.sha256"
$Port       = if ($env:PORT) { $env:PORT } else { "8000" }

function Log  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "OK  $m" -ForegroundColor Green }
function Warn ($m) { Write-Host "!   $m" -ForegroundColor Yellow }
function Err  ($m) { Write-Host "ERR $m" -ForegroundColor Red }
function Have ($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

# --- Python ------------------------------------------------------------------
Log "Checking for Python..."
if (-not (Have "python")) {
    Warn "Python not found. Attempting to install via winget..."
    if (Have "winget") {
        winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
    }
}
if (-not (Have "python")) {
    Err "Python is required. Install it from https://www.python.org/downloads/"
    Err "(check 'Add to PATH'), reopen the terminal, then re-run this script."
    exit 1
}
Ok "Python found: $((& python --version) 2>&1)"

# --- Virtualenv + requirements ----------------------------------------------
if (-not (Test-Path $VenvDir)) {
    Log "Creating virtualenv at backend\.venv..."
    & python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Err "Failed to create virtualenv."; exit 1 }
    Ok "Virtualenv created."
} else {
    Ok "Virtualenv already exists."
}

$currentHash = (Get-FileHash $ReqFile -Algorithm SHA256).Hash
$storedHash  = if (Test-Path $ReqStamp) { (Get-Content $ReqStamp -Raw).Trim() } else { "" }

if ($currentHash -eq $storedHash) {
    Ok "Requirements already satisfied (unchanged)."
} else {
    Log "Installing requirements (TensorFlow can take a while)..."
    & $VenvPython -m pip install --upgrade pip | Out-Null
    & $VenvPython -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) {
        Err "pip failed. If this is the Windows long-path error, move the project"
        Err "to a short path like C:\scd (or run 'python run.py --enable-long-paths'"
        Err "in an admin PowerShell), then delete backend\.venv and re-run."
        exit 1
    }
    Set-Content -Path $ReqStamp -Value $currentHash
    Ok "Requirements installed."
}

if ($SetupOnly) {
    Write-Host ""
    Ok "Setup complete. Run '.\start.ps1' to launch the app."
    exit 0
}

# --- Run ---------------------------------------------------------------------
Write-Host ""
Ok "App is starting. Open http://localhost:$Port in your browser."
Write-Host "   Press Ctrl+C to stop."
Write-Host ""
$env:PORT = $Port
Push-Location $BackendDir
try { & $VenvPython app.py } finally { Pop-Location }
