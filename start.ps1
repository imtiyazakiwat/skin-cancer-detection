# start.ps1 - one-shot setup + run script for the Skin Cancer Detection app (Windows / PowerShell).
#
# What it does:
#   1. Verifies Python is installed (tries winget install if missing).
#   2. Creates a virtualenv at backend\.venv if it doesn't exist.
#   3. Installs backend requirements only when they've changed (idempotent).
#   4. Verifies Node.js/npm is installed (tries winget install if missing).
#   5. Installs frontend deps only when node_modules is missing.
#   6. Starts the FastAPI backend (port 8000) in a new window and the
#      Vite frontend (port 5173) in this window.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File .\start.ps1
#   .\start.ps1 -SetupOnly      # setup only, don't start the servers
#
# Tip: if you get an execution-policy error, double-click start.bat instead.

param(
    [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"

# --- Resolve paths -----------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir  = Join-Path $ScriptDir "backend"
$FrontendDir = Join-Path $ScriptDir "frontend"
$VenvDir     = Join-Path $BackendDir ".venv"
$VenvPython  = Join-Path $VenvDir "Scripts\python.exe"
$ReqFile     = Join-Path $BackendDir "requirements.txt"
$ReqStamp    = Join-Path $VenvDir ".requirements.sha256"

$BackendPort  = if ($env:BACKEND_PORT)  { $env:BACKEND_PORT }  else { "8000" }
$FrontendPort = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { "5173" }

# --- Pretty logging ----------------------------------------------------------
function Log  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "OK  $m" -ForegroundColor Green }
function Warn ($m) { Write-Host "!   $m" -ForegroundColor Yellow }
function Err  ($m) { Write-Host "ERR $m" -ForegroundColor Red }

function Have ($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

# --- 1. Python ---------------------------------------------------------------
function Ensure-Python {
    Log "Checking for Python..."
    if (Have "python") {
        $v = (& python --version) 2>&1
        Ok "Python found: $v"
        return
    }

    Warn "Python not found. Attempting to install via winget..."
    if (Have "winget") {
        winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
    }

    if (-not (Have "python")) {
        Err "Python is required but could not be installed automatically."
        Err "Install it from https://www.python.org/downloads/ (check 'Add to PATH'),"
        Err "close and reopen the terminal, then re-run this script."
        exit 1
    }
    Ok "Python installed: $((& python --version) 2>&1)"
}

# --- 2 & 3. Virtualenv + requirements ---------------------------------------
function Ensure-Backend {
    if (-not (Test-Path $VenvDir)) {
        Log "Creating virtualenv at backend\.venv..."
        & python -m venv $VenvDir
        Ok "Virtualenv created."
    } else {
        Ok "Virtualenv already exists."
    }

    $currentHash = (Get-FileHash $ReqFile -Algorithm SHA256).Hash
    $storedHash  = if (Test-Path $ReqStamp) { Get-Content $ReqStamp -Raw } else { "" }

    if ($currentHash -eq $storedHash.Trim()) {
        Ok "Backend requirements already satisfied (unchanged)."
    } else {
        Log "Installing backend requirements..."
        & $VenvPython -m pip install --upgrade pip | Out-Null
        & $VenvPython -m pip install -r $ReqFile
        Set-Content -Path $ReqStamp -Value $currentHash
        Ok "Backend requirements installed."
    }
}

# --- 4. Node.js --------------------------------------------------------------
function Ensure-Node {
    Log "Checking for Node.js..."
    if ((Have "node") -and (Have "npm")) {
        Ok "Node found: $(& node --version)  npm: $(& npm --version)"
        return
    }

    Warn "Node.js not found. Attempting to install via winget..."
    if (Have "winget") {
        winget install -e --id OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
    }

    if (-not (Have "node")) {
        Err "Node.js could not be installed automatically."
        Err "Install the LTS version from https://nodejs.org/, close and reopen"
        Err "the terminal, then re-run this script."
        exit 1
    }
    Ok "Node installed: $(& node --version)"
}

# --- 5. Frontend deps --------------------------------------------------------
function Ensure-Frontend {
    if (Test-Path (Join-Path $FrontendDir "node_modules")) {
        Ok "Frontend dependencies already installed."
    } else {
        Log "Installing frontend dependencies (npm install)..."
        Push-Location $FrontendDir
        try { & npm install } finally { Pop-Location }
        Ok "Frontend dependencies installed."
    }
}

# --- 6. Run both servers -----------------------------------------------------
function Run-Servers {
    Log "Starting backend on http://localhost:$BackendPort (new window)..."
    $backendProc = Start-Process -FilePath $VenvPython `
        -ArgumentList "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", $BackendPort `
        -WorkingDirectory $BackendDir -PassThru
    Start-Sleep -Seconds 2
    Ok "Backend running (PID $($backendProc.Id))."

    Write-Host ""
    Ok "App is starting. Open http://localhost:$FrontendPort in your browser."
    Write-Host "   (Backend API + docs: http://localhost:$BackendPort/docs)"
    Write-Host "   Close this window or press Ctrl+C to stop the frontend."
    Write-Host ""

    Push-Location $FrontendDir
    try {
        & npm run dev -- --port $FrontendPort
    } finally {
        Pop-Location
        Log "Stopping backend..."
        if ($backendProc -and -not $backendProc.HasExited) {
            Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
        }
        Ok "Stopped."
    }
}

# --- Main --------------------------------------------------------------------
Log "Skin Cancer Detection - setup & start (Windows)"
Ensure-Python
Ensure-Backend
Ensure-Node
Ensure-Frontend

if ($SetupOnly) {
    Write-Host ""
    Ok "Setup complete. Run '.\start.ps1' to launch the app."
    exit 0
}

Write-Host ""
Run-Servers
