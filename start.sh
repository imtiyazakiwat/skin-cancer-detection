#!/usr/bin/env bash
#
# start.sh - one-shot setup + run script for the Skin Cancer Detection app.
#
# What it does:
#   1. Verifies Python 3 is installed (errors with guidance if missing).
#   2. Creates a virtualenv at backend/.venv if it doesn't exist.
#   3. Installs backend requirements only when they've changed (idempotent).
#   4. Verifies Node.js/npm is installed; tries to install it if missing.
#   5. Installs frontend deps only when node_modules is missing.
#   6. Starts the FastAPI backend (port 8000) and Vite frontend (port 5173).
#
# Usage:
#   ./start.sh            # setup everything and run both servers
#   ./start.sh --setup    # setup only, don't start the servers
#
# Press Ctrl+C to stop both servers.

set -euo pipefail

# --- Resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv"
REQ_FILE="$BACKEND_DIR/requirements.txt"
REQ_STAMP="$VENV_DIR/.requirements.sha256"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
SETUP_ONLY=0
[ "${1:-}" = "--setup" ] && SETUP_ONLY=1

# --- Pretty logging ----------------------------------------------------------
c_reset="\033[0m"; c_blue="\033[1;34m"; c_green="\033[1;32m"; c_yellow="\033[1;33m"; c_red="\033[1;31m"
log()  { printf "${c_blue}==>${c_reset} %s\n" "$1"; }
ok()   { printf "${c_green}OK ${c_reset} %s\n" "$1"; }
warn() { printf "${c_yellow}!  ${c_reset} %s\n" "$1"; }
err()  { printf "${c_red}ERR${c_reset} %s\n" "$1" >&2; }

OS="$(uname -s)"

# --- Helpers -----------------------------------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }

sha256_of() {
  if have sha256sum; then sha256sum "$1" | awk '{print $1}';
  elif have shasum;   then shasum -a 256 "$1" | awk '{print $1}';
  else echo "no-hash-tool"; fi
}

# --- 1. Python ---------------------------------------------------------------
ensure_python() {
  log "Checking for Python 3..."
  if have python3; then
    ok "Python found: $(python3 --version 2>&1)"
    return
  fi

  warn "Python 3 not found. Attempting to install..."
  if [ "$OS" = "Darwin" ] && have brew; then
    brew install python
  elif [ "$OS" = "Linux" ] && have apt-get; then
    sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
  fi

  if ! have python3; then
    err "Python 3 is required but could not be installed automatically."
    err "Install it from https://www.python.org/downloads/ and re-run this script."
    exit 1
  fi
  ok "Python installed: $(python3 --version 2>&1)"
}

# --- 2 & 3. Virtualenv + requirements ---------------------------------------
ensure_backend() {
  if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtualenv at backend/.venv..."
    python3 -m venv "$VENV_DIR"
    ok "Virtualenv created."
  else
    ok "Virtualenv already exists."
  fi

  # Activate the venv for this script's process.
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  local current_hash stored_hash
  current_hash="$(sha256_of "$REQ_FILE")"
  stored_hash="$(cat "$REQ_STAMP" 2>/dev/null || echo '')"

  if [ "$current_hash" = "$stored_hash" ]; then
    ok "Backend requirements already satisfied (unchanged)."
  else
    log "Installing backend requirements..."
    python -m pip install --upgrade pip >/dev/null
    python -m pip install -r "$REQ_FILE"
    echo "$current_hash" > "$REQ_STAMP"
    ok "Backend requirements installed."
  fi
}

# --- 4. Node.js --------------------------------------------------------------
ensure_node() {
  log "Checking for Node.js..."
  if have node && have npm; then
    ok "Node found: $(node --version)  npm: $(npm --version)"
    return
  fi

  warn "Node.js not found. Attempting to install..."
  if [ "$OS" = "Darwin" ]; then
    if have brew; then
      brew install node
    else
      err "Homebrew not found. Install Node.js from https://nodejs.org/ (LTS) and re-run."
      exit 1
    fi
  elif [ "$OS" = "Linux" ]; then
    if have apt-get; then
      curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
      sudo apt-get install -y nodejs
    else
      err "Could not auto-install Node.js. Install it from https://nodejs.org/ and re-run."
      exit 1
    fi
  else
    err "Unsupported OS for auto-install. Install Node.js from https://nodejs.org/ and re-run."
    exit 1
  fi

  if ! have node; then
    err "Node.js installation failed. Install it manually from https://nodejs.org/."
    exit 1
  fi
  ok "Node installed: $(node --version)"
}

# --- 5. Frontend deps --------------------------------------------------------
ensure_frontend() {
  if [ -d "$FRONTEND_DIR/node_modules" ]; then
    ok "Frontend dependencies already installed."
  else
    log "Installing frontend dependencies (npm install)..."
    (cd "$FRONTEND_DIR" && npm install)
    ok "Frontend dependencies installed."
  fi
}

# --- 6. Run both servers -----------------------------------------------------
BACKEND_PID=""
cleanup() {
  echo
  log "Shutting down..."
  if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
  ok "Stopped."
}

run_servers() {
  trap cleanup EXIT INT TERM

  log "Starting backend on http://localhost:$BACKEND_PORT ..."
  (
    cd "$BACKEND_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    exec uvicorn main:app --reload --host 0.0.0.0 --port "$BACKEND_PORT"
  ) &
  BACKEND_PID=$!

  sleep 2
  if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    err "Backend failed to start. See the output above."
    exit 1
  fi
  ok "Backend running (PID $BACKEND_PID)."

  log "Starting frontend on http://localhost:$FRONTEND_PORT ..."
  echo
  ok "App is starting. Open http://localhost:$FRONTEND_PORT in your browser."
  echo "   (Backend API + docs: http://localhost:$BACKEND_PORT/docs)"
  echo "   Press Ctrl+C to stop both servers."
  echo

  # Frontend runs in the foreground; when it exits, the trap stops the backend.
  (cd "$FRONTEND_DIR" && npm run dev -- --port "$FRONTEND_PORT")
}

# --- Main --------------------------------------------------------------------
main() {
  log "Skin Cancer Detection - setup & start"
  ensure_python
  ensure_backend
  ensure_node
  ensure_frontend

  if [ "$SETUP_ONLY" -eq 1 ]; then
    echo
    ok "Setup complete. Run './start.sh' to launch the app."
    exit 0
  fi

  echo
  run_servers
}

main "$@"
