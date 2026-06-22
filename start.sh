#!/usr/bin/env bash
#
# start.sh - one-shot setup + run script for the Skin Cancer Detection app
#            (macOS / Linux). Tip: 'python3 run.py' works everywhere too.
#
# What it does:
#   1. Verifies Python 3 is installed.
#   2. Creates a virtualenv at backend/.venv if it doesn't exist.
#   3. Installs backend requirements only when they've changed (idempotent).
#   4. Starts the Flask app at http://localhost:8000.
#
# No Node.js needed - the UI is served by Flask as plain HTML/CSS.
#
# Usage:
#   ./start.sh            # setup everything and run the app
#   ./start.sh --setup    # setup only, don't start the server
#
# Press Ctrl+C to stop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"
REQ_FILE="$BACKEND_DIR/requirements.txt"
REQ_STAMP="$VENV_DIR/.requirements.sha256"

PORT="${PORT:-8000}"
SETUP_ONLY=0
[ "${1:-}" = "--setup" ] && SETUP_ONLY=1

c_reset="\033[0m"; c_blue="\033[1;34m"; c_green="\033[1;32m"; c_yellow="\033[1;33m"; c_red="\033[1;31m"
log()  { printf "${c_blue}==>${c_reset} %s\n" "$1"; }
ok()   { printf "${c_green}OK ${c_reset} %s\n" "$1"; }
warn() { printf "${c_yellow}!  ${c_reset} %s\n" "$1"; }
err()  { printf "${c_red}ERR${c_reset} %s\n" "$1" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }
OS="$(uname -s)"

sha256_of() {
  if have sha256sum; then sha256sum "$1" | awk '{print $1}';
  elif have shasum;   then shasum -a 256 "$1" | awk '{print $1}';
  else echo "no-hash-tool"; fi
}

# --- Python ------------------------------------------------------------------
log "Checking for Python 3..."
if ! have python3; then
  warn "Python 3 not found. Attempting to install..."
  if [ "$OS" = "Darwin" ] && have brew; then
    brew install python
  elif [ "$OS" = "Linux" ] && have apt-get; then
    sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
  fi
fi
if ! have python3; then
  err "Python 3 is required. Install it from https://www.python.org/downloads/ and re-run."
  exit 1
fi
ok "Python found: $(python3 --version 2>&1)"

# --- Virtualenv + requirements ----------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtualenv at backend/.venv..."
  python3 -m venv "$VENV_DIR"
  ok "Virtualenv created."
else
  ok "Virtualenv already exists."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

current_hash="$(sha256_of "$REQ_FILE")"
stored_hash="$(cat "$REQ_STAMP" 2>/dev/null || echo '')"
if [ "$current_hash" = "$stored_hash" ]; then
  ok "Requirements already satisfied (unchanged)."
else
  log "Installing requirements..."
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r "$REQ_FILE"
  echo "$current_hash" > "$REQ_STAMP"
  ok "Requirements installed."
fi

if [ "$SETUP_ONLY" -eq 1 ]; then
  echo
  ok "Setup complete. Run './start.sh' to launch the app."
  exit 0
fi

# --- Run ---------------------------------------------------------------------
echo
ok "App is starting. Open http://localhost:$PORT in your browser."
echo "   Press Ctrl+C to stop."
echo
cd "$BACKEND_DIR"
PORT="$PORT" exec python app.py
