#!/usr/bin/env python3
"""
run.py - cross-platform setup + run script for the Skin Cancer Detection app.

Works on Windows, macOS, and Linux. Only needs Python to bootstrap.

What it does:
  1. Checks the Python version.
  2. (Windows) Detects the long-path limitation that breaks TensorFlow installs
     and tells you how to fix it.
  3. Creates a virtualenv at backend/.venv (or recreates it with --recreate).
  4. Installs backend requirements, only when requirements.txt has changed.
  5. Checks for Node.js/npm (tries to auto-install where possible).
  6. Runs `npm install` only when node_modules is missing.
  7. Starts the FastAPI backend (port 8000) and the Vite frontend (port 5173),
     and shuts both down cleanly on Ctrl+C.

Usage (from anywhere):
  python run.py                 # setup everything and run both servers
  python run.py --setup-only    # setup only, don't start servers
  python run.py --recreate      # delete and rebuild backend/.venv first
  python run.py --backend-port 8001 --frontend-port 5174
  python run.py --enable-long-paths   # (Windows, admin) flip the registry flag
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
VENV_DIR = BACKEND_DIR / ".venv"
REQ_FILE = BACKEND_DIR / "requirements.txt"
REQ_STAMP = VENV_DIR / ".requirements.sha256"

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# --- Pretty logging ----------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and not IS_WINDOWS


def _c(code: str, msg: str) -> str:
    return f"\033[{code}m{msg}\033[0m" if _USE_COLOR else msg


def log(m: str) -> None:
    print(_c("1;34", "==> ") + m)


def ok(m: str) -> None:
    print(_c("1;32", "OK  ") + m)


def warn(m: str) -> None:
    print(_c("1;33", "!   ") + m)


def err(m: str) -> None:
    print(_c("1;31", "ERR ") + m, file=sys.stderr)


def fail(m: str, code: int = 1) -> "None":
    err(m)
    sys.exit(code)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(cmd, cwd: Path | None = None) -> int:
    """Run a command, streaming its output. Returns the exit code."""
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


# --- 1. Python ---------------------------------------------------------------
def check_python() -> None:
    log("Checking Python version...")
    if sys.version_info < (3, 8):
        fail(
            f"Python 3.8+ required, but this is {platform.python_version()}. "
            "Install a newer Python from https://www.python.org/downloads/."
        )
    ok(f"Python {platform.python_version()} ({sys.executable})")


# --- 2. Windows long-path check ---------------------------------------------
def long_paths_enabled() -> bool | None:
    """Return True/False on Windows, or None if it can't be determined."""
    if not IS_WINDOWS:
        return None
    try:
        import winreg  # type: ignore

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        )
        value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return bool(value)
    except Exception:
        return None


def enable_long_paths() -> None:
    """Flip the Windows long-path registry flag (needs admin)."""
    if not IS_WINDOWS:
        fail("--enable-long-paths only applies to Windows.")
    try:
        import winreg  # type: ignore

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
            0,
            winreg.KEY_SET_VALUE,
        )
        winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
        ok("Long path support enabled. Restart your PC for it to take full effect.")
    except PermissionError:
        fail(
            "Could not edit the registry (admin required). Re-run this command "
            "in a PowerShell started with 'Run as administrator'."
        )
    except Exception as exc:  # pragma: no cover
        fail(f"Failed to enable long paths: {exc}")


def warn_if_long_path_risk() -> None:
    """On Windows, TensorFlow's deep file tree breaks if paths get too long."""
    if not IS_WINDOWS:
        return
    enabled = long_paths_enabled()
    # The deepest path pip writes is roughly: <venv>/Lib/site-packages/... ~180 chars.
    projected = len(str(VENV_DIR)) + 180
    risky = projected > 255
    if enabled:
        ok("Windows long-path support is enabled.")
        return
    if risky or enabled is False:
        warn("Windows long-path support is OFF and your project path is long.")
        warn(f"   Project: {ROOT}")
        warn("   TensorFlow may fail to install with a 'No such file or directory' error.")
        warn("   Fix it with EITHER of these, then re-run:")
        warn("     1) Move this folder somewhere short, e.g.  C:\\scd")
        warn("     2) Enable long paths (admin PowerShell):")
        warn("        python run.py --enable-long-paths")
        print()


# --- 3 & 4. venv + requirements ---------------------------------------------
def ensure_venv(recreate: bool) -> None:
    if recreate and VENV_DIR.exists():
        log("Removing existing virtualenv (--recreate)...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    if not VENV_DIR.exists():
        log("Creating virtualenv at backend/.venv...")
        if run([sys.executable, "-m", "venv", str(VENV_DIR)]) != 0:
            fail("Failed to create the virtualenv.")
        ok("Virtualenv created.")
    else:
        ok("Virtualenv already exists.")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_requirements() -> None:
    py = venv_python()
    if not py.exists():
        fail(f"venv Python not found at {py}. Try: python run.py --recreate")

    current = file_hash(REQ_FILE)
    stored = REQ_STAMP.read_text().strip() if REQ_STAMP.exists() else ""

    if current == stored:
        ok("Backend requirements already satisfied (unchanged).")
        return

    log("Installing backend requirements (this can take a while for TensorFlow)...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    code = run([str(py), "-m", "pip", "install", "-r", str(REQ_FILE)])
    if code != 0:
        # Do NOT write the stamp on failure, so the next run retries.
        if IS_WINDOWS and not long_paths_enabled():
            err("pip failed. This is almost certainly the Windows long-path limit.")
            err("Move the project to a short path (e.g. C:\\scd) OR run:")
            err("   python run.py --enable-long-paths   (in an admin PowerShell)")
            err("then:  python run.py --recreate")
        fail("Backend requirements installation failed (see output above).")

    REQ_STAMP.write_text(current)
    ok("Backend requirements installed.")


# --- 5. Node.js --------------------------------------------------------------
def ensure_node() -> None:
    log("Checking for Node.js...")
    if have("node") and have("npm"):
        node_v = subprocess.check_output(["node", "--version"], text=True).strip()
        ok(f"Node found: {node_v}")
        return

    warn("Node.js not found. Attempting to install...")
    try:
        if IS_WINDOWS and have("winget"):
            run(["winget", "install", "-e", "--id", "OpenJS.NodeJS.LTS",
                 "--accept-source-agreements", "--accept-package-agreements"])
        elif IS_MAC and have("brew"):
            run(["brew", "install", "node"])
        elif IS_LINUX and have("apt-get"):
            run(["sudo", "apt-get", "update"])
            run(["sudo", "apt-get", "install", "-y", "nodejs", "npm"])
    except Exception:
        pass

    if not have("node"):
        fail(
            "Node.js is required but could not be installed automatically.\n"
            "    Install the LTS version from https://nodejs.org/ (keep 'Add to PATH'),\n"
            "    close and reopen your terminal, then re-run this script."
        )
    ok("Node.js installed.")


# --- 6. Frontend deps --------------------------------------------------------
def ensure_frontend() -> None:
    if (FRONTEND_DIR / "node_modules").exists():
        ok("Frontend dependencies already installed.")
        return
    log("Installing frontend dependencies (npm install)...")
    npm = shutil.which("npm")
    if npm is None:
        fail("npm not found on PATH.")
    if run([npm, "install"], cwd=FRONTEND_DIR) != 0:
        fail("npm install failed (see output above).")
    ok("Frontend dependencies installed.")


# --- 7. Run both servers -----------------------------------------------------
def _popen(cmd, cwd: Path):
    kwargs = {"cwd": str(cwd)}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def _terminate(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_servers(backend_port: int, frontend_port: int) -> None:
    py = venv_python()
    npm = shutil.which("npm")

    log(f"Starting backend on http://localhost:{backend_port} ...")
    backend = _popen(
        [str(py), "-m", "uvicorn", "main:app", "--reload",
         "--host", "0.0.0.0", "--port", str(backend_port)],
        cwd=BACKEND_DIR,
    )
    time.sleep(2)
    if backend.poll() is not None:
        fail("Backend failed to start (see output above).")
    ok(f"Backend running (PID {backend.pid}).")

    log(f"Starting frontend on http://localhost:{frontend_port} ...")
    frontend = _popen([npm, "run", "dev", "--", "--port", str(frontend_port)],
                      cwd=FRONTEND_DIR)

    print()
    ok(f"App is starting. Open http://localhost:{frontend_port} in your browser.")
    print(f"    Backend API + docs: http://localhost:{backend_port}/docs")
    print("    Press Ctrl+C to stop both servers.\n")

    try:
        while True:
            if backend.poll() is not None:
                warn("Backend exited.")
                break
            if frontend.poll() is not None:
                warn("Frontend exited.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
        log("Shutting down...")
    finally:
        _terminate(frontend)
        _terminate(backend)
        ok("Stopped.")


# --- Main --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and run the Skin Cancer Detection app.")
    parser.add_argument("--setup-only", action="store_true", help="Set up but don't start servers.")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild backend/.venv.")
    parser.add_argument("--enable-long-paths", action="store_true",
                        help="(Windows, admin) enable long path support and exit.")
    parser.add_argument("--backend-port", type=int, default=int(os.getenv("BACKEND_PORT", "8000")))
    parser.add_argument("--frontend-port", type=int, default=int(os.getenv("FRONTEND_PORT", "5173")))
    args = parser.parse_args()

    if args.enable_long_paths:
        enable_long_paths()
        return

    log("Skin Cancer Detection - setup & run")
    check_python()
    warn_if_long_path_risk()
    ensure_venv(args.recreate)
    ensure_requirements()
    ensure_node()
    ensure_frontend()

    if args.setup_only:
        print()
        ok("Setup complete. Run 'python run.py' to launch the app.")
        return

    print()
    run_servers(args.backend_port, args.frontend_port)


if __name__ == "__main__":
    main()
