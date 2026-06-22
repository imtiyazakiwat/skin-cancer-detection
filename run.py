#!/usr/bin/env python3
"""
run.py - cross-platform setup + run script for the Skin Cancer Detection app.

Works on Windows, macOS, and Linux. Only needs Python (no Node.js required -
the UI is served by Flask as plain HTML/CSS).

What it does:
  1. Checks the Python version.
  2. (Windows) Detects the long-path limitation that breaks TensorFlow installs
     and tells you how to fix it.
  3. Creates a virtualenv at backend/.venv (or recreates it with --recreate).
  4. Installs backend requirements, only when requirements.txt has changed.
  5. Starts the Flask app (http://localhost:8000) and stops it cleanly on Ctrl+C.

Usage (from anywhere):
  python run.py                 # setup everything and run the app
  python run.py --setup-only    # setup only, don't start the server
  python run.py --recreate      # delete and rebuild backend/.venv first
  python run.py --port 8001
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


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(cmd, cwd: Path | None = None, env=None) -> int:
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None, env=env)


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
def long_paths_enabled() -> "bool | None":
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
    if not IS_WINDOWS:
        return
    enabled = long_paths_enabled()
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

    log("Installing requirements (this can take a while for TensorFlow)...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    code = run([str(py), "-m", "pip", "install", "-r", str(REQ_FILE)])
    if code != 0:
        if IS_WINDOWS and not long_paths_enabled():
            err("pip failed. This is almost certainly the Windows long-path limit.")
            err("Move the project to a short path (e.g. C:\\scd) OR run:")
            err("   python run.py --enable-long-paths   (in an admin PowerShell)")
            err("then:  python run.py --recreate")
        fail("Requirements installation failed (see output above).")

    REQ_STAMP.write_text(current)
    ok("Requirements installed.")


# --- 5. Run the Flask app ----------------------------------------------------
def _popen(cmd, cwd: Path, env=None):
    kwargs = {"cwd": str(cwd), "env": env}
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


def run_server(port: int) -> None:
    py = venv_python()
    env = os.environ.copy()
    env["PORT"] = str(port)

    log(f"Starting the app on http://localhost:{port} ...")
    print()
    ok(f"App is starting. Open http://localhost:{port} in your browser.")
    print("    Press Ctrl+C to stop.\n")

    proc = _popen([str(py), "app.py"], cwd=BACKEND_DIR, env=env)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print()
        log("Shutting down...")
    finally:
        _terminate(proc)
        ok("Stopped.")


# --- Main --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and run the Skin Cancer Detection app.")
    parser.add_argument("--setup-only", action="store_true", help="Set up but don't start the server.")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild backend/.venv.")
    parser.add_argument("--enable-long-paths", action="store_true",
                        help="(Windows, admin) enable long path support and exit.")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()

    if args.enable_long_paths:
        enable_long_paths()
        return

    log("Skin Cancer Detection - setup & run")
    check_python()
    warn_if_long_path_risk()
    ensure_venv(args.recreate)
    ensure_requirements()

    if args.setup_only:
        print()
        ok("Setup complete. Run 'python run.py' to launch the app.")
        return

    print()
    run_server(args.port)


if __name__ == "__main__":
    main()
