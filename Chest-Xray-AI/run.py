#!/usr/bin/env python3
"""One-command launcher for Windows / macOS / Linux (standard library only).

    python run.py              start backend + website, wait until healthy, open the browser
    python run.py --dev        development mode: Vite dev server (5173) + API (8000)
    python run.py --setup      create .venv and install backend requirements (explicit opt-in)
    python run.py --stop       stop processes started by this script
Press Ctrl+C to stop everything cleanly.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND, FRONTEND = ROOT / "backend", ROOT / "frontend"
RUN_DIR = ROOT / ".run"
PID_FILE = RUN_DIR / "pids.json"
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"
REQUIRED = {"fastapi": "fastapi", "uvicorn": "uvicorn", "multipart": "python-multipart", "dotenv": "python-dotenv",
            "tensorflow": "tensorflow-cpu", "tf_keras": "tf-keras", "numpy": "numpy", "cv2": "opencv-python-headless",
            "PIL": "Pillow", "reportlab": "reportlab"}


def say(msg: str = "") -> None:
    print(msg, flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0 if IS_WIN else 1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def kill_tree(pid: int) -> None:
    try:
        if IS_WIN:
            subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def stop_recorded() -> int:
    if not PID_FILE.exists():
        say("Nothing to stop (no processes recorded).")
        return 0
    for name, pid in json.loads(PID_FILE.read_text()).items():
        say(f"Stopping {name} (pid {pid})")
        kill_tree(int(pid))
    PID_FILE.unlink(missing_ok=True)
    return 0


def missing_modules() -> list[str]:
    return [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]


def pip_install() -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(BACKEND / "requirements.txt")]
    say("Running: " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        sys.exit("pip install failed. Fix the error above (Python 3.10-3.12 is recommended for TensorFlow) and retry.")


def setup_venv() -> None:
    if not VENV.exists():
        say(f"Creating virtual environment at {VENV} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    cmd = [str(venv_python()), "-m", "pip", "install", "-r", str(BACKEND / "requirements.txt")]
    say("Installing backend requirements into .venv:\n  " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        sys.exit("pip install failed. See the message above.")
    say("Setup complete. Now run:  python run.py")


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 240) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            sys.exit(f"The backend stopped unexpectedly (exit code {proc.returncode}). See messages above.")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(1)
    sys.exit("The backend did not become healthy within %d s." % timeout)


def spawn(cmd: list[str], cwd: Path, env: dict) -> subprocess.Popen:
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WIN else {"start_new_session": True}
    return subprocess.Popen(cmd, cwd=cwd, env=env, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dev", action="store_true"); ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--setup", action="store_true"); ap.add_argument("--stop", action="store_true")
    ap.add_argument("--install", action="store_true", help="install missing Python packages without asking")
    ap.add_argument("--no-browser", action="store_true"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    if a.stop:
        return stop_recorded()
    if a.setup:
        setup_venv()
        return 0
    if VENV.exists() and not in_venv():                     # transparently use the project's venv
        return subprocess.call([str(venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]])

    say("== CXR-AI launcher ==")
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        say(f"WARNING: Python {sys.version.split()[0]} detected; TensorFlow supports 3.10-3.13 (3.11/3.12 recommended).")

    miss = missing_modules()
    if miss:
        say("Missing Python packages: " + ", ".join(miss))
        say(f"One-line fix:  {sys.executable} -m pip install -r backend/requirements.txt   (or:  python run.py --setup)")
        if a.install or (sys.stdin.isatty() and input("Install them now? [y/N] ").strip().lower() == "y"):
            pip_install()
            miss = missing_modules()
        if miss:
            return 1
    say("[ok] Python packages present")

    mdir = Path(os.environ.get("MODEL_DIR", BACKEND / "models")).expanduser()
    cfg_file, weights = mdir / "model_config.json", mdir / "final_model.h5"
    have_model = cfg_file.is_file() and weights.is_file()
    say("[ok] Model artifacts found" if have_model else
        "[!!] Model artifacts NOT found in backend/models/ (model_config.json, final_model.h5).\n"
        "     The website will start, but predictions stay disabled until you train and export a model:\n"
        "     see README.md > 'Train the model'.")

    api_port, web_port = a.port, 5173
    ports = [api_port] + ([web_port] if a.dev else [])
    for p in ports:
        if not port_free(p):
            say(f"Port {p} is already in use. Close the other program, run 'python run.py --stop', or pick another port with --port.")
            return 1
    say("[ok] Ports free: " + ", ".join(map(str, ports)))

    dist = FRONTEND / "dist" / "index.html"
    if a.dev or a.rebuild or not dist.is_file():
        npm = shutil.which("npm.cmd" if IS_WIN else "npm")
        if not npm:
            if not a.dev and dist.is_file():
                pass
            else:
                say("Node.js/npm is required to build the website (https://nodejs.org, LTS) and was not found.")
                return 1
        if npm and not (FRONTEND / "node_modules").is_dir():
            say("Installing frontend packages (npm install) - one-time ...")
            if subprocess.call([npm, "install", "--no-audit", "--no-fund"], cwd=FRONTEND) != 0:
                return 1
        if npm and not a.dev:
            say("Building the website (npm run build) ...")
            if subprocess.call([npm, "run", "build"], cwd=FRONTEND) != 0:
                return 1

    env = dict(os.environ, TF_USE_LEGACY_KERAS="1", PYTHONUNBUFFERED="1")
    if a.dev:
        env["FRONTEND_URL"] = f"http://localhost:{web_port},http://127.0.0.1:{web_port}"
    procs: dict[str, subprocess.Popen] = {}
    try:
        say("Starting backend (loading the model can take up to a minute) ...")
        procs["backend"] = spawn([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(api_port)], BACKEND, env)
        if a.dev:
            npm = shutil.which("npm.cmd" if IS_WIN else "npm")
            procs["frontend"] = spawn([npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(web_port), "--strictPort"], FRONTEND, env)
        RUN_DIR.mkdir(exist_ok=True)
        PID_FILE.write_text(json.dumps({k: p.pid for k, p in procs.items()}))

        health = wait_healthy(f"http://127.0.0.1:{api_port}/api/health", procs["backend"])
        site = f"http://127.0.0.1:{web_port if a.dev else api_port}"
        if a.dev:
            wait_healthy(site, procs["frontend"], 90)
        say("")
        if health.get("model_loaded"):
            say(f"READY - open {site}   (trained model loaded; Ctrl+C to stop)")
        else:
            say(f"READY (model NOT loaded - predictions disabled) - open {site}\n  Reason: {health.get('detail')}")
        if not a.no_browser:
            webbrowser.open(site)
        while all(p.poll() is None for p in procs.values()):
            time.sleep(1)
        say("A process exited; shutting down.")
        return 1
    except KeyboardInterrupt:
        say("\nStopping ...")
        return 0
    finally:
        for p in procs.values():
            kill_tree(p.pid)
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
