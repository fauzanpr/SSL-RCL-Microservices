"""Helper bersama untuk launch_ex1.py dan check_ex1.py (status file, cek proses, matikan proses)."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
IS_WIN = os.name == "nt"


def status_path(model: str, exp: str = "ex1") -> Path:
    return LOG_DIR / f"{exp}_{model}_status.json"


def log_path(source: str, model: str, exp: str = "ex1") -> Path:
    return LOG_DIR / f"{exp}_{source}_{model}.log"


def read_status(model: str, exp: str = "ex1"):
    p = status_path(model, exp)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_status(model: str, st: dict, exp: str = "ex1"):
    LOG_DIR.mkdir(exist_ok=True)
    tmp = status_path(model, exp).with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    os.replace(tmp, status_path(model, exp))          # atomik, supaya check_ex1.py tidak membaca file setengah jadi


def pid_alive(pid) -> bool:
    if not pid:
        return False
    if IS_WIN:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out and "python" in out.lower()
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_tree(pid):
    if not pid:
        return
    if IS_WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass
