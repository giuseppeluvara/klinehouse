#!/usr/bin/env python3
"""Manage only the dedicated, time-bounded macOS shadow pilot service."""

import argparse
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def service(run_dir):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())
    label = "com.klinehouse.shadow." + str(config["start_at"])
    path = Path.home() / "Library/LaunchAgents" / (label + ".plist")
    program = run_dir / "runtime/scripts/run_shadow.py"
    python = ROOT / ".venv-timestral/bin/python"
    if not program.is_file() or not python.exists():
        raise ValueError("frozen runtime and existing Python environment required")
    settings = {
        "Label": label,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            "-s",
            str(python),
            str(program),
            "run",
            "--run-dir",
            str(run_dir),
        ],
        "WorkingDirectory": str(run_dir),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 60,
        "ProcessType": "Background",
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        },
        "StandardOutPath": str(run_dir / "stdout.log"),
        "StandardErrorPath": str(run_dir / "stderr.log"),
    }
    return config, label, path, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "stop"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("macOS LaunchAgent only")
    config, label, path, settings = service(args.run_dir)
    domain = f"gui/{os.getuid()}"
    if args.command == "start":
        if time.time() >= config["end_at"]:
            parser.error("cannot start an expired pilot")
        if path.exists():
            if plistlib.loads(path.read_bytes()) != settings:
                parser.error("existing service differs; refusing overwrite")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                plistlib.dump(settings, handle)
            path.chmod(0o600)
        present = subprocess.run(
            ["launchctl", "print", f"{domain}/{label}"], capture_output=True
        )
        subprocess.run(["launchctl", "enable", f"{domain}/{label}"], check=True)
        if present.returncode:
            subprocess.run(["launchctl", "bootstrap", domain, str(path)], check=True)
        print(
            json.dumps(
                {
                    "label": label,
                    "plist": str(path),
                    "start_utc": config["start_utc"],
                    "end_utc": config["end_utc"],
                    "run_dir": str(args.run_dir.resolve()),
                }
            )
        )
    elif args.command == "stop":
        subprocess.run(["launchctl", "disable", f"{domain}/{label}"], check=True)
        subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], check=True)
        print(json.dumps({"stopped": label, "evidence_preserved": True}))
    else:
        subprocess.run(["launchctl", "print", f"{domain}/{label}"], check=True)


if __name__ == "__main__":
    main()
