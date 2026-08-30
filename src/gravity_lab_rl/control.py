from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMMANDS = {"run", "pause", "stop"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def read_control(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"requested": "run", "state": "starting"}


def initialize_control(path: str | Path, run_id: str) -> None:
    atomic_write_json(path, {
        "format": "gravity-lab-rl-control-v1", "run_id": run_id,
        "requested": "run", "state": "starting", "updated_at": utc_now(),
    })


def request_control(path: str | Path, command: str) -> dict[str, Any]:
    requested = "run" if command == "resume" else command
    if requested not in COMMANDS:
        raise ValueError(f"unknown control command: {command}")
    data = read_control(path)
    data["requested"] = requested
    data["requested_at"] = utc_now()
    atomic_write_json(path, data)
    return data


def update_status(path: str | Path, status: dict[str, Any]) -> dict[str, Any]:
    current = read_control(path)
    current.update(status)
    current["updated_at"] = utc_now()
    atomic_write_json(path, current)
    return current


def artifacts_root() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts"


def resolve_run(run_id: str | None = None, latest: bool = False) -> Path:
    root = artifacts_root()
    if run_id and latest:
        raise ValueError("choose either --run-id or --latest")
    if run_id:
        run = root / run_id
        if not run.is_dir():
            raise FileNotFoundError(f"run not found: {run}")
        return run
    runs = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no runs found under {root}")
    return max(runs, key=lambda path: path.stat().st_mtime_ns)
