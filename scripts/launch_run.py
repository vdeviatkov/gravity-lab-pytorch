#!/usr/bin/env python3
"""Create a run directory from a config and start it under the stall watchdog, detached.

    scripts/launch_run.py --config configs/classic_all_tracks_demo.json --run-id my_run --duration-seconds 14400

The trainer is constructed once (this also builds the demonstration replay and derives the
observation normalization when the config asks for it) and checkpointed without training, so
`train_watchdog.py` can drive the whole run through `resume`, restarting past native-engine
stalls. `caffeinate` keeps macOS awake for the run's duration where available.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gravity-lab" / "python"))

from gravity_lab_rl.cli import _trainer_class  # noqa: E402
from gravity_lab_rl.config import configured, load_config  # noqa: E402
from gravity_lab_rl.control import atomic_write_json  # noqa: E402
from gravity_lab_rl.trainer import make_run_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stall-timeout", type=float, default=180.0)
    parser.add_argument("--max-restarts", type=int, default=20)
    args = parser.parse_args()

    config = configured(load_config(args.config), duration_seconds=args.duration_seconds, device=args.device)
    run_id = args.run_id or make_run_id()
    run_dir = ROOT / "artifacts" / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    trainer = _trainer_class(config)(config, run_dir)
    trainer.save(export=False)
    command = [sys.executable, "-u", str(ROOT / "scripts" / "train_watchdog.py"), "--run-id", run_id,
               "--duration-seconds", str(args.duration_seconds), "--device", args.device,
               "--stall-timeout", str(args.stall_timeout), "--max-restarts", str(args.max_restarts)]
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-i", *command]
    log = (run_dir / "trainer.log").open("ab")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), start_new_session=True)
    atomic_write_json(run_dir / "launch.json", {"run_id": run_id, "run_dir": str(run_dir), "launcher_pid": process.pid,
                                                "command": command, "launched_at": time.time()})
    print(json.dumps({"run_id": run_id, "watchdog_pid": process.pid, "log": str(run_dir / "trainer.log")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
