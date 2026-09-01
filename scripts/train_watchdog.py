#!/usr/bin/env python3
"""Run `gravity-lab-rl resume` under a stall watchdog.

The vendored classic physics engine can occasionally hang inside its native constraint
solver on certain physics states (observed on the "Hole" track during an all-tracks run).
Because the hang is inside a synchronous C++ call, the trainer's own SIGINT/SIGTERM
handling cannot interrupt it. This wrapper detects a stalled subprocess from the outside
(no control-file update for --stall-timeout seconds) and kills and resumes it, nudging
`completed_episode_count` so a resume does not immediately re-enter the same curriculum
seed. `--duration-seconds` is the cumulative active-training target already used by
`gravity-lab-rl resume`, so restarts do not add extra time on top of what is done.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gravity_lab_rl.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from gravity_lab_rl.control import resolve_run  # noqa: E402


def _active_seconds(checkpoint_path: Path) -> float:
    return float(load_checkpoint(checkpoint_path)["active_training_duration_seconds"])


def _nudge_past_stall(checkpoint_path: Path) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint["completed_episode_count"] = int(checkpoint["completed_episode_count"]) + 1
    save_checkpoint(checkpoint_path, checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True,
                        help="cumulative active-training target, same meaning as `resume`'s flag")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--stall-timeout", type=float, default=45.0,
                        help="seconds without a control-file update before declaring a stall")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-restarts", type=int, default=20)
    args = parser.parse_args()

    run_dir = resolve_run(args.run_id, latest=False)
    checkpoint_path = run_dir / "latest.pt"
    control_path = run_dir / "control.json"
    gravity_lab_rl = str(ROOT / ".venv" / "bin" / "gravity-lab-rl")

    restarts = 0
    while True:
        active = _active_seconds(checkpoint_path)
        if active >= args.duration_seconds:
            print(f"target reached: {active:.1f}s / {args.duration_seconds:.1f}s active training")
            return 0
        print(f"launching resume ({active:.1f}s / {args.duration_seconds:.1f}s so far, "
              f"restart {restarts}/{args.max_restarts})")
        process = subprocess.Popen([
            gravity_lab_rl, "resume", "--run-id", args.run_id,
            "--duration-seconds", str(args.duration_seconds), "--device", args.device,
        ])
        stalled = False
        last_seen_mtime = control_path.stat().st_mtime if control_path.is_file() else time.time()
        last_change = time.monotonic()
        while process.poll() is None:
            time.sleep(args.poll_interval)
            mtime = control_path.stat().st_mtime if control_path.is_file() else last_seen_mtime
            if mtime != last_seen_mtime:
                last_seen_mtime = mtime
                last_change = time.monotonic()
            elif time.monotonic() - last_change > args.stall_timeout:
                stalled = True
                break
        if stalled:
            print(f"stall detected (no control-file update for {args.stall_timeout:.0f}s); "
                  "killing and resuming past it")
            process.kill()
            process.wait()
            restarts += 1
            if restarts > args.max_restarts:
                print(f"giving up after {restarts} stalls", file=sys.stderr)
                return 1
            _nudge_past_stall(checkpoint_path)
            continue
        code = process.returncode
        if code != 0:
            print(f"resume exited with code {code}; not a stall, stopping", file=sys.stderr)
            return code
        # A clean exit before reaching the target duration should not happen in practice,
        # but loop back and let the active-seconds check above decide whether to continue.


if __name__ == "__main__":
    raise SystemExit(main())
