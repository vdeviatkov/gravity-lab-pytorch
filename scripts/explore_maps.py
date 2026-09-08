#!/usr/bin/env python3
"""Find one finishing action sequence per map with Go-Explore search (no neural network).

Reads the curriculum from a training config so demos are searched under exactly the league,
frame skip, ray count, and episode limit the trainer will replay them with. One subprocess per
map (the engine allows one active environment per process), `--workers` of them in parallel,
each under a hard wall-clock timeout because the vendored physics engine can hang inside a
native call on rare states.

    scripts/explore_maps.py --config configs/classic_all_tracks_demo.json --workers 8
    scripts/explore_maps.py --config ... --maps 0:5,1:1 --time-budget 900

Existing demos are skipped unless --force is given. Output: demos/lg<G>_t<T>.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gravity-lab" / "python"))

from gravity_lab_rl import DEFAULT_OBSTACLE_RAY_COUNT  # noqa: E402
from gravity_lab_rl.config import curriculum_environments, load_config  # noqa: E402
from gravity_lab_rl.explore import CellSpec, MapSearch, SearchConfig, demo_filename, verify_demo  # noqa: E402


def archive_path_for(output: Path, rng_seed: int) -> Path:
    # Per search variant (base rng seed), so parallel variants neither clobber each other's
    # archive nor mask each other's liveness heartbeat.
    return output.with_name(f"{output.stem}.archive_{rng_seed % 1000}.json")


def search_one(env_cfg: dict, seed: int, output: Path, time_budget: float, rng_seed: int,
               search: SearchConfig) -> int:
    from gravity_lab import ClassicConfig, ClassicGravityEnv

    classic = ClassicConfig(env_cfg["level_group"], env_cfg["track"], env_cfg["league"],
                            env_cfg["frame_skip"], env_cfg["max_episode_steps"], seed,
                            env_cfg.get("obstacle_ray_count", DEFAULT_OBSTACLE_RAY_COUNT))
    label = f"{env_cfg['level_group']}:{env_cfg['track']}"
    with ClassicGravityEnv(classic, env_cfg.get("level_pack")) as env:
        name = env.track_name
        started = time.monotonic()

        def report(s: MapSearch) -> None:
            elapsed = time.monotonic() - started
            print(f"[{label} {name}] t={elapsed:5.0f}s iter={s.iterations} cells={len(s.archive)} "
                  f"best_progress={s.best_progress:.3f} steps/s={s.steps / max(elapsed, 1e-9):.0f}", flush=True)

        searcher = MapSearch(env, env_cfg, seed, search, rng_seed)
        archive = archive_path_for(output, rng_seed)
        if archive.exists():
            loaded = searcher.load_archive(archive)
            print(f"[{label} {name}] resumed archive: {loaded} cells, best_progress={searcher.best_progress:.3f}", flush=True)
        demo = searcher.run(time_budget, progress_callback=report, archive_path=archive)
        if demo is None:
            print(f"[{label} {name}] NOT SOLVED in {time_budget:.0f}s: iter={searcher.iterations} "
                  f"cells={len(searcher.archive)} best_progress={searcher.best_progress:.3f}", flush=True)
            return 1
        if not verify_demo(env, demo):
            print(f"[{label} {name}] demo failed verification replay", flush=True)
            return 2
        demo.save(output)
        archive.unlink(missing_ok=True)
        print(f"[{label} {name}] SOLVED: {len(demo.actions)} steps, {demo.iterations} iterations, "
              f"{demo.search_seconds:.0f}s -> {output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "classic_all_tracks_demo.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "demos"))
    parser.add_argument("--maps", help="comma-separated group:track pairs; default: every curriculum map")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-budget", type=float, default=600.0, help="search seconds per map")
    parser.add_argument("--force", action="store_true", help="re-search maps that already have a demo")
    parser.add_argument("--explore-steps", type=int, default=SearchConfig.explore_steps)
    parser.add_argument("--hold-mean", type=float, default=SearchConfig.hold_mean)
    parser.add_argument("--hold-choices", default="", help="comma-separated hold means drawn per iteration, e.g. 3,8,20")
    parser.add_argument("--stall-timeout", type=float, default=90.0,
                        help="seconds without archive heartbeat before a worker is killed and resumed (engine hang)")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--progress-bin", type=float, default=CellSpec.progress_bin,
                        help="cell width in progress units; smaller = more archive diversity, slower")
    parser.add_argument("--single", help=argparse.SUPPRESS)  # internal: run one map in this process
    args = parser.parse_args()

    config = load_config(args.config)
    environments = curriculum_environments(config)
    seed = int(config["seeds"]["environment"])
    output_dir = Path(args.output_dir)
    hold_choices = tuple(float(v) for v in args.hold_choices.split(",") if v)
    search = SearchConfig(explore_steps=args.explore_steps, hold_mean=args.hold_mean, hold_choices=hold_choices,
                          cells=CellSpec(progress_bin=args.progress_bin))

    if args.single:
        group, track = (int(v) for v in args.single.split(":"))
        env_cfg = next(e for e in environments if e["level_group"] == group and e["track"] == track)
        return search_one(env_cfg, seed, output_dir / demo_filename(group, track), args.time_budget,
                          args.rng_seed, search)

    wanted = None
    if args.maps:
        wanted = {tuple(int(v) for v in pair.split(":")) for pair in args.maps.split(",")}
    pending = []
    for env_cfg in environments:
        pair = (env_cfg["level_group"], env_cfg["track"])
        if wanted is not None and pair not in wanted:
            continue
        if not args.force and (output_dir / demo_filename(*pair)).exists():
            print(f"[{pair[0]}:{pair[1]}] demo exists, skipping (use --force to redo)")
            continue
        pending.append(pair)
    if not pending:
        print("nothing to do")
        return 0

    # Per map: the worker process, when it started, how much search budget the map has used across
    # restarts, and the restart count. A worker whose archive heartbeat stops (native engine hang)
    # is killed and resumed from its archive with a fresh rng seed until the budget is spent.
    running: dict[tuple[int, int], dict] = {}
    used: dict[tuple[int, int], float] = {pair: 0.0 for pair in pending}
    restarts: dict[tuple[int, int], int] = {pair: 0 for pair in pending}
    results: dict[tuple[int, int], int] = {}

    def spawn(pair: tuple[int, int]) -> None:
        remaining = args.time_budget - used[pair]
        command = [sys.executable, __file__, "--config", args.config, "--output-dir", str(output_dir),
                   "--single", f"{pair[0]}:{pair[1]}", "--time-budget", str(remaining),
                   "--explore-steps", str(args.explore_steps), "--hold-mean", str(args.hold_mean),
                   "--hold-choices", args.hold_choices, "--progress-bin", str(args.progress_bin),
                   "--rng-seed", str(args.rng_seed + 1000 * restarts[pair])]
        running[pair] = {"process": subprocess.Popen(command), "started": time.monotonic()}

    def heartbeat_age(pair: tuple[int, int], started: float) -> float:
        archive = archive_path_for(output_dir / demo_filename(*pair), args.rng_seed)
        if archive.exists():
            return time.time() - archive.stat().st_mtime
        return time.monotonic() - started

    while pending or running:
        while pending and len(running) < args.workers:
            spawn(pending.pop(0))
        for pair, state in list(running.items()):
            process, started = state["process"], state["started"]
            code = process.poll()
            if code is None and heartbeat_age(pair, started) > args.stall_timeout:
                used[pair] += time.monotonic() - started
                process.kill()
                process.wait()
                del running[pair]
                if (output_dir / demo_filename(*pair)).exists():
                    results[pair] = 0
                elif used[pair] < args.time_budget:
                    restarts[pair] += 1
                    print(f"[{pair[0]}:{pair[1]}] worker stalled (engine hang); restart {restarts[pair]} "
                          f"from archive, {args.time_budget - used[pair]:.0f}s budget left", flush=True)
                    spawn(pair)
                else:
                    print(f"[{pair[0]}:{pair[1]}] budget exhausted after {restarts[pair]} restarts", flush=True)
                    results[pair] = 3
                continue
            if code is not None:
                results[pair] = code
                del running[pair]
        time.sleep(0.5)

    solved = sorted(p for p, c in results.items() if c == 0)
    unsolved = sorted(p for p, c in results.items() if c != 0)
    print(f"\nsolved {len(solved)}/{len(results)}: {' '.join(f'{g}:{t}' for g, t in solved)}")
    if unsolved:
        print(f"unsolved: {' '.join(f'{g}:{t}' for g, t in unsolved)}")
    (output_dir / "search_summary.json").write_text(json.dumps(
        {"solved": solved, "unsolved": unsolved, "time_budget": args.time_budget}, indent=1) + "\n")
    return 0 if not unsolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
