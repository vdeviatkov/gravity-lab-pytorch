from __future__ import annotations

import os
import platform
import subprocess
import json
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .control import resolve_run
from .export import export_checkpoint


DEFAULT_GAME_REPO = Path(__file__).resolve().parents[2] / "gravity-lab"
DEFAULT_BUNDLED_POLICY = Path(__file__).resolve().parents[2] / "policies" / "classic_intro.gdp"


def game_repo() -> Path:
    return Path(os.environ.get("GRAVITY_LAB_REPO", DEFAULT_GAME_REPO)).expanduser().resolve()


def integration_paths() -> tuple[Path, Path, Path]:
    root = game_repo()
    package = root / "python" / "gravity_lab"
    system = platform.system()
    library_name = {"Darwin": "libgravity_lab_classic.dylib",
                    "Windows": "gravity_lab_classic.dll"}.get(system, "libgravity_lab_classic.so")
    executable_name = "gravity_lab_classic_viewer.exe" if system == "Windows" else "gravity_lab_classic_viewer"
    build = root / "build-classic-rl"
    library_override = os.environ.get("GRAVITY_LAB_CLASSIC_LIBRARY")
    library_candidates = ([Path(library_override).expanduser()] if library_override else []) + [
        build / library_name, build / "lib" / library_name,
        build / "Release" / library_name, build / "Debug" / library_name,
    ]
    viewer_candidates = [build / executable_name, build / "Release" / executable_name,
                         build / "Debug" / executable_name]
    library = next((path.resolve() for path in library_candidates if path.is_file()),
                   library_candidates[0].resolve())
    viewer = next((path.resolve() for path in viewer_candidates if path.is_file()),
                  viewer_candidates[0].resolve())
    return package, library, viewer


def require_integration(require_viewer: bool = True) -> tuple[Path, Path, Path]:
    package, library, viewer = integration_paths()
    missing = []
    if not package.is_dir(): missing.append(f"Python bindings: {package}")
    if not library.is_file(): missing.append(f"native library: {library}")
    if require_viewer and not viewer.is_file(): missing.append(f"graphical viewer: {viewer}")
    if missing:
        raise RuntimeError("Gravity Lab integration is incomplete. Missing:\n  " + "\n  ".join(missing) +
                           "\nInitialize the submodule and run scripts/bootstrap.sh. "
                           "GRAVITY_LAB_REPO and GRAVITY_LAB_CLASSIC_LIBRARY are optional overrides.")
    os.environ.setdefault("GRAVITY_LAB_CLASSIC_LIBRARY", str(library))
    return package, library, viewer


def arcade_executable() -> Path:
    candidate = integration_paths()[2].parent / ("gravity_lab_ai_arcade.exe" if platform.system() == "Windows"
                                                  else "gravity_lab_ai_arcade")
    if not candidate.is_file():
        raise RuntimeError(f"AI arcade executable is missing: {candidate}\nRun scripts/bootstrap.sh to build it.")
    return candidate


def bundled_policy() -> Path:
    if not DEFAULT_BUNDLED_POLICY.is_file():
        raise FileNotFoundError(f"bundled policy is missing: {DEFAULT_BUNDLED_POLICY}")
    return DEFAULT_BUNDLED_POLICY


def _policy_settings(path: str | Path) -> tuple[Path, dict[str, Any], int]:
    from gravity_lab import DenseQPolicy

    policy_path = Path(path).expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"policy not found: {policy_path}")
    loaded = DenseQPolicy.load(policy_path)
    if loaded.environment_id != "gravity-lab-classic-v1" or loaded.observation_size != 28 or loaded.action_count != 9:
        raise ValueError("policy must target gravity-lab-classic-v1 with 28 observations and 9 actions")
    environment: dict[str, Any] = {
        "level_group": 0, "track": 0, "league": 0, "frame_skip": 2,
        "max_episode_steps": 2000, "level_pack": None,
    }
    seed = 2_000_007
    sidecar = policy_path.with_suffix(policy_path.suffix + ".json")
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        recorded = metadata.get("training_environment") or metadata.get("configuration", {}).get("environment", {})
        environment.update({key: recorded[key] for key in environment if key in recorded})
        seed = int(metadata.get("configuration", {}).get("seeds", {}).get("final_evaluation", seed))
    return policy_path, environment, seed


def _check_source(run_id: str | None, checkpoint: str | Path | None,
                  policy: str | Path | None) -> None:
    if sum(value is not None for value in (run_id, checkpoint, policy)) > 1:
        raise ValueError("choose only one of run_id, checkpoint, or policy")


def play(run_id: str | None = None, checkpoint: str | Path | None = None,
         policy: str | Path | None = None, episodes: int = 5,
         group: int | None = None, track: int | None = None, league: int | None = None,
         fps: int | None = None, seed: int | None = None, validate_only: bool = False,
         wait: bool = True) -> subprocess.Popen[bytes] | subprocess.CompletedProcess[bytes]:
    _check_source(run_id, checkpoint, policy)
    _, _, viewer = require_integration()
    saved = None
    policy_seed = 2_000_007
    if policy is not None:
        policy_path, env, policy_seed = _policy_settings(policy)
    else:
        try:
            run = resolve_run(run_id, latest=run_id is None) if checkpoint is None else Path(checkpoint).resolve().parent
        except FileNotFoundError:
            if run_id is not None or checkpoint is not None:
                raise
            policy_path, env, policy_seed = _policy_settings(bundled_policy())
        else:
            checkpoint_path = Path(checkpoint) if checkpoint else run / "latest.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            policy_path = run / ("final.gdp" if checkpoint_path.name == "final.pt" else "latest.gdp")
            export_checkpoint(checkpoint_path, policy_path)
            saved = load_checkpoint(checkpoint_path)
            env = saved["config"]["environment"]
            policy_seed = int(saved["config"]["seeds"]["final_evaluation"])
    args = [str(viewer), "--policy", str(policy_path)]
    if validate_only:
        args.append("--validate-only")
    else:
        actual_fps = fps if fps is not None else (25 if int(env["frame_skip"]) == 2 else 50)
        args += ["--group", str(env["level_group"] if group is None else group),
                 "--track", str(env["track"] if track is None else track),
                 "--league", str(env["league"] if league is None else league),
                 "--frame-skip", str(env["frame_skip"]), "--max-steps", str(env["max_episode_steps"]),
                 "--episodes", str(episodes), "--fps", str(actual_fps),
                 "--seed", str(policy_seed if seed is None else seed)]
        if env.get("level_pack"):
            args += ["--level-pack", str(env["level_pack"])]
    return subprocess.run(args, check=True) if wait else subprocess.Popen(args)


def arcade(run_id: str | None = None, checkpoint: str | Path | None = None,
           policy: str | Path | None = None,
           seed: int | None = None, wait: bool = True) -> subprocess.Popen[bytes] | subprocess.CompletedProcess[bytes]:
    _check_source(run_id, checkpoint, policy)
    require_integration()
    executable = arcade_executable()
    saved = None
    policy_seed = 2_000_007
    if policy is not None:
        policy_path, env, policy_seed = _policy_settings(policy)
    else:
        try:
            run = resolve_run(run_id, latest=run_id is None) if checkpoint is None else Path(checkpoint).resolve().parent
        except FileNotFoundError:
            if run_id is not None or checkpoint is not None:
                raise
            policy_path, env, policy_seed = _policy_settings(bundled_policy())
        else:
            checkpoint_path = Path(checkpoint) if checkpoint else run / "latest.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            policy_path = run / ("final.gdp" if checkpoint_path.name == "final.pt" else "latest.gdp")
            export_checkpoint(checkpoint_path, policy_path)
            saved = load_checkpoint(checkpoint_path)
            env = saved["config"]["environment"]
            policy_seed = int(saved["config"]["seeds"]["final_evaluation"])
    args = [str(executable), "--policy", str(policy_path),
            "--frame-skip", str(env["frame_skip"]), "--max-steps", str(env["max_episode_steps"]),
            "--seed", str(policy_seed if seed is None else seed)]
    return subprocess.run(args, check=True) if wait else subprocess.Popen(args)
