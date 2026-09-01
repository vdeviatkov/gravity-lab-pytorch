from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint
from .config import configured, load_config
from .control import read_control, request_control, resolve_run
from .evaluation import evaluate_model
from .export import export_checkpoint
from .model import DenseQNetwork, select_device
from .playback import arcade, play
from .trainer import Trainer, make_run_id


def _run_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id")
    parser.add_argument("--latest", action="store_true")


def _print_summary(summary: dict[str, Any]) -> None:
    evaluation = summary["final_evaluation"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Run: {summary['run_id']}")
    print(f"Final checkpoint: {summary['paths']['final_checkpoint']}")
    print(f"Final policy: {summary['paths']['final_policy']}")
    print("Final evaluation: mean reward={:.6g}, mean progress={:.6g}, finish={:.1%}, crash={:.1%}, truncation={:.1%}".format(
        evaluation["mean_reward"], evaluation["mean_progress"], evaluation["finish_rate"],
        evaluation["crash_rate"], evaluation["truncation_rate"]))
    best = summary.get("best_evaluation")
    if best is not None:
        print(f"Best checkpoint: {summary['paths']['best_checkpoint']}")
        print(f"Best policy: {summary['paths']['best_policy']}")
        print("Best evaluation: mean reward={:.6g}, mean progress={:.6g}, finish={:.1%}, crash={:.1%}, truncation={:.1%}".format(
            best["mean_reward"], best["mean_progress"], best["finish_rate"],
            best["crash_rate"], best["truncation_rate"]))
    print(f"Resume: gravity-lab-rl resume --run-id {summary['run_id']}")
    print(f"Play: ./scripts/play_latest.sh --run-id {summary['run_id']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gravity-lab-rl")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/classic_intro.json"))
    train.add_argument("--duration-seconds", type=float)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    train.add_argument("--run-id")
    train.add_argument("--initialize-policy", help="start from a compatible portable .gdp policy")
    resume = sub.add_parser("resume")
    _run_arg(resume)
    resume.add_argument("--duration-seconds", type=float)
    resume.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    evaluate = sub.add_parser("evaluate")
    _run_arg(evaluate)
    evaluate.add_argument("--episodes", type=int)
    evaluate.add_argument("--seed", type=int)
    export = sub.add_parser("export")
    _run_arg(export)
    export.add_argument("--checkpoint")
    export.add_argument("--output")
    playback = sub.add_parser("play")
    playback_source = playback.add_mutually_exclusive_group()
    playback_source.add_argument("--run-id")
    playback_source.add_argument("--latest", action="store_true")
    playback_source.add_argument("--checkpoint")
    playback_source.add_argument("--policy")
    playback.add_argument("--episodes", type=int, default=5)
    playback.add_argument("--group", type=int); playback.add_argument("--track", type=int)
    playback.add_argument("--league", type=int); playback.add_argument("--fps", type=int)
    playback.add_argument("--seed", type=int); playback.add_argument("--validate-only", action="store_true")
    ai_arcade = sub.add_parser("arcade")
    arcade_source = ai_arcade.add_mutually_exclusive_group()
    arcade_source.add_argument("--run-id")
    arcade_source.add_argument("--latest", action="store_true")
    arcade_source.add_argument("--checkpoint")
    arcade_source.add_argument("--policy")
    ai_arcade.add_argument("--seed", type=int)
    control = sub.add_parser("control")
    control.add_argument("action", choices=("status", "pause", "resume", "stop"))
    _run_arg(control)
    status = sub.add_parser("status")
    _run_arg(status)
    args = parser.parse_args(argv)

    if args.command == "train":
        cfg = configured(load_config(args.config), duration_seconds=args.duration_seconds, device=args.device)
        run_id = args.run_id or make_run_id()
        initial_policy = Path(args.initialize_policy) if args.initialize_policy else None
        summary = Trainer(cfg, Path(__file__).resolve().parents[2] / "artifacts" / run_id,
                          initial_policy=initial_policy).run()
        _print_summary(summary)
        return 0
    if args.command == "resume":
        run = resolve_run(args.run_id, args.latest or args.run_id is None)
        checkpoint = run / "latest.pt"
        saved = load_checkpoint(checkpoint)
        cfg = configured(saved["config"], duration_seconds=args.duration_seconds, device=args.device)
        summary = Trainer(cfg, run, checkpoint).run()
        _print_summary(summary)
        return 0
    if args.command in ("status", "control"):
        run = resolve_run(args.run_id, args.latest or args.run_id is None)
        action = "status" if args.command == "status" else args.action
        data = read_control(run / "control.json") if action == "status" else request_control(run / "control.json", action)
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    if args.command == "export":
        run = resolve_run(args.run_id, args.latest or args.run_id is None)
        source = Path(args.checkpoint) if args.checkpoint else run / "latest.pt"
        output = Path(args.output) if args.output else run / "latest.gdp"
        print(export_checkpoint(source, output))
        return 0
    if args.command == "evaluate":
        run = resolve_run(args.run_id, args.latest or args.run_id is None)
        saved = load_checkpoint(run / "latest.pt")
        cfg, norm = saved["config"], saved["normalization"]
        device = select_device(cfg["experiment"]["device"])
        model = DenseQNetwork(cfg["seeds"]["parameter_initialization"], norm["input_scale"], norm["input_bias"]).to(device)
        model.load_state_dict(saved["online_network"])
        print(json.dumps(evaluate_model(model, cfg, args.episodes, args.seed, device), indent=2, sort_keys=True))
        return 0
    if args.command == "play":
        play(args.run_id, args.checkpoint, args.policy, args.episodes, args.group, args.track, args.league,
             args.fps, args.seed, args.validate_only)
        return 0
    if args.command == "arcade":
        arcade(args.run_id, args.checkpoint, args.policy, args.seed)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
