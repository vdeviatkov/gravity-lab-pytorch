from __future__ import annotations

import json
import os
import platform
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, restore_rng_state, rng_state, save_checkpoint
from .config import curriculum_environment_index, curriculum_environments, model_input_size
from .control import atomic_write_json, initialize_control, read_control, update_status
from .evaluation import double_dqn_targets, evaluate_model
from .export import export_checkpoint, load_policy_into_model
from .model import DenseQNetwork, select_device
from .playback import game_repo, require_integration
from .replay import ReplayBuffer


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except ValueError:
        return path.name


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"


def make_metadata(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    gravity_path = game_repo()
    return {
        "format": "gravity-lab-rl-metadata-v1",
        "experiment_repository": ".",
        "experiment_repository_commit": _git_commit(_repo_root()),
        "gravity_lab_repository": _portable_path(gravity_path),
        "gravity_lab_repository_commit": _git_commit(gravity_path),
        "environment_id": config["environment_id"],
        "environment_configuration": config["environment"],
        "curriculum": config.get("curriculum"),
        "algorithm_configuration": config["algorithm"],
        "normalization": config["normalization"], "seeds": config["seeds"],
        "python_version": sys.version, "pytorch_version": torch.__version__,
        "operating_system": platform.platform(), "cpu_architecture": platform.machine(),
        "processor": platform.processor(), "selected_device": str(device),
        "checkpoint_selection_rule": config["experiment"]["checkpoint_selection_rule"],
        "training_start_timestamp": _now(), "training_end_timestamp": None,
        "active_training_duration_seconds": 0.0,
        "resume_begins_fresh_episode": True,
        "deterministic_accelerator_algorithms_enabled": False,
    }


class Trainer:
    def __init__(self, config: dict[str, Any], run_dir: Path,
                 resume_checkpoint: Path | None = None,
                 initial_policy: Path | None = None) -> None:
        if resume_checkpoint is not None and initial_policy is not None:
            raise ValueError("resume_checkpoint and initial_policy are mutually exclusive")
        require_integration(require_viewer=False)
        self.config, self.run_dir = config, run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(int(config["experiment"].get("torch_num_threads", 1)))
        self.device = select_device(config["experiment"]["device"])
        norm, seeds, algo = config["normalization"], config["seeds"], config["algorithm"]
        self.online = DenseQNetwork(seeds["parameter_initialization"], norm["input_scale"],
                                    norm["input_bias"]).to(self.device)
        self.target = DenseQNetwork(seeds["parameter_initialization"], norm["input_scale"],
                                    norm["input_bias"]).to(self.device)
        if initial_policy is not None:
            loaded_normalization = load_policy_into_model(self.online, initial_policy)
            self.config["normalization"] = loaded_normalization
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=algo["learning_rate"])
        self.observation_size = model_input_size(self.config)
        self.replay = ReplayBuffer(algo["replay_capacity"], seeds["replay_sampling"],
                                   observation_size=self.observation_size)
        self.exploration_rng = random.Random(seeds["epsilon_exploration"])
        self.transition_count = self.optimizer_update_count = self.completed_episode_count = 0
        self.active_elapsed = 0.0
        self.latest_metrics: dict[str, Any] = {}
        self.resume_checkpoint = resume_checkpoint
        self.metadata = make_metadata(config, self.device)
        self.metadata["torch_num_threads"] = torch.get_num_threads()
        if initial_policy is not None:
            self.metadata["initialized_from_policy"] = _portable_path(initial_policy)
        self.control_path = run_dir / "control.json"
        self._stop_signal: str | None = None
        self._active_since = time.monotonic()
        self._last_checkpoint_active = 0.0
        self._last_status_wall = 0.0
        # Environment construction, checkpoint I/O, evaluation, and pause time are not active
        # training. The clock starts only when the first training-loop boundary is entered.
        self._paused = True
        if resume_checkpoint:
            self._restore(resume_checkpoint)
        atomic_write_json(run_dir / "config.json", config)
        atomic_write_json(run_dir / "metadata.json", self.metadata)
        initialize_control(self.control_path, run_dir.name)

    def _restore(self, path: Path) -> None:
        saved = load_checkpoint(path, self.device)
        self.online.load_state_dict(saved["online_network"])
        self.target.load_state_dict(saved["target_network"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.replay.load_state_dict(saved["replay_buffer"])
        self.transition_count = int(saved["transition_count"])
        self.optimizer_update_count = int(saved["optimizer_update_count"])
        self.completed_episode_count = int(saved["completed_episode_count"])
        self.active_elapsed = float(saved["active_training_duration_seconds"])
        self.latest_metrics = saved.get("latest_metrics", {})
        self.exploration_rng.setstate(saved["epsilon_exploration_rng_state"])
        restore_rng_state(saved["rng_state"])
        prior = saved.get("metadata", {})
        self.metadata["training_start_timestamp"] = prior.get(
            "training_start_timestamp", self.metadata["training_start_timestamp"]
        )
        self.metadata["resume_timestamps"] = [*prior.get("resume_timestamps", []), _now()]
        self.metadata["resumed_from"] = _portable_path(path)
        self._last_checkpoint_active = self.active_elapsed
        self._active_since = time.monotonic()

    def epsilon(self) -> float:
        algo = self.config["algorithm"]
        fraction = min(1.0, self.transition_count / algo["epsilon_decay_transitions"])
        return float(algo["epsilon_start"] + fraction * (algo["epsilon_end"] - algo["epsilon_start"]))

    def current_active_elapsed(self) -> float:
        return self.active_elapsed + (0.0 if self._paused else time.monotonic() - self._active_since)

    def _checkpoint_payload(self) -> dict[str, Any]:
        environments = curriculum_environments(self.config)
        curriculum_index = curriculum_environment_index(self.config, self.completed_episode_count)
        return {
            "online_network": self.online.state_dict(), "target_network": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "replay_buffer": self.replay.state_dict(),
            "transition_count": self.transition_count,
            "optimizer_update_count": self.optimizer_update_count,
            "completed_episode_count": self.completed_episode_count,
            "epsilon_schedule_state": {"current_epsilon": self.epsilon(),
                                       "transition_count": self.transition_count},
            "epsilon_exploration_rng_state": self.exploration_rng.getstate(),
            "rng_state": rng_state(), "config": self.config, "latest_metrics": self.latest_metrics,
            "normalization": self.config["normalization"], "metadata": self.metadata,
            "active_training_duration_seconds": self.current_active_elapsed(),
            "curriculum_state": {"environment_index": curriculum_index,
                                 "environment": environments[curriculum_index]},
            "saved_at": _now(),
        }

    def save(self, final: bool = False, export: bool = True) -> Path:
        name = "final.pt" if final else "latest.pt"
        path = self.run_dir / name
        save_checkpoint(path, self._checkpoint_payload())
        if not final:
            # latest is always a complete atomic snapshot, never a symlink to a file being written.
            self._last_checkpoint_active = self.current_active_elapsed()
        if export:
            export_checkpoint(path, self.run_dir / ("final.gdp" if final else "latest.gdp"))
        return path

    def _status(self, state: str, checkpoint: Path | None = None) -> None:
        environments = curriculum_environments(self.config)
        curriculum_index = curriculum_environment_index(self.config, self.completed_episode_count)
        update_status(self.control_path, {
            "state": state, "transitions": self.transition_count,
            "optimizer_updates": self.optimizer_update_count, "episodes": self.completed_episode_count,
            "active_training_seconds": self.current_active_elapsed(),
            "epsilon": self.epsilon(), "latest_metrics": self.latest_metrics,
            "checkpoint": _portable_path(checkpoint or self.run_dir / "latest.pt"),
            "pid": os.getpid(), "device": str(self.device),
            "curriculum_environment_index": curriculum_index,
            "environment": environments[curriculum_index],
        })

    def _pause_if_requested(self) -> bool:
        request = read_control(self.control_path).get("requested", "run")
        if request == "stop":
            self._stop_signal = "control-stop"
            return True
        if request != "pause":
            if self._paused:
                self._paused = False
                self._active_since = time.monotonic()
                self._status("running")
            return False
        if not self._paused:
            self.active_elapsed = self.current_active_elapsed()
            self._paused = True
            checkpoint = self.save()
            self._status("paused", checkpoint)
        while read_control(self.control_path).get("requested") == "pause" and not self._stop_signal:
            self._status("paused")
            time.sleep(0.2)
        if read_control(self.control_path).get("requested") == "stop":
            self._stop_signal = "control-stop"
            return True
        self._paused = False
        self._active_since = time.monotonic()
        self._status("running")
        return False

    def _optimize(self) -> float:
        algo = self.config["algorithm"]
        batch = self.replay.sample(algo["batch_size"], self.device)
        predicted = self.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
        targets = double_dqn_targets(batch.rewards, batch.next_observations, batch.terminated,
                                     self.online, self.target, algo["gamma"])
        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), algo["gradient_clip_norm"])
        self.optimizer.step()
        self.optimizer_update_count += 1
        if self.optimizer_update_count % algo["target_update_every"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def run(self) -> dict[str, Any]:
        from gravity_lab import ClassicConfig, ClassicGravityEnv

        environments = curriculum_environments(self.config)
        algo, seeds = self.config["algorithm"], self.config["seeds"]
        duration = float(self.config["experiment"]["duration_seconds"])
        metrics_path = self.run_dir / "metrics.jsonl"
        old_handlers: dict[int, Any] = {}

        def handle_signal(signum: int, _frame: Any) -> None:
            self._stop_signal = signal.Signals(signum).name

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, handle_signal)
        graceful_reason = "duration-expired"
        failure: BaseException | None = None
        self._status("running")
        env: ClassicGravityEnv | None = None
        try:
            with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics_stream:
                active_index = curriculum_environment_index(self.config,
                                                            self.completed_episode_count)
                env_cfg = environments[active_index]

                def open_environment(configuration: dict[str, Any]) -> ClassicGravityEnv:
                    classic = ClassicConfig(
                        configuration["level_group"], configuration["track"],
                        configuration["league"], configuration["frame_skip"],
                        configuration["max_episode_steps"], seeds["environment"])
                    return ClassicGravityEnv(classic, configuration.get("level_pack"))

                env = open_environment(env_cfg)
                track_name = env.track_name
                # The environment always returns OBSERVATION_SIZE values; truncate to this
                # model's actual input width (BASE_OBSERVATION_SIZE for the legacy network,
                # OBSERVATION_SIZE for the obstacle-sensor network) via the shared compatible
                # prefix.
                observation = env.reset(seeds["environment"] + self.completed_episode_count)[
                    :self.observation_size]
                episode_reward, episode_length, last_loss = 0.0, 0, None
                while self.current_active_elapsed() < duration and not self._stop_signal:
                    if self._pause_if_requested():
                        break
                    epsilon = self.epsilon()
                    if self.exploration_rng.random() < epsilon:
                        action = self.exploration_rng.randrange(9)
                    else:
                        with torch.inference_mode():
                            values = self.online(torch.tensor(observation, dtype=torch.float32,
                                                              device=self.device))
                            action = int(torch.argmax(values).item())
                    step = env.step(action)
                    next_observation = step.observation[:self.observation_size]
                    self.replay.add(observation, action, step.reward, next_observation,
                                    step.terminated, step.truncated)
                    observation = next_observation
                    self.transition_count += 1
                    episode_reward += step.reward
                    episode_length += 1
                    if (len(self.replay) >= algo["replay_warmup"] and
                            self.transition_count % algo["update_every"] == 0):
                        last_loss = self._optimize()
                    if step.terminated or step.truncated:
                        self.completed_episode_count += 1
                        self.latest_metrics = {
                            "episode": self.completed_episode_count, "reward": episode_reward,
                            "length": episode_length, "progress": float(step.observation[0]),
                            "finished": step.finished, "crashed": step.crashed,
                            "truncated": step.truncated, "epsilon": epsilon, "loss": last_loss,
                            "transitions": self.transition_count,
                            "optimizer_updates": self.optimizer_update_count,
                            "active_training_seconds": self.current_active_elapsed(),
                            "level_group": env_cfg["level_group"], "track": env_cfg["track"],
                            "league": env_cfg["league"], "track_name": track_name,
                            "timestamp": _now(),
                        }
                        metrics_stream.write(json.dumps(self.latest_metrics, sort_keys=True) + "\n")
                        metrics_stream.flush()
                        next_index = curriculum_environment_index(self.config,
                                                                  self.completed_episode_count)
                        if next_index != active_index:
                            env.close()
                            env = None
                            active_index = next_index
                            env_cfg = environments[active_index]
                            env = open_environment(env_cfg)
                            track_name = env.track_name
                        observation = env.reset(seeds["environment"] + self.completed_episode_count)[
                            :self.observation_size]
                        episode_reward, episode_length = 0.0, 0
                    now = time.monotonic()
                    if now - self._last_status_wall >= self.config["experiment"]["status_interval_seconds"]:
                        self.latest_metrics.update({"current_episode_reward": episode_reward,
                                                    "current_episode_length": episode_length,
                                                    "current_progress": float(observation[0])})
                        self._status("running")
                        self._last_status_wall = now
                    if (self.current_active_elapsed() - self._last_checkpoint_active >=
                            self.config["experiment"]["checkpoint_interval_seconds"]):
                        self.save()
                graceful_reason = self._stop_signal or "duration-expired"
        except KeyboardInterrupt:
            graceful_reason = "KeyboardInterrupt"
        except BaseException as error:
            failure = error
            graceful_reason = f"exception: {type(error).__name__}: {error}"
        finally:
            if env is not None:
                try:
                    env.close()
                except BaseException as close_error:
                    if failure is None:
                        failure = close_error
            if not self._paused:
                self.active_elapsed = self.current_active_elapsed()
                self._paused = True
            try:
                latest = self.save(final=False)
                if failure is None:
                    final = self.save(final=True)
                    self._status("evaluating", final)
                else:
                    self._status("error-checkpointed", latest)
            except BaseException as save_error:
                if failure is None:
                    failure = save_error
                else:
                    print(f"warning: could not preserve checkpoint after error: {save_error}", file=sys.stderr)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        if failure is not None:
            raise failure

        evaluation = evaluate_model(self.online, self.config, device=self.device)
        summary = {
            "format": "gravity-lab-rl-summary-v1", "run_id": self.run_dir.name,
            "reason": graceful_reason, "training_start_timestamp": self.metadata["training_start_timestamp"],
            "training_end_timestamp": _now(), "active_training_duration_seconds": self.active_elapsed,
            "transition_count": self.transition_count,
            "optimizer_update_count": self.optimizer_update_count,
            "completed_episode_count": self.completed_episode_count,
            "checkpoint_selection_rule": self.config["experiment"]["checkpoint_selection_rule"],
            "final_evaluation": evaluation,
            "paths": {"final_checkpoint": _portable_path(self.run_dir / "final.pt"),
                      "final_policy": _portable_path(self.run_dir / "final.gdp"),
                      "metrics": _portable_path(metrics_path),
                      "metadata": _portable_path(self.run_dir / "metadata.json")},
        }
        atomic_write_json(self.run_dir / "summary.json", summary)
        self.metadata.update({"training_end_timestamp": summary["training_end_timestamp"],
                              "active_training_duration_seconds": self.active_elapsed,
                              "transition_count": self.transition_count,
                              "optimizer_update_count": self.optimizer_update_count,
                              "final_evaluation": evaluation})
        atomic_write_json(self.run_dir / "metadata.json", self.metadata)
        # Refresh the final source-of-truth checkpoint and deployment sidecar with end time and
        # evaluation metadata. The learned state is unchanged.
        self.save(final=True)
        self._status("stopped", self.run_dir / "final.pt")
        return summary
