from __future__ import annotations

import json
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import DEFAULT_OBSTACLE_RAY_COUNT, TRACKS_PER_LEVEL_GROUP
from .checkpoint import load_checkpoint, restore_rng_state, rng_state, save_checkpoint
from .config import curriculum_environments, model_input_size
from .control import atomic_write_json, initialize_control, read_control, update_status
from .evaluation import evaluate_model
from .export import export_checkpoint
from .model import ActorCriticNetwork, select_device
from .playback import game_repo, require_integration
from .trainer import _now, _portable_path, make_metadata


class PPOTrainer:
    """On-policy PPO trainer for gravity-lab-classic-v1, as an alternative to the off-policy
    Trainer (Double DQN). Reuses the same operational infrastructure (control.json, checkpoint
    format base, best-checkpoint tracking, progressive curriculum gating, the .gdp export path)
    but replaces the replay buffer / target network / epsilon-greedy machinery with rollout
    collection, GAE advantage estimation, and the clipped surrogate objective -- an on-policy
    algorithm doesn't suffer the off-policy replay-buffer track-imbalance issue that motivated
    balanced sampling in the DQN trainer (see docs/training-runs.md, "v3"), since each update only
    ever trains on the rollout just collected, not a large accumulated buffer.
    """

    def __init__(self, config: dict[str, Any], run_dir: Path,
                 resume_checkpoint: Path | None = None,
                 initial_policy: Path | None = None) -> None:
        if initial_policy is not None:
            raise ValueError("initial_policy warm-start is not yet supported for PPO")
        require_integration(require_viewer=False)
        self.config, self.run_dir = config, run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(int(config["experiment"].get("torch_num_threads", 1)))
        self.device = select_device(config["experiment"]["device"])
        norm, seeds, algo = config["normalization"], config["seeds"], config["algorithm"]
        hidden_sizes = tuple(algo["hidden_sizes"])
        self.model = ActorCriticNetwork(seeds["parameter_initialization"], norm["input_scale"],
                                        norm["input_bias"], hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=algo["learning_rate"])
        self.observation_size = model_input_size(self.config)
        self.shuffle_rng = np.random.default_rng(seeds["replay_sampling"])
        self.transition_count = self.optimizer_update_count = self.completed_episode_count = 0
        self.active_elapsed = 0.0
        # Adaptive curriculum: track selection is weighted toward whatever tracks currently have
        # the lowest recent success rate, instead of uniform round-robin. Diagnosed from run #12/13
        # (docs/training-runs.md, "sparse-success plateau"): most "stuck" tracks were provably
        # solvable (finished at least once in thousands of attempts) but the rare success never
        # got consolidated into reliable behavior under uniform per-track exposure. Weighting
        # toward low-success tracks gives a rare win more repeated practice to reinforce.
        self.curriculum_rng = random.Random(seeds["replay_sampling"])
        self.track_success_ema: dict[int, float] = {}
        self._episodes_since_switch = 0
        self._current_env_cfg: dict[str, Any] | None = None
        self.best_score: tuple[float, float] | None = None
        self.best_metrics: dict[str, Any] | None = None
        self._last_best_eval_active = 0.0
        # See Trainer.unlocked_stages (same progressive-difficulty-gating mechanism).
        self.unlocked_stages = 1
        self.latest_metrics: dict[str, Any] = {}
        self.resume_checkpoint = resume_checkpoint
        self.metadata = make_metadata(config, self.device)
        self.metadata["torch_num_threads"] = torch.get_num_threads()
        self.control_path = run_dir / "control.json"
        self._stop_signal: str | None = None
        self._active_since = time.monotonic()
        self._last_checkpoint_active = 0.0
        self._last_status_wall = 0.0
        self._paused = True
        if resume_checkpoint:
            self._restore(resume_checkpoint)
        atomic_write_json(run_dir / "config.json", config)
        atomic_write_json(run_dir / "metadata.json", self.metadata)
        initialize_control(self.control_path, run_dir.name)

    def _restore(self, path: Path) -> None:
        saved = load_checkpoint(path, self.device)
        self.model.load_state_dict(saved["online_network"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.shuffle_rng.bit_generator.state = saved["shuffle_rng_state"]
        self.transition_count = int(saved["transition_count"])
        self.optimizer_update_count = int(saved["optimizer_update_count"])
        self.completed_episode_count = int(saved["completed_episode_count"])
        self.active_elapsed = float(saved["active_training_duration_seconds"])
        self.latest_metrics = saved.get("latest_metrics", {})
        restore_rng_state(saved["rng_state"])
        prior = saved.get("metadata", {})
        self.metadata["training_start_timestamp"] = prior.get(
            "training_start_timestamp", self.metadata["training_start_timestamp"]
        )
        self.metadata["resume_timestamps"] = [*prior.get("resume_timestamps", []), _now()]
        self.metadata["resumed_from"] = _portable_path(path)
        self._last_checkpoint_active = self.active_elapsed
        self._last_best_eval_active = float(saved.get("last_best_eval_active", self.active_elapsed))
        self.unlocked_stages = int(saved.get("unlocked_stages", 1))
        self.track_success_ema = saved.get("track_success_ema", {})
        if "curriculum_rng_state" in saved:
            self.curriculum_rng.setstate(saved["curriculum_rng_state"])
        best_score = saved.get("best_score")
        self.best_score = tuple(best_score) if best_score is not None else None
        self.best_metrics = saved.get("best_metrics")
        self._active_since = time.monotonic()

    def _track_id(self, env_cfg: dict[str, Any]) -> int:
        return int(env_cfg["level_group"]) * TRACKS_PER_LEVEL_GROUP + int(env_cfg["track"])

    def _update_track_success(self, env_cfg: dict[str, Any], finished: bool) -> None:
        # Slow-moving EMA (alpha=0.05, ~20-episode time constant): a single success shouldn't spike
        # the estimate and immediately deprioritize a track that's still mostly failing.
        track_id = self._track_id(env_cfg)
        prior = self.track_success_ema.get(track_id, 0.5)
        self.track_success_ema[track_id] = 0.95 * prior + 0.05 * float(finished)

    def _select_next_environment(self, environments: list[dict[str, Any]]) -> dict[str, Any]:
        # +0.05 floor keeps a never-yet-attempted or 0%-success track's weight bounded (20x a fully
        # mastered track's, not infinite), so mastered tracks still get occasional refresher practice.
        weights = [1.0 / (self.track_success_ema.get(self._track_id(env), 0.5) + 0.05)
                  for env in environments]
        return self.curriculum_rng.choices(environments, weights=weights, k=1)[0]

    def current_active_elapsed(self) -> float:
        return self.active_elapsed + (0.0 if self._paused else time.monotonic() - self._active_since)

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "online_network": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "shuffle_rng_state": self.shuffle_rng.bit_generator.state,
            "curriculum_rng_state": self.curriculum_rng.getstate(),
            "track_success_ema": self.track_success_ema,
            "transition_count": self.transition_count,
            "optimizer_update_count": self.optimizer_update_count,
            "completed_episode_count": self.completed_episode_count,
            "rng_state": rng_state(), "config": self.config, "latest_metrics": self.latest_metrics,
            "normalization": self.config["normalization"], "metadata": self.metadata,
            "active_training_duration_seconds": self.current_active_elapsed(),
            "best_score": list(self.best_score) if self.best_score is not None else None,
            "best_metrics": self.best_metrics,
            "last_best_eval_active": self._last_best_eval_active,
            "unlocked_stages": self.unlocked_stages,
            "curriculum_state": {"environment": self._current_env_cfg},
            "saved_at": _now(),
        }

    def save(self, final: bool = False, export: bool = True) -> Path:
        name = "final.pt" if final else "latest.pt"
        path = self.run_dir / name
        save_checkpoint(path, self._checkpoint_payload())
        if not final:
            self._last_checkpoint_active = self.current_active_elapsed()
        if export:
            export_checkpoint(path, self.run_dir / ("final.gdp" if final else "latest.gdp"))
        return path

    def _status(self, state: str, checkpoint: Path | None = None) -> None:
        update_status(self.control_path, {
            "state": state, "transitions": self.transition_count,
            "optimizer_updates": self.optimizer_update_count, "episodes": self.completed_episode_count,
            "active_training_seconds": self.current_active_elapsed(),
            "latest_metrics": self.latest_metrics,
            "checkpoint": _portable_path(checkpoint or self.run_dir / "latest.pt"),
            "best_score": list(self.best_score) if self.best_score is not None else None,
            "unlocked_stages": self.unlocked_stages,
            "pid": __import__("os").getpid(), "device": str(self.device),
            "environment": self._current_env_cfg,
            "track_success_ema": self.track_success_ema,
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

    def _ppo_update(self, observations: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray,
                    advantages: np.ndarray, returns: np.ndarray) -> float:
        algo = self.config["algorithm"]
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_logp_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        count = len(observations)
        minibatch_size = int(algo["minibatch_size"])
        clip_epsilon = float(algo["clip_epsilon"])
        last_loss = 0.0
        for _ in range(int(algo["ppo_epochs"])):
            order = np.arange(count)
            self.shuffle_rng.shuffle(order)
            for start in range(0, count, minibatch_size):
                batch = order[start:start + minibatch_size]
                batch_t = torch.as_tensor(batch, dtype=torch.int64, device=self.device)
                logits = self.model(obs_t[batch_t])
                distribution = torch.distributions.Categorical(logits=logits)
                new_logp = distribution.log_prob(actions_t[batch_t])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_logp - old_logp_t[batch_t])
                batch_adv = advantages_t[batch_t]
                surrogate = torch.min(ratio * batch_adv,
                                      torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_adv)
                policy_loss = -surrogate.mean()
                value_loss = F.mse_loss(self.model.forward_value(obs_t[batch_t]), returns_t[batch_t])
                loss = (policy_loss + float(algo["value_coef"]) * value_loss
                       - float(algo["entropy_coef"]) * entropy)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), algo["gradient_clip_norm"])
                self.optimizer.step()
                self.optimizer_update_count += 1
                last_loss = float(loss.detach().cpu())
        return last_loss

    def run(self) -> dict[str, Any]:
        from gravity_lab import ClassicConfig, ClassicGravityEnv

        algo, seeds = self.config["algorithm"], self.config["seeds"]
        duration = float(self.config["experiment"]["duration_seconds"])
        rollout_length = int(algo["rollout_length"])
        gamma, gae_lambda = float(algo["gamma"]), float(algo["gae_lambda"])
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
                env_cfg = curriculum_environments(self.config, self.unlocked_stages)[0]
                self._current_env_cfg = env_cfg
                episodes_per_track = int(self.config["curriculum"]["episodes_per_track"]) if (
                    self.config.get("curriculum", {}).get("enabled", False)) else 1

                def open_environment(configuration: dict[str, Any]) -> ClassicGravityEnv:
                    classic = ClassicConfig(
                        configuration["level_group"], configuration["track"],
                        configuration["league"], configuration["frame_skip"],
                        configuration["max_episode_steps"], seeds["environment"],
                        configuration.get("obstacle_ray_count", DEFAULT_OBSTACLE_RAY_COUNT))
                    return ClassicGravityEnv(classic, configuration.get("level_pack"))

                env = open_environment(env_cfg)
                track_name = env.track_name
                observation = env.reset(seeds["environment"] + self.completed_episode_count)[
                    :self.observation_size]
                episode_reward, episode_length, last_loss = 0.0, 0, None
                stop_early = False

                while self.current_active_elapsed() < duration and not self._stop_signal:
                    observations: list[Any] = []
                    actions: list[int] = []
                    log_probs: list[float] = []
                    values: list[float] = []
                    rewards: list[float] = []
                    dones: list[bool] = []
                    while len(observations) < rollout_length:
                        if self._pause_if_requested():
                            stop_early = True
                            break
                        with torch.inference_mode():
                            obs_t = torch.tensor(observation, dtype=torch.float32, device=self.device)
                            logits = self.model(obs_t)
                            value = self.model.forward_value(obs_t)
                            distribution = torch.distributions.Categorical(logits=logits)
                            action = distribution.sample()
                            log_prob = distribution.log_prob(action)
                        step = env.step(int(action.item()))
                        observations.append(observation)
                        actions.append(int(action.item()))
                        log_probs.append(float(log_prob.item()))
                        values.append(float(value.item()))
                        rewards.append(step.reward)
                        done = step.terminated or step.truncated
                        dones.append(done)
                        observation = step.observation[:self.observation_size]
                        self.transition_count += 1
                        episode_reward += step.reward
                        episode_length += 1
                        if done:
                            self.completed_episode_count += 1
                            self.latest_metrics = {
                                "episode": self.completed_episode_count, "reward": episode_reward,
                                "length": episode_length, "progress": float(step.observation[0]),
                                "finished": step.finished, "crashed": step.crashed,
                                "truncated": step.truncated, "loss": last_loss,
                                "transitions": self.transition_count,
                                "optimizer_updates": self.optimizer_update_count,
                                "active_training_seconds": self.current_active_elapsed(),
                                "level_group": env_cfg["level_group"], "track": env_cfg["track"],
                                "league": env_cfg["league"], "track_name": track_name,
                                "timestamp": _now(),
                            }
                            metrics_stream.write(json.dumps(self.latest_metrics, sort_keys=True) + "\n")
                            metrics_stream.flush()
                            self._update_track_success(env_cfg, step.finished)
                            self._episodes_since_switch += 1
                            if self._episodes_since_switch >= episodes_per_track:
                                self._episodes_since_switch = 0
                                candidates = curriculum_environments(self.config, self.unlocked_stages)
                                new_cfg = self._select_next_environment(candidates)
                                if new_cfg != env_cfg:
                                    env.close()
                                    env = None
                                    env_cfg = new_cfg
                                    self._current_env_cfg = env_cfg
                                    env = open_environment(env_cfg)
                                    track_name = env.track_name
                            observation = env.reset(
                                seeds["environment"] + self.completed_episode_count)[
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
                        best_eval_interval = float(
                            self.config["experiment"].get("best_checkpoint_eval_interval_seconds", 90.0))
                        if self.current_active_elapsed() - self._last_best_eval_active >= best_eval_interval:
                            self._last_best_eval_active = self.current_active_elapsed()
                            # This abandons the in-progress episode without a real terminal/
                            # truncated signal, so the just-recorded transition must be treated as
                            # a rollout boundary (no bootstrap across the discontinuity) even though
                            # its `done` flag says otherwise -- see NStepAccumulator.reset() in
                            # trainer.py for the equivalent DQN-side concern.
                            if dones:
                                dones[-1] = True
                            env.close()
                            self.model.eval()
                            eval_result = evaluate_model(self.model, self.config, episodes=1,
                                                         device=self.device)
                            self.model.train()
                            score = (eval_result["finish_rate"], eval_result["mean_progress"])
                            if self.best_score is None or score > self.best_score:
                                self.best_score = score
                                self.best_metrics = eval_result
                                save_checkpoint(self.run_dir / "best.pt", self._checkpoint_payload())
                                export_checkpoint(self.run_dir / "best.pt", self.run_dir / "best.gdp")
                            curriculum = self.config.get("curriculum")
                            if curriculum and curriculum.get("enabled", False):
                                stages = curriculum["stages"]
                                if self.unlocked_stages < len(stages):
                                    current_group = stages[self.unlocked_stages - 1]["level_group"]
                                    stage_rows = [row for row in eval_result["episodes"]
                                                 if row["level_group"] == current_group]
                                    stage_finish = (sum(1.0 for row in stage_rows if row["finished"])
                                                   / len(stage_rows)) if stage_rows else 0.0
                                    threshold = float(curriculum.get("stage_advance_finish_rate", 0.5))
                                    if stage_finish >= threshold:
                                        self.unlocked_stages += 1
                            env = open_environment(env_cfg)
                            observation = env.reset(
                                seeds["environment"] + self.completed_episode_count)[
                                :self.observation_size]
                            episode_reward, episode_length = 0.0, 0
                        if self.current_active_elapsed() >= duration:
                            break
                    if stop_early or not observations:
                        break
                    with torch.inference_mode():
                        bootstrap_value = 0.0 if dones[-1] else float(
                            self.model.forward_value(torch.tensor(
                                observation, dtype=torch.float32, device=self.device)).item())
                    values_arr = np.asarray(values, dtype=np.float32)
                    rewards_arr = np.asarray(rewards, dtype=np.float32)
                    dones_arr = np.asarray(dones, dtype=np.bool_)
                    advantages = np.zeros_like(rewards_arr)
                    last_gae = 0.0
                    for t in reversed(range(len(rewards_arr))):
                        next_non_terminal = 0.0 if dones_arr[t] else 1.0
                        next_value = bootstrap_value if t == len(rewards_arr) - 1 else values_arr[t + 1]
                        delta = rewards_arr[t] + gamma * next_value * next_non_terminal - values_arr[t]
                        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                        advantages[t] = last_gae
                    returns = advantages + values_arr
                    last_loss = self._ppo_update(np.asarray(observations, dtype=np.float32),
                                                 np.asarray(actions, dtype=np.int64),
                                                 np.asarray(log_probs, dtype=np.float32),
                                                 advantages, returns)
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

        evaluation = evaluate_model(self.model, self.config, device=self.device)
        final_score = (evaluation["finish_rate"], evaluation["mean_progress"])
        if self.best_score is None or final_score > self.best_score:
            self.best_score = final_score
            self.best_metrics = evaluation
            save_checkpoint(self.run_dir / "best.pt", self._checkpoint_payload())
            export_checkpoint(self.run_dir / "best.pt", self.run_dir / "best.gdp")
        summary = {
            "format": "gravity-lab-rl-summary-v1", "run_id": self.run_dir.name,
            "reason": graceful_reason, "training_start_timestamp": self.metadata["training_start_timestamp"],
            "training_end_timestamp": _now(), "active_training_duration_seconds": self.active_elapsed,
            "transition_count": self.transition_count,
            "optimizer_update_count": self.optimizer_update_count,
            "completed_episode_count": self.completed_episode_count,
            "checkpoint_selection_rule": "best (finish_rate, mean_progress) seen during periodic "
                                        "evaluation; see best_evaluation and paths.best_checkpoint",
            "final_evaluation": evaluation,
            "best_evaluation": self.best_metrics,
            "paths": {"final_checkpoint": _portable_path(self.run_dir / "final.pt"),
                      "final_policy": _portable_path(self.run_dir / "final.gdp"),
                      "best_checkpoint": _portable_path(self.run_dir / "best.pt"),
                      "best_policy": _portable_path(self.run_dir / "best.gdp"),
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
        return summary
