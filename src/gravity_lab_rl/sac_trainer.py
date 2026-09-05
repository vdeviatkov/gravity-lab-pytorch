from __future__ import annotations

import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ACTION_COUNT, DEFAULT_OBSTACLE_RAY_COUNT, TRACKS_PER_LEVEL_GROUP
from .checkpoint import load_checkpoint, restore_rng_state, rng_state, save_checkpoint
from .config import curriculum_environments, model_input_size
from .control import atomic_write_json, initialize_control, read_control, update_status
from .evaluation import evaluate_model
from .export import export_checkpoint
from .model import DenseQNetwork, select_device
from .playback import require_integration
from .replay import ReplayBuffer
from .trainer import NStepAccumulator, _now, _portable_path, make_metadata


class SACREDQTrainer:
    """Discrete Soft Actor-Critic with Randomized Ensembled Double Q-learning (REDQ).

    Reuses the same operational infrastructure as Trainer/PPOTrainer (control.json, checkpoint
    format, best-checkpoint tracking, progressive curriculum gating, per-track-balanced replay,
    n-step returns, the .gdp export path -- the actor is a plain DenseQNetwork, identical in shape
    to the DQN online network, so evaluation/export need no algorithm-specific code at all: argmax
    over the actor's logits is exactly evaluate_model's existing deterministic policy).

    Algorithm (Christodoulou 2019 discrete SAC + Chen et al. 2021 REDQ, combined): a categorical
    actor and an ensemble of `ensemble_size` critics, each an independent DenseQNetwork over the
    full discrete action set (no action sampling needed for the critic target -- the expectation
    over actions is computed exactly since the action space is discrete and small). Every critic
    update re-draws a random `subset_size`-of-`ensemble_size` subset of the *target* critics and
    bootstraps off their elementwise minimum (REDQ's in-target subsampling, which controls the
    overestimation that a naive high-update-ratio ensemble would otherwise amplify); all ensemble
    members are trained toward that same subsampled target. The actor is updated against the mean
    Q over the *entire* ensemble (lower-variance than the subsampled minimum, standard in REDQ).
    The entropy temperature (`alpha`) is auto-tuned toward a configured target entropy, replacing
    epsilon-greedy exploration entirely -- action selection during rollout is a stochastic sample
    from the actor's categorical distribution, not argmax.
    """

    def __init__(self, config: dict[str, Any], run_dir: Path,
                 resume_checkpoint: Path | None = None,
                 initial_policy: Path | None = None) -> None:
        if initial_policy is not None:
            raise ValueError("initial_policy warm-start is not yet supported for sac_redq")
        require_integration(require_viewer=False)
        self.config, self.run_dir = config, run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(int(config["experiment"].get("torch_num_threads", 1)))
        self.device = select_device(config["experiment"]["device"])
        norm, seeds, algo = config["normalization"], config["seeds"], config["algorithm"]
        hidden_sizes = tuple(algo["hidden_sizes"])
        init_seed = int(seeds["parameter_initialization"])
        self.ensemble_size = int(algo["ensemble_size"])
        self.subset_size = int(algo["subset_size"])

        self.actor = DenseQNetwork(init_seed, norm["input_scale"], norm["input_bias"],
                                   hidden_sizes).to(self.device)
        # Distinct initialization seeds per ensemble member so the critics start decorrelated --
        # REDQ's variance-reduction benefit depends on the ensemble actually disagreeing early on.
        self.critics = nn.ModuleList([
            DenseQNetwork(init_seed + 100 + i, norm["input_scale"], norm["input_bias"], hidden_sizes)
            for i in range(self.ensemble_size)
        ]).to(self.device)
        self.critic_targets = nn.ModuleList([
            DenseQNetwork(init_seed + 100 + i, norm["input_scale"], norm["input_bias"], hidden_sizes)
            for i in range(self.ensemble_size)
        ]).to(self.device)
        for critic, target in zip(self.critics, self.critic_targets):
            target.load_state_dict(critic.state_dict())
            target.eval()

        self.log_alpha = torch.tensor(math.log(float(algo["initial_alpha"])), device=self.device,
                                      requires_grad=True)
        self.target_entropy = float(algo["target_entropy_ratio"]) * math.log(ACTION_COUNT)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=algo["actor_learning_rate"])
        self.critics_optimizer = torch.optim.Adam(self.critics.parameters(), lr=algo["critic_learning_rate"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=algo["alpha_learning_rate"])

        self.observation_size = model_input_size(self.config)
        self.replay = ReplayBuffer(algo["replay_capacity"], seeds["replay_sampling"],
                                   observation_size=self.observation_size)
        # Separate stream from the replay buffer's own sampling RNG so REDQ's per-update target
        # subset draw doesn't correlate with which transitions get sampled.
        self.subset_rng = np.random.default_rng(int(seeds["replay_sampling"]) + 1)
        self.transition_count = self.optimizer_update_count = self.completed_episode_count = 0
        self.active_elapsed = 0.0
        self.best_score: tuple[float, float] | None = None
        self.best_metrics: dict[str, Any] | None = None
        self._last_best_eval_active = 0.0
        # Progressive difficulty gating -- see Trainer.unlocked_stages.
        self.unlocked_stages = 1
        # Adaptive curriculum: track selection weighted toward whatever tracks currently have the
        # lowest recent success rate, instead of plain round-robin -- ported from PPOTrainer (see
        # docs/training-runs.md, "sparse-success plateau" and "Adaptive curriculum + peak-based
        # progress"). A struggling track gets picked far more often, giving it more rehearsal
        # instead of the same fixed share every mastered track gets under round-robin.
        self.curriculum_rng = random.Random(seeds["replay_sampling"])
        self.track_success_ema: dict[int, float] = {}
        self._episodes_since_switch = 0
        self._current_env_cfg: dict[str, Any] | None = None
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
        self.actor.load_state_dict(saved["online_network"])
        for critic, state in zip(self.critics, saved["critics"]):
            critic.load_state_dict(state)
        for target, state in zip(self.critic_targets, saved["critic_targets"]):
            target.load_state_dict(state)
        self.actor_optimizer.load_state_dict(saved["actor_optimizer"])
        self.critics_optimizer.load_state_dict(saved["critics_optimizer"])
        self.alpha_optimizer.load_state_dict(saved["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(saved["log_alpha"].to(self.device))
        self.replay.load_state_dict(saved["replay_buffer"])
        self.subset_rng.bit_generator.state = saved["subset_rng_state"]
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

    def current_active_elapsed(self) -> float:
        return self.active_elapsed + (0.0 if self._paused else time.monotonic() - self._active_since)

    def _track_id(self, env_cfg: dict[str, Any]) -> int:
        return int(env_cfg["level_group"]) * TRACKS_PER_LEVEL_GROUP + int(env_cfg["track"])

    def _update_track_success(self, env_cfg: dict[str, Any], finished: bool) -> None:
        # Slow-moving EMA (alpha=0.05, ~20-episode time constant): a single success shouldn't spike
        # the estimate and immediately deprioritize a track that's still mostly failing.
        track_id = self._track_id(env_cfg)
        prior = self.track_success_ema.get(track_id, 0.5)
        self.track_success_ema[track_id] = 0.95 * prior + 0.05 * float(finished)

    def _select_next_environment(self, environments: list[dict[str, Any]]) -> dict[str, Any]:
        # +0.15 floor (not PPO's +0.05 -- see docs/training-runs.md, "Adaptive curriculum outcome
        # (run #21)") caps a 0%-success track's weight at ~7x a ~90%-mastered one, not ~20x. At
        # +0.05, every track in a newly-unlocked stage ties at the same near-maximal weight the
        # moment it starts failing, which starves already-mastered tracks of refresher practice in
        # a self-reinforcing loop (losing -> picked more -> more losses -> weight stays maxed) --
        # observed to actively degrade live training quality in run #21's second half, not just
        # plateau. The higher floor keeps real priority for struggling tracks while guaranteeing
        # mastered ones a much larger residual share.
        weights = [1.0 / (self.track_success_ema.get(self._track_id(env), 0.5) + 0.15)
                  for env in environments]
        return self.curriculum_rng.choices(environments, weights=weights, k=1)[0]

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "online_network": self.actor.state_dict(),
            "critics": [critic.state_dict() for critic in self.critics],
            "critic_targets": [target.state_dict() for target in self.critic_targets],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critics_optimizer": self.critics_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "replay_buffer": self.replay.state_dict(),
            "subset_rng_state": self.subset_rng.bit_generator.state,
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
            "alpha": float(self.log_alpha.exp().item()), "latest_metrics": self.latest_metrics,
            "checkpoint": _portable_path(checkpoint or self.run_dir / "latest.pt"),
            "best_score": list(self.best_score) if self.best_score is not None else None,
            "unlocked_stages": self.unlocked_stages,
            "pid": os.getpid(), "device": str(self.device),
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

    def _optimize_once(self) -> float:
        algo = self.config["algorithm"]
        gamma = float(algo["gamma"])
        batch = self.replay.sample(algo["batch_size"], self.device)

        with torch.no_grad():
            next_logits = self.actor(batch.next_observations)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()
            subset = self.subset_rng.choice(self.ensemble_size, size=self.subset_size, replace=False)
            subset_q = torch.stack([self.critic_targets[i](batch.next_observations) for i in subset], dim=0)
            min_q = subset_q.min(dim=0).values
            alpha = self.log_alpha.exp()
            next_value = (next_probs * (min_q - alpha * next_log_probs)).sum(dim=-1)
            discount = gamma ** batch.steps.to(batch.rewards.dtype)
            target = batch.rewards + discount * (~batch.terminated).to(batch.rewards.dtype) * next_value

        self.critics_optimizer.zero_grad(set_to_none=True)
        critic_loss = torch.zeros((), device=self.device)
        for critic in self.critics:
            predicted = critic(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
            critic_loss = critic_loss + F.smooth_l1_loss(predicted, target)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), algo["gradient_clip_norm"])
        self.critics_optimizer.step()

        logits = self.actor(batch.observations)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        with torch.no_grad():
            q_mean = torch.stack([critic(batch.observations) for critic in self.critics], dim=0).mean(dim=0)
        actor_loss = (probs * (self.log_alpha.exp().detach() * log_probs - q_mean)).sum(dim=-1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), algo["gradient_clip_norm"])
        self.actor_optimizer.step()

        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        tau = float(algo["tau"])
        with torch.no_grad():
            for critic, target_net in zip(self.critics, self.critic_targets):
                for param, target_param in zip(critic.parameters(), target_net.parameters()):
                    target_param.mul_(1.0 - tau).add_(param, alpha=tau)

        self.optimizer_update_count += 1
        return float(critic_loss.detach().cpu())

    def _optimize(self) -> float:
        last_loss = 0.0
        for _ in range(int(self.config["algorithm"]["utd_ratio"])):
            last_loss = self._optimize_once()
        return last_loss

    def run(self) -> dict[str, Any]:
        from gravity_lab import ClassicConfig, ClassicGravityEnv

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
                env_cfg = curriculum_environments(self.config, self.unlocked_stages)[0]
                self._current_env_cfg = env_cfg
                track_id = self._track_id(env_cfg)
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
                n_step = NStepAccumulator(int(algo.get("n_step", 1)), algo["gamma"])
                while self.current_active_elapsed() < duration and not self._stop_signal:
                    if self._pause_if_requested():
                        break
                    with torch.inference_mode():
                        logits = self.actor(torch.tensor(observation, dtype=torch.float32,
                                                         device=self.device))
                        action = int(torch.distributions.Categorical(logits=logits).sample().item())
                    step = env.step(action)
                    next_observation = step.observation[:self.observation_size]
                    for ready in n_step.push(observation, action, step.reward, next_observation,
                                             step.terminated, step.truncated):
                        self.replay.add(*ready, track_id=track_id)
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
                            "truncated": step.truncated,
                            "alpha": float(self.log_alpha.exp().item()), "loss": last_loss,
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
                                track_id = self._track_id(env_cfg)
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
                    best_eval_interval = float(
                        self.config["experiment"].get("best_checkpoint_eval_interval_seconds", 90.0))
                    if self.current_active_elapsed() - self._last_best_eval_active >= best_eval_interval:
                        self._last_best_eval_active = self.current_active_elapsed()
                        env.close()
                        self.actor.eval()
                        eval_episodes = int(self.config["experiment"].get("evaluation_episodes", 1))
                        eval_result = evaluate_model(self.actor, self.config, episodes=eval_episodes,
                                                     device=self.device)
                        self.actor.train()
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
                        observation = env.reset(seeds["environment"] + self.completed_episode_count)[
                            :self.observation_size]
                        episode_reward, episode_length = 0.0, 0
                        n_step.reset()
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

        evaluation = evaluate_model(self.actor, self.config, device=self.device)
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
        self.save(final=True)
        self._status("stopped", self.run_dir / "final.pt")
        return summary
