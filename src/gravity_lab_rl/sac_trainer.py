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
from .config import with_experiment_defaults, curriculum_environments, model_input_size
from .control import atomic_write_json, initialize_control, read_control, update_status
from .evaluation import evaluate_model
from .export import export_checkpoint, policy_from_model
from .model import build_network, select_device
from .demo_curriculum import DemoCurriculum, compute_normalization, demo_transitions
from .playback import require_integration
from .replay import ReplayBatch, ReplayBuffer
from .practice import PracticeBank
from .recording import begin_recording_session, record_environment
from .reward import EpisodeReward, RewardConfig
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
        config = with_experiment_defaults(config)
        self.config, self.run_dir = config, run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(int(config["experiment"].get("torch_num_threads", 1)))
        self.device = select_device(config["experiment"]["device"])
        norm, seeds, algo = config["normalization"], config["seeds"], config["algorithm"]
        init_seed = int(seeds["parameter_initialization"])
        self.ensemble_size = int(algo["ensemble_size"])
        self.subset_size = int(algo["subset_size"])

        self.observation_size = model_input_size(self.config)
        # Demonstrations (searched by scripts/explore_maps.py) drive the backward start curriculum
        # and a permanent behavior-cloning replay; see demo_curriculum.py. Loaded before the
        # networks because the observation normalization can be derived from the demo data.
        self.demos = DemoCurriculum(config.get("demos", {}), curriculum_environments(config),
                                    int(seeds.get("demo_curriculum", 29)))
        self.demo_replay: ReplayBuffer | None = None
        if self.demos.enabled:
            self._build_demo_replay()
            norm = self.config["normalization"]
        self.actor = build_network(self.config, "actor", init_seed).to(self.device)
        # Distinct initialization seeds per ensemble member so the critics start decorrelated --
        # REDQ's variance-reduction benefit depends on the ensemble actually disagreeing early on.
        self.critics = nn.ModuleList([
            build_network(self.config, "critic", init_seed + 100 + i) for i in range(self.ensemble_size)
        ]).to(self.device)
        self.critic_targets = nn.ModuleList([
            build_network(self.config, "critic", init_seed + 100 + i) for i in range(self.ensemble_size)
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
        curriculum = config.get('curriculum', {})
        self.unlocked_stages = len(curriculum.get('stages', [None])) if curriculum.get('unlock_all', False) else 1
        self.track_episode_counts: dict[int, int] = {}
        self.track_full_start_counts: dict[int, int] = {}
        self.best_elapsed = 0.0
        self.practice = PracticeBank(config.get('practice', {}), seeds.get('obstacle_practice', 23))
        # Adaptive curriculum: track selection weighted toward whatever tracks currently have the
        # lowest recent success rate, instead of plain round-robin -- ported from PPOTrainer (see
        # docs/training-runs.md, "sparse-success plateau" and "Adaptive curriculum + peak-based
        # progress"). A struggling track gets picked far more often, giving it more rehearsal
        # instead of the same fixed share every mastered track gets under round-robin.
        self.curriculum_rng = random.Random(seeds["replay_sampling"])
        self.sticky_rng = random.Random(seeds["epsilon_exploration"])
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
        self._last_timelapse_active = 0.0
        self._paused = True
        if resume_checkpoint:
            self._restore(resume_checkpoint)
        atomic_write_json(run_dir / "config.json", config)
        atomic_write_json(run_dir / "metadata.json", self.metadata)
        initialize_control(self.control_path, run_dir.name)

    def _open_plain_environment(self, configuration: dict[str, Any]):
        from gravity_lab import ClassicConfig, ClassicGravityEnv

        classic = ClassicConfig(configuration["level_group"], configuration["track"], configuration["league"],
                                configuration["frame_skip"], configuration["max_episode_steps"],
                                self.config["seeds"]["environment"],
                                configuration.get("obstacle_ray_count", DEFAULT_OBSTACLE_RAY_COUNT))
        return ClassicGravityEnv(classic, configuration.get("level_pack"))

    def _build_demo_replay(self) -> None:
        """Replay every demo once into a permanent, per-track-balanced buffer. Rebuilt on resume
        (deterministic, cheap) rather than checkpointed. Also derives the fixed observation
        normalization when the config asks for `normalization.kind = "demo_statistics"`."""
        algo = self.config["algorithm"]
        transitions = demo_transitions(self.demos.demos, curriculum_environments(self.config),
                                       RewardConfig.from_config(self.config), int(algo.get("n_step", 1)),
                                       float(algo["gamma"]), self.observation_size, self._open_plain_environment)
        if not transitions:
            raise ValueError("demos are enabled but no demo matched the curriculum; run scripts/explore_maps.py")
        norm = self.config["normalization"]
        if norm.get("kind") == "demo_statistics":
            observations = np.stack([t[0] for t in transitions])
            scale, bias = compute_normalization(observations, self.actor_track_region()[0], self.actor_track_region()[1])
            self.config["normalization"] = {"kind": "fixed", "input_scale": scale, "input_bias": bias,
                                            "source": "demo_statistics"}
        self.demo_replay = ReplayBuffer(len(transitions), int(self.config["seeds"]["replay_sampling"]) + 2,
                                        observation_size=self.observation_size)
        for observation, action, reward, next_observation, terminated, truncated, steps, track_id in transitions:
            self.demo_replay.add(observation, action, reward, next_observation, terminated, truncated, steps, track_id)

    @staticmethod
    def actor_track_region() -> tuple[int, int]:
        from . import ACCELERATION_REGION_END, TRACK_ID_REGION_END
        return ACCELERATION_REGION_END, TRACK_ID_REGION_END

    def _start_episode(self, env, env_cfg: dict[str, Any], track_id: int):
        seed = self.config["seeds"]["environment"] + self.completed_episode_count
        if self.demos.enabled:
            return self.demos.start(env, env_cfg, track_id, seed)
        return self.practice.start(env, env_cfg, seed)

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
        if not self.config.get('curriculum', {}).get('unlock_all', False):
            self.unlocked_stages = int(saved.get("unlocked_stages", 1))
        self.track_episode_counts = saved.get('track_episode_counts', {})
        self.track_full_start_counts = saved.get('track_full_start_counts', {})
        self.best_elapsed = saved.get('best_elapsed', 0.0)
        if 'practice_bank' in saved:
            self.practice.load_state_dict(saved['practice_bank'])
        if 'demo_curriculum' in saved and self.demos.enabled:
            self.demos.load_state_dict(saved['demo_curriculum'])
        if 'sticky_rng_state' in saved:
            self.sticky_rng.setstate(saved['sticky_rng_state'])
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
        # Alternate coverage turns with focused turns. Coverage turns choose
        # least-visited maps, guaranteeing access instead of relying on chance.
        if self.config.get('curriculum', {}).get('guaranteed_coverage', False):
            if self.completed_episode_count % 2 == 0:
                demos_enabled = getattr(self, 'demos', None) is not None and self.demos.enabled
                counts = self.track_episode_counts if demos_enabled else self.track_full_start_counts
                return min(environments, key=lambda env: counts.get(self._track_id(env), 0))
        weights = [1.0 / (self.track_success_ema.get(self._track_id(env), 0.5) + 0.15)
                  for env in environments]
        focus = set(self.config.get('curriculum', {}).get('focus_tracks', []))
        boost = float(self.config.get('curriculum', {}).get('focus_weight', 1.0))
        weights = [weight * (boost if f"{env['level_group']}:{env['track']}" in focus else 1)
                   for env, weight in zip(environments, weights)]
        return self.curriculum_rng.choices(environments, weights=weights, k=1)[0]

    def _record_evaluation(self, evaluation: dict[str, Any]) -> None:
        row = {'active_training_seconds': self.current_active_elapsed(),
               'training_episodes_per_track': self.track_episode_counts,
               'full_start_episodes_per_track': self.track_full_start_counts,
               'practice_episodes': self.practice.restored_episodes,
               'demo_curriculum': self.demos.summary() if self.demos.enabled else None,
               'best_finish_rate_before_evaluation': self.best_score[0] if self.best_score else None,
               'evaluation': evaluation}
        with (self.run_dir / 'evaluation_history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "practice_bank": self.practice.state_dict(),
            "demo_curriculum": self.demos.state_dict(),
            "sticky_rng_state": self.sticky_rng.getstate(),
            "best_elapsed": self.best_elapsed,
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
            "track_episode_counts": self.track_episode_counts,
            "track_full_start_counts": self.track_full_start_counts,
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
            "track_episode_counts": self.track_episode_counts,
            "track_full_start_counts": self.track_full_start_counts,
            "demo_curriculum": self.demos.summary() if self.demos.enabled else None,
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
        demo_batch = None
        if self.demo_replay is not None and int(self.demos.config["bc_batch_size"]) > 0:
            demo_batch = self.demo_replay.sample(int(self.demos.config["bc_batch_size"]), self.device)
            batch = ReplayBatch(*(torch.cat([a, b]) for a, b in zip(batch.__dict__.values(), demo_batch.__dict__.values())))

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
        if demo_batch is not None and float(self.demos.config["bc_weight"]) > 0.0:
            # Behavior cloning on the demonstration slice of the batch (it was concatenated last).
            demo_log_probs = log_probs[-len(demo_batch.actions):]
            bc_loss = F.nll_loss(demo_log_probs, demo_batch.actions)
            actor_loss = actor_loss + float(self.demos.config["bc_weight"]) * bc_loss
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

        # Preserve the initial (or resumed) policy before the first update.
        if self.config["experiment"].get("timelapse_interval_seconds"):
            timelapse_dir = self.run_dir / "timelapse"
            timelapse_dir.mkdir(exist_ok=True)
            snapshot = timelapse_dir / f"t_{int(self.active_elapsed):07d}.gdp"
            if not snapshot.exists():
                policy_from_model(self.actor).save(snapshot)

        algo, seeds = self.config["algorithm"], self.config["seeds"]
        duration = float(self.config["experiment"]["duration_seconds"])
        metrics_path = self.run_dir / "metrics.jsonl"
        begin_recording_session(self.run_dir, self.config, self.transition_count, self.active_elapsed)
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
                    return record_environment(
                        ClassicGravityEnv(classic, configuration.get("level_pack")),
                        self.run_dir, configuration, self.config, self.current_active_elapsed)

                env = open_environment(env_cfg)
                track_name = env.track_name
                episode_start = self._start_episode(env, env_cfg, track_id)
                observation = episode_start.observation[:self.observation_size]
                episode_actions = list(episode_start.actions)
                practice_prefix_steps = len(episode_actions)
                reward_config = RewardConfig.from_config(self.config)
                peak_progress = episode_start.peak_progress
                reward_tracker = EpisodeReward(reward_config, peak_progress)
                episode_reward, episode_length, last_loss = 0.0, 0, None
                n_step = NStepAccumulator(int(algo.get("n_step", 1)), algo["gamma"])
                sticky = float(self.demos.config["sticky_action_probability"]) if self.demos.enabled else 0.0
                previous_action: int | None = None
                while self.current_active_elapsed() < duration and not self._stop_signal:
                    if self._pause_if_requested():
                        break
                    greedy_episode = bool(getattr(episode_start, "greedy", False))
                    if (not greedy_episode and previous_action is not None and sticky > 0.0
                            and self.sticky_rng.random() < sticky):
                        # Temporally extended exploration: a setup maneuver spans dozens of
                        # consecutive 0.04 s decisions, which per-step sampling almost never repeats.
                        action = previous_action
                    else:
                        with torch.inference_mode():
                            logits = self.actor(torch.tensor(observation, dtype=torch.float32,
                                                             device=self.device))
                            if greedy_episode:
                                action = int(torch.argmax(logits).item())
                            else:
                                action = int(torch.distributions.Categorical(logits=logits).sample().item())
                    previous_action = action
                    step = env.step(action)
                    episode_actions.append(action)
                    self.practice.observe(env_cfg, episode_start.seed, episode_actions, step)
                    next_observation = step.observation[:self.observation_size]
                    reward, peak_progress = reward_tracker.step(step.observation[0], step.finished, step.crashed)
                    for ready in n_step.push(observation, action, reward, next_observation,
                                             step.terminated, step.truncated):
                        self.replay.add(*ready, track_id=track_id)
                    observation = next_observation
                    self.transition_count += 1
                    episode_reward += reward
                    episode_length += 1
                    if (len(self.replay) >= algo["replay_warmup"] and
                            self.transition_count % algo["update_every"] == 0):
                        last_loss = self._optimize()
                    if step.terminated or step.truncated:
                        self.completed_episode_count += 1
                        self.track_episode_counts[track_id] = self.track_episode_counts.get(track_id, 0) + 1
                        if not practice_prefix_steps:
                            self.track_full_start_counts[track_id] = self.track_full_start_counts.get(track_id, 0) + 1
                        self.latest_metrics = {
                            "episode": self.completed_episode_count, "reward": episode_reward,
                            "practice_prefix_steps": practice_prefix_steps,
                            "demo_prefix": self.demos.prefix.get(track_id) if self.demos.enabled else None,
                            "greedy": bool(getattr(episode_start, "greedy", False)),
                            "peak_progress": peak_progress,
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
                        if not practice_prefix_steps:
                            self._update_track_success(env_cfg, step.finished)
                        self.demos.record(track_id, practice_prefix_steps, step.finished,
                                          bool(getattr(episode_start, "greedy", False)))
                        previous_action = None
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
                        episode_start = self._start_episode(env, env_cfg, track_id)
                        observation = episode_start.observation[:self.observation_size]
                        episode_actions = list(episode_start.actions)
                        practice_prefix_steps = len(episode_actions)
                        peak_progress = episode_start.peak_progress
                        reward_tracker = EpisodeReward(reward_config, peak_progress)
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
                    timelapse_interval = self.config["experiment"].get("timelapse_interval_seconds")
                    if (timelapse_interval and self.current_active_elapsed() - self._last_timelapse_active
                            >= float(timelapse_interval)):
                        self._last_timelapse_active = self.current_active_elapsed()
                        timelapse_dir = self.run_dir / "timelapse"
                        timelapse_dir.mkdir(exist_ok=True)
                        policy_from_model(self.actor).save(
                            timelapse_dir / f"t_{int(self.current_active_elapsed()):07d}.gdp")
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
                        self._record_evaluation(eval_result)
                        score = (eval_result["finish_rate"], eval_result["mean_progress"])
                        if self.best_score is None or score > self.best_score:
                            self.best_elapsed = self.current_active_elapsed()
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
                        episode_start = self._start_episode(env, env_cfg, track_id)
                        observation = episode_start.observation[:self.observation_size]
                        episode_actions = list(episode_start.actions)
                        practice_prefix_steps = len(episode_actions)
                        peak_progress = episode_start.peak_progress
                        reward_tracker = EpisodeReward(reward_config, peak_progress)
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
        self._record_evaluation(evaluation)
        final_score = (evaluation["finish_rate"], evaluation["mean_progress"])
        if self.best_score is None or final_score > self.best_score:
            self.best_elapsed = self.active_elapsed
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
            "best_policy_active_training_seconds": self.best_elapsed,
            "track_training_episodes": self.track_episode_counts,
            "track_full_start_episodes": self.track_full_start_counts,
            "practice_episodes": self.practice.restored_episodes,
            "practice_reconstructed_steps": self.practice.reconstructed_steps,
            "demo_curriculum": self.demos.summary() if self.demos.enabled else None,
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
        from .video import generate_training_videos
        if graceful_reason == 'duration-expired' or self.config['experiment'].get('map_overlay_on_stop', False):
            generate_training_videos(self.run_dir, self.config)
        return summary
