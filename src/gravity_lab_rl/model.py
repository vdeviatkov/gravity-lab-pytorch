from __future__ import annotations

import torch
from torch import nn

from . import ACTION_COUNT, OBSERVATION_SIZE


class DenseQNetwork(nn.Module):
    def __init__(self, initialization_seed: int, input_scale: list[float] | None = None,
                 input_bias: list[float] | None = None,
                 hidden_sizes: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        # The observation width is derived from the normalization vectors rather than fixed to
        # OBSERVATION_SIZE, so a model can target either the legacy 28-value observation or the
        # current OBSERVATION_SIZE (36, with the obstacle-ray sensor); the environment's raw
        # observation is a superset, and the leading `input_size` values are always a compatible
        # prefix (see docs/policy-comparison.md).
        input_size = len(input_scale) if input_scale is not None else OBSERVATION_SIZE
        hidden1, hidden2 = hidden_sizes
        # Linear constructors initialize parameters, so isolate even that temporary work from
        # PyTorch's process-global RNG before applying our named local-generator initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.fc1 = nn.Linear(input_size, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.q = nn.Linear(hidden2, ACTION_COUNT)
        self.register_buffer("input_scale", torch.tensor(input_scale or [1.0] * OBSERVATION_SIZE,
                                                         dtype=torch.float32))
        self.register_buffer("input_bias", torch.tensor(input_bias or [0.0] * OBSERVATION_SIZE,
                                                        dtype=torch.float32))
        self.reset_parameters(initialization_seed)

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for layer in (self.fc1, self.fc2, self.q):
            nn.init.kaiming_uniform_(layer.weight, a=5 ** 0.5, generator=generator)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(layer.bias, -bound, bound, generator=generator)

    def trunk(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations * self.input_scale + self.input_bias
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.q(self.trunk(observations))


class ActorCriticNetwork(DenseQNetwork):
    """Shares DenseQNetwork's trunk + `q` head (here read as action logits, not Q-values) so a
    trained actor exports through the unchanged `policy_from_model`/.gdp path: argmax over raw
    logits is identical to argmax over softmax(logits), so inference code needs no PPO-awareness at
    all. Adds a `value` head used only during training (GAE / the critic loss); never exported.
    """

    def __init__(self, initialization_seed: int, input_scale: list[float] | None = None,
                 input_bias: list[float] | None = None,
                 hidden_sizes: tuple[int, int] = (128, 128)) -> None:
        super().__init__(initialization_seed, input_scale, input_bias, hidden_sizes)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed + 1)
            self.value = nn.Linear(hidden_sizes[1], 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(initialization_seed + 1)
        nn.init.kaiming_uniform_(self.value.weight, a=5 ** 0.5, generator=generator)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.value.weight)
        nn.init.uniform_(self.value.bias, -1 / fan_in**0.5, 1 / fan_in**0.5, generator=generator)

    def forward_value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value(self.trunk(observations)).squeeze(-1)


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _default_excluded_inputs() -> tuple[int, ...]:
    # Index 4-5: physics point 0's offset relative to itself, always (0, 0). Indices 60-71: the
    # per-component "acceleration" region, which the engine reads from an integrator slot whose
    # force accumulators are never written, so it is always zero (see docs/training-runs.md).
    from . import ACCELERATION_REGION_END, OBSTACLE_REGION_END
    return (4, 5, *range(OBSTACLE_REGION_END, ACCELERATION_REGION_END))


DEFAULT_EXCLUDED_INPUTS = _default_excluded_inputs()


class TrackConditionedNetwork(nn.Module):
    """Shared trunk with the track one-hot re-injected at every hidden layer, plus one output
    head per track.

    Why (see docs/training-runs.md, "Session synthesis"): a single MLP fed the 30-track one-hot
    only through its first layer plateaued at 7-9/30 under DQN, PPO and SAC alike, and the set of
    maps it solved *flickered* between evaluations -- gradient updates for one map overwrote the
    output-layer solution of another. Here the last layer is a per-track head (30 x 9 logits,
    selected by the track id already present in the observation), so the layer where most of that
    interference landed is no longer shared at all, and each hidden layer receives the one-hot as
    extra inputs, which is exactly a learned per-track bias per layer (a linear map of a one-hot
    is an embedding lookup). Optional LayerNorm is for the SAC critics only: the actor must stay
    exportable to the plain relu/linear `.gdp` format, and `export.py` does that exactly by
    carrying the one-hot through the trunk as pass-through units and selecting the head with a
    relu mask (see `policy_from_track_conditioned`).

    The observation's track-id region is always consumed raw (its normalization is forced to
    scale 1 / bias 0) so that the one-hot stays a one-hot both here and in the exported policy.
    """

    def __init__(self, initialization_seed: int, input_scale: list[float] | None = None,
                 input_bias: list[float] | None = None,
                 hidden_sizes: tuple[int, ...] = (512, 512, 256), layer_norm: bool = False,
                 excluded_inputs: tuple[int, ...] | None = None, track_conditioning: bool = True) -> None:
        super().__init__()
        from . import ACCELERATION_REGION_END, TRACK_ID_REGION_END, TRACK_ID_SIZE

        input_size = len(input_scale) if input_scale is not None else OBSERVATION_SIZE
        if input_size < TRACK_ID_REGION_END:
            raise ValueError("track-conditioned network needs an observation width that includes the track id")
        self.track_start, self.track_end, self.track_count = ACCELERATION_REGION_END, TRACK_ID_REGION_END, TRACK_ID_SIZE
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.layer_norm = bool(layer_norm)
        # With track_conditioning=False the network is a plain MLP over the remaining inputs: no
        # one-hot injection, a single 9-way head, and the track-id region is excluded from the
        # inputs entirely, so map identity must be inferred from the terrain sensors.
        self.track_conditioning = bool(track_conditioning)
        scale = list(input_scale) if input_scale is not None else [1.0] * input_size
        bias = list(input_bias) if input_bias is not None else [0.0] * input_size
        for index in range(self.track_start, self.track_end):
            scale[index], bias[index] = 1.0, 0.0
        self.register_buffer("input_scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("input_bias", torch.tensor(bias, dtype=torch.float32))
        # Observation entries the network never reads (constant-zero regions of the engine's
        # layout, see DEFAULT_EXCLUDED_INPUTS). The exported policy keeps the full observation
        # width and simply has zero first-layer weights on these columns.
        excluded = set(DEFAULT_EXCLUDED_INPUTS if excluded_inputs is None else excluded_inputs)
        excluded &= set(range(input_size))
        if not self.track_conditioning:
            excluded |= set(range(self.track_start, self.track_end))
        elif excluded & set(range(self.track_start, self.track_end)):
            raise ValueError("the track-id region cannot be excluded from a track-conditioned network")
        active = [index for index in range(input_size) if index not in excluded]
        self.register_buffer("active_inputs", torch.tensor(active, dtype=torch.int64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            layers, previous = [], len(active)
            for index, size in enumerate(self.hidden_sizes):
                extra = self.track_count if (index and self.track_conditioning) else 0
                layers.append(nn.Linear(previous + extra, size))
                previous = size
            self.layers = nn.ModuleList(layers)
            self.norms = nn.ModuleList([nn.LayerNorm(size) for size in self.hidden_sizes]) if self.layer_norm else None
            self.heads = nn.Linear(previous, (self.track_count if self.track_conditioning else 1) * ACTION_COUNT)
        self.reset_parameters(initialization_seed)

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for layer in (*self.layers, self.heads):
            nn.init.kaiming_uniform_(layer.weight, a=5 ** 0.5, generator=generator)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def track_one_hot(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[..., self.track_start:self.track_end]

    def trunk(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations * self.input_scale + self.input_bias
        one_hot = self.track_one_hot(x)
        h = x.index_select(-1, self.active_inputs)
        for index, layer in enumerate(self.layers):
            h = layer(h if (index == 0 or not self.track_conditioning) else torch.cat([h, one_hot], dim=-1))
            if self.norms is not None:
                h = self.norms[index](h)
            h = torch.relu(h)
        return h

    def all_track_logits(self, observations: torch.Tensor) -> torch.Tensor:
        """(..., track_count, ACTION_COUNT) outputs of every head; `forward` selects one row."""
        return self.heads(self.trunk(observations)).reshape(*observations.shape[:-1], self.track_count, ACTION_COUNT)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.track_conditioning:
            return self.heads(self.trunk(observations))
        logits = self.all_track_logits(observations)
        track = self.track_one_hot(observations).argmax(dim=-1)
        index = track[..., None, None].expand(*track.shape, 1, ACTION_COUNT)
        return logits.gather(-2, index).squeeze(-2)


def build_network(config: dict, role: str = "actor", seed: int | None = None,
                  normalization: dict | None = None) -> nn.Module:
    """The network a config describes: `algorithm.network` is "dense" (default, DenseQNetwork /
    ActorCriticNetwork) or "track_conditioned". `role` is "actor" (exported) or "critic"
    (training-only; gets `algorithm.critic_layer_norm` when set)."""
    algo, norm = config["algorithm"], normalization or config["normalization"]
    init_seed = int(seed if seed is not None else config["seeds"]["parameter_initialization"])
    hidden_sizes = tuple(int(size) for size in algo["hidden_sizes"])
    kind = algo.get("network", "dense")
    if kind == "track_conditioned":
        layer_norm = role == "critic" and bool(algo.get("critic_layer_norm", False))
        excluded = algo.get("excluded_inputs")
        return TrackConditionedNetwork(init_seed, norm["input_scale"], norm["input_bias"], hidden_sizes, layer_norm,
                                       None if excluded is None else tuple(int(i) for i in excluded),
                                       bool(algo.get("track_conditioning", True)))
    if kind != "dense":
        raise ValueError(f"unknown network kind: {kind!r}")
    if algo.get("kind", "dqn") == "ppo":
        return ActorCriticNetwork(init_seed, norm["input_scale"], norm["input_bias"], hidden_sizes)
    return DenseQNetwork(init_seed, norm["input_scale"], norm["input_bias"], hidden_sizes)
