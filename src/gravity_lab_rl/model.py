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
