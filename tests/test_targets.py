import torch
from torch import nn

from gravity_lab_rl.evaluation import double_dqn_targets


class FixedNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, observations):
        return self.values.expand(observations.shape[0], -1)


def test_double_dqn_selects_online_action_and_uses_target_value():
    online = FixedNetwork([1.0, 5.0, 2.0])
    target = FixedNetwork([100.0, 7.0, 200.0])
    result = double_dqn_targets(torch.tensor([2.0]), torch.zeros(1, 36),
                                torch.tensor([False]), online, target, 0.5)
    assert torch.allclose(result, torch.tensor([5.5]))


def test_terminated_disables_bootstrap_but_truncation_does_not():
    online = FixedNetwork([0.0, 1.0])
    target = FixedNetwork([3.0, 10.0])
    rewards = torch.tensor([2.0, 2.0])
    terminated = torch.tensor([True, False])
    truncated = torch.tensor([False, True])
    result = double_dqn_targets(rewards, torch.zeros(2, 36), terminated, online, target, 0.9)
    assert truncated.tolist() == [False, True]  # documents the paired transition conditions
    assert torch.allclose(result, torch.tensor([2.0, 11.0]))

