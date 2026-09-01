import torch

from gravity_lab_rl.model import DenseQNetwork


def test_model_dimensions():
    model = DenseQNetwork(11)
    assert model(torch.zeros(4, 72)).shape == (4, 9)
    assert model(torch.zeros(72)).shape == (9,)


def test_named_seed_is_deterministic_and_does_not_change_global_rng():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    first = DenseQNetwork(77)
    after = torch.get_rng_state()
    second = DenseQNetwork(77)
    third = DenseQNetwork(78)
    assert torch.equal(before, after)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(first.parameters(), third.parameters()))

