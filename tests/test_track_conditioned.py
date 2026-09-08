import numpy as np
import torch
from gravity_lab import DenseQPolicy

from gravity_lab_rl import ACCELERATION_REGION_END, ACTION_COUNT, OBSERVATION_SIZE, TRACK_ID_SIZE
from gravity_lab_rl.export import policy_from_model
from gravity_lab_rl.model import TrackConditionedNetwork, build_network


def _observations(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(count, OBSERVATION_SIZE)) * 3.0
    obs[:, ACCELERATION_REGION_END:ACCELERATION_REGION_END + TRACK_ID_SIZE] = 0.0
    for row in range(count):
        obs[row, ACCELERATION_REGION_END + rng.integers(TRACK_ID_SIZE)] = 1.0
    return obs


def test_forward_selects_the_observed_track_head():
    model = TrackConditionedNetwork(5, hidden_sizes=(32, 16)).eval()
    obs = torch.from_numpy(_observations(6, 1)).float()
    logits = model(obs)
    assert logits.shape == (6, ACTION_COUNT)
    every = model.all_track_logits(obs)
    tracks = obs[:, ACCELERATION_REGION_END:ACCELERATION_REGION_END + TRACK_ID_SIZE].argmax(dim=1)
    for row in range(6):
        torch.testing.assert_close(logits[row], every[row, tracks[row]])
    assert model(obs[0]).shape == (ACTION_COUNT,)


def test_track_id_normalization_is_forced_to_identity():
    model = TrackConditionedNetwork(5, [0.5] * OBSERVATION_SIZE, [0.25] * OBSERVATION_SIZE, (8,))
    region = slice(ACCELERATION_REGION_END, ACCELERATION_REGION_END + TRACK_ID_SIZE)
    assert torch.all(model.input_scale[region] == 1.0) and torch.all(model.input_bias[region] == 0.0)
    assert float(model.input_scale[0]) == 0.5


def test_export_is_exact_and_loads_from_file(tmp_path):
    model = TrackConditionedNetwork(3, [0.7] * OBSERVATION_SIZE, [0.1] * OBSERVATION_SIZE, (64, 48, 32)).double().eval()
    obs = _observations(25, 2)
    with torch.inference_mode():
        expected = model(torch.from_numpy(obs)).numpy()
    policy = policy_from_model(model)
    policy.save(tmp_path / "policy.gdp")
    loaded = DenseQPolicy.load(tmp_path / "policy.gdp")
    actual = np.asarray([loaded.evaluate(row) for row in obs])
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    assert [loaded.action(row) for row in obs] == list(expected.argmax(axis=1))


def test_layer_norm_critic_cannot_be_exported_but_actor_can():
    config = {"algorithm": {"network": "track_conditioned", "hidden_sizes": [16, 8], "critic_layer_norm": True,
                            "kind": "sac_redq"},
              "normalization": {"input_scale": [1.0] * OBSERVATION_SIZE, "input_bias": [0.0] * OBSERVATION_SIZE},
              "seeds": {"parameter_initialization": 1}}
    critic, actor = build_network(config, "critic"), build_network(config, "actor")
    assert critic.norms is not None and actor.norms is None
    policy_from_model(actor)
    try:
        policy_from_model(critic)
    except ValueError:
        pass
    else:
        raise AssertionError("LayerNorm network exported")


def test_excluded_inputs_are_ignored_by_model_and_export():
    from gravity_lab_rl.model import DEFAULT_EXCLUDED_INPUTS

    model = TrackConditionedNetwork(9, hidden_sizes=(24, 16)).double().eval()
    assert model.layers[0].in_features == OBSERVATION_SIZE - len(DEFAULT_EXCLUDED_INPUTS)
    obs = _observations(10, 5)
    perturbed = obs.copy()
    perturbed[:, list(DEFAULT_EXCLUDED_INPUTS)] += 100.0
    with torch.inference_mode():
        torch.testing.assert_close(model(torch.from_numpy(obs)), model(torch.from_numpy(perturbed)))
    policy = policy_from_model(model)
    first = np.asarray(policy.layers[0].weights)
    assert np.all(first[:, list(DEFAULT_EXCLUDED_INPUTS)] == 0.0)
    expected = model(torch.from_numpy(obs)).detach().numpy()
    np.testing.assert_allclose([policy.evaluate(r) for r in perturbed], expected, atol=1e-9)


def test_plain_mlp_variant_ignores_track_id_and_exports_exactly():
    model = TrackConditionedNetwork(4, hidden_sizes=(32, 24, 16), excluded_inputs=(1,),
                                    track_conditioning=False).double().eval()
    region = list(range(ACCELERATION_REGION_END, ACCELERATION_REGION_END + TRACK_ID_SIZE))
    assert not set(region) & set(model.active_inputs.tolist()) and 1 not in model.active_inputs.tolist()
    assert model.heads.out_features == ACTION_COUNT
    obs = _observations(12, 8)
    other = obs.copy()
    other[:, region] = 0.0
    other[:, region[0]] = 1.0  # every row now claims track 0
    with torch.inference_mode():
        expected = model(torch.from_numpy(obs))
        torch.testing.assert_close(expected, model(torch.from_numpy(other)))
    policy = policy_from_model(model)
    assert [len(layer.bias) for layer in policy.layers] == [32, 24, 16, ACTION_COUNT]
    np.testing.assert_allclose([policy.evaluate(r) for r in obs], expected.numpy(), atol=1e-9)
