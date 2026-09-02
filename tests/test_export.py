import json

import numpy as np
import torch
from gravity_lab import DenseQPolicy

from gravity_lab_rl import ACTION_COUNT, ENVIRONMENT_ID, OBSERVATION_SIZE
from gravity_lab_rl.checkpoint import save_checkpoint
from gravity_lab_rl.export import export_checkpoint, load_policy_into_model, policy_from_model
from gravity_lab_rl.model import DenseQNetwork


def test_pytorch_dense_policy_q_value_parity():
    model = DenseQNetwork(31, [0.5] * OBSERVATION_SIZE, [-0.25] * OBSERVATION_SIZE).eval().double()
    policy = policy_from_model(model)
    observations = np.random.default_rng(99).normal(size=(8, OBSERVATION_SIZE)).astype(np.float64)
    with torch.inference_mode():
        expected = model(torch.from_numpy(observations)).numpy()
    actual = np.asarray([policy.evaluate(row) for row in observations])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_exported_contract_and_atomic_file(tmp_path):
    model = DenseQNetwork(11)
    checkpoint = tmp_path / "latest.pt"
    config = {"seeds": {"parameter_initialization": 11}, "environment": {},
              "algorithm": {"hidden_sizes": [128, 128]}}
    normalization = {"kind": "identity", "input_scale": [1.0] * OBSERVATION_SIZE,
                     "input_bias": [0.0] * OBSERVATION_SIZE}
    save_checkpoint(checkpoint, {
        "online_network": model.state_dict(), "config": config, "normalization": normalization,
        "metadata": {}, "transition_count": 4, "optimizer_update_count": 0,
    })
    output = export_checkpoint(checkpoint, tmp_path / "latest.gdp")
    policy = DenseQPolicy.load(output)
    assert policy.environment_id == ENVIRONMENT_ID
    assert policy.observation_size == OBSERVATION_SIZE
    assert policy.action_count == ACTION_COUNT
    sidecar = json.loads((tmp_path / "latest.gdp.json").read_text())
    assert sidecar["environment_id"] == ENVIRONMENT_ID
    assert sidecar["observation_size"] == OBSERVATION_SIZE and sidecar["action_count"] == 9
    assert not (tmp_path / "latest.gdp.tmp").exists()


def test_portable_policy_can_initialize_training_model(tmp_path):
    source = DenseQNetwork(41, [0.5] * OBSERVATION_SIZE, [-0.25] * OBSERVATION_SIZE).eval()
    policy_path = tmp_path / "source.gdp"
    policy_from_model(source).save(policy_path)
    restored = DenseQNetwork(99).eval()
    normalization = load_policy_into_model(restored, policy_path)
    observations = torch.randn(7, OBSERVATION_SIZE, generator=torch.Generator().manual_seed(12))
    with torch.inference_mode():
        torch.testing.assert_close(restored(observations), source(observations), rtol=0, atol=0)
    assert normalization["input_scale"] == [0.5] * OBSERVATION_SIZE
    assert normalization["input_bias"] == [-0.25] * OBSERVATION_SIZE
