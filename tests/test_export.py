import json

import numpy as np
import torch
from gravity_lab import DenseQPolicy

from gravity_lab_rl import ACTION_COUNT, ENVIRONMENT_ID, OBSERVATION_SIZE
from gravity_lab_rl.checkpoint import save_checkpoint
from gravity_lab_rl.export import export_checkpoint, policy_from_model
from gravity_lab_rl.model import DenseQNetwork


def test_pytorch_dense_policy_q_value_parity():
    model = DenseQNetwork(31, [0.5] * 28, [-0.25] * 28).eval().double()
    policy = policy_from_model(model)
    observations = np.random.default_rng(99).normal(size=(8, 28)).astype(np.float64)
    with torch.inference_mode():
        expected = model(torch.from_numpy(observations)).numpy()
    actual = np.asarray([policy.evaluate(row) for row in observations])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_exported_contract_and_atomic_file(tmp_path):
    model = DenseQNetwork(11)
    checkpoint = tmp_path / "latest.pt"
    config = {"seeds": {"parameter_initialization": 11}, "environment": {}, "algorithm": {}}
    normalization = {"kind": "identity", "input_scale": [1.0] * 28, "input_bias": [0.0] * 28}
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
    assert sidecar["observation_size"] == 28 and sidecar["action_count"] == 9
    assert not (tmp_path / "latest.gdp.tmp").exists()
