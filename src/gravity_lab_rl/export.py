from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from gravity_lab import DenseLayer, DenseQPolicy

from . import ACTION_COUNT, ENVIRONMENT_ID, OBSERVATION_SIZE
from .checkpoint import load_checkpoint
from .model import DenseQNetwork


def policy_from_model(model: DenseQNetwork) -> DenseQPolicy:
    policy = DenseQPolicy(
        environment_id=ENVIRONMENT_ID,
        input_scale=model.input_scale.detach().cpu(),
        input_bias=model.input_bias.detach().cpu(),
        layers=[
            DenseLayer.from_values(model.fc1.weight, model.fc1.bias, "relu"),
            DenseLayer.from_values(model.fc2.weight, model.fc2.bias, "relu"),
            DenseLayer.from_values(model.q.weight, model.q.bias, "linear"),
        ],
    )
    if policy.observation_size != OBSERVATION_SIZE or policy.action_count != ACTION_COUNT:
        raise ValueError("exported policy dimensions do not match the environment")
    return policy


def load_policy_into_model(model: DenseQNetwork, path: str | Path) -> dict[str, Any]:
    """Initialize a training model from a compatible portable policy."""
    import torch

    policy = DenseQPolicy.load(path)
    if policy.environment_id != ENVIRONMENT_ID or policy.observation_size != OBSERVATION_SIZE or policy.action_count != ACTION_COUNT:
        raise ValueError("initial policy is incompatible with gravity-lab-classic-v1")
    if len(policy.layers) != 3 or [len(layer.bias) for layer in policy.layers] != [128, 128, 9]:
        raise ValueError("initial policy architecture must be 28x128x128x9")
    with torch.no_grad():
        model.input_scale.copy_(torch.tensor(policy.input_scale, dtype=model.input_scale.dtype,
                                             device=model.input_scale.device))
        model.input_bias.copy_(torch.tensor(policy.input_bias, dtype=model.input_bias.dtype,
                                            device=model.input_bias.device))
        for module, layer in zip((model.fc1, model.fc2, model.q), policy.layers):
            module.weight.copy_(torch.tensor(layer.weights, dtype=module.weight.dtype,
                                             device=module.weight.device))
            module.bias.copy_(torch.tensor(layer.bias, dtype=module.bias.dtype,
                                           device=module.bias.device))
    return {"kind": "fixed", "input_scale": list(policy.input_scale),
            "input_bias": list(policy.input_bias)}


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path,
                      sidecar: bool = True) -> Path:
    checkpoint = load_checkpoint(checkpoint_path)
    norm = checkpoint["normalization"]
    seed = checkpoint["config"]["seeds"]["parameter_initialization"]
    model = DenseQNetwork(seed, norm["input_scale"], norm["input_bias"])
    model.load_state_dict(checkpoint["online_network"])
    model.eval()
    destination = Path(output_path)
    policy_from_model(model).save(destination)
    if sidecar:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        data: dict[str, Any] = {
            "format": "gravity-lab-dense-q-policy-sidecar-v1",
            "policy": destination.name, "policy_sha256": digest,
            "checkpoint": Path(checkpoint_path).name,
            "environment_id": ENVIRONMENT_ID, "observation_size": OBSERVATION_SIZE,
            "action_count": ACTION_COUNT, "configuration": checkpoint["config"],
            "normalization": norm, "metadata": checkpoint.get("metadata", {}),
            "transition_count": checkpoint["transition_count"],
            "optimizer_update_count": checkpoint["optimizer_update_count"],
        }
        target = destination.with_suffix(destination.suffix + ".json")
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    return destination
