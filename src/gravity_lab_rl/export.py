from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from gravity_lab import DenseLayer, DenseQPolicy

from . import ACTION_COUNT, ENVIRONMENT_ID
from .checkpoint import load_checkpoint
from .config import valid_observation_size
from .model import DenseQNetwork, TrackConditionedNetwork, build_network

# Head-selection mask offset for exported track-conditioned policies (see
# policy_from_track_conditioned). Any head output whose magnitude stays below this is selected
# exactly; the loader computes in double precision, so the offset costs no accuracy on the
# selected head (M*1 - M is exactly zero) and only needs to exceed |logit| for the others.
HEAD_MASK_OFFSET = 1.0e5


def policy_from_dense(model: DenseQNetwork) -> DenseQPolicy:
    return DenseQPolicy(
        environment_id=ENVIRONMENT_ID,
        input_scale=model.input_scale.detach().cpu(),
        input_bias=model.input_bias.detach().cpu(),
        layers=[
            DenseLayer.from_values(model.fc1.weight, model.fc1.bias, "relu"),
            DenseLayer.from_values(model.fc2.weight, model.fc2.bias, "relu"),
            DenseLayer.from_values(model.q.weight, model.q.bias, "linear"),
        ],
    )


def policy_from_track_conditioned(model: TrackConditionedNetwork) -> DenseQPolicy:
    """Exact plain-MLP form of a TrackConditionedNetwork for the relu/linear `.gdp` format.

    - The track one-hot is carried through every hidden layer as extra pass-through units
      (identity weights; relu leaves 0/1 values unchanged), so later layers can read it.
    - The per-track heads become one relu layer of 2 * tracks * actions units: for head (t, a)
      with pre-activation z, unit `pos` = relu(z + M*(onehot_t - 1)) and unit `neg` =
      relu(-z + M*(onehot_t - 1)). For the selected track both see +0 and pos - neg = z exactly;
      for every other track both see -M and are 0 as long as |z| < M.
    - A final linear layer sums pos - neg per action.
    Requires `layer_norm=False` (the format has no normalization layer); critics are never exported.
    """
    if model.norms is not None:
        raise ValueError("a LayerNorm network cannot be exported to the dense policy format")
    tracks, actions = model.track_count, ACTION_COUNT
    if not model.track_conditioning:
        # Plain MLP: sequential relu layers (zero weight on excluded observation entries) + linear head.
        layers = []
        for index, layer in enumerate(model.layers):
            weight = layer.weight.detach().cpu().double().numpy()
            if index == 0:
                block = np.zeros((weight.shape[0], len(model.input_scale)))
                block[:, model.active_inputs.detach().cpu().numpy()] = weight
                weight = block
            layers.append(DenseLayer.from_values(weight, layer.bias.detach().cpu().double().numpy(), "relu"))
        layers.append(DenseLayer.from_values(model.heads.weight.detach().cpu().double().numpy(),
                                             model.heads.bias.detach().cpu().double().numpy(), "linear"))
        return DenseQPolicy(environment_id=ENVIRONMENT_ID, input_scale=model.input_scale.detach().cpu(),
                            input_bias=model.input_bias.detach().cpu(), layers=layers)
    input_size = len(model.input_scale)
    one_hot_columns = np.arange(model.track_start, model.track_end)
    active_inputs = model.active_inputs.detach().cpu().numpy()
    layers: list[DenseLayer] = []
    previous_width = input_size
    for index, layer in enumerate(model.layers):
        weight = layer.weight.detach().cpu().double().numpy()
        bias = layer.bias.detach().cpu().double().numpy()
        hidden = weight.shape[0]
        block = np.zeros((hidden + tracks, previous_width))
        if index == 0:
            block[:hidden, active_inputs] = weight  # excluded observation entries keep zero weight
        else:
            block[:hidden, :weight.shape[1]] = weight
        # Pass-through rows copy the one-hot: from the raw observation for the first layer, from
        # the previous layer's own pass-through units afterwards.
        source = one_hot_columns if index == 0 else np.arange(previous_width - tracks, previous_width)
        block[hidden + np.arange(tracks), source] = 1.0
        layers.append(DenseLayer.from_values(block, np.concatenate([bias, np.zeros(tracks)]), "relu"))
        previous_width = hidden + tracks
    head_weight = model.heads.weight.detach().cpu().double().numpy()  # (tracks*actions, hidden)
    head_bias = model.heads.bias.detach().cpu().double().numpy()
    hidden = head_weight.shape[1]
    units = tracks * actions
    mask = np.zeros((2 * units, previous_width))
    mask_bias = np.zeros(2 * units)
    for track in range(tracks):
        for action in range(actions):
            row = track * actions + action
            for sign, offset in ((1.0, 0), (-1.0, units)):
                mask[offset + row, :hidden] = sign * head_weight[row]
                mask[offset + row, hidden + track] = HEAD_MASK_OFFSET
                mask_bias[offset + row] = sign * head_bias[row] - HEAD_MASK_OFFSET
    layers.append(DenseLayer.from_values(mask, mask_bias, "relu"))
    output = np.zeros((actions, 2 * units))
    for track in range(tracks):
        for action in range(actions):
            output[action, track * actions + action] = 1.0
            output[action, units + track * actions + action] = -1.0
    layers.append(DenseLayer.from_values(output, np.zeros(actions), "linear"))
    return DenseQPolicy(environment_id=ENVIRONMENT_ID, input_scale=model.input_scale.detach().cpu(),
                        input_bias=model.input_bias.detach().cpu(), layers=layers)


def policy_from_model(model: Any) -> DenseQPolicy:
    if isinstance(model, TrackConditionedNetwork):
        policy = policy_from_track_conditioned(model)
    else:
        policy = policy_from_dense(model)
    if not valid_observation_size(policy.observation_size) or policy.action_count != ACTION_COUNT:
        raise ValueError("exported policy dimensions do not match the environment")
    return policy


def load_policy_into_model(model: DenseQNetwork, path: str | Path) -> dict[str, Any]:
    """Initialize a training model from a compatible portable policy."""
    import torch

    policy = DenseQPolicy.load(path)
    if (policy.environment_id != ENVIRONMENT_ID
            or not valid_observation_size(policy.observation_size)
            or policy.action_count != ACTION_COUNT):
        raise ValueError("initial policy is incompatible with gravity-lab-classic-v1")
    if not isinstance(model, DenseQNetwork):
        raise ValueError("only plain dense networks can be initialized from a portable policy")
    expected_hidden = [len(model.fc1.bias), len(model.fc2.bias), ACTION_COUNT]
    if len(policy.layers) != 3 or [len(layer.bias) for layer in policy.layers] != expected_hidden:
        found = "x".join(str(len(layer.bias)) for layer in policy.layers)
        raise ValueError(
            f"initial policy hidden architecture {found} does not match the target model's "
            f"{'x'.join(str(n) for n in expected_hidden)}; use a config with matching hidden_sizes"
        )
    if policy.observation_size != len(model.input_scale):
        raise ValueError(
            "initial policy observation width does not match the target model "
            f"({policy.observation_size} vs {len(model.input_scale)}); use a config whose "
            "normalization vectors match the policy being loaded"
        )
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
    # PPO checkpoints carry an extra value head (ActorCriticNetwork); policy_from_model only ever
    # reads fc1/fc2/q regardless of subclass, so the exported .gdp is identical in shape either way
    # -- the value head is training-only and never exported.
    model = build_network(checkpoint["config"], "actor", normalization=norm)
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
            "environment_id": ENVIRONMENT_ID, "observation_size": len(norm["input_scale"]),
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
