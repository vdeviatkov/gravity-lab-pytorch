import numpy as np
import torch

from gravity_lab_rl import OBSERVATION_SIZE
from gravity_lab_rl.checkpoint import load_checkpoint, save_checkpoint
from gravity_lab_rl.model import DenseQNetwork
from gravity_lab_rl.replay import ReplayBuffer


def test_checkpoint_round_trip_preserves_optimizer_and_replay(tmp_path):
    model = DenseQNetwork(11)
    target = DenseQNetwork(12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, OBSERVATION_SIZE)).sum()
    loss.backward(); optimizer.step()
    replay = ReplayBuffer(10, 17)
    replay.add(np.zeros(OBSERVATION_SIZE), 3, 1.5, np.ones(OBSERVATION_SIZE), False, True)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, {
        "online_network": model.state_dict(), "target_network": target.state_dict(),
        "optimizer": optimizer.state_dict(), "replay_buffer": replay.state_dict(),
        "transition_count": 1, "optimizer_update_count": 1,
    })
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()
    saved = load_checkpoint(path)
    restored_model = DenseQNetwork(999)
    restored_model.load_state_dict(saved["online_network"])
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=9e-3)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_replay = ReplayBuffer(10, 999)
    restored_replay.load_state_dict(saved["replay_buffer"])
    assert optimizer.state_dict()["state"].keys() == restored_optimizer.state_dict()["state"].keys()
    assert len(restored_replay) == 1 and restored_replay.position == 1
    assert restored_replay.actions[0] == 3 and restored_replay.truncated[0]
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored_model.parameters()))

