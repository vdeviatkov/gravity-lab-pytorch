import json

from gravity_lab_rl.control import initialize_control, read_control, request_control, update_status


def test_atomic_control_state_changes(tmp_path):
    path = tmp_path / "control.json"
    initialize_control(path, "run-1")
    request_control(path, "pause")
    update_status(path, {"state": "paused", "transitions": 42})
    assert read_control(path)["requested"] == "pause"
    request_control(path, "resume")
    data = json.loads(path.read_text())
    assert data["requested"] == "run" and data["transitions"] == 42
    assert not path.with_suffix(".json.tmp").exists()

