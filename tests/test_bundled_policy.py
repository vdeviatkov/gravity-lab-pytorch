import hashlib
import json
from pathlib import Path

from gravity_lab import DenseQPolicy

from gravity_lab_rl import ACTION_COUNT, BASE_OBSERVATION_SIZE, ENVIRONMENT_ID, OBSERVATION_SIZE
from gravity_lab_rl.playback import bundled_policy
import gravity_lab_rl.playback as playback


def test_bundled_policy_contract_and_checksum():
    path = bundled_policy()
    assert path == Path(__file__).resolve().parents[1] / "policies" / "classic_intro.gdp"
    policy = DenseQPolicy.load(path)
    metadata = json.loads(path.with_suffix(".gdp.json").read_text(encoding="utf-8"))
    assert policy.environment_id == ENVIRONMENT_ID
    # The default bundled policy is the legacy (pre-obstacle-sensor) network.
    assert policy.observation_size == BASE_OBSERVATION_SIZE
    assert policy.action_count == ACTION_COUNT
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["policy_sha256"]


def test_bundled_sensor_policy_contract_and_checksum():
    path = Path(__file__).resolve().parents[1] / "policies" / "classic_intro_sensor.gdp"
    policy = DenseQPolicy.load(path)
    metadata = json.loads(path.with_suffix(".gdp.json").read_text(encoding="utf-8"))
    assert policy.environment_id == ENVIRONMENT_ID
    assert policy.observation_size == OBSERVATION_SIZE
    assert policy.action_count == ACTION_COUNT
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["policy_sha256"]


def test_arcade_falls_back_to_bundled_policy_without_runs(monkeypatch, tmp_path):
    executable = tmp_path / "gravity_lab_ai_arcade"
    executable.touch()
    captured = {}
    monkeypatch.setattr(playback, "require_integration", lambda: None)
    monkeypatch.setattr(playback, "arcade_executable", lambda: executable)
    monkeypatch.setattr(playback, "resolve_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        FileNotFoundError("no runs")))
    monkeypatch.setattr(playback.subprocess, "run", lambda args, check: captured.update(
        {"args": args, "check": check}))
    playback.arcade()
    assert captured["check"] is True
    assert captured["args"][captured["args"].index("--policy") + 1] == str(bundled_policy())


def test_arcade_accepts_explicit_portable_policy(monkeypatch, tmp_path):
    executable = tmp_path / "gravity_lab_ai_arcade"
    executable.touch()
    captured = {}
    monkeypatch.setattr(playback, "require_integration", lambda: None)
    monkeypatch.setattr(playback, "arcade_executable", lambda: executable)
    monkeypatch.setattr(playback.subprocess, "run", lambda args, check: captured.update({"args": args}))
    playback.arcade(policy=bundled_policy())
    assert captured["args"][captured["args"].index("--policy") + 1] == str(bundled_policy().resolve())


def test_model_source_options_are_mutually_exclusive():
    try:
        playback.arcade(run_id="one", policy=bundled_policy())
    except ValueError as error:
        assert "choose only one" in str(error)
    else:
        raise AssertionError("ambiguous model source was accepted")
