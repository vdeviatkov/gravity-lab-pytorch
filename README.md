# Gravity Lab PyTorch

A small, standalone PyTorch Double DQN experiment repository for the faithful
`gravity-lab-classic-v1` environment. The `gravity-lab/` Git submodule provides the game, native
physics, Python bindings, portable policy format, and graphical viewer. This repository owns all
neural-network, replay-buffer, training, control, and experiment code.

## Setup

```sh
git clone --recurse-submodules https://github.com/vdeviatkov/gravity-lab-pytorch.git
cd gravity-lab-pytorch
./scripts/bootstrap.sh
```

If the repository was cloned without `--recurse-submodules`, `bootstrap.sh` initializes the
submodule automatically. The equivalent manual command is:

```sh
git submodule update --init --recursive
```

The script builds the submodule's native classic library and graphical viewer when absent, creates
`.venv`, and installs this project, Gravity Lab, PyTorch, NumPy, and pytest. No Gymnasium or RL
framework is used. On a platform where `python3` is unsuitable, use
`PYTHON=python3.12 ./scripts/bootstrap.sh`.

Native build prerequisites are CMake 3.20+, a C++20 compiler, pkg-config, SDL2, SDL2_image, and
SDL2_ttf. On macOS, install them with:

```sh
brew install cmake pkg-config sdl2 sdl2_image sdl2_ttf
```

On Debian or Ubuntu:

```sh
sudo apt install cmake pkg-config build-essential libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev
```

For advanced development, `GRAVITY_LAB_REPO=/path/to/another/checkout` can override the bundled
submodule. `GRAVITY_LAB_CLASSIC_LIBRARY=/path/to/library` can override the native library.

## Train

The quickest complete pipeline test is:

```sh
./scripts/train_one_minute.sh
```

This means approximately 60 seconds of *active* training, excluding paused time. One minute only
tests the pipeline; it is not expected to learn a reliably successful motorcycle policy.

Other common commands:

```sh
./scripts/train.sh --duration-seconds 3600
./scripts/train_and_play.sh --duration-seconds 60
.venv/bin/gravity-lab-rl resume --latest
```

`resume` restores both networks, Adam state, replay contents and position, counters, schedule,
normalization, named exploration/replay RNGs, and recorded Python/NumPy/PyTorch RNG states. The
classic API cannot serialize exact mid-episode physics, so a resumed trainer starts a fresh
episode while preserving all learning state. A duration override is the total active-duration
target for that run; omit it to continue toward the saved configuration's target.

The default device is `auto` (CUDA, then MPS, then CPU). This network is small enough that CPU can
be faster than Apple MPS because accelerator dispatch overhead may dominate. Force a backend with
`--device cpu`, `--device mps`, or `--device cuda`.

## Pause, inspect, resume, and stop

From a second terminal:

```sh
./scripts/control.sh status
./scripts/control.sh pause
./scripts/control.sh resume
./scripts/control.sh stop
```

Commands default to the most recently modified run. Add `--run-id RUN_ID` to select one. Pause is
acknowledged at a safe environment-step boundary, writes an atomic checkpoint/export, stops both
collection and optimization, and keeps the process alive without counting paused time. Stop,
Ctrl-C, and SIGTERM preserve `latest.pt`, write `final.pt` and `final.gdp`, evaluate, summarize, and
exit. An unexpected exception attempts to preserve `latest.pt`/`latest.gdp` before re-raising.
Status reports the state, PID, transitions, optimizer updates, episodes, active time, epsilon,
latest reward/progress, and checkpoint path.

## Evaluate and watch

Greedy evaluation uses epsilon exactly zero:

```sh
.venv/bin/gravity-lab-rl evaluate --latest --episodes 5
./scripts/play_latest.sh
./scripts/play_latest.sh --run-id RUN_ID --episodes 5 --fps 25 --seed 2000007
```

Playback can explicitly override `--group`, `--track`, or `--league`. Otherwise it uses the saved
training configuration. With `frame_skip=2`, playback defaults to 25 FPS so displayed agent steps
roughly follow simulated real time. The viewer runs as a separate OS process and greedily loads an
atomic snapshot, so `play_latest.sh` is safe while training continues and never creates a second
`ClassicGravityEnv` in the trainer process.

The classic engine stores process-global state: create only one active `ClassicGravityEnv` per
process. Training and evaluation are sequential; graphical viewing is a separate executable.

The current classic environment is deterministic. Reset seeds are recorded and independently
named for future compatibility, but different reset seeds currently do not diversify physics on
the same track. Repeated evaluation of one deterministic greedy policy and configuration may
therefore produce identical episodes. Future generalization evaluation should use a predefined
suite of tracks and leagues. Final-evaluation seeds are never used to select a checkpoint.

## Checkpoints and artifacts

Each run is under `artifacts/<run-id>/`:

```text
config.json             resolved configuration
metadata.json           commits, platform, versions, seeds, device, timing
metrics.jsonl           flushed episode records
control.json            atomic requested command and live status
latest.pt               latest resumable atomic checkpoint
latest.gdp              latest portable greedy policy
latest.gdp.json         deployment sidecar and SHA-256
final.pt                graceful-final resumable checkpoint
final.gdp               graceful-final portable policy
final.gdp.json          final deployment sidecar
summary.json            counters, timing, paths, and final evaluation
```

`.pt` is the PyTorch source of truth for resuming and contains the online/target weights, optimizer,
replay buffer, counters, epsilon state, RNG state, configuration, metrics, metadata, and exact input
normalization. `.gdp` is the dependency-free `gravity-lab-dense-q-policy-v1` deployment artifact
used by the C++ viewer. It contains the same normalization and online dense Q-network, but no
optimizer or replay state. `latest` is selected by recency—not by final evaluation results.

Manual export and viewer validation:

```sh
.venv/bin/gravity-lab-rl export --latest
./gravity-lab/build-classic-rl/gravity_lab_classic_viewer \
  --policy artifacts/RUN_ID/latest.gdp --validate-only
```

## Configuration and algorithm

Edit or copy `configs/classic_intro.json`, then pass `--config FILE`. Defaults are one environment,
28→128→128→9 ReLU Q-network, Double DQN, Adam at `1e-3`, gamma `0.99`, batch 128, replay 100,000,
5,000-transition warm-up, one update per four transitions, hard target copy per 2,000 optimizer
updates, Huber loss, gradient clipping at 10, and epsilon 1.0→0.05 over 200,000 transitions. Input
normalization is explicitly identity. Environment, parameter initialization, epsilon exploration,
replay sampling, validation, and final-evaluation seeds are separate configuration fields.

## Troubleshooting

- “Python bindings missing”: initialize the submodule and rerun `bootstrap.sh`.
- “native library missing”: build the game with `GRAVITY_LAB_BUILD_CLASSIC=ON` as documented in
  `game/docs/classic-rl.md`, or set `GRAVITY_LAB_CLASSIC_LIBRARY`.
- “viewer missing”: build the `gravity_lab_classic_viewer` target in `build-classic-rl`.
- SDL window/library errors: install SDL2, SDL2_image, SDL2_ttf and run from a graphical login
  session. Headless training and tests do not need a display.
- macOS MPS trouble or poor throughput: use `--device cpu`.

Run tests with `.venv/bin/pytest`. The optional native integration test skips with its missing path
when the classic library or Python binding is unavailable.
