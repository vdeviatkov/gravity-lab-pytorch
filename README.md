# Gravity Lab PyTorch

PyTorch Double DQN training and graphical policy playback for
`gravity-lab-classic-v1`. The faithful game and physics are included as the `gravity-lab/` Git
submodule.

## Quick start

Install native dependencies.

```sh
# macOS
brew install cmake pkg-config sdl2 sdl2_image sdl2_ttf

# Debian/Ubuntu
sudo apt install cmake pkg-config build-essential libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev
```

Windows 11 runs natively; WSL is not required. In PowerShell, install Git, clone, and run setup:

```powershell
winget install --exact --id Git.Git
git clone --recurse-submodules https://github.com/vdeviatkov/gravity-lab-pytorch.git
cd gravity-lab-pytorch
.\scripts\bootstrap.cmd
```

The bootstrap automatically installs Python, CMake, MSYS2, the compiler, and SDL dependencies;
builds the native `.exe`/`.dll` files; creates `.venv`; and installs PyTorch. A UAC prompt may
appear. After installing Git, reopen PowerShell if the `git` command is not found.

On macOS or Linux, clone, set up, train for one minute, and open the unlocked AI player:

```sh
git clone --recurse-submodules https://github.com/vdeviatkov/gravity-lab-pytorch.git
cd gravity-lab-pytorch
./scripts/bootstrap.sh
./scripts/train_one_minute.sh
./scripts/ai_arcade.sh
```

`bootstrap.sh` initializes a missing submodule, builds the native tools, creates `.venv`, and
installs the Python dependencies.

On native Windows, use the matching `.cmd` scripts:

```powershell
.\scripts\train_one_minute.cmd
.\scripts\ai_arcade.cmd
.\scripts\control.cmd status
```

The one-minute run is a pipeline test, not a generally reliable policy.

## Train

```sh
# One-minute smoke test
./scripts/train_one_minute.sh

# Longer run
./scripts/train.sh --duration-seconds 3600

# Train, then open playback
./scripts/train_and_play.sh --duration-seconds 60

# Resume the latest interrupted run
.venv/bin/gravity-lab-rl resume --latest
```

CPU can be faster than MPS for this small model. Add `--device cpu` when desired.

## Watch the model

Use AI Arcade to choose any track, league, playback speed, and episode count:

```sh
./scripts/ai_arcade.sh
./scripts/ai_arcade.sh --run-id RUN_ID
```

Controls:

- Arrow keys: navigate and change values
- Enter: start AI playback
- Escape during playback: return to track selection
- Escape in the selector: exit

For direct playback without the selector:

```sh
./scripts/play_latest.sh
./scripts/play_latest.sh --group 0 --track 2 --league 0 --episodes 20 --fps 100
```

Models may perform poorly on tracks they were not trained on.

## Control training

From another terminal:

```sh
./scripts/control.sh status
./scripts/control.sh pause
./scripts/control.sh resume
./scripts/control.sh stop
```

Add `--run-id RUN_ID` to select a run. Pause, stop, Ctrl-C, and SIGTERM preserve an atomic
checkpoint. Playback runs in a separate process and is safe while training continues.

## Results

Evaluate the latest model:

```sh
.venv/bin/gravity-lab-rl evaluate --latest --episodes 5
```

Runs are stored under `artifacts/<run-id>/`:

- `latest.pt` / `final.pt`: resumable training checkpoints
- `latest.gdp` / `final.gdp`: portable viewer policies
- `metrics.jsonl`: episode metrics
- `metadata.json`: configuration and reproducibility data
- `summary.json`: final evaluation and counters

The environment is deterministic on a fixed track, so repeated greedy evaluations can be
identical.

## Troubleshooting

- Missing submodule: `git submodule update --init --recursive`
- Missing native tools: rerun `./scripts/bootstrap.sh`
- SDL errors: install the native dependencies above and use a graphical desktop session
- Tests: `.venv/bin/pytest`
