# Full-map training videos

`assets/map_plates/` contains 30 empty, complete game maps (groups 0–2, tracks 0–9).
Open `assets/map_plates/index.jpg` for an overview. Plates use the original game's
terrain renderer and flag sprites, with one fixed perspective across the whole map.
They do not require a successful policy or stitched playthroughs.

Build the updated viewer and generate the assets:

```sh
cmake --build gravity-lab/build-classic-rl -j 4
.venv/bin/python -m pip install -e '.[video]'
# Install ffmpeg with your system package manager if it is not already available.
.venv/bin/python scripts/generate_map_plates.py
```

## Recorded attempts and automatic finalization

DQN, PPO, and SAC/REDQ save **every actual training attempt on every visited map**
by default. At the training time limit, final evaluation is followed automatically
by a results plot and three videos: the first map in each group (`0:0,1:0,2:0`).
Only maps with recorded attempts produce a video. Training remains headless; PNG
capture and video encoding run afterward and do not consume the training budget.

Each video shows attempts 1–20 together, then 21–40, and so on, in recorded order.
Every environment step is replayed with its original action and reset seed. The
full map and complete paths remain visible; bikes stay at their final position
until the batch ends. The final batch can contain fewer than 20 bikes. Attempts
interrupted by stopping or evaluation are included and labeled partial. Practice
reconstruction actions are included and their prefix length is labeled separately.
These recordings cover training rollouts, not separate evaluation episodes.

One command regenerates both the plot and videos without starting training:

```sh
.venv/bin/python scripts/finalize_training.py --run-id RUN_ID
# All recorded maps, still one video per map:
.venv/bin/python scripts/finalize_training.py --run-id RUN_ID --tracks all
# Specific maps:
.venv/bin/python scripts/finalize_training.py --run-id RUN_ID --tracks 0:0,1:2
```

User-stopped runs skip expensive final rendering unless `map_overlay_on_stop` is
explicitly true. Their recordings are retained, so the same command works later.
The stopped historical run can still produce a plot and checkpoint replays, but
its unrecorded exploratory actions cannot be recovered. If an old run is resumed,
only subsequent training is recorded; `recording_sessions.jsonl` and the video
manifest disclose whether recording was enabled from the beginning.

For video-only generation or advanced playback settings:

```sh
.venv/bin/python scripts/generate_map_overlay.py --run-id RUN_ID --tracks all
```

To make a chronological training video from up to 200 attempts, with 20 bikes
visible at the same time in each batch, use:

```sh
# macOS/Linux
.venv/bin/python scripts/generate_map_overlay.py \
  --run-id RUN_ID --tracks 2:0 \
  --source training --batch-size 20 --max-attempts-per-map 200 \
  --step-stride 1 --speedup 4

# Windows PowerShell
.venv\Scripts\python.exe scripts\generate_map_overlay.py `
  --run-id RUN_ID --tracks 2:0 `
  --source training --batch-size 20 --max-attempts-per-map 200 `
  --step-stride 1 --speedup 4
```

When more than 200 attempts are available, the renderer selects 200 attempts
evenly across the recorded training timeline. When fewer are available, it
uses every recorded attempt; it does not invent or duplicate runs. Therefore,
the result may contain fewer than 10 batches, and the final batch may contain
fewer than 20 bikes. Use `--tracks 2:0,2:1,...` or `--tracks all` to render
multiple maps.

`--source training` requires actual action recordings. The default `--source auto`
uses these when present and otherwise generates clearly labeled legacy checkpoint
replays. `--source checkpoints` explicitly replays saved policies instead. Legacy
replays include every `timelapse/t_*.gdp`, `final.gdp`, and distinct `best.gdp`.
They are deterministic evaluations, not historical training attempts.

Optional experiment settings (these are the defaults):

```json
{
  "record_training_episodes": true,
  "map_overlay_after_training": true,
  "map_overlay_tracks": "0:0,1:0,2:0",
  "map_overlay_batch_size": 20,
  "training_plot_after_training": true
}
```

Set `map_overlay_tracks` to `"all"` for every map. Set `map_overlay_after_training`
to false for manual rendering while retaining recordings. The standalone finalizer
explicitly enables plots and videos for its invocation. `--batch-size` overrides
20; `--run-dir PATH` accepts a run outside `artifacts/`.

`training_episodes/` stores each attempt's configuration, seed, start time, outcome,
and one-byte-per-action file. Actions are written after each successful step without
userspace buffering. A process crash leaves a replayable incomplete prefix. Resumes
create unique files without overwriting earlier attempts. Disk/power failures are
not protected by per-step fsync. Keep the same native physics build and custom level
pack for faithful replay. No trained network inference is needed for action replay.

Outputs are `map_overlay_lgG_tT.mp4`, JSON sidecars listing every attempt and batch
size, and `map_overlay_manifest.json`. Different bike leagues get a `_leagueN`
suffix. Each batch uses the full game-rendered map and expands to contain off-map
movements, then fits into a shared output size without cropping. Native captures
are removed after each batch, bounding temporary image storage. Batches are joined
into one MP4 without another encoding pass; existing videos are replaced only
when the complete map succeeds.

Manual rendering uses two independent map processes by default (`--jobs N`).
Automatic rendering uses up to four, configurable with `map_overlay_jobs`. Each
encoder uses two threads. `--speedup 3` speeds playback without dropping steps;
`--trail-length 60` shortens paths, and `--keep-frames` retains captures. Actual
training videos require `--step-stride 1` to retain every recorded movement.
`--league N` filters recorded attempts by their original league; legacy policy
replays instead use it as an override.

Progress and failures are reported in `map_overlay_generation.log` and
`map_overlay_status.json`. A render failure preserves training results and action
recordings. The finalizer exits unsuccessfully if either plot or video generation
fails. The native viewer accepts `--actions FILE --episodes 1` for direct action
replay using the original group, track, league, frame skip, episode limit, and seed.

## Coordinates and assets

A plate's sidecar gives `min_ox`, `min_oy`, width, and height. World pixel `(x,y)`
maps to `(x - min_ox, -y - min_oy)`. Level vertices have half the physics fixed-point
scale, so the geometry accessor uses the game's own `<<3>>16` conversion.

The viewer's legacy `positions.csv` columns `bike_x,bike_y` contain the viewport
origin, not the bike center. With recording look-ahead disabled, add `(320,240)`
after flipping the vertical coordinate to locate the bike. `--bike-only` renders
an isolated bike layer, so masking it requires no guesses about terrain colors.

The renderer supports `--map-plate PATH.png` without a policy. It renders the whole
level into an SDL target texture and preserves the full target's clipping bounds
when drawing flag sprites. No physics steps, teleports, or policy recordings are
used to make a plate. Assets derive from the GPL-2.0-only vendored classic game;
see `gravity-lab/classic/LICENSE.md` and `GRAVITY_LAB_CHANGES.md` for attribution.


## Training results plot

Every completed run also generates `progress.png` and a scalable `progress.svg`,
independently of whether video generation is enabled. The plot shows fixed full-start
evaluation finishes and mean progress, rolling training reward and peak progress,
map coverage over time, and completed episodes per map. Full-start training and
obstacle-practice suffixes are displayed separately. When a stored equal-time baseline
evaluation is present, it appears as a reference line on the evaluation panels.

Regenerate a plot without rendering videos:

```sh
.venv/bin/python scripts/plot_progress.py --run-id RUN_ID
```

Set `experiment.training_plot_after_training` to `false` to disable automatic plots.
Plot generation logs and status are in `training_plot_generation.log` and
`training_plot_status.json`. Older runs without evaluation history show their final
evaluation where available; the plot does not invent intermediate evaluation points.
