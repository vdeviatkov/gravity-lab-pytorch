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

Generate checkpoint replays for selected maps in an existing training run:

```sh
.venv/bin/python scripts/generate_map_overlay.py \
  --run-id sac_redq_finishbonus_only_20260906_021909 --tracks 1:2
```

Omit `--tracks` to render the first map in each level group: `0:0,1:0,2:0`
(three videos). `--tracks all` explicitly selects all 30 maps.
Use `--league N` to override bike league. Nonmatching group/league pairs get a
`_leagueN` suffix so their outputs do not overwrite each other. `--run-dir PATH`
accepts a run outside the standard artifacts directory.

Every `timelapse/t_*.gdp` policy, `final.gdp`, and (when different) `best.gdp` is
replayed from reset to its terminal
frame. Each attempt has a color and numbered legend; complete paths stay visible,
and finished/crashed bikes stay at their last position. The camera shows the whole
map and expands when a bike moves beyond its bounds. The simulation uses the run's
frame skip, episode limit, league, and custom level pack. Custom-pack plates are
stored inside the run instead of replacing the built-in assets.

The default records every environment step at real-time playback speed. Two maps
render concurrently in independent native processes; use `--jobs 1` to serialize
rendering or another positive value to change concurrency. Each encoder uses two
threads. Automatic post-training batches use up to four jobs, configurable with
`experiment.map_overlay_jobs`. A successful batch writes `map_overlay_manifest.json`. Optional
`--speedup 3` speeds playback up; `--step-stride 2` reduces the animation's temporal
resolution; `--trail-length 60` shows only the latest 60 environment steps instead
of the entire path. `--keep-frames` retains raw captures for inspection.

Outputs are `map_overlay_lgG_tT.mp4` and a JSON sidecar listing each policy, its
training timestamp, outcome, and rendering settings. PNG frames are streamed to
FFmpeg as raw RGB, avoiding a second large sequence of intermediate images. A
failed recording fails the job instead of silently omitting a policy. Existing
videos are replaced only after encoding succeeds.

## Automatic generation after training

For DQN, PPO, and SAC/REDQ, every run automatically generates a results plot and
three videos (`0:0,1:0,2:0`) after final evaluation when training reaches its time budget. User-stopped
runs skip rendering unless `experiment.map_overlay_on_stop` is explicitly true. Missing settings default to
`map_overlay_after_training: true`, `map_overlay_tracks: "0:0,1:0,2:0"`,
`training_plot_after_training: true`, and a policy
snapshot every 300 seconds. New runs also save
the initial policy, and resumed runs save their starting policy. The final policy
is always included, even if training ends before the next snapshot interval.

Optional experiment settings:

```json
{
  "timelapse_interval_seconds": 300,
  "map_overlay_after_training": true,
  "map_overlay_tracks": "0:0,1:2"
}
```

Omit `map_overlay_tracks` to generate the three default videos. Set it to `"all"`
for all 30, or a list such as `"0:0,1:2"` for specific maps. Set `timelapse_interval_seconds` to a shorter
interval for more snapshots (the 10-minute all-map run uses 120 seconds). Set
`map_overlay_after_training` to `false` to generate videos manually. Rendering runs
in a separate process after training and does not count against the training time
budget. Progress is logged to `map_overlay_generation.log`; completion or failure
is recorded in `map_overlay_status.json`. A rendering failure preserves all saved
training results.

These videos show deterministic evaluations of saved policies. They are not a
recording of the exploratory actions taken during training. Historical episodes
cannot be reconstructed from policy snapshots alone.

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
