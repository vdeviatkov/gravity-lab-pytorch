# Training one network for all 30 maps: search-found demos + backward curriculum

This document describes the approach behind `configs/classic_all_tracks_demo_noid.json`, the
current default for training a single policy that should clear every map. It replaces the
plain "train SAC on all maps and hope" loop described in `docs/training-runs.md`, which
plateaued at 7-9 of 30 maps under every algorithm, reward and curriculum tried.

Contents:

1. [Why the old approach stalled](#1-why-the-old-approach-stalled)
2. [Step 1: search one demonstration per map](#2-step-1-search-one-demonstration-per-map)
3. [Step 2: train one network from the demos](#3-step-2-train-one-network-from-the-demos)
4. [The network and its inputs](#4-the-network-and-its-inputs)
5. [Evaluation and checkpoint selection](#5-evaluation-and-checkpoint-selection)
6. [How to run it](#6-how-to-run-it)
7. [Knobs and what they do](#7-knobs-and-what-they-do)
8. [Known issues](#8-known-issues)
9. [Results so far](#9-results-so-far)

## 1. Why the old approach stalled

Two facts about this environment shape everything below.

**The game is deterministic.** `gravity-lab-classic-v1` has no randomness: the seed exists for
bookkeeping only, and three evaluation seeds always produce byte-identical episodes. A map is
therefore *solved* the moment one action sequence reaches the finish, and evaluation is a
single deterministic run per map.

**A shared network fails in two unrelated ways.** Reading the 8-hour coverage run
(`saved_runs/sac_redq_8hr_coverage_20260906_022918`) and the per-map specialist runs showed:

- *Exploration.* On maps such as Deep, Spikeholes or Dantes Peak the policy crashed at the same
  obstacle in 95-100% of thousands of attempts. The move needed there is a *setup*: brake, lean
  back, hold throttle for a second or two. At frame skip 2 that is 25-50 consecutive identical
  decisions, which per-step policy noise essentially never produces. Reward shaping cannot
  reward a maneuver the policy never performs.
- *Retention.* With uniform exposure (about 278 episodes per map) the set of maps the network
  could finish flickered between 3 and 7 from one evaluation to the next. Gradient updates for
  one map overwrote another map's solution.

Reward constants, gamma, staged unlocking and adaptive track weighting touch neither problem,
which is why every variant landed in the same 7-9 range. The approach below attacks each
problem separately: search for the moves, then train one network from what search found.

## 2. Step 1: search one demonstration per map

Code: `src/gravity_lab_rl/explore.py`, driver `scripts/explore_maps.py`. No neural network.

Because the game is deterministic, *one* finishing action sequence is a complete solution, and
any state can be reconstructed exactly by resetting with the fixed seed and replaying the
actions that led there (the same reconstruction `practice.py` used). That makes Go-Explore
(Ecoffet et al. 2019) a natural fit:

1. **Archive of cells.** A cell is a coarse bucket of bike state: `(progress in 2% bins,
   center horizontal velocity in 0.4 bins clipped to +-4, wheel pitch in 30 degree bins)`.
   Each cell remembers the *shortest* action prefix that reached it.
2. **Pick a cell.** Half the picks come from the *frontier*, i.e. cells within 6% progress of
   the best progress seen so far; the other half are drawn with weight `1 / sqrt(1 + times
   chosen)` so rarely tried cells get another chance. This keeps pressure on the furthest
   obstacle while still revisiting earlier situations that might lead past it with a
   different approach speed or angle.
3. **Return, then explore.** Reset, replay the cell's prefix, then take random actions for up
   to 150 steps. Each random action is *held* for a geometric number of steps, mean 8, or a
   mean drawn from `{3, 8, 20}` in the later passes. Holding is the point: it makes the setup
   maneuvers reachable. Every state visited during exploration is offered to the archive; new
   cells are added and existing cells keep the shorter prefix.
4. **Finish.** The first prefix whose last step reports `finished` is the map's demonstration.
   It is replayed once more to verify, then saved to `demos/lg<G>_t<T>.json` along with the
   league, frame skip, ray count, episode limit and seed it was found under. `load_demos`
   refuses a demo whose settings differ from the training config.

Practicalities: the engine allows one environment per process, so the driver runs one worker
process per map (`--workers` in parallel) at 6-10k simulator steps per second. The vendored
physics engine occasionally hangs inside a native call; the worker writes its archive to disk
every 15 seconds as a heartbeat, and the driver kills a worker whose archive stops updating for
90 seconds and restarts it from that archive with a new random seed, so a hang costs at most
90 seconds. Archives are per search variant (`lg<G>_t<T>.archive_<seed>.json`) so parallel
variants do not mask each other's heartbeat.

All 30 maps were solved on 2026-09-07: 24 within a 900-second first pass (Deep in 52 s, Hole
48 s, Spikeholes 97 s), the remaining six with more time or the variable hold lengths ("100%"
took 1545 s and an 1885-step sequence). Demos are between 437 and 1885 steps long.

The demo is **not the player**. It is a jerky, random-looking sequence that happens to work
from one exact start state. Its job is to tell training where the finish is reachable from.

## 3. Step 2: train one network from the demos

Code: `src/gravity_lab_rl/demo_curriculum.py`, wired into `src/gravity_lab_rl/sac_trainer.py`
under the `demos` config block. The base algorithm is unchanged discrete SAC with REDQ critics
(runs 18-24 in `docs/training-runs.md`): one categorical actor, four Q critics updated toward a
target built from a random pair of them, entropy temperature auto-tuned to a target entropy,
n-step (3) returns, per-track-balanced replay, gamma 0.99, batch 128, one update per
environment step. Three things are added around it.

### 3.1 Backward start curriculum

Every episode picks a map (all 30 are active from the first minute; coverage turns pick the
least-trained map and the other turns weight maps by inverse recent success). Then one of two
things happens:

- **Demo-start episode** (80% of episodes on a map with a demo): the simulator replays the first
  `prefix` actions of that map's demo, and the network takes over from there. Prefix steps are
  not learned from; the peak progress reached during the prefix is reconstructed so re-covering
  demo ground earns nothing; `mark_practice_prefix` records the prefix so training videos show
  where the policy took over.
- **Full-start episode** (the remaining 20%, and every episode on a map without a demo):
  a normal start from the start line.

The takeover point per map begins `initial_remaining` (50) demo steps before the finish, so the
network first has to learn only the last two seconds of each map. When at least 3 of the last 4
*greedy* demo-start episodes on that map finish, the point moves `step_back` (80) steps toward
the start. Six greedy failures in a row move it forward again. When it reaches step 0 the map
has **graduated** and is trained entirely from the real start. The network therefore only ever
learns a short new stretch in front of something it can already do (Salimans & Chen 2018).

### 3.2 Greedy gating of the curriculum

Half of the demo-start episodes run **greedily** (argmax action, no exploration), and only those
count toward advancing or retreating the takeover point. The other half sample from the actor
and repeat the previous action with probability `sticky_action_probability` (0.3) for
temporally extended exploration, and are used only as training data.

This matters more than it sounds. Before greedy gating, stages were judged on the sampled
policy, which drifts off the demo somewhere in almost every 100-step stage; success was 30-40%
and the walk-back stalled at 36% with retreats climbing. Meanwhile an offline check showed the
actor already picked the demo action in 98% of demo states. Judging stages on greedy episodes,
the same policy that formal evaluation measures, lifted stage success to about 70% and the
walk-back pace from 2 to 5 percentage points per 15 minutes.

### 3.3 Demonstration replay and behavior cloning

At startup every demo is replayed once and its transitions, with the same reward function and
n-step treatment as online data, are stored in a second, permanent, per-track-balanced replay
buffer. Each optimizer step:

- concatenates `bc_batch_size` (64) demo transitions to the 128 online transitions for the
  critic update, so the critics see finishing trajectories for every map at every step;
- adds `bc_weight` (1.0) times the cross-entropy between the actor's distribution and the demo
  action on that demo slice (DQfD-style), which pins the actor to the demo along the demo's own
  states. This is the anchor that stops one map's learning from erasing another's: it is a fixed
  supervised target that never moves, unlike Q targets.

Reward, for completeness (`reward.py`, values from the config): +50 on finish, -5 on crash,
-0.1 on any step without new peak progress, +0.1 per percent of new peak progress scaled by
`1 + progress`, plus a quadratic speed bonus (0.25) on that progress.

## 4. The network and its inputs

Code: `TrackConditionedNetwork` in `src/gravity_lab_rl/model.py`, selected by
`algorithm.network: "track_conditioned"`. Two variants exist.

**No-map-identity variant** (`track_conditioning: false`, the current default). A plain MLP:
89 inputs, hidden layers `[512, 512, 256]` with ReLU, one 9-way head. The critics have the same
shape plus LayerNorm; the actor has none because it must export. The 89 inputs are the engine's
134-value observation minus the entries listed under `excluded_inputs`:

| Observation index | Count | Meaning |
|---|---|---|
| 0 | 1 | Progress along the track, about -0.2 at the start line to ~0.97 at the finish |
| 2 | 1 | Race started flag |
| 3 | 1 | Bike league / 3 (engine class) |
| 6, 7 | 2 | Center point velocity x, y |
| 8 to 27 | 20 | Five other physics points: offset x, y from the center, velocity x, y (front wheel, rear wheel, three frame and rider points) |
| 28 to 59 | 32 | Obstacle rays from the bike center, every 11.25 degrees, distance to the ground polyline in [0, 1] |
| 102 to 133 | 32 | Head-clearance rays from the rider's head, same directions |

Excluded: index 1 (`1 - progress`, redundant); indices 4-5 (point 0's offset from itself,
always zero); 60-71 (the "acceleration" region, always zero because the engine reads it from an
integrator slot whose force accumulators are never written); 72-101 (the track one-hot).

The one-hot was dropped deliberately. With map identity plus progress, the network can
memorize each map as a position-indexed action sequence, which scores on the deterministic
evaluation but is not a player. Without it, the terrain must be read from the rays, so what the
network learns transfers between maps. In a side-by-side run the no-identity variant was ahead
of the identity variant on every metric after 27 minutes, so the identity run was stopped.

**Map-identity variant** (`track_conditioning: true`, kept for comparison). The one-hot is
re-injected as extra inputs to every hidden layer (a learned per-map bias per layer) and the
output is one 9-way head *per map*, selected by the observation's track id, so the layer where
cross-map interference lands is not shared. `export.py` writes this as an exact plain-MLP
`.gdp`: the one-hot is carried through the trunk as pass-through units and the heads become a
relu mask layer (`tests/test_track_conditioned.py` checks parity to 1e-9).

**Normalization.** `normalization.kind: "demo_statistics"` computes a fixed per-feature
scale and shift from the demo transitions at the first start, freezes it into the checkpoint
and export, and leaves constant features and the one-hot region untouched. The exported policy
keeps the full 134-wide observation (zero first-layer weights on excluded entries), so the
AI Arcade, viewer and C++ loader need no change.

## 5. Evaluation and checkpoint selection

Formal evaluation is unchanged: every 600 s of active training, one greedy episode per map
from the real start, seed 2000007. The best `(finish_rate, mean_progress)` checkpoint is kept as
`best.pt` / `best.gdp`; `final.gdp` is whatever the run ends with. `evaluation_episodes` is 1
because repeated episodes are identical in this environment.

While the curriculum is still walking back, the full-start score understates the policy: it
can drive the *end* of every map long before it can drive the start. Read
`control.json -> demo_curriculum` for the real progress signal: `prefix` (current takeover
point per track id), `demo_length`, `graduated`, `advances`, `retreats`.

## 6. How to run it

```sh
# 1. Search demos for every map in the curriculum (skips maps that already have one)
scripts/explore_maps.py --config configs/classic_all_tracks_demo_noid.json --workers 10 --time-budget 900

# Retry the stubborn ones with more time and varied hold lengths
scripts/explore_maps.py --config configs/classic_all_tracks_demo_noid.json \
    --maps 2:7,2:9 --time-budget 2400 --hold-choices 3,8,20 --explore-steps 200 --progress-bin 0.01

# 2. Create the run and start it under the stall watchdog (builds the demo replay and the
#    normalization, then trains via `resume` so engine hangs are survived)
scripts/launch_run.py --config configs/classic_all_tracks_demo_noid.json --run-id my_run --duration-seconds 28800

# Watch
./scripts/control.sh status --run-id my_run
```

Demos are reloaded from `demos/` every time the run starts, so a demo added later is picked up
by stopping and resuming. `resume` reads the config from `latest.pt`, not from the config file;
to change a setting on an existing run, stop it, patch `checkpoint["config"]` in `latest.pt`
(`gravity_lab_rl.checkpoint.load_checkpoint` / `save_checkpoint`), set `requested` back to
`run` in `control.json`, and relaunch `scripts/train_watchdog.py`. Stopping triggers the final
evaluation and plot, about a minute.

## 7. Knobs and what they do

`demos` block (`DEFAULTS` in `demo_curriculum.py`):

| Key | Current | Effect |
|---|---|---|
| `initial_remaining` | 50 | First takeover point, in demo steps before the finish |
| `step_back` | 80 | Demo steps the takeover point moves per advance; larger = fewer stages, harder stages |
| `advance_window`, `advance_successes` | 4, 3 | Advance when this many of the last N greedy demo-start episodes finished |
| `retreat_window` | 6 | Retreat after this many consecutive greedy failures |
| `greedy_probability` | 0.5 | Share of demo-start episodes run greedily; only they move the takeover point |
| `full_start_probability` | 0.2 | Share of episodes started from the real start on maps with a demo |
| `sticky_action_probability` | 0.3 | Exploration episodes repeat the previous action with this probability |
| `bc_weight`, `bc_batch_size` | 1.0, 64 | Behavior-cloning weight and demo transitions per optimizer step |

`algorithm` additions: `network` (`dense` or `track_conditioned`), `track_conditioning`,
`critic_layer_norm`, `excluded_inputs`, `target_entropy_ratio` (0.3 here, down from 0.7,
since exploration now comes from demos and sticky actions rather than a near-uniform policy).

## 8. Known issues

- **Native engine hangs.** The vendored physics can hang inside a step on some states (seen on
  Hole, Pillar, "Trial again"). Search workers and the training watchdog both detect this by a
  stopped heartbeat and restart; training loses at most the stall timeout (180 s) plus the
  current episode.
- **Throughput.** About 80 environment steps per second on CPU with four torch threads, one
  environment per process. Walking all 30 maps back takes a few hours; the curriculum pace,
  not evaluation, is the number to watch.
- **Dead observation entries.** Indices 4-5 and 60-71 are always zero (see section 4). Fixing
  the acceleration region would be an engine change plus a rebuild of both CMake trees and
  would shift every existing policy's inputs, so it is excluded at the model instead.
- **Demos are arbitrary.** Search finds *a* solution, not a clean one. Behavior cloning pulls the
  policy toward those jerky inputs along the demo states; RL from the takeover points is what
  smooths and robustifies it. Long demos (1400-1900 steps) take the most stages.

## 9. Results so far

Run `demo_curriculum_noid_20260907_175241`, still in progress at the time of writing:

| Active training | Best full-start evaluation | Graduated maps | Walk-back done |
|---|---|---|---|
| 30 min | 5 / 30 | 0 | 22% |
| 60 min | 7 / 30 | 3 | 34% |
| 90 min | 10 / 30 | 3 | 41% |
| 110 min | 13 / 30, mean progress 0.590 | 3 | 46% |

The 13 maps at 110 minutes included Deep, Hole, Savvy, Floorboards and Undertaker, none of
which any previous shared network had finished. Every prior approach in this repository
plateaued at 9 of 30. See `docs/training-runs.md` for the running log.
