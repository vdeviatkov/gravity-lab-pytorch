# Model documentation

This document describes the neural network used to play `gravity-lab-classic-v1`
(`src/gravity_lab_rl/model.py`), the exact meaning of its input and output vectors, and every
training hyperparameter, for both bundled configs (`configs/classic_intro.json` and
`configs/classic_all_tracks.json`).

## 1. Architecture

`DenseQNetwork` is a plain feed-forward Q-network — 3 fully-connected layers, ReLU between them:

```
input (36)  -->  fc1 (128, ReLU)  -->  fc2 (128, ReLU)  -->  q (9, linear)
```

| Layer | Shape        | Activation | Parameters      |
|-------|--------------|-----------|-----------------|
| fc1   | 36 → 128     | ReLU       | 36*128 + 128 = 4,736 |
| fc2   | 128 → 128    | ReLU       | 128*128 + 128 = 16,512 |
| q     | 128 → 9      | linear     | 128*9 + 9 = 1,161 |

Total: 22,409 parameters.

Before the first layer, inputs are affine-transformed by a per-feature scale/bias that is stored
as model buffers (not trained):

```
x = observation * input_scale + input_bias
```

Both configs ship with `normalization.kind = "identity"` (`input_scale = [1]*36`,
`input_bias = [0]*36`), so in the bundled setup this is a no-op — the network consumes the raw
observation values described in section 2.

**Initialization**: weights use Kaiming-uniform (`a = sqrt(5)`, matching `nn.Linear`'s own
default init formula, but driven by a *named, seeded* `torch.Generator` instead of the global
RNG); biases are uniform in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`. Both are seeded from
`seeds.parameter_initialization` in the config, so a given seed always reproduces the same initial
weights. Two networks are created at training start with identical initialization: the **online**
network (trained every step) and the **target** network (its weights are hard-copied from the
online network every `target_update_every` optimizer steps).

## 2. Input — the observation vector

`OBSERVATION_SIZE = 36` (`src/gravity_lab_rl/__init__.py`). The network's `forward()` takes a
single `torch.float32` tensor of shape `(36,)` (or `(batch, 36)`). The values are produced by
`Environment::make_observation()` in `gravity-lab/src/classic_environment.cpp`:

| Index | Meaning |
|-------|---------|
| 0 | Race progress: `(bike_x - track_start_x) / track_span`, in `[0, 1]` (can slightly exceed 1 near the finish) |
| 1 | `1 - progress` (index 0 mirrored) |
| 2 | `1.0` if the race timer/track hasn't started yet, else `0.0` |
| 3 | League, normalized: `league / 3.0` (league is 0–3: e.g. 100cc/175cc/220cc/250cc tiers) |
| 4–7 | Physics point 0 (center reference) relative to itself: `dx`, `dy`, `field_382`, `field_383` — always `(0, 0, field_382, field_383)` |
| 8–11 | Physics point 1 (front wheel) relative to center: `dx`, `dy`, `field_382`, `field_383` |
| 12–15 | Physics point 2 (rear wheel) relative to center: `dx`, `dy`, `field_382`, `field_383` |
| 16–19 | Physics point 3 (frame/rider constraint point) relative to center: `dx`, `dy`, `field_382`, `field_383` |
| 20–23 | Physics point 4 (frame/rider constraint point) relative to center: `dx`, `dy`, `field_382`, `field_383` |
| 24–27 | Physics point 5 (frame/rider constraint point) relative to center: `dx`, `dy`, `field_382`, `field_383` |
| 28–35 | Obstacle-distance sensor: 8 rays cast from the center point (see below) |

The 6 physics points (indices 0–5 in the engine's internal `field_29[]` array) are documented in
`gravity-lab/docs/classic-rl.md`: point 0 is the center reference, 1 is the front wheel, 2 is the
rear wheel, and 3–5 are the remaining frame/rider constraint points from the original engine. For
each point `i`, the 4 features are:

- `dx = (point.x - center.x) / (65536 * 10)` — horizontal offset from the center point, in track
  units (fixed-point 16.16 internally, scaled down by 10 track-units-worth of fixed-point range)
- `dy = (point.y - center.y) / (65536 * 10)` — vertical offset from the center point, same scaling
- `field_382 / (65536 * 20)` — the point's per-axis motion/velocity-like state (x-component),
  scaled down
- `field_383 / (65536 * 20)` — the point's per-axis motion/velocity-like state (y-component),
  scaled down

All values are plain `double`s; there is no clipping, so extreme physics states (e.g. right after
a crash) can in principle produce values outside the "typical" range implied by the scaling
constants above.

### Obstacle-distance sensor (indices 28–35)

Indices 28–35 are a fixed 8-ray "lidar" sensor over the track's ground polyline, added so the
network can see upcoming terrain instead of only the bike's current physics state. It is
implemented in `Environment::Impl::make_observation()` / `cast_obstacle_ray()`
(`gravity-lab/src/classic_environment.cpp`, constants in `gravity-lab/include/gravity_lab/classic_environment.hpp`):

- The track's ground is a polyline of `(x, y)` points (`GameLevel::pointPositions`), strictly
  increasing in `x`. Each consecutive pair of points is one **bounded** obstacle segment — a ray
  only counts as hitting a segment if the intersection point falls within that segment's own two
  endpoints (parameter `s ∈ [0, 1]`), never on the segment's infinite-line extension.
- 8 rays are cast from the bike's center point (`kObstacleRayCount = 8`), evenly spaced by full
  turns (45° apart). Ray 0 points along the direction of increasing progress (`+x`, i.e. toward
  the finish); the rest follow counter-clockwise from there.
- For each ray, only track segments near the bike's current position are searched
  (`kObstacleSearchRadius = 64` segments on each side of the bike's current segment) — this bounds
  the cost per step regardless of total track length.
- Each output value is `min(hit_distance, kObstacleMaxRange) / kObstacleMaxRange`, i.e. `0.0` means
  "touching an obstacle right now" and `1.0` means "nothing within sensor range" (either genuinely
  no hit, or the nearest hit is at/beyond `kObstacleMaxRange`). `kObstacleMaxRange` is currently
  `kFixed * 10.0 * 5.0` — five times the divisor already used for the position-delta features in
  indices 4–27, so ray distances land in a comparable numeric range to the rest of the observation.

All three constants (ray count, search radius, max range) are compile-time constants next to
`kObservationSize` and are meant to be tuned empirically (e.g. via the AI Arcade viewer) rather
than treated as fixed; changing any of them changes `kObservationSize`/`OBSERVATION_SIZE` and
therefore requires retraining (see the compatibility note below).

**Practical input contract**: pass a length-36 `float32` array/tensor built from one call to
`env.reset(...)` or `env.step(action)`. Do not reorder, rescale, or drop any of the 36 values —
the trained weights (and the `.gdp` policy export) assume this exact layout. **Compatibility
note**: `OBSERVATION_SIZE` changed from 28 to 36 when the obstacle sensor was added. Any policy
checkpoint or `.gdp` file trained against the old 28-value observation is no longer compatible and
must be retrained; `load_policy_into_model`/`_policy_settings` reject a size mismatch with a clear
error rather than silently misinterpreting the vector.

## 3. Output — Q-values and action selection

The network outputs `ACTION_COUNT = 9` raw (linear, unbounded) Q-values, one per discrete action.
There is no softmax — action selection is `argmax` over the 9 outputs (greedy), or, during
training, epsilon-greedy (random action with probability `epsilon`, else `argmax`).

Actions decode to `(drive, lean)` controls (`gravity-lab/src/classic_environment.cpp`):

| Index | Action              | drive | lean |
|-------|---------------------|-------|------|
| 0 | Coast                   |  0 |  0 |
| 1 | Throttle                |  1 |  0 |
| 2 | Brake                   | -1 |  0 |
| 3 | LeanBack                |  0 | -1 |
| 4 | LeanForward             |  0 |  1 |
| 5 | ThrottleLeanBack        |  1 | -1 |
| 6 | ThrottleLeanForward     |  1 |  1 |
| 7 | BrakeLeanBack           | -1 | -1 |
| 8 | BrakeLeanForward        | -1 |  1 |

`drive`: `+1` = throttle, `-1` = brake, `0` = coast. `lean`: `-1` = lean back, `+1` = lean forward,
`0` = neutral. Each action holds for `frame_skip` physics ticks (2 by default) before the agent
observes and chooses again.

### Reward (for context, not part of the network itself)

```
reward = (bike_x_after - bike_x_before) * 0.1 - 0.001      # per step
reward += 10.0   if the track was finished this step
reward -= 5.0    if the bike crashed this step
```

## 4. Training algorithm

Double DQN (`src/gravity_lab_rl/trainer.py`, `evaluation.py`):

- **Loss**: Smooth L1 (Huber) between predicted Q(s, a) and the Double-DQN target.
- **Target**: `target = r + gamma * (1 - terminated) * Q_target(s', argmax_a' Q_online(s', a'))`.
  Truncation (time-limit) does **not** zero out bootstrapping — only `terminated`
  (crash/finish) does.
- **Optimizer**: Adam, `lr = algorithm.learning_rate`.
- **Gradient clipping**: `clip_grad_norm_` to `algorithm.gradient_clip_norm`.
- **Target network sync**: hard copy of online → target every `target_update_every` optimizer
  updates (not a soft/Polyak update).
- **Replay buffer**: uniform random sampling (not prioritized), capacity
  `algorithm.replay_capacity`; training starts only after `algorithm.replay_warmup` transitions
  have been collected.
- **Update cadence**: one optimizer step every `algorithm.update_every` environment transitions.
- **Exploration**: epsilon-greedy, linearly annealed from `epsilon_start` to `epsilon_end` over
  `epsilon_decay_transitions` transitions.

## 5. Hyperparameters by config

| Hyperparameter | `classic_intro.json` | `classic_all_tracks.json` |
|---|---|---|
| `hidden_sizes` | `[128, 128]` | `[128, 128]` |
| `learning_rate` | `0.001` | `0.0005` |
| `gamma` (discount) | `0.99` | `0.99` |
| `batch_size` | `128` | `128` |
| `replay_capacity` | `100,000` | `300,000` |
| `replay_warmup` | `5,000` | `10,000` |
| `update_every` | `4` steps | `4` steps |
| `target_update_every` | `2,000` updates | `2,000` updates |
| `gradient_clip_norm` | `10.0` | `10.0` |
| `epsilon_start` | `1.0` | `0.3` |
| `epsilon_end` | `0.05` | `0.05` |
| `epsilon_decay_transitions` | `200,000` | `1,000,000` |
| `frame_skip` | `2` | `2` |
| `max_episode_steps` | `2,000` | `2,000` |
| `normalization.kind` | `identity` | `identity` |
| curriculum | none (single track: level_group 0, track 0, league 0) | 3 stages × 10 tracks each (level_group/league 0,1,2), 5 episodes/track before advancing, repeats |
| `duration_seconds` | `3,600` | `1,800` |
| `evaluation_episodes` | `5` | `1` |
| `device` | `auto` | `cpu` (`torch_num_threads = 1`) |

**Seeds** (identical in both configs): `environment=7`, `parameter_initialization=11`,
`epsilon_exploration=13`, `replay_sampling=17`, `validation=1000007`, `final_evaluation=2000007`.
These make a training run fully reproducible given the same config and code.

`classic_all_tracks.json` is typically used together with
`--initialize-policy policies/classic_intro.gdp` (see `README.md`), i.e. as a fine-tuning/sweep
config seeded from an already-trained intro policy rather than from scratch.

## 6. Portable export format (`.gdp`)

Trained checkpoints (`.pt`, containing optimizer/replay state) are exported to a dependency-free
`gravity-lab-dense-q-policy-v1` text format (`gravity-lab/python/gravity_lab/dense_policy.py`,
`src/gravity_lab_rl/export.py`) for use outside PyTorch (e.g. the C++ viewer/arcade app). It stores,
per layer: input/output size, activation name (`relu`/`tanh`/`linear`), weights, and biases, plus
the top-level `input_scale`/`input_bias` vectors — i.e. exactly the same 36→128→128→9,
relu/relu/linear network described above, just serialized as plain text instead of a PyTorch
state dict. A `.gdp.json` sidecar records provenance: architecture, activations, training
environment, transition/update counts, and reference evaluation stats (see
`policies/classic_intro.gdp.json` for the bundled policy's own metadata).
