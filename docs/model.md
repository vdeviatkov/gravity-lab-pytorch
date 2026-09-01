# Model documentation

This document describes the neural network used to play `gravity-lab-classic-v1`
(`src/gravity_lab_rl/model.py`), the exact meaning of its input and output vectors, and every
training hyperparameter, for both bundled configs (`configs/classic_intro.json` and
`configs/classic_all_tracks.json`).

## 1. Architecture

`DenseQNetwork` is a plain feed-forward Q-network — 3 fully-connected layers, ReLU between them:

```
input (28)  -->  fc1 (128, ReLU)  -->  fc2 (128, ReLU)  -->  q (9, linear)
```

| Layer | Shape        | Activation | Parameters      |
|-------|--------------|-----------|-----------------|
| fc1   | 28 → 128     | ReLU       | 28*128 + 128 = 3,712 |
| fc2   | 128 → 128    | ReLU       | 128*128 + 128 = 16,512 |
| q     | 128 → 9      | linear     | 128*9 + 9 = 1,161 |

Total: 21,385 parameters.

Before the first layer, inputs are affine-transformed by a per-feature scale/bias that is stored
as model buffers (not trained):

```
x = observation * input_scale + input_bias
```

Both configs ship with `normalization.kind = "identity"` (`input_scale = [1]*28`,
`input_bias = [0]*28`), so in the bundled setup this is a no-op — the network consumes the raw
observation values described in section 2.

**Initialization**: weights use Kaiming-uniform (`a = sqrt(5)`, matching `nn.Linear`'s own
default init formula, but driven by a *named, seeded* `torch.Generator` instead of the global
RNG); biases are uniform in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`. Both are seeded from
`seeds.parameter_initialization` in the config, so a given seed always reproduces the same initial
weights. Two networks are created at training start with identical initialization: the **online**
network (trained every step) and the **target** network (its weights are hard-copied from the
online network every `target_update_every` optimizer steps).

## 2. Input — the observation vector

`OBSERVATION_SIZE = 28` (`src/gravity_lab_rl/__init__.py`). The network's `forward()` takes a
single `torch.float32` tensor of shape `(28,)` (or `(batch, 28)`). The values are produced by
`Environment::make_observation()` in `gravity-lab/src/classic_environment.cpp`:

| Index | Meaning |
|-------|---------|
| 0 | Race progress: `(bike_x - track_start_x) / track_span`, in `[0, 1]` (can slightly exceed 1 near the finish) |
| 1 | `1 - progress` (index 0 mirrored) |
| 2 | `1.0` if the race timer/track hasn't started yet, else `0.0` |
| 3 | League, normalized: `league / 3.0` (league is 0–3: e.g. 100cc/175cc/220cc/250cc tiers) |
| 4–7 | Body part 0 relative to the bike's center part: `dx`, `dy`, `field_382`, `field_383` |
| 8–11 | Body part 1 relative to center: `dx`, `dy`, `field_382`, `field_383` |
| 12–15 | Body part 2 relative to center: `dx`, `dy`, `field_382`, `field_383` |
| 16–19 | Body part 3 relative to center: `dx`, `dy`, `field_382`, `field_383` |
| 20–23 | Body part 4 relative to center: `dx`, `dy`, `field_382`, `field_383` |
| 24–27 | Body part 5 (the center part itself) relative to center: `dx`, `dy`, `field_382`, `field_383` — always `(0, 0, field_382, field_383)` |

The 6 "body parts" (indices 0–5 in the engine's internal `field_29[]` array) are the bike's
physics bodies (front wheel, rear wheel, chassis/frame, rider torso, etc. — the exact per-index
identity lives in the decompiled physics engine and isn't named there either, but part 5 is always
the reference/"center" body). For each part `i`, the 4 features are:

- `dx = (part.x - center.x) / (65536 * 10)` — horizontal offset from the center body, in track
  units (fixed-point 16.16 internally, scaled down by 10 track-units-worth of fixed-point range)
- `dy = (part.y - center.y) / (65536 * 10)` — vertical offset from the center body, same scaling
- `field_382 / (65536 * 20)` — the part's per-axis motion/velocity-like state (x-component),
  scaled down
- `field_383 / (65536 * 20)` — the part's per-axis motion/velocity-like state (y-component),
  scaled down

All values are plain `double`s; there is no clipping, so extreme physics states (e.g. right after
a crash) can in principle produce values outside the "typical" range implied by the scaling
constants above.

**Practical input contract**: pass a length-28 `float32` array/tensor built from one call to
`env.reset(...)` or `env.step(action)`. Do not reorder, rescale, or drop any of the 28 values —
the trained weights (and the `.gdp` policy export) assume this exact layout.

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
the top-level `input_scale`/`input_bias` vectors — i.e. exactly the same 28→128→128→9,
relu/relu/linear network described above, just serialized as plain text instead of a PyTorch
state dict. A `.gdp.json` sidecar records provenance: architecture, activations, training
environment, transition/update counts, and reference evaluation stats (see
`policies/classic_intro.gdp.json` for the bundled policy's own metadata).
