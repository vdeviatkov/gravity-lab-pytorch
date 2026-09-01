# Two policies, one racetrack

A head-to-head between the legacy 28-input network already on `main` and the new 36-input
network with an obstacle-ray sensor, across all 30 tracks of the Easy/Medium/Pro curriculum —
and a late-training collapse that the sensor run's final checkpoint did not survive.

- **Legacy** — `origin/main`, commit `8671956`
- **Sensor** — local branch, run `20260831-202029-13831`
- **As of** 2026-09-01, sensor run complete (1,800s)

> **Headline finding.** The sensor run's final checkpoint — the one this repository would
> actually ship — has collapsed: 0% finish rate on all 30 tracks, including Intro, which it was
> solving 93% of the time moments earlier in training. This happened in the last ~140 seconds of
> a 1,800-second run. Everything below reports that collapse honestly rather than substituting a
> healthier earlier reading for it.

Both policies are Double DQN networks trained on the identical vendored physics engine, the
identical 30-track Easy/Medium/Pro curriculum, and the identical reward. The only structural
difference is what the network is allowed to see: the legacy network sees only the bike's own
state; the sensor network additionally sees eight ray-cast distances to the track surface ahead.
Everything below is measured, not estimated — see the closing note for exactly how.

Both approaches live side by side in this repository and are both runnable — the native
environment always returns the full 36-value observation; indices 0–27 are an unchanged,
compatible prefix, so a 28-input model simply ignores the last 8 values. `gravity_lab_rl.model`
sizes a network's input layer from its config's `normalization.input_scale` length (28 or 36),
so the same CLI works for either.

## How to run each approach

| | Legacy (28-in) | Sensor (36-in) |
|---|---|---|
| Config | `configs/classic_intro_legacy.json`, `configs/classic_all_tracks_legacy.json` | `configs/classic_intro.json`, `configs/classic_all_tracks.json` |
| Bundled policy | `policies/classic_intro.gdp` | `policies/classic_intro_sensor.gdp` |

**Train from scratch:**

```sh
# Legacy (28-input)
.venv/bin/gravity-lab-rl train --config configs/classic_intro_legacy.json --duration-seconds 60

# Sensor (36-input)
.venv/bin/gravity-lab-rl train --config configs/classic_intro.json --duration-seconds 60
```

**Continue training from a bundled policy** (config's normalization width must match the policy
being loaded):

```sh
# Legacy
.venv/bin/gravity-lab-rl train --config configs/classic_all_tracks_legacy.json \
  --initialize-policy policies/classic_intro.gdp --duration-seconds 1800 --device cpu

# Sensor
.venv/bin/gravity-lab-rl train --config configs/classic_all_tracks.json \
  --initialize-policy policies/classic_intro_sensor.gdp --duration-seconds 1800 --device cpu
```

**Play a bundled policy in the graphical viewer or AI Arcade** (both accept either width
directly — no config needed, the policy file is self-describing):

```sh
./scripts/play_latest.sh --policy policies/classic_intro.gdp           # legacy
./scripts/play_latest.sh --policy policies/classic_intro_sensor.gdp    # sensor

./scripts/ai_arcade.sh --policy policies/classic_intro.gdp             # legacy
./scripts/ai_arcade.sh --policy policies/classic_intro_sensor.gdp      # sensor
```

**Formally evaluate a bundled policy** (1 episode × all 30 tracks, ε=0 — this is how every
number in this document was produced):

```python
from gravity_lab_rl.config import load_config
from gravity_lab_rl.model import DenseQNetwork
from gravity_lab_rl.export import load_policy_into_model
from gravity_lab_rl.evaluation import evaluate_model

cfg = load_config("configs/classic_all_tracks_legacy.json")  # or classic_all_tracks.json for sensor
model = DenseQNetwork(cfg["seeds"]["parameter_initialization"],
                      cfg["normalization"]["input_scale"], cfg["normalization"]["input_bias"])
load_policy_into_model(model, "policies/classic_intro.gdp")  # or classic_intro_sensor.gdp
print(evaluate_model(model, cfg))
```

A run already in `artifacts/<run-id>/` can instead be evaluated directly by CLI —
`gravity-lab-rl evaluate --run-id <id>` — since its own config on disk already carries the
correct observation width.

## 1. Legacy policy — 28 inputs

*No obstacle sensing.*

### Input → output

```
observation[28] → fc1(128, ReLU) → fc2(128, ReLU) → q(9, linear)
```

The 28 inputs are entirely *proprioceptive*: race progress (2), a start flag and league (2), then
six physics points — center, front wheel, rear wheel, and three frame/rider constraint points —
each contributing position-relative-to-center and velocity (4 values × 6 points = 24). The
network knows exactly how the bike is currently balanced and moving. It has no information about
the track beyond the bike's current contact point. Output is nine raw Q-values, one per
throttle×lean combination, chosen by argmax.

### Hyperparameters

| Parameter | Config default | Actual sweep run |
|---|---:|---:|
| Learning rate | 0.0005 | 0.0003 |
| Epsilon decay (transitions) | 1,000,000 | 2,000,000 |
| Batch size | 128 | 128 |
| Gamma | 0.99 | 0.99 |
| Replay capacity / warmup | 300,000 / 10,000 | same |
| Target sync every | 2,000 updates | same |

The checked-in `configs/classic_all_tracks.json` is a starting point; the bundled policy on
`main` is actually the winner of a 12-way hyperparameter sweep off that base, warm-started from a
single-track policy and continued twice more as better sweep candidates turned up.

### Compute spent

| | |
|---|---:|
| Active training | 4,287s |
| Transitions | 7.69M |
| Optimizer updates | 1.92M |

### Formal evaluation — 1 episode × all 30 tracks, ε=0

| Metric | Value |
|---|---:|
| Finish rate | 26.7% |
| Mean progress | 0.509 |
| Mean reward | 11.00 |
| Crash rate | 70.0% |
| Truncation rate | 3.3% |

**Solves 8 of 30 tracks**: Intro, Slope, Crackle, Knolls, Cliff, Hole, Original (all Easy), plus
one Medium track, Spikehops. Nothing in Pro.

## 2. Sensor policy — 36 inputs

*8-ray obstacle sensor.*

### Input → output

```
observation[36] → fc1(128, ReLU) → fc2(128, ReLU) → q(9, linear)
```

Indices 0–27 are identical to the legacy network. Indices 28–35 are new: eight rays cast from
the bike's center, 45° apart, ray 0 pointing toward the finish. Each ray is intersected against
the track's ground polyline treated as **bounded segments** — a hit only counts inside a
segment's own two endpoints, never its infinite-line extension — searched within 64 segments of
the bike's position. Output is `min(hit distance, max range) / max range`, so `1.0` means nothing
in range and `0.0` means touching something now. This is the one thing the legacy network
structurally cannot have: a look ahead.

### Hyperparameters

| Parameter | Value |
|---|---:|
| Learning rate | 0.0005 |
| Epsilon decay (transitions) | 1,000,000 |
| Epsilon start → end | 0.3 → 0.05 |
| Batch size | 128 |
| Gamma | 0.99 |
| Replay capacity / warmup | 300,000 / 10,000 |
| Target sync every | 2,000 updates |
| Obstacle rays / search radius / max range | 8 / 64 segments / 5× position scale |

Same curriculum config as the legacy run's checked-in default (untuned — no sweep has been run
on this architecture yet), warm-started from a 1-minute single-track smoke policy rather than
from scratch.

### Compute spent

| | |
|---|---:|
| Active training | 1,800s |
| Transitions | 10.69M |
| Optimizer updates | 2.67M |

### Formal evaluation — final checkpoint, 1 episode × all 30 tracks, ε=0

| Metric | Value |
|---|---:|
| Finish rate | **0.0%** |
| Mean progress | **0.033** |
| Mean reward | **−0.21** |
| Crash rate | 23.3% |
| Truncation rate | 76.7% |

> **This is a collapsed policy, not a weak one.** Every one of the 30 evaluation episodes ran
> the full 2,000 steps without finishing *or* crashing — including Intro, which the training log
> below shows this same network solving 93% of the time just minutes earlier. Inspecting the raw
> Q-values directly on Intro at this checkpoint: all nine actions score within `3.8–4.2`, a
> spread too small to encode a real preference, so `argmax` is effectively picking noise. Loss
> and training-time reward stayed normal right up to the last logged episode — nothing about it
> looked like a blow-up while it was happening. This is consistent with value-function collapse
> under multi-task curriculum interference, worsened by this project's checkpoint policy:
> `configs/classic_all_tracks.json` ships "latest online network; final evaluation is never used
> for selection," so whatever state the network is in at the exact end of the clock is what gets
> deployed, healthy or not.

### What the network could do before it collapsed

A checkpoint taken at ~92% of the training budget (roughly 140 seconds, ~1.3M transitions before
the collapse) told a very different story on a formal 30-track evaluation: **20.0% finish rate**,
mean progress **0.444**, 6 tracks solved (Intro, Shorty, Slope, Crackle, Cliff, Original). That
checkpoint's weights weren't saved anywhere durable — the run overwrites `latest.pt` in place —
so it can't be recovered or shipped from this run. It's reported here as evidence of capability,
not as a substitute for the real final number above.

> The training log is a larger, lower-variance sample than either single 30-episode snapshot:
> across the full run, with exploration noise on, **21 of 30 tracks were finished at least
> once** — including seven Medium/Pro tracks the legacy network has never once completed. That
> capability was real during training. It just wasn't what survived to the final checkpoint.

### Finish rate by track (full run, training log, exploration included)

| Track | Finish rate |
|---|---:|
| Intro | 94% |
| Shorty | 83% |
| Crackle | 75% |
| Knolls | 59% |
| Slope | 49% |
| Cliff | 19% |
| Hole | 18% |
| Original | 17% |
| Savvy | 6% |
| Indoor | 5% |
| Spikehops | 2% |
| Blocks, Downhill, Undertaker, Deep, Intense, Hillclimb, Modesty, Bumps, Floorboards, Abrupt | <1% each |

9 Pro/Medium tracks (Spikeholes, Pillar, Trenches, Tip top, Dantes Peak, 100%, Training day,
Trial again, Liberty) were never finished.

## 3. Head to head

| Metric | Legacy (28-in) | Sensor (36-in) |
|---|---:|---:|
| Observation size | 28 | 36 |
| Parameters | 21,385 | 22,409 |
| Active training time | 4,287s | 1,800s |
| Tracks ever finished (full training log) | 8 / 30 | **21 / 30** |
| Medium/Pro tracks ever finished | 1 | **7** |
| Formal eval — final checkpoint | 26.7% finish | **0.0% finish (collapsed)** |
| Formal eval — best checkpoint seen | 26.7% finish | 20.0% finish (unrecoverable) |

## 4. Conclusion

**Why the legacy network isn't stuck at zero.** Even with no forward vision, the 28-dimensional
state is a complete description of how the bike is currently balanced — the tilt and velocity of
every constraint point relative to the frame. That's enough to ride reactively: correct a
wheelie, lean into a landing, throttle out of a dip, the instant it happens. On Easy tracks,
whose slopes and bumps are gentle and telegraphed by the bike's own motion a fraction of a second
in advance, reactive control plus enough repeated exposure to the same 30 tracks is sufficient to
learn track-specific timing by feel — effectively memorized reflexes rather than foresight. That
ceiling is visible in the results: every legacy success is Easy-tier (plus one lucky Medium
track), and the policy crashes 70% of the time overall, because a cliff edge or a gap that needs
braking *before* the bike gets there is invisible until the bike is already falling into it.

**Why the sensor network isn't a win yet, but the evidence for it is real.** The obstacle rays
are the one capability structurally impossible to reactively substitute for — and the training
log backs that up: this run touched **21 of 30 tracks** and **7 Medium/Pro tracks** the legacy
network has never once finished, on well under half the training time, on an unswept
configuration. But capability during training is not the same as a deployable result. The
checkpoint this run actually ends on has collapsed to a 0% finish rate across every track,
including ones it had just been solving reliably. That collapse is the real story of this run:
something in this project's training loop — almost certainly the interaction between
curriculum-driven task-switching and Double DQN's bootstrapped value estimates, sharpened by
shipping whichever checkpoint happens to exist when the clock runs out — can silently destroy a
healthy policy in the space of a couple of minutes, with no visible warning in the training-time
loss or reward.

**What this means for next steps.** The obstacle-sensor architecture is worth pursuing; this
specific run is not something to bundle or ship. The single highest-leverage fix, for either
architecture, is to stop trusting "whatever the online network is at the end": run periodic
formal evaluations during training (the machinery for this already exists in `evaluate_model`)
and checkpoint the best-scoring network seen, not the last one. Without that safeguard, comparing
final checkpoints is close to comparing lottery tickets — and this run is the demonstration of
exactly that risk.

---

Legacy figures read from `origin/main` commit `8671956` (`policies/classic_intro.gdp.json`,
`configs/classic_all_tracks.json`). Sensor figures read from local run
`20260831-202029-13831`, completed at its full 1,800s budget: final-checkpoint eval from
`summary.json`'s `final_evaluation`; training-log rates from `metrics.jsonl` (19,449 episodes);
the ~92%-of-budget interim eval and raw Q-value inspection were run manually against
`latest.pt`/`final.pt` via `evaluate_model`, identical protocol and seed (2000007) to the legacy
figure.
