# Training run log

A running record of every trained checkpoint produced for `gravity-lab-classic-v1`, so later
comparisons don't require re-deriving numbers from `artifacts/`. All finish rates are from the
**formal evaluation protocol**: 1 episode per curriculum environment (all 30 Easy/Medium/Pro
tracks), deterministic (ε=0), seed `2000007` — the same protocol every run below uses, so the
`finish_rate` column is directly comparable across rows.

| # | Approach | Obs width | Config | Active training | Transitions | Best finish rate | Mean progress | Status | Bundled as |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| 1 | Legacy (28-in), sweep-tuned | 28 | `configs/classic_all_tracks_legacy.json` (lr=0.0003 sweep variant) | 4,287s | 7.69M | 26.7% (8/30) | 0.509 | complete | `policies/classic_intro.gdp` |
| 2 | 8-ray sensor, single-track smoke | 36 | `configs/classic_intro.json` | 60s | 385k | 100% (1/1 trained track only) | 0.977 | complete, superseded by #4 | — |
| 3 | 8-ray sensor, all-tracks (no best-tracking) | 36 | `configs/classic_all_tracks.json` | 1,800s | 10.69M | 0% final / 20.0% (6/30) at ~92% of budget, unrecoverable | 0.033 final / 0.444 peak | complete — **final checkpoint collapsed**, motivated best-checkpoint tracking | — (historical only, see `docs/policy-comparison.md`) |
| 4 | 8-ray sensor, all-tracks + best-checkpoint tracking | 36 | `configs/classic_all_tracks.json` | 1,800s | 10.69M | 10.0% (3/30) final / **23.3% (7/30) best** | 0.454 | complete | `policies/classic_intro_sensor.gdp` |
| 5 | 32-ray + acceleration, single-track smoke | 72 | `configs/classic_intro_32ray_accel.json` | 60s | 314k | 100% (1/1 trained track only) | 0.970 | complete, superseded by #6 | — |
| 6 | 32-ray + acceleration, all-tracks + best-checkpoint tracking | 72 | `configs/classic_all_tracks_32ray_accel.json`, warm-started from #5 | 2,745.6s (manually stopped, `control-stop`, before the 3,600s target) | 11.97M | 3.3% (1/30) final / **23.3% (7/30) best** | 0.437 | complete (stopped early) | `policies/classic_intro_32ray_accel.gdp` |
| 7 | Reward attempt 1: per-step penalty 0.001→0.003 (undiscounted math, **wrong**), warm-started from #6 | 72 | `configs/classic_all_tracks_32ray_accel.json` | 1,421.0s (manually stopped) | 6.01M | 3.3% (1/30) final / 20.0% (6/30) best — never recovered past #6's 23.3% starting point | 0.434 | complete — **reward fix ineffective**, see "Reward tuning experiments" below | `policies/classic_intro_32ray_accel_rewardfix.gdp` (superseded by #8) |
| 8 | Reward attempt 2: per-step penalty →0.1 (corrected discounted math), warm-started from #6 | 72 | `configs/classic_all_tracks_32ray_accel.json` | 10,051.1s (manually stopped, plateaued) | *not recorded* | 16.7% (5/30) final and best, plateaued for final ~2.6h of run | 0.369 | complete — **regressed previously-solved tracks**, see "Reward tuning experiments" below | `policies/classic_intro_32ray_accel_rewardfix2.gdp` (superseded by #9) |
| 9 | Reward attempt 3: per-step penalty →0.06 (cross-term-scale corrected), warm-started from #6 | 72 | `configs/classic_all_tracks_32ray_accel.json` | 280.1s (abandoned) | 1.25M | 10.0% (3/30) at abandonment | 0.304 | **abandoned** — root cause was never the per-step constant; see "Redesign: v2" below | — (superseded by v2) |
| 10 | v2 redesign, single-track validation (track id + shaping + n-step + wide net) | 102 | `configs/classic_intro_v2.json` | 600s | 566.9k | 100% (1/1 trained track), 0% crash | 0.966 | complete — validated the redesign before committing to the full curriculum | `artifacts/v2_intro_validate/best.gdp` |
| 11 | v2 redesign, all-tracks curriculum | 102 | `configs/classic_all_tracks_v2.json` | *pending* | *pending* | *pending* | *pending* | **in progress** | *pending* |

## Notes

- **Run #3 is why best-checkpoint tracking (`best_checkpoint_eval_interval_seconds` in
  `Trainer`) exists at all**: its final checkpoint silently collapsed to 0% from a 20% peak in
  the last ~140s of training, with no warning in training-time loss or reward. Full writeup in
  `docs/policy-comparison.md`. The same collapse recurred in runs #6 and #7 (final checkpoint
  well below the tracked best both times) — it's a repeatable property of this training loop on
  the all-tracks curriculum, not a one-off.
- Runs #4, #6, #7, #8 all use that fix: periodic deterministic evaluation during training,
  keeping the best-scoring `(finish_rate, mean_progress)` checkpoint rather than whatever exists
  when the clock runs out. `best_checkpoint_eval_interval_seconds: 90` throughout (a full
  30-track eval costs ~0.2s, so this is cheap).
- Observation layout reference: `[0,28)` base bike/track state, `[28, 28+ray_count)` obstacle
  rays, `[60,72)` per-component acceleration (always present in the 72-wide layout). Every
  shorter width above is an exact, unchanged prefix of every longer one — see
  `gravity-lab/include/gravity_lab/classic_environment.hpp` and `docs/policy-comparison.md`.
- Hyperparameters shared by rows 3, 4, 6, 7, 8 (the all-tracks curriculum configs): lr=0.0005,
  γ=0.99, batch=128, replay capacity 300k / warmup 10k, target sync every 2,000 updates,
  ε 0.3→0.05 over 1M transitions, 5 episodes/track before curriculum advances, evaluated/trained
  on CPU with `torch_num_threads=1`.

## Reward tuning experiments

Motivated by hands-on play of run #6's bundled policy (`classic_intro_32ray_accel.gdp`): on maps
it hasn't mastered, the bike freezes in place rather than attempting the obstacle. Two attempts
to fix this by changing the environment's per-step reward penalty
(`kPerStepPenalty` in `gravity-lab/src/classic_environment.cpp`), one wrong, one corrected.

**The reward** (`Environment::step`):
`r = 0.1 * (center_x_after - center_x_before) - kPerStepPenalty`, `+10` on finish, `-5` on crash
(`kCrashPenalty`). Trained with Double DQN, discount `γ = 0.99`.

### Attempt 1 (run #7) — wrong math, no effect

Reasoning at the time: over one full truncated episode (`max_episode_steps = 2000`), the original
`kPerStepPenalty = 0.001` costs only `0.001 × 2000 = 2.0` total — less than the `5.0` crash
penalty — so freezing for the whole episode is cheaper than risking a crash. Raised to `0.003` so
the same flat sum (`0.003 × 2000 = 6.0`) would exceed the crash penalty.

**Why this was wrong**: the agent doesn't optimize a flat sum over 2000 steps — it optimizes the
*discounted* return, and with `γ = 0.99` a persistent per-step cost is worth the geometric series
`Σ_t γ^t · (-c) → -c / (1 - γ)` as the horizon grows, not `-c × 2000`. At `c = 0.001` that's `-0.1`;
at `c = 0.003` it's `-0.3`. Both are trivial next to the `-5.0` crash penalty — the fix moved the
number that actually matters by only `0.2`, not the `4.0` the flat-sum math implied.

**Outcome**: trained 1,421.0s (warm-started from run #6's 23.3%/7-30 best). Best checkpoint
recovered only to 20.0% (6/30) — it never got back past its own 7/30 starting point — and hands-on
play confirmed the freezing behavior was unchanged. Stopped manually once this was clear.

### Attempt 2 (run #8) — corrected math

For idling-forever to discount to *more* negative than one crash:
`kPerStepPenalty / (1 - γ) > kCrashPenalty` → `kPerStepPenalty > kCrashPenalty × (1 - γ) = 5.0 × 0.01 = 0.05`.

Set `kPerStepPenalty = 0.1` (2× the breakeven point, asymptotic idle cost ≈ `-10`, clearly worse
than one `-5.0` crash), warm-started from run #6's best again.

**First attempt at run #8 hit a second, unrelated bug**: `Trainer._restore()` reset
`_last_best_eval_active` to `self.active_elapsed` on every resume — i.e. "an eval just happened"
— instead of preserving how much active time had actually passed since the real last eval. This
run's warm-started policy happened to trigger the native physics-engine stall (the same one
`scripts/train_watchdog.py` exists for) unusually often, and each stall-triggered restart reset
the countdown before it ever reached `best_checkpoint_eval_interval_seconds` (90s) — so 234s of
active training passed with the best-checkpoint eval never firing even once. Fixed by persisting
`last_best_eval_active` itself in the checkpoint payload and restoring that value instead of
`active_elapsed`. Verified with a short train/resume smoke test: with the bug, an eval due at
cumulative 20s would be pushed to 28s (past a 25s run, i.e. never); fixed, it fires at exactly
20s across the resume boundary. This bug is orthogonal to the reward change and would have
affected any run that stalls-and-restarts more often than its eval interval.

Run #8 relaunched after this fix and trained 10,051.1s (~2.8h) before being stopped manually.

**Outcome**: best checkpoint reached 5/30 (0.369 mean progress) and then **plateaued exactly there
for the final ~2.6h of the run** — the best-checkpoint file's SHA256 was identical at the 508s
mark and at every later check, so no new best was found for the majority of the run's wall time.
Hands-on play still showed the freezing behavior on unsolved maps, and per-track finish rates in
`metrics.jsonl` showed `kPerStepPenalty = 0.1` had **regressed tracks the policy had already
solved** under the old reward: Shorty 74–84% → 36%, Crackle 65–80% → 25%, Slope 44–52% → 2%, while
training loss stayed stable throughout (so this wasn't visible from loss curves alone — it required
checking per-track finish rates directly).

**Why**: `kPerStepPenalty = 0.1` satisfies the idle-vs-crash discounted-value constraint
(`> 0.05`) but is the *same order of magnitude* as the forward-progress reward term
(`0.1 × Δx` per step). A typical successful step earns roughly `0.1` from progress alone (a
~226-step Intro finish totals ~21.7 reward across the episode), so a flat `-0.1` per step made most
genuinely good steps net-negative too — not just idling. The two prior attempts each satisfied one
constraint while ignoring the other: attempt 1 used undiscounted math and didn't actually clear the
idle-vs-crash threshold; attempt 2 cleared that threshold but didn't check the constant against the
progress term's own scale.

### Attempt 3 (run #9) — cross-term-scale corrected

Kept the discounted-value threshold from attempt 2 (`kPerStepPenalty > 0.05`) but added the second
constraint discovered from attempt 2's regression: `kPerStepPenalty` must stay well under the
progress term's per-step scale (`~0.1`) so it doesn't compete with ordinary good driving. Set
`kPerStepPenalty = 0.06` — clears the idle-vs-crash threshold with a 20% margin
(`0.06 / 0.01 = 6.0 > 5.0`) while staying clearly below the `~0.1` progress-term ceiling.
Warm-started from run #6's baseline (`classic_intro_32ray_accel.gdp`, 23.3%/7-30, pre-reward-tuning),
**not** from run #8's plateaued/regressed checkpoint.

**Build note**: rebuilding just the `gravity_lab_classic` CMake target after this source edit is
**not sufficient** — `gravity_lab_classic_viewer` links the static `gravity_lab_classic_core`
library directly as its own target and is not part of that target's dependency closure, so a
narrow `--target gravity_lab_classic` build leaves the viewer running the old reward constant
(confirmed via a verification play showing the stale `-0.0997513` reward value, and via `otool -L`
showing the viewer doesn't link the shared `.dylib` at all). A full `cmake --build . -j4` (no
`--target` restriction) is required after any change to `classic_environment.cpp` to keep the
viewer, arcade, and test binaries in sync with the shared library. Verified afterward: viewer
reward for the fixed one-step test trajectory is `-0.0597513`, matching the predicted
`-0.059751282`; C++ (`gravity_lab_classic_tests`) and Python (`pytest tests`, 19/19) suites pass.

Run #9 launched after this fix and rebuild, but was **abandoned after 280.1s** (3/30, 0.304 mean
progress) once it became clear the per-step-penalty constant was never the real lever — see
"Redesign: v2" below.

## Redesign: v2 (track id, potential-based shaping, n-step returns, wider network)

Three reward-constant attempts (0.003, 0.1, 0.06) each fixed one piece of the discounted-value math
and broke something else, and none of them changed the underlying symptom (freezing on unsolved
maps) or the instability (late-training collapse, regression on already-solved tracks when another
track's gradient updates overwrite it). That pattern — not the specific constants — is what
prompted a step back: this is fundamentally a **30-track multi-task problem being trained as a
single task**, and the reward constant was never going to fix that.

**Root cause.** The observation carried no explicit track identity — only bike geometry, relative
component state, and ray-cast distances — so the network had to infer "which of the 30 tracks am I
on" purely from local geometry, sharing one small MLP's weights across wildly different track
layouts. Combined with a single 300k-capacity replay buffer mixing transitions from all 30 tracks
under a round-robin curriculum (5 episodes/track, forever), this is close to a textbook setup for
catastrophic interference in off-policy DQN: gradient updates from one track's data overwrite
Q-values learned for another. That explains every symptom seen across runs #3, #6, #7, #8, #9
without needing a separate explanation for each.

**Four changes**, all in `gravity-lab/src/classic_environment.cpp` (environment) and
`src/gravity_lab_rl/{model,replay,evaluation,trainer,config,export}.py` (training stack):

1. **Track identity in the observation.** A new one-hot region over `(level_group, track)` — 30
   values, `kAccelerationRegionEnd..kObservationSize` (`72..102`) — always computed, appended after
   the existing acceleration region. `kObservationSize` is now `102`. Backward compatible by the
   same fixed-region-prefix design used for the ray-count and acceleration extensions: every
   previously-trained policy (28-wide legacy, 36-wide 8-ray, 72-wide 32-ray+accel) is still an
   exact, unchanged prefix of the new vector and remains runnable unmodified.

2. **Potential-based reward shaping, replacing the per-step-penalty constant entirely.** The reward
   is now `kProgressShapingScale * (gamma * progress_after - progress_before)` plus terminal
   crash/finish bonuses — no flat per-step term. This is Ng, Harada & Russell's (1999) potential-
   based shaping using `progress` (fraction of track completed, already in the observation) as the
   potential function: `F(s,a,s') = gamma*Phi(s') - Phi(s)` provably never changes which policy is
   optimal, so unlike the old constant it cannot be "too large" relative to the progress term (it
   *is* the progress term, formalized correctly) or distort genuinely good driving the way the 0.1
   attempt did.

3. **A behavior-conditioned idle penalty, replacing the crash-vs-idle half of the old constraint.**
   Fires when `|progress delta|` this step is below `kIdleProgressDeltaThreshold = 0.001`, not when
   raw physics velocity is low. This required an empirical detour: the first implementation used
   the bike center's raw velocity magnitude and picked a threshold assuming velocity is
   "translational speed," but measurement showed a component's raw velocity is dominated by
   suspension/wheel-bounce noise — 2-5 world-units/frame even coasting at a dead stop — so a
   velocity-based threshold would have almost never fired, repeating the same "designed but never
   actually calibrated" mistake as reward attempts 1-2. Progress delta (scale-invariant across
   tracks of different lengths, immune to vertical bounce noise) was measured instead: coasting
   produces `|progress delta|` up to ~0.0007/step from settling/slope-slip alone, while genuine
   throttling crosses `0.001`/step within a handful of steps and keeps climbing, so `0.001` cleanly
   separates the two. `kIdlePenalty = 0.1` keeps the same idle-vs-crash discounted-value margin
   derived for attempt 2 (`kIdlePenalty / (1-gamma) > kCrashPenalty` requires `> 0.05`).

4. **n-step returns and a wider network**, both in the Python training stack only (no environment
   change): `NStepAccumulator` (`trainer.py`) turns the 1-step transition stream into n-step
   returns (`n_step: 3` in the new configs) before they reach the replay buffer, so the sparse `+10`
   finish bonus propagates back over 3 steps per sample instead of relying purely on 1-step
   bootstrapping — aimed at credit assignment on the specific "should I attempt this risky obstacle"
   decision. `ReplayBuffer` gained a `steps` field per transition (defaults to 1 for older
   checkpoints) so the Bellman target's bootstrap discount is `gamma ** steps`, not a fixed `gamma`,
   correctly handling the shorter windows flushed at episode boundaries.
   `DenseQNetwork.hidden_sizes` is now config-driven (`hidden_sizes` in `algorithm`, was hardcoded
   to `[128, 128]`) so the new configs can use `[256, 256]` for the harder 30-track/102-input
   problem; `export.py`'s architecture-match check was changed from a hardcoded `[128, 128, 9]`
   literal to comparing against the actual target model's layer sizes, since two valid hidden-layer
   widths now exist.

**New configs**: `configs/classic_intro_v2.json` (single-track validation) and
`configs/classic_all_tracks_v2.json` (full curriculum), both `102`-wide with `hidden_sizes: [256,
256]` and `n_step: 3`; otherwise identical hyperparameters to run #6's config so the effect of this
redesign isn't confounded with an unrelated hyperparameter change. No warm start: the new
observation width and network width aren't compatible with any prior checkpoint, so v2 starts from
random initialization.

**Single-track validation (run #10), before committing to a long curriculum run**: 600s on Intro
alone. Reached **100% finish rate, 0% crash rate** in the final 50 episodes (mean reward 10.97,
climbing to a final formal-eval mean reward of 13.0 and best-checkpoint mean reward of 13.9, mean
progress 0.966-0.967) — a clean, stable convergence, not intermittent luck. This is a materially
different outcome from every reward-tuning attempt (1-3), none of which ever got the policy past
intermittent freezing on any track. Run #11 (full 30-track curriculum) launched immediately after.
