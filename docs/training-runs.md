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
| 11 | v2 redesign, all-tracks curriculum | 102 | `configs/classic_all_tracks_v2.json` | 2,946.6s (manually stopped, user judged results bad) | 10.76M | 10.0% (3/30) final and best, plateaued from ~2280s | 0.334 | complete — **stopped**, see "v2 all-tracks outcome" below | `policies/classic_intro_v2.gdp` |
| 12a | v3 smoke test: balanced replay + gated curriculum | 102 | `configs/classic_all_tracks_v3.json` | 1,800s | 6.32M | 16.7% (5/30) best / 13.3% (4/30) final | 0.377 best / 0.320 final | complete — validated the fix, no plateau, see "v3" below | — |
| 12 | v3, all-tracks curriculum (full run) | 102 | `configs/classic_all_tracks_v3.json` | 1,089.4s (manually stopped, user judged approach not working) | 3.89M | 13.3% (4/30) best | 0.335 | complete — **stopped**, mechanism verified working (see below) but too slow for the stated goal | — |
| 13a | PPO, single-track validation | 102 | `configs/classic_intro_ppo.json` | 90s | ~85k | 100% (1/1 trained track), 0% crash | 0.968 | complete — 100% deterministic finish by 90s, ~6x faster than DQN's run #10 | — |
| 13 | PPO, all-tracks curriculum (gated, entropy_coef 0.01) | 102 | `configs/classic_all_tracks_ppo.json` | 1,606.0s (manually stopped) | *not recorded* | 16.7% (5/30) best, plateaued from ~470s | 0.395 | complete — **stopped**, same sparse-success plateau as run #12 (4/10 stage-0 tracks mastered, 6 stuck); diagnosed as an exploration problem, not algorithm choice, see below | — |
| 14 | Adaptive-curriculum smoke test | 102 | `configs/classic_all_tracks_ppo.json` (entropy_coef 0.03 + inverse-success-weighted track selection) | 60s | 364k | 3.3% (1/30) | 0.248 | complete — validated track-selection shift toward hard tracks before a long run, see below | — |
| 15 | PPO, all-tracks with adaptive curriculum (full run) | 102 | `configs/classic_all_tracks_ppo.json` | 784.4s (manually stopped, user chose to return to DQN) | *not recorded* | 6.7% (2/30) best | 0.242 | complete — **stopped before verdict**: training attention had correctly shifted toward hard tracks (as designed) but not enough active time had passed to see whether they actually improved before the pivot back to DQN | — |
| 16 | DQN, two-bonus reward, all-tracks (track id + n-step + wide net + balanced replay + gated curriculum) | 102 | `configs/classic_all_tracks_v3.json` | 1,376.6s (manually stopped, user wanted to try the progress ramp) | *not recorded* | 26.7% (8/30) best, stage 1 unlocked at ~101s | 0.393 | complete — **stopped**, broke through the sparse-success plateau on 9/10 stage-0 tracks (best result yet), see below | `policies/classic_intro_twobonus.gdp` |
| 17 | DQN, two-bonus + progress-ramped bonus, all-tracks | 102 | `configs/classic_all_tracks_v3.json` | 7,958.8s (manually stopped, plateaued) | *not recorded* | 23.3% (7/30) best, plateaued from ~532s | 0.447 | complete — **stopped**, best formal result across every approach this session but never broke past 7/30 despite ~7,400s more training, a mid-run eval-robustness fix, and a 9h target; see "Session synthesis" below | `policies/classic_intro_ramped.gdp` |
| 18 | SAC + REDQ (discrete, "REDQ-lite": ensemble=4/subset=2, UTD=1), all-tracks | 102 | `configs/classic_all_tracks_sac.json` | 4,039.8s (manually stopped, mid-run pivot to implement the head-clearance sensor, **not** a plateau stop) | 1.38M | 20.0% (6/30) best / 16.7% (5/30) final, stage 1 unlocked at ~675s | 0.351 best / 0.298 final | complete — **stopped**, see "SAC + REDQ: a new algorithm family" below | `policies/classic_sac_redq_interim.gdp` (interim) |
| 19 | SAC + REDQ, all-tracks + head-clearance sensor | 134 | `configs/classic_all_tracks_sac.json` | 1,302.4s (manually stopped, mid-run pivot to add the speed bonus reward term, **not** a plateau stop) | 402k | 26.7% (8/30) best / 20.0% (6/30) final, stage 1 unlocked at ~773s | 0.334 best / 0.370 final | complete — **stopped**, see "Head-clearance sensor outcome" below | `policies/classic_sac_redq_headclear_interim.gdp` (interim) |
| 20 | SAC + REDQ, all-tracks + head-clearance sensor + speed-bonus reward | 134 | `configs/classic_all_tracks_sac.json` | 12,513.99s (~3.48h; manually stopped, plateaued) | 3.46M | 30.0% (9/30) best / 20.0% (6/30) final, stage 1 unlocked by first eval | 0.456 best / 0.362 final | complete — **stopped**, best score plateaued at 9/30 from ~t=5211s (~85 min in) through the rest of the run (~2.9h with no improvement); see "Speed bonus reward term" outcome below | `policies/classic_sac_redq_headclear_speedbonus_interim.gdp` (interim) |
| 21 | SAC + REDQ, all-tracks + peak-based progress reward + adaptive curriculum (`success_ema` floor 0.05) | 134 | `configs/classic_all_tracks_sac.json` | 6,634.15s (~1.84h; manually stopped, curriculum over-concentration identified) | 2.12M | 26.7% (8/30) best / 3.3% (1/30) final, 0/30 stage-1 throughout | 0.414 best / 0.246 final | complete — **stopped**, adaptive curriculum over-concentrated on stage-1 once it tied near 0% success, degrading live training performance without a stage-1 payoff; see "Adaptive curriculum outcome" below | none deployed (did not beat run #20) |
| 22 | SAC + REDQ, all-tracks + peak-based progress reward + adaptive curriculum (`success_ema` floor 0.15) | 134 | `configs/classic_all_tracks_sac.json` | 6,660.29s (~1.85h; manually stopped by user, no further diagnosis pending) | 2.14M | 30.0% (9/30) best / 13.3% (4/30) final, stage 1 3/30-episodes | 0.454 best / 0.424 final | complete — **stopped**, matched run #20's peak (9/30/0.456→0.454) but reached it ~3x faster (t=1614s vs t=5211s) with a healthier stage0/stage1 attention split (~35%/65%, no decline recurrence); plateaued at 9/30 for the remaining ~1.3h with no further breakthrough; see "Rebalanced curriculum outcome" below | `policies/classic_sac_redq_curriculum_v2_interim.gdp` |

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

**Build note, correction to the note above**: a full `cmake --build . -j4` from
`gravity-lab/build-classic-rl` does **not** actually cover `gravity_lab_ai_arcade`, despite the
claim in the run #9 build note above. `apps/ai_arcade.cpp` is a target of the **repo-root**
`CMakeLists.txt` (a separate CMake project that `add_subdirectory(gravity-lab gravity-lab-build)`s
the game as a nested subproject), built from a completely different build tree at `build-native/`
in the repo root -- not `gravity-lab/build-classic-rl`. The two trees write their executables into
the *same* output directory (the root `CMakeLists.txt` sets
`CMAKE_RUNTIME_OUTPUT_DIRECTORY` to `gravity-lab/build-classic-rl`), which is exactly what made
this easy to miss: `gravity_lab_ai_arcade` sits right next to `gravity_lab_classic_viewer` in that
directory, but only the latter is rebuilt by `cmake --build gravity-lab/build-classic-rl`. Caught
when `./scripts/ai_arcade.sh --policy policies/classic_intro_v2.gdp` failed with "policy is
incompatible ... (28 to 72 observations)" against a 102-wide v2 policy, even though the viewer and
all C++/Python tests were already passing against the new width. Fix: `cmake --build build-native
-j4` from the repo root, in addition to (not instead of) building `gravity-lab/build-classic-rl`.
Both trees must be rebuilt after any change to `classic_environment.cpp` or the two headers it
defines (`classic_environment.hpp`, `classic_c_api.h`).

**Single-track validation (run #10), before committing to a long curriculum run**: 600s on Intro
alone. Reached **100% finish rate, 0% crash rate** in the final 50 episodes (mean reward 10.97,
climbing to a final formal-eval mean reward of 13.0 and best-checkpoint mean reward of 13.9, mean
progress 0.966-0.967) — a clean, stable convergence, not intermittent luck. This is a materially
different outcome from every reward-tuning attempt (1-3), none of which ever got the policy past
intermittent freezing on any track. Run #11 (full 30-track curriculum) launched immediately after.

### v2 all-tracks outcome (run #11) — stopped, single-track validation did not transfer

Trained 2,946.6s (~49 min) before being stopped manually once the plateau was clear. Best checkpoint
reached **3/30 (Intro, Shorty, Crackle — the first three tracks in curriculum order)** by ~2280s
and never improved past that for the remaining ~670s it ran.

**Per-track breakdown** (last 3,000 training-time episodes, ε=0.05, pulled from `metrics.jsonl`
before stopping) told a clearer story than the aggregate score: Intro 86% finish, Shorty 79%,
Crackle 56% — genuinely mastered, not lucky — but **every one of the other 27 tracks was at 0%
finish**, including league-0 tracks (Slope, Knolls, Deep, Cliff, Hole, Original, Savvy) that
earlier runs under the *old* reward and *without* track-id conditioning had gotten partway to
(e.g. Slope was 44-52% training-time finish under run #6's setup, vs. 0% here). All of league 1 and
league 2 (20 tracks) were at 0% with mean progress mostly 0.03-0.42, i.e. crashing early rather than
inching toward the finish.

**What this means for the diagnosis**: the single-track validation (run #10) confirmed the redesign
*can* learn a clean policy end to end — track-id one-hot, potential-based shaping, idle penalty,
n-step returns, and the wider 256x256 network all work correctly and are not the problem in
isolation. But run #11 shows the core multi-task-interference hypothesis from the "Redesign: v2"
section above was not actually resolved by these changes: the network still mastered only the
tracks earliest and most-repeated in curriculum order and made no progress on the rest, and by one
measure (Slope) did *worse* than a smaller, non-track-conditioned network trained on the old
reward. Track-id conditioning gives the network the *information* to distinguish tracks, but
doesn't by itself solve catastrophic interference in the shared trunk's gradients, and a bigger
network (256x256 vs 128x128) plus richer input (102 vs 72 dims) needs more data/updates to fit
before it can be expected to generalize across 30 tracks — it's possible this run was simply
stopped too early to tell such data-hunger apart from a genuine dead end, but the user's call to
stop and reassess was made on the actual evidence in hand (a 27/30 wall of zeros), not on giving it
more time speculatively.

Deployed for reference/comparison: `policies/classic_intro_v2.gdp` (mid-run snapshot at 3/30,
0.334 mean progress — not competitive with run #6's 23.3%/7-30, kept only as a checkpoint of what
this architecture looks like before further iteration).

## v3: per-track-balanced replay sampling + gated curriculum

Diagnosis of run #11's failure, from its per-track breakdown: the round-robin curriculum exposes
all 30 tracks (including league-2 "Pro" tracks) from episode 1 with no difficulty gating, and the
single shared replay buffer is sampled uniformly at random -- but a mastered track produces long
episodes (finishes take hundreds of steps) while an unsolved one crashes fast, so the buffer (and
every training batch) is structurally biased toward whatever's already easy. That's a rich-get-
richer loop: solved tracks get reinforced harder, everything else stays underrepresented in its own
training data. Track-id conditioning (the v2 redesign) gives the network the *information* to tell
tracks apart but doesn't fix this on its own. Two changes, both in
`src/gravity_lab_rl/{replay,trainer,config}.py` (no environment/C++ changes):

1. **Per-track-balanced replay sampling** (`ReplayBuffer` in `replay.py`). Each transition now
   carries a `track_id`; every `sample()` call draws close to `batch_size / (number of tracks
   currently represented)` from each track's own bucket instead of uniformly at random from the
   whole buffer, so a struggling track gets as much training signal per batch as a mastered one
   regardless of episode-length imbalance. Bucket membership is maintained in O(1) per insertion
   via swap-remove (an O(buffer size) scan per optimizer step would be far too slow at these
   transition rates) -- verified in isolation with a synthetic 900/50/10-transition imbalance
   across 3 tracks, confirming ~30/30/30 balanced draws from a 90-sample batch, plus bucket
   integrity after wraparound eviction and after a checkpoint state_dict round-trip. With a single
   track (the default `track_id=0`), this degenerates to plain uniform sampling, so single-track
   configs (`classic_intro*.json`) are unaffected.

2. **Progressive difficulty gating** (`Trainer.unlocked_stages`, `curriculum_environments`'s new
   `unlocked_stages` parameter in `config.py`). Training starts with only curriculum stage 0 (10
   league-0 tracks) in rotation; the next stage unlocks only once the current hardest unlocked
   stage's finish rate clears `curriculum.stage_advance_finish_rate` (0.5 in the new config),
   checked at each existing best-checkpoint eval (no extra evaluation cost). Formal evaluation
   (`evaluate_model`) is unaffected and always covers the full 30-track protocol regardless of
   gating state, so `best_score`/`finish_rate` numbers stay comparable across every run in this
   log. `unlocked_stages` is persisted in the checkpoint so it survives watchdog restarts.

New config: `configs/classic_all_tracks_v3.json` (`classic_all_tracks_v2.json` plus
`curriculum.stage_advance_finish_rate: 0.5`); `src/gravity_lab_rl/model.py`, the observation
layout, and the reward are unchanged from v2.

**Smoke test** (`v3_gating_smoke`, 1800s): aggregate stage-0 (10-track) training-time finish rate
climbed steadily and monotonically, 18.6% (645s) -> 20.7% (368s window, an earlier read) ->
33.7% (final) -- no plateau, unlike run #11 which was flat by this point. Four of the ten stage-0
tracks were genuinely solved by the end (Intro 96.7%, Shorty 90.0%, Crackle 83.3%, Cliff 66.7%) and
every other stage-0 track showed real mean progress (0.16-0.49) rather than run #11's wall of
near-zero. Best-checkpoint formal eval reached 5/30 (16.7%), mean progress 0.377 -- already above
run #11's entire 2,946.6s result (3/30, 0.334) in a third of the time. Did not clear the 50%
stage-0 gate within the 1800s smoke window (`unlocked_stages` stayed at 1 throughout), so stage 1
was never exercised in this smoke test, but the trend justified proceeding to a full run rather
than iterating further on the smoke test itself.

## Return to DQN with an explicit two-bonus reward

After the PPO adaptive-curriculum run also showed the same underlying pattern (shifting attention
toward hard tracks costs finish rate on easy ones in the short term, with the real test -- whether
the hard tracks actually improve -- still pending), the user asked to return to DQN with a reward
redesigned around two simple, literal bonuses instead of the derived potential-based-shaping
formula from "Redesign: v2": a big one-time bonus for finishing, and a small bonus for every
percent of forward progress made, plus a small penalty on any step with no forward progress
(explicitly kept, on the user's confirmation, to avoid reintroducing the original freezing bug --
without it, standing still is free while attempting an obstacle risks a crash penalty).

**New formula** (`gravity-lab/src/classic_environment.cpp`, `Environment::step`):
```cpp
const double percent_moved = (current_progress - previous_progress) * 100.0;
double reward = percent_moved > 0.0 ? kProgressPercentBonus * percent_moved : -kIdlePenalty;
if (reached_finish) reward += kFinishBonus;
if (did_crash) reward -= kCrashPenalty;
```
`kFinishBonus = 10.0` (unchanged), `kProgressPercentBonus = 0.1` (a full clean traversal, ~100
percentage points of progress, contributes ~10 -- comparable in scale to the finish bonus, not
dominant), `kIdlePenalty = 0.1` (same idle-vs-crash discounted-value margin derived earlier:
`kIdlePenalty / (1-gamma) > kCrashPenalty` requires `> 0.05`). Deliberately simpler than the
potential-based shaping it replaces -- no discount factor baked into the environment, no
progress-delta dead-zone threshold (a tiny positive settling-noise tick now earns a tiny positive
bonus rather than triggering the idle penalty, verified via the same coast/throttle check used
throughout this log: coast produces mostly small positive/negative single-digit-millipoint
rewards, throttle produces a clearly growing positive trend).

Everything else from the v2/v3 redesign is kept: track-id one-hot, 256x256 network, n-step
returns, per-track-balanced replay sampling, and progressive curriculum gating -- only the reward
formula changed. Rebuilt both CMake trees (`gravity-lab/build-classic-rl` and the repo-root
`build-native`, per the two-build-tree gotcha documented above), verified with
`gravity_lab_classic_tests` and `pytest tests` (19/19).

## Progress-ramped bonus

Run #16 (two-bonus reward, DQN) broke through the sparse-success plateau on most stage-0 tracks
(9 of 10 showing nonzero finish rate by ~950s, vs. 4/10 under every prior reward design) but still
had exactly one track stuck at 0% (Deep, the same track that was uniquely unsolvable under PPO's
adaptive curriculum too) and flattened out around 8/30 for several hundred seconds while stage 1
got its first exposure. Stopped by user request to try scaling the per-percent bonus by how far
along the track it's earned -- the flat per-percent bonus makes the first 10% of a hard track worth
exactly the same as the last 10%, so a policy that dies early and restarts cheaply on easy early
percentage points has no built-in incentive to push further into the specific late-track obstacle
that's actually unsolved.

**Change** (`gravity-lab/src/classic_environment.cpp`): added `kProgressRampFactor = 1.0` and a
`progress_multiplier = 1.0 + kProgressRampFactor * current_progress` applied to the positive
per-percent bonus only (not the idle penalty, finish bonus, or crash penalty):
```cpp
double reward = percent_moved > 0.0
    ? kProgressPercentBonus * percent_moved * progress_multiplier
    : -kIdlePenalty;
```
1x multiplier at the very start of a track, ~2x right at the finish line. Verified numerically:
`reward / percent_moved` tracked `0.1 * (1 + progress)` exactly across a live episode (0.093 at
progress -0.07, climbing to 0.125 at progress 0.25). Rebuilt both CMake trees, `pytest tests`
19/19, `gravity_lab_classic_tests` pass.

## Eval-robustness bug: `evaluation_episodes` was hardcoded to 1

While digging into why run #17's formal score stayed flat at 7/30 for ~1,250s despite steadily
improving, broad cumulative per-track finish rates (most stage-0 tracks in the 15-90% range, not
solved-or-not), found that the periodic best-checkpoint eval in both `trainer.py` and
`ppo_trainer.py` hardcoded `episodes=1` in the `evaluate_model(...)` call, **ignoring the config's
`evaluation_episodes` field entirely** (which was itself set to `1` in every all-tracks config).
With most tracks now genuinely probabilistic (a "60%-likely" track fails its one formal-eval shot
40% of the time), a single-episode snapshot across all 30 tracks is a very high-variance estimate
of true competence, and the strict `score > self.best_score` gate meant an unlucky snapshot could
never overwrite a lucky earlier one -- so the tracked "best" could lag far behind what the policy
could actually do.

**Fix**: both trainers now read `int(self.config["experiment"].get("evaluation_episodes", 1))` for
the periodic eval instead of a hardcoded `1`; bumped `evaluation_episodes` to `3` in
`classic_all_tracks_v3.json` and `classic_all_tracks_ppo.json` (different seed per episode per
track via `evaluate_model`'s existing `actual_seed + episode` logic, so more episodes genuinely
reduces variance rather than repeating the same deterministic trajectory). Since a checkpoint's
config is embedded at save time, resuming from an existing checkpoint does *not* pick up a config
file edit -- had to patch the saved checkpoint's embedded `config.experiment.evaluation_episodes`
directly before resuming, to keep run #17's progress rather than restart from scratch. Verified
active on the first post-fix eval (`best_metrics.episodes_per_environment == 3`,
`episode_count == 90`).

**Outcome**: the fairer measurement did not reveal a hidden higher score -- run #17's formal best
stayed at 7/30 even under the 3-episode average, and cumulative per-track stats themselves
flattened out over the following ~1,600s (several tracks' cumulative rates ticked down slightly),
consistent with a genuine plateau rather than a measurement artifact. The bug was real and worth
fixing regardless (any future run benefits from a less noisy best-checkpoint signal), but it was
not the explanation for this particular plateau.

## Session synthesis: why every shared-network approach lands in the same range

Across the whole session, every variation tried on a single shared network across all 30 tracks --
2 algorithms (DQN, PPO), 3 reward designs (potential-based shaping, two-bonus, progress-ramped
two-bonus), replay balancing, progressive curriculum gating, adaptive inverse-success track
selection, and a more robust evaluation metric -- converged to essentially the same **7-9/30
(23-30%)** formal ceiling, with the same qualitative shape underneath: a handful of tracks reach
80-95% cumulative finish rate, several reach double digits, and several stay near-zero (one, Deep,
literally zero across thousands of attempts under both algorithms). Meanwhile, single-track
training (no competition for network capacity or replay/rollout attention) is fast and reliable
under both algorithms -- DQN ~530s, PPO ~90s -- every time it's been tried.

That pattern -- consistent ceiling regardless of training-dynamics changes, combined with trivially
reliable single-track convergence -- points at a capacity/interference ceiling in the *shared
network across 30 tasks* premise itself, not in how that shared network is trained. Having now
exhausted the training-dynamics levers available within that premise, the strongest remaining,
evidence-backed option is the one set aside earlier in the session: per-track specialist models,
dispatched by track at inference time (which the game's fixed 30-track roster makes trivial, unlike
a setting requiring generalization to unseen tracks). At the demonstrated single-track convergence
rate, training all 30 specialists is a bounded, achievable amount of compute (a few hours), and each
one is the easy single-task problem this session has repeatedly proven this pipeline solves well --
unlike the shared-network approach, which has now plateaued in the same range under every
combination of fixes tried.

## Pivot to PPO

After two DQN redesign iterations (v2, v3) each measurably improved the failure mode but were
judged too slow to reach the actual goal, switched to PPO (on-policy, clipped surrogate objective,
GAE advantage estimation) as a different algorithm family rather than continuing to iterate on
off-policy DQN's replay/curriculum mechanics. New module `src/gravity_lab_rl/ppo_trainer.py`
(`PPOTrainer`), `ActorCriticNetwork` in `model.py` (shares `DenseQNetwork`'s trunk and `q` head --
read as action logits, not Q-values -- plus a training-only `value` head), dispatched via
`algorithm.kind: "ppo"` in the config (`cli.py`, `export.py` branch on it). No environment/C++
changes. Reuses checkpoint format, control.json, best-checkpoint tracking, and the v3 progressive
curriculum gating mechanism unchanged. `evaluate_model` (deterministic argmax) works unmodified for
a PPO actor since argmax over logits equals argmax over softmax(logits).

### Sparse-success plateau and the adaptive curriculum fix

Run #12 (v3 DQN, gated) and run #13 (PPO, gated) both plateaued at a near-identical point: 4 of 10
stage-0 tracks fully mastered (Intro, Shorty, Crackle, Cliff), the other 6 stuck. Since the plateau
was identical across two structurally different algorithms (off-policy DQN with replay/target
network vs. on-policy PPO with GAE), the bottleneck could not be an algorithm-specific mechanic --
it had to be something about training *dynamics* shared by both, or the tracks themselves.

Inspecting run #13's full history (44,215 episodes, not just the recent window) settled it:
5 of the 6 "stuck" tracks were **provably solvable** -- Slope 9/4,425, Knolls 37/4,420, Hole
41/4,420, Original 6/4,420, Savvy 104/4,420 finishes (0.1-2.4%) -- but the policy never
consolidated a rare success into reliable behavior. Deep was the outlier: 0/4,420 finishes, never
reaching past 72% progress. Direct inspection of the best checkpoint's deterministic behavior on
Deep showed it holding `ThrottleLeanForward` constant from step 40 to the crash at step 110, with
the obstacle-ray sensor clearly showing a closing obstacle (min ray distance 0.22 -> 0.13 over the
last 8 steps) that it never reacted to -- the same "locked on one action regardless of changing
state" signature as the very first PPO smoke test, just now on an otherwise well-trained
checkpoint. This is a sparse-success exploration problem specific to certain obstacles, not a
training-algorithm defect.

**Fix, in `src/gravity_lab_rl/ppo_trainer.py`**: replaced uniform round-robin track selection with
an adaptive curriculum weighted by inverse recent success rate (`track_success_ema`, a slow EMA per
track, `alpha=0.05`), so a track that's mostly failing gets picked more often -- giving a rare
success more repeated practice to reinforce into reliable behavior, instead of the policy rotating
straight back to a track it's already good at. Weight is `1 / (success_rate + 0.05)`, so a
0%-success track gets ~20x a 100%-success track's selection frequency, not unboundedly more (still
leaves room for mastered tracks to get occasional refresher practice). Also raised `entropy_coef`
0.01 -> 0.03 to encourage more action variety at exactly the obstacles where the greedy policy was
getting stuck repeating one action. State (`track_success_ema`, curriculum RNG) persists across
checkpoints.

Verified in isolation (weighted-sampling unit check: 3 tracks at 95%/50%/2% success produced
~6%/10%/84% selection share over 6,000 draws, matching the `1/(p+0.05)` weighting exactly) and via
a 60s smoke test: track exposure shifted sharply toward the historically-hard tracks (Deep 275,
Savvy 275, Original 250, Hole 240, Knolls 220 episodes) and away from the mastered ones (Intro 45,
Shorty 45) -- confirming the mechanism engages correctly before committing to a long run.

**Single-track validation** (`ppo_smoke_test`, `configs/classic_intro_ppo.json`): reached 100%
deterministic finish rate by **90s** active training (vs. DQN's ~530s for the same task, run #10) --
verified via the exported .gdp in the viewer (3/3 clean finishes, reward 12.85, no crash). One false
alarm during validation: a 60s snapshot showed strong stochastic-rollout rewards (+9) but a
degenerate deterministic (argmax) policy (always the same action regardless of state, crashing at
step 62) -- investigated by comparing stochastic vs. greedy behavior directly, found the action
distribution was still nearly flat (max probability ~0.41) this early, consistent with
"undertrained," not a bug; confirmed by extending to 600s where deterministic performance caught up
by 90s. Not a code defect, just too small a training budget for the first data point.

Run #12 (`v3_all_tracks_20260902`, full 6h curriculum) launched immediately after, then stopped
twice: first at a 10-minute fail-fast checkpoint (3/30 formal eval -- but this bar is structurally
near-unreachable under a gated curriculum, since formal eval is capped by how much of stage 0 is
solved until it unlocks; see the stage-aware-bar discussion below), resumed under a corrected
stage-aware checkpoint (stage-0-only aggregate finish rate vs. a 25% bar, which it was clearing:
32% by 821s), then stopped again by explicit user judgment at 1,089.4s (4/30 best, 0.335 mean
progress) as "not working" -- not because a checkpoint rule failed, but because progress at this
rate was judged too slow to reach the actual goal (30/30) in acceptable time, even though the
mechanism itself (balanced replay + gating) was doing what it was designed to do.

**Assessment**: v3's fixes were not wrong -- they measurably worked (no plateau, steady per-track
improvement, healthier per-track distribution than v2) -- but "correct and slow" isn't a passing
grade against the actual goal. The single clearest empirical fact from this whole redesign arc is
that **single-track training is fast and reliable**: run #10 hit 100% finish on Intro alone in
~530s. Multi-task training (v2, v3) divides gradient/batch attention across many tracks
simultaneously, which is inherently slower per-track than single-task training regardless of how
well the multi-task interference itself is managed. This motivates moving away from one generalist
network across all 30 tracks entirely -- see the next section.

## SAC + REDQ: a new algorithm family

After the per-track specialist pivot (`scripts/train_specialists.py`, one small network per track
instead of one shared network across all 30) demonstrated the shared-network ceiling could be
sidestepped entirely, the user asked to try one more single-shared-network approach before
committing further to specialists: Soft Actor-Critic with Randomized Ensembled Double Q-learning
(REDQ), a different algorithm family from both DQN (off-policy, epsilon-greedy, single critic) and
PPO (on-policy, clipped surrogate) tried earlier this session.

**Implementation** (`src/gravity_lab_rl/sac_trainer.py`, `algorithm.kind: "sac_redq"`): discrete SAC
(Christodoulou 2019) -- a categorical actor plus an ensemble of independent critics, entropy
temperature auto-tuned toward a target entropy instead of epsilon-greedy exploration -- combined
with REDQ's (Chen et al. 2021) in-target subsampling: every critic update draws a random
`subset_size`-of-`ensemble_size` subset of the *target* critics and bootstraps off their elementwise
minimum, which controls the overestimation a naive high-update-ratio ensemble would otherwise
amplify. Because the action space is discrete and small (9 actions), the target's expectation over
actions is computed exactly (`Σ_a π(a|s')·(min_Q(s',a) − α·log π(a|s'))`) rather than by sampling.
The actor is a plain `DenseQNetwork` -- identical in shape to the DQN/PPO networks -- so it exports
through the existing `.gdp` path and `evaluate_model`'s argmax-based deterministic evaluation
unchanged; critics are separate `DenseQNetwork` instances never exported. Reuses this session's
existing infrastructure unchanged: per-track-balanced replay (`replay.py`), n-step returns
(`NStepAccumulator`), progressive curriculum gating, best-checkpoint tracking, control.json
pause/stop, checkpoint/resume. New config validation branch in `config.py`
(`ensemble_size`, `subset_size`, `utd_ratio`, `tau`, `target_entropy_ratio`, `initial_alpha`).

**Compute tradeoff, deliberately scaled down from the REDQ paper's defaults.** REDQ's headline
result combines a large ensemble (paper default `N=10`) with a high update-to-data ratio (`G≈20`
gradient updates per environment step) -- the ensemble subsampling is specifically what makes that
high UTD safe from overestimation blowup. Both are expensive: this repo trains single-process on
CPU, and every prior run in this log needed several million transitions of curriculum exposure to
reach even the 7-9/30 ceiling, so an implementation that reproduced REDQ's full UTD would either
need far more than 2h wall-clock or would starve the curriculum of transitions entirely. Shipped
values were cut to `ensemble_size=4`, `subset_size=2`, `utd_ratio=1` (`configs/classic_all_tracks_sac.json`)
-- keeping REDQ's core mechanism (randomized subset-of-ensemble target minimization) while dropping
the extreme UTD, explicitly trading away most of REDQ's sample-efficiency headline for a throughput
this hardware can sustain over a multi-track curriculum.

**Validation before the long run**: a 45s single-track smoke test (`configs/classic_intro_sac.json`)
reached 100% deterministic finish rate on Intro -- faster than PPO's 90s baseline (run #13a) and
DQN's ~530s (run #10) -- and a resume/checkpoint round-trip preserved that result exactly. Full test
suite (19/19) unaffected.

**All-tracks outcome (run #18)**: launched under a watchdog (`scripts/train_watchdog.py`) against
the same 2h target used throughout this session, using the unchanged two-bonus + progress-ramped
reward from runs #16/#17 (no environment change this run). Cleared the stage-0 gate (50% aggregate
finish rate) at **675s active training** -- unlocking stage 1 faster than every DQN/PPO run in this
log except run #16's hand-tuned reward (~101s) -- and reached a best formal-eval score of **6/30
(20.0%), mean progress 0.351** by the time it was stopped. Per-track: Intro, Shorty, Slope, Crackle,
Knolls, Original solved at least once; stage-0's other 4 tracks (Deep, Cliff, Hole, Savvy) and all 10
stage-1 tracks unsolved, though several stage-1 tracks (Hillclimb, Blocks, Spikehops, Pillar) showed
real, growing mean progress (~0.33) rather than instant crashes by the time training stopped -- the
same "engaging with the obstacle but not yet completing it" phase every prior run went through after
its own stage-1 unlock.

**This was a deliberate mid-run pivot, not a plateau stop** -- stopped at 4,039.8s (1.38M
transitions) specifically to implement the head-clearance ray sensor below and start a fresh
comparison run, not because SAC+REDQ's own trajectory had flattened out (best score was still
climbing: 5/30 → 6/30 in the 30 minutes right before the stop). So 6/30 is a lower bound on what this
run would have reached given the full 2h, not a verdict on the algorithm.

**Throughput vs. sample efficiency**: this run's transition rate was ~342/s (1.38M transitions /
4,039.8s active), 10-13x slower than every DQN/PPO all-tracks run that recorded a transition count
(3,500-4,500/s, e.g. runs #6, #11, #12) -- expected, since one REDQ update round costs roughly 12
forward + 5 backward passes across the actor and 4-critic ensemble per environment step (vs. DQN's
~3 forward + 1 backward passes only once every 4 steps). But per-transition, SAC+REDQ reached 6/30
using 4.6-8.7x *fewer* transitions than the DQN runs that reached comparable-or-better formal scores
(run #6: 11.97M transitions for 7/30; run #12a v3 smoke: 6.32M for 5/30) -- consistent with REDQ's
core sample-efficiency premise surviving even the scaled-down ensemble/UTD, but not enough of a
wall-clock win under this session's CPU/2h constraints to make it a clear-cut improvement over DQN
without a longer run to confirm where it actually plateaus.

Deployed as an interim snapshot for hands-on testing: `policies/classic_sac_redq_interim.gdp`
(best.gdp at the time of the stop, 6/30/0.351 -- superseded once the head-clearance run below
produces its own best checkpoint).

## Head-clearance sensor

Hands-on play of run #18's policy, plus a direct question about how the obstacle sensor works,
prompted tracing the decompiled physics engine (`gravity-lab/classic/src/GamePhysics.cpp`,
`LevelLoader.cpp`) to check whether crash detection was really per-point circle-vs-track-segment
collision, as suspected. It is: `GamePhysics::const175_1_half = {114688, 65536, 32768}` (F16
fixed-point, ≈1.75/1.0/0.5 track-units) gives three radius classes, and body-part offsets in
`GamePhysics::reset`'s per-point switch show physics point 5 sits directly above the center
(~5 units up, all other points offset lower/sideways), has the *smallest* radius (0.5), and is the
only point whose ground contact sets an immediate hard-crash flag (`field_36`) rather than the
graduated bounce handling every other point gets -- strong evidence point 5 is the rider's head.
Point 5's position was already in the observation (it's one of the "six physics points" every prior
run trained against, just not individually labeled), so the actual gap was that the 32-ray obstacle
sensor casts only from the *center* point, ~5 units below the head, and reports raw geometric
distance rather than clearance (distance minus the colliding point's own radius).

**Change** (`gravity-lab/src/classic_environment.cpp`, `include/gravity_lab/classic_environment.hpp`):
a new, appended observation region -- same ray count and angles as the existing obstacle sensor
(`Config::obstacle_ray_count`, deliberately *not* a second independently-configurable ray count, so
every existing derivation of ray count from a policy's declared `observation_size` in
`apps/ai_arcade.cpp`/`classic_policy_viewer.cpp` stayed correct with zero code changes) -- cast from
physics point 5 instead of point 0, each entry `max(0, distance - head_radius) / kObstacleMaxRange`.
`OBSERVATION_SIZE` grew 102 -> 134 (`kMaxObstacleRayCount` more slots, mirroring how the obstacle
region itself is sized). Verified live in Python before training: head-clearance values track real
geometry, diverge meaningfully from the same-angle obstacle-sensor value (sometimes tighter,
sometimes looser, since the head is offset up and to the same x as the center), and sharpen sharply
in the steps immediately before a crash. Both CMake trees rebuilt; `gravity_lab_classic_tests` and
`pytest tests` (19/19) pass; old 28-wide and 102-wide policies still validate against the rebuilt
shared library (the new region is a pure append, same prefix-compatibility guarantee as every prior
region addition).

### Head-clearance sensor outcome (run #19)

Fresh run (no warm start -- 134-wide input is incompatible with run #18's 102-wide network), same
`configs/classic_all_tracks_sac.json` hyperparameters and reward as run #18 (two-bonus +
progress-ramped, unchanged) so the only variable was the new sensor region. Cleared the stage-0 gate
at **773s** (vs run #18's 675s -- comparable) and reached **8/30 (26.7%), mean progress 0.334** by
1,302.4s, when it was stopped -- already beating run #18's entire final result (6/30, reached only
after 4,039.8s) in under a third of the time. Per-track: the same 5 tracks from run #18 (Intro,
Shorty, Slope, Crackle, Knolls) plus 3 more (Cliff, Hole, Original) solved; stage-1 still at 0/30.
Transition throughput was ~309/s (402k transitions / 1,302.4s), about 10% lower than run #18's
~342/s -- the extra ray-casting and wider network input do cost a little, as expected, but nowhere
near enough to explain the large finish-rate improvement.

**Caveat, same as run #18**: this was also a deliberate mid-run pivot (to add the speed-bonus reward
term below), stopped while still climbing, not at a plateau -- so this is a lower bound and a
promising early signal, not a controlled, converged A/B result. Two runs is not enough to rule out
noise (both runs' curriculum/replay/critic-subset randomness differs even with the same top-level
seeds, since episode counts and timing differ), but an 8/30-by-1,302s vs 6/30-by-4,040s gap is large
enough relative to the run-to-run variance seen elsewhere in this log (e.g. run #18 alone moved
5/30->6/30 in its last 30 minutes) to be worth taking as a real, if unconfirmed, improvement.

Deployed as an interim snapshot: `policies/classic_sac_redq_headclear_interim.gdp` (best.gdp at the
time of the stop, 8/30/0.334).

## Speed bonus reward term

User request, after watching both SAC+REDQ runs: make the reward explicitly pay more for covering
the same distance *faster*, on top of the existing linear per-percent-progress bonus -- e.g. moving
from 5% to 20% progress in 5s should earn much more than doing it in 10s, not the same total.

**Why the existing reward didn't already do this**: `kProgressPercentBonus * percent_moved` is
linear in `percent_moved`, and `percent_moved` summed across the steps between two progress points
always totals the same distance-covered regardless of how many steps that took -- so two attempts
covering the same 15 percentage points earned essentially the same total linear-term reward whether
spread over few big steps or many small ones. The only pre-existing pull toward speed was indirect
(fewer steps means less accumulated idle penalty, and discounting mildly favors reward received
sooner) -- nothing rewarded speed *explicitly*.

**Change** (`gravity-lab/src/classic_environment.cpp`): since each environment step already spans a
fixed slice of game time (`Config::frame_skip` physics ticks), `percent_moved` this step already
*is* an instantaneous speed measurement, not just a distance one. Added a second, quadratic term on
top of the existing linear bonus, applied only on positive-progress steps (so it doesn't touch the
idle-vs-crash discounted-value math):
```cpp
double reward = percent_moved > 0.0
    ? kProgressPercentBonus * percent_moved * progress_multiplier
          + kSpeedBonusScale * percent_moved * percent_moved
    : -kIdlePenalty;
```
Because this term is quadratic rather than linear, covering a given span of track in fewer/bigger
steps earns strictly more total reward than the same total distance spread over more/smaller steps
-- directly implementing "faster earns much more," without needing to track wall-clock time
explicitly (it falls out of the fixed per-step time slice).

**Calibration**: `kSpeedBonusScale = 0.25`, chosen against a measured full-throttle run (Intro,
`ThrottleLeanForward` every step) rather than picked blind, continuing this log's established
practice of calibrating reward constants against measured behavior instead of assumed constants
(see the three reward-tuning attempts earlier in this log, each of which initially skipped this
step). Typical positive `percent_moved` during full throttle measured **0.36-0.46/step**; at 0.4 the
quadratic term (`0.25 * 0.4^2 = 0.04`) is almost exactly equal to the linear term
(`0.1 * 0.4 = 0.04`) -- roughly doubling reward for genuinely fast movement, while at a slow/cautious
`percent_moved = 0.05` the quadratic term is 1/8th of the linear one (negligible). Verified live: the
quadratic term's share of total positive-step reward grows from 37% (`percent_moved=0.21`) to 52%
(`percent_moved=0.40`) as speed increases toward the measured full-throttle range, and the engine's
reward output matches the closed-form formula exactly at every sampled step. Both CMake trees
rebuilt; `gravity_lab_classic_tests` and `pytest tests` (19/19) pass.

Combined with the head-clearance sensor for the next run (run #20) -- unlike the sensor (an
observation change, orthogonal to the reward), this changes what every prior run in this log
optimized for, so run #20 is not a clean A/B against run #19 for the sensor's effect alone, only
against the reward-tuning attempts much earlier in this log for the reward's effect. Warm-starting
from run #19 was not an option regardless (different reward changes what a "good" Q-value/critic
means, and the actor/critic architecture is unchanged but the target distribution they were trained
against is not).

### Combined outcome (run #20) -- new best, then a long plateau

Same config as runs #18/#19, no warm start (reward changed, same reasoning as above). Initially
extended to an 8h target mid-run (see "Wall-clock note" below) but stopped at 12,513.99s (~3.48h)
once the plateau below was clear. Reached a new session-best of **9/30 (30.0%), mean progress
0.456** -- ahead of both #18 (6/30) and #19 (8/30) -- and stage 1 registered its first-ever finishes
across all three SAC+REDQ attempts (Downhill and Spikehops each solved at least once, though not
always both simultaneously in the same eval snapshot -- see per-track flicker below). Per-track:
the same 5 core stage-0 tracks from #19 (Intro, Shorty, Slope, Crackle, Knolls) plus Cliff, Original,
and Savvy (all stage-0), plus Spikehops (stage-1).

**The plateau**: best score reached 9/30 by t=5211s (~87 min in) and never improved again for the
remaining ~2.9h of active training, despite the training-time finish rate and mean progress staying
healthy and non-declining throughout (consistently 27-43% / 0.49-0.60 across 150-episode windows,
right up to the stop) -- this is a genuine plateau on the *eval* metric specifically, not a collapse:
several stage-1 tracks kept making real, growing progress in live training without ever converting to
an eval-time finish (Blocks reached 13.3% training-time finish rate in the final 300 episodes;
Downhill and Hillclimb sat at 0.46-0.48 mean progress). This is the same qualitative shape as every
DQN/PPO plateau earlier in this log's "Session synthesis": a handful of tracks solid, several close
but not converting, most others untouched.

**Reward-hacking check, prompted by a user question**: does the per-step, previous-step-relative
progress bonus (especially the new quadratic speed term) create an incentive to retreat cheaply
(flat idle penalty, not proportional to retreat distance) then surge forward in one big step
(quadratic-in-distance bonus), farming reward via oscillation instead of genuinely finishing tracks?
Checked empirically: rolled the interim policy out deterministically on 5 stuck/near-miss tracks
(Deep, Hillclimb, Downhill, Spikehops, Hole) and tracked progress against its own running peak.
Max retreat-from-peak across all 5 was 0.0000-0.0021 -- negligible, essentially physics-settling
noise, not a deliberate exploit. **Not the cause of this plateau**, but confirmed as a real latent
risk in the reward's design (nothing prevents it structurally), addressed going into run #21 below.

**Wall-clock note**: this run was extended from a 2h to an 8h target mid-run (at t=4572.8s) once
the fail-fast gate had cleared and it looked promising. Discovered during that extension that most of
the elapsed *wall-clock* time between check-ins was not landing as *active* training time -- 15
consecutive gaps, mostly a repeating ~900s (15-minute) pattern, cost ~144 minutes of wall time with
almost no training progress. Diagnosed as **not** actual macOS sleep (`pmset -g` showed the host
continuously awake throughout, 166-day uptime, no reboot) but the session's own sandboxed execution
environment pausing between conversation turns -- confirmed by a live 15-second check showing
`active_training_seconds` advancing exactly 1:1 with wall-clock time while a turn was active, but
losing hours to gaps between turns. `caffeinate` (tried first, tied to the watchdog process) had no
effect, consistent with this being environment-level rather than guest-OS power management. No fix
available from inside the session; noted here since it means "N hours of active training" can take
substantially longer in calendar time than N hours if there are long gaps between check-ins, and
because it explains why this run's real-world timestamps span far more than 3.48h.

Deployed as an interim snapshot: `policies/classic_sac_redq_headclear_speedbonus_interim.gdp`
(best.gdp at the time of the stop, 9/30/0.456).

## Adaptive curriculum + peak-based progress (run #21 setup)

Motivated by the run #20 plateau and the user's question about forcing more experimentation on hard
tracks: two changes, one closing the reward-hacking risk found above, one attacking the plateau
directly by reallocating training attention.

**Peak-based progress bonus** (`gravity-lab/src/classic_environment.cpp`): the per-step progress
bonus (both the linear per-percent term and the quadratic speed term) now fires only on progress
*past the episode's peak so far*, not merely past the immediately preceding step -- retreating and
re-covering already-visited ground earns nothing (still pays the flat idle penalty, same as before),
so the only way to earn any positive reward is to reach genuinely new track position. This closes the
retreat-then-surge loophole structurally (not just empirically-currently-unobserved) and adds a mild
intrinsic push toward unexplored territory, which is also a partial, indirect answer to the "pay for
trying something new" question -- new peak progress is the only thing that pays.

**Adaptive curriculum** (`src/gravity_lab_rl/sac_trainer.py`): ported the inverse-success-weighted
track selection already proven in `ppo_trainer.py` (see "Sparse-success plateau and the adaptive
curriculum fix" above) -- replaces SAC's plain round-robin curriculum with selection weighted by
`1 / (track_success_ema + 0.05)`, a slow EMA (alpha=0.05) per track. A struggling track gets picked
far more often than a mastered one (up to ~20x at the 0%-vs-100%-success extremes), giving the
stuck stage-1 tracks (Downhill, Hillclimb, Blocks, Bumps, Pillar -- all showing real but
not-yet-converted progress in run #20) more rehearsal instead of the same fixed share every
mastered track gets under round-robin. `track_success_ema` persists across checkpoints, same as PPO.

### Adaptive curriculum outcome (run #21) -- over-concentration, stopped and re-tuned

Fresh run (no warm start -- new reward scale), otherwise identical to run #20's config. Early
signs were positive: stage-0 gate held, and **Hole was solved for the first time ever** across
every SAC+REDQ attempt (runs #18-20 never converted a single Hole finish, training-time or eval) by
t=1861s (31 min), reaching 8/30 -- roughly run #20's pace to the same count, but via a genuinely
different, previously-unsolvable track rather than the same easy set.

**Then it got worse, not just flat.** By t=3781s, `track_success_ema` showed every single stage-1
track pinned at 0.0-0.08 -- not merely "still hard," but uniformly zero, because the weighting
formula's `+0.05` floor gives a 0%-success track ~20x a mastered track's selection weight, and once
*every* newly-unlocked stage-1 track ties at that floor simultaneously (as happens right when a
whole stage unlocks at once, unlike PPO's original validation context of individually-differentiated
per-track success rates), the curriculum dumps nearly all its budget there and starves stage-0 of
refresher practice. This is a **runaway feedback loop**, not a one-time imbalance: a track that's
losing gets picked more, which produces more losses (or at best no signal), which keeps its weight
maxed, which keeps it getting picked. Training-time finish rate and mean progress both *declined*
over the run's second half (finish rate 6-10% -> 0-3.5%, mean progress 0.38-0.40 -> 0.25-0.36 across
150-200-episode windows) despite ~76 minutes of concentrated stage-1 rehearsal producing zero
stage-1 formal-eval finishes. Stopped at 6,634.15s once this decline (not just a flat best-score) was
clear -- final result **8/30 (26.7%), mean progress 0.414 best** -- essentially a wash against run
#20's 9/30/0.456, i.e. this run's changes did not net-improve on their own combined merits, and the
adaptive curriculum specifically regressed live training quality in its second half. (Final,
non-best evaluation was much worse, 1/30/0.246 -- expected instability from stopping mid-training on
whatever policy the last few concentrated-hard-track episodes happened to leave behind, not a
meaningful number; `best_evaluation` is the one that matters, per this log's usual convention.)

**Fix going into run #22**: raised the weighting floor from `0.05` to `0.15` in
`_select_next_environment` (`sac_trainer.py` only -- `ppo_trainer.py` is unchanged, since this
specific failure mode wasn't observed there and this session has no evidence it needs the same
fix). Effect: caps the maximum weight ratio between a 0%-success track and a ~90%-mastered one at
roughly `(0.9+0.15)/0.15 ≈ 7x` instead of `~20x`, leaving real priority for struggling tracks while
guaranteeing mastered tracks a much larger residual share than the near-zero share they were getting
under the runaway state observed above. Peak-based progress reward is unchanged and kept -- it
wasn't implicated in this specific regression (no evidence of reward-hacking recurrence, and the
mechanism is sound by construction regardless of curriculum behavior).

### Rebalanced curriculum outcome (run #22) -- fix confirmed, same ceiling

Fresh run, `success_ema` floor raised to `0.15`, otherwise identical to run #21. The fix held for
the run's full 6,660.29s (~1.85h): stage0/stage1 attention split stayed in a healthy 30-38%/62-70%
range throughout (verified at four separate checkpoints across the run), and training-time finish
rate/mean progress stayed noisy-but-flat (5.5-18.5% / 0.36-0.47 across dozens of 200-episode
windows) with no repeat of run #21's sustained decline. Reached **9/30 (30.0%), mean progress
0.454** by t=1614s (27 min) -- matching run #20's eventual peak (9/30/0.456) in roughly a third of
run #20's time to get there (t=5211s), and via a different, healthier route: Downhill (stage-1)
instead of Spikehops rounds out the ninth track, and stage-1 registered 3/30-episodes in the best
eval (vs. run #20's exact-same 3/30 stage-1 count, so comparable stage-1 depth reached faster, not
just stage-0 reshuffled).

**But it plateaued at the same ceiling.** No improvement past 9/30 for the remaining ~1.3h of active
training despite the healthier attention allocation -- stopped by explicit user request, not a
fail-fast or decline trigger. Near-miss tracks (Indoor briefly hit 50% training-time finish rate in
one window, Hillclimb reached 13.3%) showed real movement without ever converting to a formal-eval
finish, the same "engaging but not yet consolidating" pattern seen in every prior run once it
reaches this territory.

**Assessment**: the curriculum fix (0.15 floor) is validated -- it reaches the same peak
meaningfully faster with a demonstrably healthier training dynamic (no decline, balanced attention)
-- but reaching the peak faster did not, on its own, break through *past* the peak. The ~9/30
(30%) ceiling that DQN, PPO, and now three SAC+REDQ configurations have all separately converged to
(see "Session synthesis" above) continues to hold across yet another set of changes. Deployed:
`policies/classic_sac_redq_curriculum_v2_interim.gdp` (9/30/0.454, this run's best).
