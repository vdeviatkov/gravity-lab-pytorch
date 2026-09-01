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
| 8 | Reward attempt 2: per-step penalty →0.1 (corrected discounted math), warm-started from #6 | 72 | `configs/classic_all_tracks_32ray_accel.json` | *pending* | *pending* | *pending* | *pending* | **in progress** | *pending* |

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

Run #8 relaunched after this fix. Result pending — see the table row above once it completes.
