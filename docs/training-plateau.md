# Diagnosing the hard-map plateau

Evidence from `sac_redq_finishbonus_only_20260906_021909` (30-minute SAC/REDQ run):

| Observation | Implication |
| --- | --- |
| Group 0: 615 training episodes; group 1: 410; group 2: **0** | Group 2 results measure transfer to unseen maps. The curriculum never unlocked that group. |
| Map 0:9: 185 training episodes, no finishes | This is a practiced but unsolved obstacle, not merely a lack of map exposure. |
| Map 1:1: 90 training episodes, no finishes | Repeating the same failed behavior needs a more targeted experiment. |
| Best evaluation: 8/30 maps finished; final: 4/30 | Training regressed after the best checkpoint. Keep best-policy evaluation alongside final-policy evaluation. |

These counts come from `metrics.jsonl` and `summary.json`. They are one training run,
not evidence that any proposed change will reliably improve performance.

## Suggested experiment order

1. **Separate exposure from skill.** Log training episode counts beside evaluation
   results for each map. Try an all-map curriculum with a small guaranteed sampling
   share for every map, while retaining extra practice on near-solvable maps. The
   current strict group-unlock threshold can indefinitely exclude group 2.
2. **Concentrate practice.** Train one failing map such as 0:9 separately. For a
   larger improvement, implement complete simulation snapshots and restart some
   practice episodes just before the obstacle. Save and restore all bike positions,
   velocities, integration state, and level collision state; simply teleporting the
   bike is not a valid restart. Keep a share of full-start episodes and evaluate only
   from the original start. Adapt practice difficulty as success improves. This is
   motivated by [start-state curriculum research](https://arxiv.org/abs/1707.05300),
   but its effectiveness on this game remains to be tested.
3. **Lengthen the reward horizon.** At frame skip 2, actions span 0.04 game seconds.
   With gamma 0.99, a reward 20 seconds away is multiplied by `0.99^500 = 0.00657`.
   Compare gamma 0.995 (`0.0816` at 20 seconds), holding other settings fixed. This
   gives delayed success more influence; it can also make value learning harder.
4. **Check whether the reward discourages obstacle setup.** Current shaping rewards
   squared forward speed and penalizes every step without new peak progress,
   including braking and backing up. A cautious maneuver may therefore look worse
   than a fast crash. First compare `speed_bonus_scale: 0` against 0.25. A separate
   experiment could delay the idle penalty until sustained inactivity. Watch for
   standing-still behavior if changing that penalty.
5. **Measure regressions, not just the latest result.** Retain a fixed test suite of
   easy maps while working on hard maps. Use `best.gdp` for the strongest measured
   checkpoint. Track-balanced replay already exists; investigate practice coverage,
   critic targets, and within-track replay coverage before adding another sampler.

For each comparison, change one factor, use the same maps and training budget, and
repeat with at least three independent training seeds. Report finish rate, peak
progress, and the obstacle where attempts fail. Changing evaluation seeds alone
may provide little diversity when a deterministic policy starts from the same state.

A higher REDQ update-to-data ratio is a later experiment (for example 1 versus 2).
The [original REDQ paper](https://arxiv.org/abs/2101.05982) combines a high ratio with
an ensemble and random target subsets, primarily on continuous-control benchmarks.
This repository uses discrete actions and CPU training: more updates cost time and
cannot supply successful transitions the replay buffer has never seen. Compare both
performance per environment transition and performance per minute.

The first 10-minute run retained the prior algorithm, rewards, and curriculum as a
baseline, but was stopped after about 163 seconds before the combined experiment.

## Implemented combined experiment

The baseline run was stopped at the user's request. The revised 10-minute run uses
`configs/classic_all_tracks_sac_obstacle_practice.json`:

- All three groups are available immediately. Every other map-selection turn picks
  a map with the fewest completed full-start episodes; remaining turns use adaptive
  failure weighting. Maps 0:9, 1:1, and 1:2 receive a 3x focus weight on those adaptive
  turns. Easy tracks retain guaranteed coverage for regression prevention.
- `practice.enabled: true`, with probability 0.5 and candidate states recorded every
  25 actions. The bank keeps short valid routes in 5% progress bins, up to 90% of the
  map, leaving at least 100 environment steps for further practice. Half the practice
  selections target the furthest saved bin; half rehearse other reachable bins.
- Practice reconstructs the complete simulator state by resetting with the saved
  environment seed and replaying the saved action prefix. The resulting observation
  must match exactly. This avoids introducing a partial native snapshot format or
  teleporting the bike. Original episode time limits still apply. Prefix steps are
  counted separately and are not added again to the replay buffer. The original
  peak progress is reconstructed too, preventing previously covered ground from
  earning a fresh progress bonus.
- The practice bank and its independently seeded RNG are checkpointed. Evaluation
  and videos always use full-start episodes. Practice success does not raise the
  scheduler's full-start success estimate. Metrics identify practice prefix length,
  peak progress, per-map completed episodes, and per-map full-start episodes.
- Gamma is 0.995, speed bonus is zero, and `idle_grace_steps: 25` allows one game
  second without new peak progress before idle penalties start. Genuine new peak
  progress resets that timer; recovering only to the old peak does not.
- REDQ uses two updates per collected step. The 10-minute wall-clock training budget
  includes practice-prefix reconstruction and periodic evaluation.
- `evaluation_history.jsonl` preserves complete fixed-protocol evaluations with map
  exposure counts. The best checkpoint's training time is recorded. Videos include
  `best.gdp` when it differs from `final.gdp`.

Use `practice.enabled: false` to disable practice, `curriculum.unlock_all: false`
to restore staged unlocking, and `curriculum.guaranteed_coverage: false` to restore
fully adaptive selection. These curriculum/practice additions currently apply to
SAC/REDQ. Episode reward grace is supported by all three trainers and evaluation.

The combined run is a functionality and outcome check, not an ablation study or a
multi-seed estimate. It cannot establish which individual change caused a result.
