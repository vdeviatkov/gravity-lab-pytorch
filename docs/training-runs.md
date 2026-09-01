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

## Notes

- **Run #6 was stopped manually** (`gravity-lab-rl control stop`) at 2,745.6s of its 3,600s target,
  after hands-on play revealed the trained policy tends to freeze in place on maps it hasn't
  mastered rather than attempt them. That's explained by the reward scale, not the network: an
  idling episode costs at most `0.001 * 2000 steps = 2.0`, less than the `5.0` crash penalty, so
  standing still is the reward-minimizing choice on any section the policy hasn't learned to pass
  reliably. Its own final checkpoint had also regressed to 3.3% by the stop point, the same
  late-training collapse tendency seen in run #3 — best-checkpoint tracking is what protects
  #4 and #6 from that. See the reward rebalance below.

- **Run #3 is why best-checkpoint tracking (`best_checkpoint_eval_interval_seconds` in
  `Trainer`) exists at all**: its final checkpoint silently collapsed to 0% from a 20% peak in
  the last ~140s of training, with no warning in training-time loss or reward. Full writeup in
  `docs/policy-comparison.md`.
- Runs #4 and #6 both use the fix: periodic deterministic evaluation during training, keeping
  the best-scoring `(finish_rate, mean_progress)` checkpoint rather than whatever exists when
  the clock runs out. `best_checkpoint_eval_interval_seconds: 90` for both (a full 30-track
  eval costs ~0.2s, so this is cheap).
- Run #6's row will be updated in place once the in-progress run completes, rather than adding
  a new row for the same run.
- Observation layout reference: `[0,28)` base bike/track state, `[28, 28+ray_count)` obstacle
  rays, `[60,72)` per-component acceleration (always present in the 72-wide layout). Every
  shorter width above is an exact, unchanged prefix of every longer one — see
  `gravity-lab/include/gravity_lab/classic_environment.hpp` and `docs/policy-comparison.md`.
- Hyperparameters shared by rows 3, 4, and 6 (the all-tracks curriculum configs): lr=0.0005,
  γ=0.99, batch=128, replay capacity 300k / warmup 10k, target sync every 2,000 updates,
  ε 0.3→0.05 over 1M transitions, 5 episodes/track before curriculum advances, evaluated/trained
  on CPU with `torch_num_threads=1`.
