# Training trajectories

The learning curves of this work, exported from the raw TensorBoard logs so they can be read
without the 155 MB of event files (377 of them) that produced them.

## Contents

| File | What |
|---|---|
| `<alias>.csv` | One file per run: one row per update, one column per scalar tag. 15 runs. |
| `index.json` | Per run: the upstream log directory, the CSV hash, the tag list with step ranges, and the SHA256 of every source event file. Also holds the `/time`-suffixed series, whose x-axis is wall-clock seconds rather than the update index. |
| `screening.json` | The **screening evaluations** — 64 first episodes, seed 70601, every 100 updates — for the 13 runs that have them. These are the numbers the acceptance gate and `REPORT.md` judge. |

Everything here is machine-extracted; no number was typed by hand. `index.json` and
`screening.json` carry the hashes, so any row can be traced back to the run directory it
came from.

## Two different curves, and why both are here

The `Episode_Metrics/*` columns in the CSV are **4096-environment training aggregates**,
logged every update by the trainer. They are the raw learning signal.

`screening.json` is a **separate evaluation**: 64 episodes, one fixed seed, deterministic
actions, run every 100 updates. It is not a subset of the CSV — the numbers genuinely differ
because they measure different populations. At u1000 of `glide-glide-02`:

| | training aggregate (CSV) | screening (JSON / REPORT §7.1) |
|---|---:|---:|
| credible ≥1.0 s | 0.774 | 0.969 |

Neither is "the" number. The screening series is the one the gate uses; the training
aggregate is the one available for every update, including the runs that were never screened.
`screening.json` also keeps the `updates: 0` pre-training baseline that `evidence/milestones.json`
omits, which is what makes a run's starting point visible.

## The runs

`updates` is the iteration range. "Assisted" runs start from a measured mid-motion state and
can never certify a from-rest result; only `full` starts at HOME with exactly zero velocity.

| Alias | Stage | Assisted | Updates | Screening pts | Best credible ≥1.0 s | Best glide |
|---|---|---|---:|---:|---:|---:|
| `main-expert` | — | — | 16400–16699 | — | the frozen upstream waist-balance expert | — |
| `retained-bridge-pilot` | — | — | 0–299 | — | first bridge connection | — |
| `unload` | unload | yes | 300–1299 | 11 | 0 % | 0.72 s |
| `unload-extension-01` | unload | yes | 1300–1668 | 4 | 0 % | 0.72 s |
| `unload-extension-02` | unload | yes | 1600–2599 | 11 | 0 % | 0.76 s |
| `unload-extension-04` | unload | yes | 2600–3199 | 7 | 0 % — **passed its 0.5 s gate** | 0.92 s |
| `transfer-near-extension-04` | transfer-near | yes | 3200–3499 | 4 | 0 % — **passed its 0.5 s gate** | 0.92 s |
| `transfer-extension-04` | transfer | yes | 3500–3899 | 5 | 0 % — **passed its 0.5 s gate** | 0.88 s |
| `full-extension-04` | full | **no** | 3900–4899 | 11 | 18.8 % | 1.24 s |
| `full-reboot-01` | full | **no** | 4900–5899 | 11 | 51.6 % | 1.36 s |
| `full-v16` | full | **no** | 5900–6899 | 11 | 68.8 % | 1.66 s |
| `full-v16-02` | full | **no** | 6900–7899 | 11 | 81.3 % | 1.82 s |
| `full-v16b` | full | **no** | 7900–8599 | 7 | 84.4 % | 1.84 s |
| `glide-glide-01` | glide | yes | 8500–9499 | 11 | 89.1 % | 1.72 s |
| `glide-glide-02` | glide | yes | 9500–10499 | 11 | **96.9 %** — first 2.0 s glides | **2.14 s** |

"Best" is the maximum over each run's screening points, not the value at its last update,
and it is the *credible* predicate — the strict one. The 0.5 s gates that passed did so on
`credible_050_rate` with 256 episodes × 2 seeds, which is a different and stricter
certification than any single screening point; see `docs/VERIFICATION.md`.

## Reading the columns

Tags are grouped by prefix. The full list is in `index.json`; this is what the groups mean.

**`Episode_Reward/<term>`** — the reward, decomposed. `task` and `shaping` are the objective,
`failure` is the −2 event, `handoff` is the one-shot entry-speed bonus added by v16. The
`curriculum_*` terms belong to the earlier, superseded recipes.

**`Episode_Termination/<term>`** — why episodes ended. These **co-fire**: one episode can
trip `body_ground` and `fell_over` together, so the columns sum to more than the episode
count and must never be read as a partition. `body_ground` and `fell_over` are falls;
`time_out` is hitting the horizon, which is a *true terminal* here, not a truncation
(`docs/METHOD.md` §6).

**`Episode_Metrics/bridge/*`** — the glide-credit machinery. `credible` is the strict
per-frame glide predicate, `credible_best_s` the longest glide so far, `entry_speed` the COM
speed at the moment one-foot support is established, `balance_debt` the accumulated lateral
and fore/aft debt, `quality` the 1/(1+debt) discount, `progress_steps` the credit counter.

**`Episode_Metrics/onefoot/*`** — the physical state. `com_speed`, `lateral`, `cross_track`,
`heading_error`, `upright`, `left_force` / `right_force`, `clearance`, and the `waist_*`
family describing the weight transfer. `success_050` / `success_100` / `success_200` are the
loose per-metric durations at 0.5 / 1.0 / 2.0 s; the `credible_*` columns in `screening.json`
are the strict ones and are the ones the gate uses.

**`Loss/*`, `Policy/mean_std`, `Perf/*`** — PPO internals. `Loss/learning_rate` should read a
constant `1e-4`: the schedule is deliberately `fixed` (`docs/METHOD.md` §6). `Perf/total_fps`
is what the run cost.

**`Train/mean_reward`, `Train/mean_episode_length`** — start one update later than the rest
(the episode aggregates are reported a step behind the training loop).

## What the curves show

The single most useful thing in these files is visible in one glance at
`Episode_Termination/time_out` across `full-v16`, `full-v16-02` and `full-v16b`: it climbs
while `body_ground` falls. The robot stops falling and starts surviving to the horizon.
Then `Episode_Metrics/onefoot/com_speed` shows *why* that was not enough — the surviving
episodes bleed to 0.00 m/s and stand still. That transition is the central finding of this
work and is written up in `REPORT.md` §8.

## Regenerating

```bash
# needs the upstream working tree with its logs/ directory intact
uv run --locked python scripts/export-training-trajectories.py --out trajectories
```

Re-running rewrites the CSVs byte-identically, since the source logs are immutable and the
values are formatted to 6 significant digits. The raw logs remain the authority for any
number quoted in `REPORT.md`.
