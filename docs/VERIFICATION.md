# Verification — what counts as evidence here

This project deliberately keeps several different notions of "it worked" apart. They are
easy to conflate and conflating them is how RL results get overstated, so this document
lists them and says which ones are load-bearing.

## 1. The predicates

### 1.1 `valid` — the strict glide predicate

Implemented once, in `curriculum_onefoot_valid(...)`, and **never loosened** as the target
duration grew:

```python
finite & left & ~right & ~forbidden
  & clearance          >= .010    # metres, measured from the tyre mesh bound (0.016 m)
  & speed              >  .10     # root link forward velocity, m/s
  & speed              <  .80
  & |lateral|          <  .15     # m/s
  & |heading|          <  .35     # rad
  & |cross_track|      <  .20     # m
  & upright            >  .80     # -projected_gravity_b.z
  & |blade_yaw|        <  .35     # rad, atan2 of the line between the two left wheels
  & phase              >= .99     # the sustained hold, not the transfer
```

with

```python
left  = both left wheels reported found  AND  each carries > 0.02 N
right = any right wheel reported found          # `~right` is required
forbidden = onefoot_bad_contact(env)            # any non-wheel ground contact
```

Two details matter and are deliberate:

* **A contact-detector hit is not evidence of loaded support.** `left` requires
  `found.all()` *and* a `> 0.02 N` normal load on **each** left wheel. A mere collision
  flag is not enough.
* **`speed` is the root-link forward velocity**, not the COM velocity — the original
  success contract. The COM speed is logged *separately*
  (`com_speed`) precisely so that a torso-motion/centroid discrepancy is visible rather
  than hidden inside the success metric.

The counter is an integer number of consecutive steps, converted to seconds by
`× step_dt`, so there is no floating-point accumulation in the duration.

### 1.2 `credible` — stricter, and the one this work is judged on

```python
credible = valid & finite & ~physical_failure & (com_speed > .10) & (skate_speed > .05)
```

i.e. `valid`, **plus** the COM is genuinely moving forward and the support skate is
genuinely rolling forward. A robot that satisfies `valid` while its torso oscillates in
place with the skate barely turning is `valid` but not `credible`.

The headline numbers in this repository are **credible** numbers. `success_*` (the plain
`valid` metric) is also reported everywhere, and is always the more flattering of the two.

### 1.3 What is *not* a success

Explicitly **not** treated as evidence of the skill:

* posture survival,
* a single favourable batch,
* success under initial assistance (see §3),
* the discounted PPO return,
* the undiscounted logged reward,
* a successful ONNX export.

The discounted return is used for *judging learning* (it is the quantity optimisation
affects), never as evidence that the physical task was completed.

## 2. The reward's integrity conditions

These are asserted at runtime on **every** evaluation, not sampled:

| Condition | Assertion |
|---|---|
| Shaping telescopes to zero | `discounted_shaping_max_abs < 2e-5` (measured 1.7e-7) |
| The horizon is a true terminal | `assert not extras['time_outs'].any()` |
| The reward manager does not double-scale | `assert allclose(reward, r['total'])` |
| No non-finite actions | `assert isfinite(action).all()` |
| No NaNs | `nan_episodes == 0` required by the gate |

Every termination term is configured with `term.time_out = False`. That is what makes the
horizon the *true end of the task attempt* rather than an artificial truncation, and it is
why RSL-RL must not bootstrap across it. Combined with all terminal potentials being zero,
this makes the return well defined and stops the finite horizon from being farmed.

## 3. Initial assistance

Every evaluation records `initial_assistance = (stage != 'full')`, and the gate requires
this flag to match the stage. This is the mechanism that keeps assisted results honest:

* Assisted stages (`near`, `unload`, `transfer-near`, `transfer`, **`glide`**) start from a
  non-HOME state with injected velocity. Their results **can never certify a from-rest
  result.**
* Only **`full`** starts at HOME with root, joint and wheel velocities exactly zero, and
  only `full` can produce a from-rest claim.

`glide` was added to `SUSTAINED_STAGES` so that it is judged against the *same* 2.0 s
metric as `full` — otherwise the two stages would certify different tasks and the
comparison would be meaningless. But being judged on the same metric does not make it the
same evidence.

## 4. The acceptance gate

```python
def bridge_stage_gate(reports, stage, checkpoint_sha256):
    duration = '200' if stage in SUSTAINED_STAGES else '050'
    if len(reports) != 2 or len({r['seed'] for r in reports}) != 2:
        return False
    return all(r['stage'] == stage
               and r['checkpoint_sha256'] == checkpoint_sha256
               and r['episodes'] >= 256
               and r['nan_episodes'] == 0
               and r['initial_assistance'] == (stage != 'full')
               and r[f'success_{duration}_rate']  >= .80
               and r[f'credible_{duration}_rate'] >= .80
               and r['discounted_shaping_max_abs'] < 2e-5
               for r in reports)
```

In words: **two independent runs of 256 first episodes on two distinct seeds, at least 80 %
on both the `valid` and the `credible` predicate, zero NaNs, shaping verified to telescope,
and the checkpoint identified by SHA256.** The gate is not a summary statistic; each of the
two reports is checked individually.

The gate is attempted by the trainer only when the 64-episode screening evaluation already
reaches 0.8 on the gate key, and only with `--exit-on-gate`. The gate key is
`credible_200_rate` for `SUSTAINED_STAGES` and `credible_050_rate` otherwise.

### What has and has not been gated

| Stage | Gate key | Screening best | Gated | Passed |
|---|---|---:|---|---|
| `near` | credible 0.5 s | 87.1 % | yes | **no** |
| `unload` | credible 0.5 s | 83.9 % | yes | **no** |
| `transfer-near` | credible 0.5 s | 94.9 % | yes | **no** |
| `transfer` | credible 0.5 s | 85.5 % | yes | **no** |
| **`glide`** (`model_10499`) | credible **2.0 s** | 1.6 % | **yes, run directly** | **no** |
| `full` | credible **2.0 s** | 0 % | never | — |

The trainer only attempts the gate once 64-episode screening reaches 0.8 on the gate key.
That never happened for the stages carrying the 2.0 s goal, so they had no gate result at
all. A dedicated runner (`scripts/run-bridge-stage-gate.py`) now calls `bridge_stage_gate`
directly, and `glide` has been gated properly.

**`glide`, `glide-glide-02-training/model_10499` (SHA256 recorded in the report file),
seeds 71301 / 71302, 256 first episodes each — FAILED**

| Condition | Required | 71301 | 71302 |
|---|---|---:|---:|
| `episodes` | ≥ 256 | 256 | 256 |
| distinct seeds | 2 | ok | ok |
| `checkpoint_sha256` | match | ok | ok |
| `initial_assistance == (stage != 'full')` | True | True | True |
| `nan_episodes` | 0 | 0 | 0 |
| `discounted_shaping_max_abs` | < 2e-5 | 7.5e-7 | 8.1e-7 |
| `success_200_rate` | ≥ 0.80 | 0.0 % | 0.4 % |
| `credible_200_rate` | ≥ 0.80 | **0.0 %** | **0.39 %** |

Everything except the 2.0 s duration passes. The same 512 first episodes give credible
0.5 s **99.6 %** and credible 1.0 s **95.3 %**. So the stage is measured, and fails on
exactly one axis, by 0.39 % against a required 80 %. Full report:
`records/glide-glide-02-gate.json`.

The four assisted transfer stages were gated earlier and all failed, because their credible
1.0 s and 2.0 s rates were 0 % under a 0.35 m/s injection ceiling. `full` still has no gate
result: its screening never approached the trigger.

## 5. Metrics as defined

All per-block numbers come from `evaluate()` in `scripts/train-credit-bridge.py`: N
environments are run, and the **first episode of each environment** is tracked, so
`episodes = N` counts first episodes.

* `credible_050/100/200_rate` — fraction of first episodes whose longest *credible* streak
  reached 0.5 / 1.0 / 2.0 s, excluding any episode that produced a NaN.
* `max_credible_glide_s`, `mean_credible_glide_s` — over the same first episodes.
* `mean_episode_s` — mean first-episode length.
* `termination_counts` — per-term counts at the first-episode termination. **A termination
  term can co-fire**: real recorded values include `fell_over,body_ground` and
  `body_ground,stalled_transfer` for the same episode. So `termination_counts` is a set of
  per-term counts, **not a partition**, and the counts sum to more than the episode count.
  Falls should be read as "`body_ground` and/or `fell_over`", not as a sum.
* `entry_speed` — COM forward speed at the first frame of single support, reduced with
  `last` for the metric and `max` over the episode for the report.
* `discounted_handoff_return` — the discounted sum of the `handoff` term, reported
  separately from the base return so the bonus can be inspected on its own.

## 6. Checkpoint and source integrity

Training verifies, every chunk:

* the SHA256 of every pinned source file is unchanged,
* the frozen-expert digest is unchanged,
* `retained_skills_digest(actor)` equals the value recorded at the start,
* all actor and critic parameters are finite,
* the learning rate is unchanged and fixed,
* the main curriculum checkpoint's hash is unchanged.

This is not decoration: it is what caught an operator editing a pinned source mid-run
(see `REPORT.md` §9). The run stopped at update 700, every recorded report verified, and it
was resumed from update 600. The incident record is `records/incident-v16b-01.json`.

`evidence/checkpoints.json` carries the SHA256 of every checkpoint shipped here, plus the
pinned-source hashes recorded by the run that produced it.

## 7. Test suite

`CUDA_VISIBLE_DEVICES='' uv run --locked --with pytest python -m pytest tests/`
→ **434 passed, 1 skipped**, CPU-only.

The tests that guard the claims in this repository:

| Test | Guards |
|---|---|
| `test_old_mdp_and_registry_bytes_are_preserved` | the first 351 782 bytes of `tasks/mdp.py` |
| `test_retained_skill_bridge.py` | the inlined `.25` gate literal equals `HOLD_AUTHORITY` |
| `test_credit_bridge_rewards.py` | handoff is zero ≤ 0.38, one-shot, needs the 0.25 s hold; balance debt is zero inside the deadband; the glide stage config and its pose evidence |
| `test_bridge_recovery.py` | the three transition kinds and actor protection |
| `test_onefoot_curriculum.py` | pitch-aware spawn quaternion, per-wheel mesh lifts, glide single-support geometry |

## 8. Provenance of the numbers in this repository

`evidence/*.json` is generated by parsing the training logs, trace files and state files
directly, not transcribed by hand:

| File | Contents |
|---|---|
| `milestones.json` | every `CREDIT_MILESTONE` record from all 12 training runs (92 records) |
| `gates.json` | every gate report, including the 15 per-seed reports |
| `coast-down.json` | per-glide entry/exit speeds, fitted deceleration, exit reason, speed budget and derived 2.0 s requirements |
| `checkpoints.json` | SHA256 of every shipped checkpoint + the pinned-source hashes of the runs |
