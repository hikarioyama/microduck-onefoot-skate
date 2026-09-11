# Microduck — one-foot skating via a retained-skill bridge

Reinforcement-learning work on **Microduck** (a ~25 cm, ~800 g biped with 14 Dynamixel
XL330 servos and four passive roller wheels), built on
[mjlab](https://github.com/mujocolab/mjlab).

The goal is a single skill: **accelerate from rest on the rollers, transfer weight onto
the left foot, steer inward with the support skate, and hold a sustained left-foot
glide** — ideally entering that glide at the robot's top speed and staying up for as
long as possible.

> ## Status: interim — the target skill is NOT achieved
>
> * Best one-foot glide: **2.14 s**. **96.9 %** of first episodes hold ≥1.0 s; **100 %**
>   hold ≥0.5 s.
> * The declared goal of a **2.0 s** credible glide has been reached — but by only
>   **1.6 %** of episodes (1 of 64), against the 80 % the gate requires.
> * **No certified stage has ever passed the formal acceptance gate** (256 first episodes ×
>   2 seeds, ≥80 % on both the original and the credible metric). Three *assisted* 0.5 s
>   stages — `unload`, `transfer-near`, `transfer` — did pass their own 0.5 s gate, and that
>   is recorded; but the two **sustained** stages (`glide`, `full`) are judged on the 2.0 s
>   metric and neither has passed it.
> * The `glide` stage is **assisted** (it is initialised from a measured mid-glide state),
>   so it can never certify a from-rest result. Only the `full` stage, which starts at
>   HOME with exactly zero velocity, can do that — and `full` has not passed either.
> * The two cheap routes to the gate are now **measured and closed**: entry speed is capped
>   by the valid predicate, and raising the correction's authority destroys the robot. The
>   remaining obstacle is a **glide deceleration of ~0.34 m/s² whose mechanism is not yet
>   identified**, against 0.058 m/s² for a four-wheel coast.
>
> This repository is a checkpoint of the *method and the measurements*, not a claim that
> the robot can skate.

---

## What is actually in here

Four things were produced, in order of how much they matter.

**1. A learnable-authority audit that found 35 % of every episode was dead.**
The policy is a trainable bridge between two frozen experts. Its correction gate is

```
blend = phase² · (3 − 2·phase)
gate  = 4·phase·(1 − phase) + 0.25 · blend
out   = (1 − blend)·public_roller + blend·onefoot_expert + gate·correction
```

At `phase = 0` — i.e. during the entire 2.0 s from-rest acceleration — the gate is
**exactly zero**, so the trainable part has no authority at all. With a 5.73 s mean
episode, that is 2.00 s = 35 % of every rollout spent with *nothing to learn*. Worse, that
2 s is not improvable by more training: freezing the accelerator and letting the public
expert run for 8 s caps the speed at **0.382 m/s**, and extending the lift window from
1.0 s to 6.0 s still caps at **0.49 m/s**. The robot stops pushing; it is not a
scheduling problem.

**2. An assisted `glide` stage built from measured data, not guessed.**
The first frame at which left-only support is established was captured across 64 rollouts
(`scripts/measure-glide-pose.py`), reduced to a median, clamped inside the joint limits,
and grounded on the *tyre mesh vertices* rather than on a nominal radius. The stage starts
at `phase_start = 1.0`, so **every episode is 100 % inside the learnable region**.

**3. Two genuine spawn-state defects, found and fixed.**
The stage initially scored 0.281 on credible-0.5 s where the same policy scored 0.969 in
`full`. A joint-level diff of the spawn state against real recorded handoffs explains why:

| Defect | Spawn was | Real handoff is |
|---|---|---|
| Servo joint velocities | all **zero** (median \|v\| 1.86, max 7.64 rad/s) | non-zero, moving |
| Wheel speeds | all four forced to `v/r` = **+34.5 rad/s** | support **+36.7 / +36.5**, swing **−2.1 / +11.1** |

The second one is the dominant error and it matters because wheel velocities are part of
the actor's observation. Fixing both moved credible-0.5 s from **0.281 → 0.672** and
tripled the mean episode length.

**4. Reward v16 (handoff bonus + two-axis balance shaping).**
A one-shot bonus for the COM speed at the moment one-foot support is established, plus a
lateral **and** fore/aft balance debt. The fore/aft axis was explicitly requested after
noting that the robot collapses backward as well as sideways.

**5. The full training record, and a surface that actually runs.**
Two things beyond the findings are preserved because a checkpoint of conclusions is not
reusable on its own.

*The record.* `trajectories/` holds the learning curves of **all 15 runs** — 6.2 MB of CSV,
one row per update — plus `screening.json`, which is the 64-episode screening series the
gate and this report judge (and which is a different population from the 4096-env training
aggregates in the CSV: at u1000 of `glide-02`, credible ≥1.0 s is 0.774 in training and
0.969 in screening). Both are exported from the 377 raw TensorBoard event files, so the
curves stay readable without the 155 MB of logs that produced them.
*The decisions.* The reasoning, alternatives and negatives are the five signed
`records/actor-transition-*.json` files, each stating what changed, which owner file moved,
what was measured to justify it, and the SHA256 of the sources on both sides of the change.
The negative results are kept deliberately: they are what makes the remaining open problem a
statement about the physics rather than about what has not been tried yet.

*The surface.* Making the repository usable by someone else turned out to require real work,
not just prose. `src/` is an overlay on upstream `53b8971`, and 13 of the 24 test modules
load a tool by file path from `local/`. Upstream's `local/` is **not tracked by git at all**
(`git ls-files local` is empty across 281 files), so this repository is the only place those
tools are published; they are vendored here in full, along with the two measured data files
three tests read and the frozen one-foot expert at the path the pinned code records. The
result: a bare clone of this repository scores **42 failed / 1 collection error**, and the
documented overlay scores **434 passed, 1 skipped, 0 failed**. The shipped ONNX policies are
also verified against the recorded actors rather than assumed — 2.15e-06 max difference,
`records/onnx-vs-actor.json` — and `docs/ONNX-INFERENCE.md` documents the 61-dim observation
layout needed to drive them without installing mjlab or MuJoCo at all. Checking that against
the robot model found two things a user has to know: the action is in **actuator order**, not
the robot's joint order (the head block is permuted), and the policy **drives 10 of the 14
joints into their hard limits**, which is how it balances but is not directly deployable.
Both are recorded in `records/joint-limits.json`.

---

## Headline results

Screening = 64 first episodes, seed 70601, evaluated every 100 updates.
"credible" is stricter than "valid": see [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

| Reward | Stage | Upd | credible 0.5 s | credible 1.0 s | credible 2.0 s | longest | mean episode | ran to horizon |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v15 | `full` | 1000 | 90.6 % | 51.6 % | 0 % | 1.36 s | 4.34 s | 0 / 64 |
| v16 | `full` | 1000 | 93.8 % | 64.1 % | 0 % | 1.56 s | 4.73 s | 7 / 64 |
| v16-02 | `full` | 1000 | 82.8 % | 70.3 % | 0 % | 1.74 s | 5.46 s | 42 / 64 |
| v16b | `full` | 600 | 96.9 % | 84.4 % | 0 % | 1.70 s | 5.77 s | 52 / 64 |
| v16b + `glide` | `glide` | 1000 | 98.4 % | 89.1 % | 0 % | 1.62 s | 3.18 s | 56 / 64 |
| **+ entry band `(.50,.76)`** | **`glide`** | **1000** | **100 %** | **96.9 %** | **1.6 %** | **2.14 s** | **3.32 s** | **61 / 64** |

The last row is the current best and carries the **first 2.0 s credible one-foot glides
ever recorded** — 2.00 s at u600 and 2.14 s at u1000 — but by 1 of 64 episodes. The
formal two-seed 256-episode gate was then run on `model_10499`: credible 0.5 s
**99.6 %**, credible 1.0 s **95.3 %**, credible 2.0 s **0.0 % / 0.39 %** — **failed**,
on the 2.0 s duration and nothing else. Raw reports: `records/glide-glide-02-gate.json`.

The `glide` rows are not comparable to the `full` rows episode-for-episode: `full` spends
2.0 s accelerating and runs a 6.0 s horizon, `glide` starts already skating and runs a
3.4 s horizon. What the comparison does show is the balance failure rate falling and the
1.0 s rate rising, which is what a dedicated glide stage was supposed to buy.

### Why 2.0 s is still out of reach

The requirement is arithmetic. To travel for `T` seconds from entry speed `v₀` at
constant deceleration `a` and stay above the 0.10 m/s credible floor:

```
v₀ ≥ 0.10 + T · |a|
```

The speed budget that buys is `(v₀ − 0.10) / |a|` seconds, and comparing the *budget* with
the *achieved* glide is what shows where the robot now stands:

| Checkpoint | mean entry | mean \|deceleration\| | speed budget | best glide achieved | budget used |
|---|---:|---:|---:|---:|---:|
| v15 `model_4899` | 0.338 m/s | 0.161 m/s² | 1.48 s | 1.16 s | **78 %** |
| v16-02 `model_7899` | 0.528 m/s | 0.243 m/s² | 1.76 s | 1.72 s | **98 %** |

**The bottleneck changed kind.** Under v15 the robot fell before it had spent its speed —
it used only 78 % of the glide its speed allowed, and 42 of 64 rollouts fell. Under v16-02
it uses 98 % of the budget: the glides now run almost exactly as long as the speed allows
and then stop at the 0.10 m/s floor. Balance is no longer the binding limit; **the speed
budget is.**

To reach 2.0 s at v16-02's numbers, exactly one of these has to give:

| Lever | Needs | Currently |
|---|---:|---:|
| Reduce mean deceleration | ≤ **0.214 m/s²** | 0.243 m/s² (needs −12 %) |
| Raise mean entry speed | ≥ **0.587 m/s** | 0.528 mean, **0.633 max** |

The best individual episodes already enter at 0.633 m/s — above the requirement — and still
do not hold two seconds, because their deceleration is no better.

Where the deceleration comes from is measured, and it is not friction:

| Window after glide start | Mean deceleration |
|---|---:|
| 0 – 0.25 s | **−0.039 m/s²** |
| 0.25 – 0.50 s | −0.193 m/s² |
| 0.50 – 0.75 s | −0.154 m/s² |
| 0.75 – 1.00 s | −0.166 m/s² |
| > 1.00 s | −0.486 m/s² |

Four-wheel coasting decelerates at about **−0.058 m/s²**, so the first quarter-second of a
glide is already at rolling-resistance levels and everything after it is the robot losing
balance. The collapse is lateral — and notably the support skate never helps: the support
blade's yaw stays between **0.03 and 0.18 rad even on the longest glides**, i.e. the
steering degree of freedom that could correct the drift is never used.

An earlier version of this analysis claimed the 2.0 s goal was "physically impossible".
That was wrong, and the correction is recorded at the top of
[`docs/GLIDE-PHYSICS.md`](docs/GLIDE-PHYSICS.md).

### The two cheap routes are now closed by measurement

* **Entry speed is capped.** The valid predicate rejects root speed ≥ 0.80 m/s. At the
  measured mean deceleration, 80 % of episodes would need ~0.94 m/s. Tightening the band to
  the ceiling still only reaches ~1.9 s.
* **Raising the correction's authority destroys the robot.** The glide is composed as
  `onefoot + 0.25 · correction`, and the preserved expert alone falls over in 0.022 s while
  the bridge glides 1.50 s — which made "give the correction more room" look like an
  obvious win. Swept directly:

  | authority | mean credible glide | falls |
  |---:|---:|---:|
  | **0.25 (shipped)** | **1.46 s** | 1 / 64 |
  | 0.50 | 0.028 s | 72 / 64 |
  | 1.00 | 0.003 s | 64 / 64 |

  The correction is a *trained equilibrium* at 0.25, not an under-scaled fixer, so it
  cannot be re-scaled after training. `HOLD_AUTHORITY` is untouched.

So the remaining obstacle is the deceleration itself — ~0.34 m/s² against 0.058 m/s² for a
four-wheel coast — and its mechanism is **not yet identified**, after testing rolling
resistance, wheel skid, lateral scrub, blade yaw, and speed-controlled frame correlations.
Several of those tests came back negative; they are kept in the record because they are
what rules the easy explanations out. Details: [`REPORT.md`](REPORT.md) §8.

---

## Layout

```
README.md                       this file
REPORT.md                       the full technical report
docs/METHOD.md                  robot, observation/action contract, two-expert architecture
docs/REWARD-V16.md              reward specification (task / shaping / failure / handoff)
docs/GLIDE-PHYSICS.md           the measured glide physics and the corrected diagnosis
docs/VERIFICATION.md            what counts as evidence: predicates, gate protocol, metrics
docs/NEXT-STEPS.md              open questions and the options that were on the table
docs/REPRODUCING.md             how to lay this over upstream and run the whole suite
docs/ONNX-INFERENCE.md          running the shipped policies: 61D in, 14 targets out
evidence/                       machine-extracted JSON: milestones, gates, checkpoint hashes
trajectories/                   the learning curves of all 15 runs (CSV + screening JSON)
src/mjlab_microduck/            the 24 files this work added or changed
tests/                          the 24 new test modules covering this work
local/                          the data + tools those tests load by path
scripts/                        training, gate, tracing, pose-measurement, inference tools
records/                        signed transition records, incidents, raw measurements
logs/                           the one checkpoint the pinned code expects, at its path
checkpoints/                    the checkpoints referenced above (git-lfs)
```

`evidence/*.json` is generated by parsing the training logs and state files directly, so
every number in this README and in `REPORT.md` can be re-derived rather than trusted. The
same is true of `trajectories/`, which is exported from the raw TensorBoard logs by
`scripts/export-training-trajectories.py`.

## Reproducing

Two things are needed: the checkpoints, and upstream underneath.

**1. The checkpoints are git-lfs.** A fresh clone gets 132-byte pointer files until LFS is
active, and the failure mode is a confusing `torch.load` error:

```bash
git lfs install      # once per machine; without it a clone leaves pointer files
git lfs pull         # if you cloned before installing it
```

**2. This repository is an overlay, not a standalone package.** `src/` holds only the 24
files this work added or changed; the robot model and its 23 MB of meshes, and ~230 other
package files, come from upstream. So the suite does **not** run in a bare clone of this
repository — and the honest number for a bare clone is 42 failed / 1 collection error.

Overlay it on upstream and it is clean. Full detail in
[`docs/REPRODUCING.md`](docs/REPRODUCING.md):

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git microduck_rl
cd microduck_rl && git checkout 53b8971b61baf5b7f3c16d135dd7cac37623de4b

DELIVERABLE=/path/to/microduck-onefoot-skate
cp -r "$DELIVERABLE/src/."  src/
cp -r "$DELIVERABLE/tests/." tests/
cp -r "$DELIVERABLE/local" local
cp -r "$DELIVERABLE/logs"  logs

CUDA_VISIBLE_DEVICES='' uv run --locked --with pytest python -m pytest tests/ -q
# 434 passed, 1 skipped in 23.60s
```

The single skip is upstream's `test_aarch64_cuda_torch.py`, which is conditional on an
aarch64 host. Tests are CPU-only; training needs a CUDA GPU, and this work used a single
RTX 5070 Ti.

**Using a shipped policy needs neither mjlab nor MuJoCo**, only `numpy` and `onnxruntime`:

```bash
uv run --with onnxruntime --with numpy --with onnx \
    python scripts/policy_inference.py checkpoints/glide/policy-glide-02.onnx
```

See [`docs/ONNX-INFERENCE.md`](docs/ONNX-INFERENCE.md) for the exact 61-dim observation
layout and the action convention. The shipped ONNX is verified against the recorded actor
(2.15e-06 max difference) by `scripts/verify-shipped-onnx.py`; the result is in
`records/onnx-vs-actor.json`.

Continuing **training** instead needs the upstream working tree, because the runner
resolves its sources and state files relative to it:

```bash
# one assisted glide block from the v16b checkpoint
BRIDGE_CHAIN_TAG=glide-01 \
BRIDGE_ACTOR_TRANSITION=records/actor-transition-glide-stage.json \
  scripts/run-bridge-extension-03.sh \
  <path-to-full-v16b-state.json> glide-01 glide
```

## Provenance

Derived from **`pollen-robotics/microduck_rl`** at commit
`53b8971b61baf5b7f3c16d135dd7cac37623de4b`, Apache-2.0. `checkpoints/frozen/` additionally
contains the two frozen experts the bridge was built on:

* `main-expert-balance100-model_16699.pt` — the upstream waist-balance curriculum policy
  that supplies the preserved one-foot expert.
* `public-roller-expert.onnx` — the public roller-acceleration expert that drives the
  `phase = 0` half of the bridge.

Environment: Python 3.12, `mjlab 1.3.0`, `mujoco 3.10.0`, `mujoco-warp 3.8.1`,
`rsl-rl-lib 5.0.1`, `torch 2.9.1+cu129`.
