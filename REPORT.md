# Technical report — one-foot skating on Microduck

Interim report, 2026-09-11. Companion documents: [`docs/METHOD.md`](docs/METHOD.md),
[`docs/REWARD-V16.md`](docs/REWARD-V16.md), [`docs/GLIDE-PHYSICS.md`](docs/GLIDE-PHYSICS.md),
[`docs/VERIFICATION.md`](docs/VERIFICATION.md), [`docs/NEXT-STEPS.md`](docs/NEXT-STEPS.md).

---

## 1. The problem, stated precisely

Microduck is a ~25 cm biped with 14 Dynamixel XL330 servos and four passive roller wheels
(two under each foot). The target skill is a chain:

```
from rest ──▶ accelerate on 4 wheels ──▶ transfer weight to the LEFT foot
          ──▶ unload and lift the right skate ──▶ steer inward with the support skate
          ──▶ hold a long one-foot glide
```

and the ambition behind it is to enter that glide at the robot's **top speed** and stay on
one foot for as long as possible, rather than to reproduce the acceleration sequence every
episode.

Two downstream constraints shaped every decision:

* The one-foot expert already existed and worked. It was a preserved policy from the
  upstream waist-balance curriculum, and it was not allowed to be modified.
* A public roller-acceleration expert already existed as a frozen ONNX export, and the
  reward, observation and action contracts around it were pinned by tests.

So the work is not "train a skating policy from scratch". It is: **hand a preserved
one-foot skill a body that is already moving, and make the handoff learnable.**

## 2. Architecture

The policy is a retained-skill bridge: two frozen experts, one trainable correction.

```
                     ┌──────────────────────┐
   raw obs (61D) ────▶│ FrozenPublicRoller   │──▶ public action (14D)   phase ≈ 0
                     ├──────────────────────┤
                     │ FrozenOnefoot (MLP)  │──▶ one-foot action (14D) phase ≈ 1
                     ├──────────────────────┤
                     │ trainable correction │──▶ residual (14D)
                     └──────────────────────┘
                                  │
        blend = phase²(3−2·phase)
        gate  = 4·phase·(1−phase) + 0.25·blend
        out   = (1−blend)·public + blend·onefoot + gate·correction
```

* `phase` is an absolute observation slot that a command term drives: it ramps 0 → 1
  across a 2.0 s acceleration window and then a 1.0 s lift window, and sits at 1.0
  thereafter.
* The bridge's last layer is zero-initialised, so at t = 0 the composed policy is exactly
  the public roller expert.
* `HOLD_AUTHORITY = 0.25` is the share of the residual the correction keeps while the
  one-foot expert is fully in charge. It exists because the strict valid-glide predicate
  requires `phase ≥ 0.99`, where `4·phase·(1−phase) = 0.04`. Without that term the frozen
  expert would cap the whole task at its own ceiling `by construction`. It was added
  through an explicit, recorded, hashed actor-forward transition
  (`records/actor-transition-hold-authority.json`).
* Joints and bodies are always resolved **by name**, never by index. The action is 14D,
  HOME-relative. Unused command slots are zero-padded.

Full contract details, including the 61D actor / 78D critic split, are in
[`docs/METHOD.md`](docs/METHOD.md).

## 3. Finding: 35 % of every episode had no learnable authority

The `full` stage starts at HOME with exactly zero velocity and runs a 6.0 s horizon with a
2.0 s acceleration window. On the schedule above, `phase = 0` for the first 2.0 s, and the
gate is **identically zero** there. The trainable correction therefore contributes nothing
during acceleration.

| Quantity | Measured |
|---|---:|
| Mean episode length (`full`) | 5.73 s |
| Of which `phase = 0`, zero learnable authority | **2.00 s = 35 %** |
| Effective learnable time | 3.73 s |

The obvious objection is that more acceleration training would eventually raise the entry
speed. Two measurements say no:

* `scripts/probe-credit-bridge-acceleration.py` with an 8 s acceleration window and
  `acceleration_s = 1e6`, holding the right foot down, produces a **plateau at 0.37–0.38
  m/s between t ≈ 2.6 s and 3.4 s**, peaking at 0.372 and decaying after:
  `t=5.0 s → 0.317`, `t=8.0 s → 0.096`. The ceiling belongs to the frozen public expert,
  not to the schedule. (The frozen expert is fed a constant twist command of 0.6 via
  observation slot 48, and still saturates near 0.38.)
* Extending `lift_s` from 1.0 s to 6.0 s caps the speed at **0.49 m/s** — the robot stops
  pushing; it is not a matter of giving it more time.

And the current schedule already hands off at the best moment: with `acceleration_s = 2.0`
and `lift_s = 1.0` the handoff lands at t = 3.0 s, the **centre of the plateau** and 95 %
of the peak. Delaying is strictly worse (t = 5.0 s gives 0.317).

**Conclusion.** Replaying the acceleration prefix every episode is not just wasteful, it is
unimprovable. Training from speed is the correct response.

## 4. The assisted `glide` stage

The existing assisted stages (`near`, `unload`, `transfer-near`, `transfer`) start from
authored poses with injected speeds, but all of them cap the injected speed at 0.35 m/s and
**none of them starts inside the glide itself** (`phase ≥ 0.99`). There was a place to
train the *transfer* and nowhere to train the *glide*.

`scripts/measure-glide-pose.py` runs the `full` stage, finds the first frame at which
left-only support is established, and records the servo pose, every joint velocity and the
root attitude. It caught the state in **64 / 64** rollouts.

| Measured quantity | Median |
|---|---|
| Support skate (left) lift | 0.000 / 0.0078 m — **grounded** |
| Swing skate (right) lift | 0.0157 / 0.0738 m — **clearly airborne** |
| Entry speed | p50 **0.510**, p90 0.544, max 0.580 m/s |
| Root pitch / roll | +0.1086 / −0.0078 rad |
| Joint velocities | max **−7.25 rad/s** (right hip pitch); every joint moving |

`scripts/add-glide-pose.py` reduces this to a median, clamps it inside the joint limits by
1e-4 rad, grounds it on the **tyre mesh vertices** (not a nominal wheel radius), and writes
`POSES['glide']`.

The stage configuration is:

```python
'glide': ('glide', phase_start=1.0, episode_length_s=3.4, speed_range=(.45, .58))
SUSTAINED_STAGES = ('glide', 'full')   # both are judged on the 2.0 s credible metric
goal_s = 2.0
```

`phase_start = 1.0` with `acceleration_s = 0` means **the first control step is already
inside the sustained hold**, so every episode is 100 % learnable. `episode_length_s` is
shortened to 3.4 s because a glide cannot usefully last longer than the horizon.

**`glide` is assisted and can never certify a from-rest result.** The
`initial_assistance = (stage != 'full')` contract records this automatically. Only `full`
— HOME, root and joint and wheel velocities exactly zero — can produce from-rest evidence.

## 5. Two spawn-state defects

The first evaluation of the new stage scored **0.281** on credible-0.5 s where the *same
checkpoint* scored 0.969 in `full`. `scripts/compare-glide-spawn-state.py` diffed the
spawn state against real recorded handoffs joint by joint and found two defects.

### 5.1 All servo joint velocities were zero

The shared spawn helper starts every servo at rest. Real handoffs have a median |joint
velocity| of **1.86 rad/s** and a maximum of **7.64 rad/s**. The spawn was therefore
"a gliding pose with completely stopped joints" — a state that does not exist. The actor
observes joint velocities, so this is not cosmetic.

### 5.2 The swing wheels were spinning in mid-air at the support-wheel rate

The helper overwrote all four wheel velocities with `v/r` = +34.5 rad/s. The real handoff:

| Joint | Spawn (forced `v/r`) | Real |
|---|---:|---:|
| `passive_LF_wheel` (support) | +34.5 | **+36.7** |
| `passive_LR_wheel` (support) | +34.5 | **+36.5** |
| `passive_RF_wheel` (swing) | +34.5 | **−2.1** |
| `passive_RR_wheel` (swing) | +34.5 | **+11.1** |

### 5.3 Effect of the fix

Same checkpoint, 64 episodes:

| Metric | Before | After |
|---|---:|---:|
| credible 0.5 s | 0.281 | **0.672** |
| credible 1.0 s | 0.109 | **0.391** |
| longest credible | 1.22 s | **1.50 s** |
| ran to horizon | 5 / 64 | **24 / 64** |
| mean episode | 1.01 s | **2.00 s** |

### 5.4 Why the fix had to live outside `mdp.py`

`tests/test_old_mdp_and_registry_bytes_are_preserved` requires the **first 351 782 bytes of
`tasks/mdp.py`** — everything before the v15 reward region — to stay byte-identical, and
the shared spawn helper `reset_curriculum_onefoot` lives inside that pinned prefix. Adding
an optional argument to it would break that invariant *permanently*: once the byte offset
shifts it cannot be restored.

So the faithful spawn was added as `reset_credit_bridge_spawn` in
`tasks/microduck_onefoot_credit_bridge_env_cfg.py`: an exact copy of the shared helper plus
`root_pitch` and `joint_vel`, installed **only** for poses that actually carry those
measured fields. Every other assisted stage still uses the original helper, byte for byte.

## 6. Reward v16

Reward v15 was `task + shaping + failure`. v16 keeps that structure and adds exactly one
term; it does not add a negative continuing reward (which would be escapable by dying
early). See [`docs/REWARD-V16.md`](docs/REWARD-V16.md) for the full specification.

* **B — handoff speed bonus.** One payment per episode, made only when a valid *and*
  credible glide survives 0.25 s, of
  `1.2 · clamp((entry_speed − 0.38) / 0.30, 0, 1)`, where `entry_speed` is the COM forward
  speed at the first frame of single support. `0.38` is the frozen public expert's measured
  ceiling, so the bonus is **exactly zero** below it: existing good behaviour is untouched
  and only new capability is rewarded. Full credit at 0.68 m/s. The gain is deliberately
  below the 2.0 failure cost so that a reckless throw-and-fall cannot pay.
  The span was later widened from 0.15 to 0.30 and the gain from 1.0 to 1.2 because the
  gradient was saturating at 0.53 m/s and the policy was pinning itself there.
* **E — two-axis balance debt.** A cost accumulated only during left-only support:
  `((|lateral| − 0.008)/0.025)² + ((|fore/aft| − 0.005)/0.020)²`, feeding the existing
  `quality = 1/(1 + smooth_debt)` discount. Since it is a discount rather than a payment,
  collapsing early does not escape it. The deadbands are the measured p50 values, so normal
  gliding accrues essentially nothing.
* **E′ — fore/aft in the potential.** `credit_bridge_potential` gained a second capture
  factor `exp(−(fore_aft_error/0.02)²)` multiplying the existing lateral one. All terminal
  potentials are 0, so the telescoping shaping sum stays exactly 0
  (`discounted_shaping_max_abs < 2e-5` is asserted in every evaluation).

The fore/aft axis was included because credible glides show the robot drifting out of
balance **backward as well as sideways**: |capture error| moves from 0.0058 → 0.0251 m
fore/aft and 0.0072 → 0.0203 m laterally over the course of a glide.

**Measured effect of B.** Entry speed rose from 0.393 (v15) to **0.465 m/s** (v16), past the
frozen expert's 0.38 ceiling, with a maximum of 0.534 — into the region where the bonus was
intended to bite. **Measured effect of E.** Lateral capture error fell from 0.0203 to
**0.0065 m**. Neither showed up as a 2.0 s glide, because the failure mode changed to the
lateral collapse described in §8.

## 7. Results

### 7.1 Screening (64 first episodes, seed 70601, every 100 updates)

| Reward | Stage | Upd | credible 0.5 s | 1.0 s | 2.0 s | longest | mean | horizon | falls (per-term) | stalled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v15 | `full` | 1000 | 90.6 % | 51.6 % | 0 % | 1.36 s | 4.34 s | 0 | 38 + 26 | 27 |
| v16 | `full` | 1000 | 93.8 % | 64.1 % | 0 % | 1.56 s | 4.73 s | 7 | 51 + 14 | 8 |
| v16-02 | `full` | 1000 | 82.8 % | 70.3 % | 0 % | 1.74 s | 5.46 s | 42 | 13 + 11 | 8 |
| v16b | `full` | 600 | 96.9 % | 84.4 % | 0 % | 1.70 s | 5.77 s | 52 | 3 + 6 | 7 |
| glide-01 | `glide` | 1000 | 98.4 % | 89.1 % | 0 % | 1.62 s | 3.18 s | 56 | 6 + 2 | 1 |
| **glide-02** | **`glide`** | **1000** | **100 %** | **96.9 %** | **1.6 %** | **2.14 s** | **3.32 s** | **61** | **3 + 1** | **0** |

"falls" is the `body_ground` and `fell_over` termination terms; termination terms can
co-fire on the same episode, so these are per-term counts, not a partition of the 64.

Full per-block series for all 12 training runs are in
[`evidence/milestones.json`](evidence/milestones.json).

The `glide-02` series over its 1000 updates (wider entry band + wheel follow, see §8.1):

| Upd | credible 0.5 s | 1.0 s | 2.0 s | longest | mean credible | horizon | falls (`body_ground`+`fell_over`) | stopped |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 100 % | 95.3 % | 0 % | 1.94 s | 1.436 s | 60 | 2 + 2 | 0 |
| 200 | 96.9 % | 89.1 % | 0 % | 1.80 s | 1.335 s | 60 | 3 + 0 | 0 |
| 300 | 96.9 % | 92.2 % | 0 % | 1.94 s | 1.421 s | 61 | 2 + 0 | 0 |
| 400 | 96.9 % | 95.3 % | 0 % | 1.98 s | 1.428 s | 61 | 1 + 1 | 0 |
| 500 | 98.4 % | 93.8 % | 0 % | 1.96 s | 1.397 s | 59 | 1 + 1 | 0 |
| 600 | 100 % | 90.6 % | **1.6 %** | **2.00 s** | 1.374 s | 62 | 1 + 1 | 0 |
| 700 | 98.4 % | 92.2 % | 0 % | 1.86 s | 1.388 s | 59 | 1 + 2 | 0 |
| 800 | 98.4 % | 89.1 % | 0 % | 1.84 s | 1.363 s | 58 | 0 + 2 | 0 |
| 900 | 98.4 % | 95.3 % | 0 % | 1.92 s | 1.433 s | 58 | 4 + 2 | 0 |
| 1000 | **100 %** | **96.9 %** | **1.6 %** | **2.14 s** | **1.433 s** | **61** | 3 + 1 | 0 |

The u1000 checkpoint is `checkpoints/glide/model_10499.pt`.

The 2.0 s barrier is crossed at u600 and again at u1000, but at 1 of 64 episodes. Against
the v16b starting point (`model_8499`: credible 1.0 s 84.4 %, longest 1.70 s) the honest
reading is that the entry-speed change bought a real but modest gain, and that the last
0.14 s to a *reliable* 2.0 s is not a speed problem.

### 7.2 Formal gate

A dedicated runner (`scripts/run-bridge-stage-gate.py`) was added because the trainer only
attempts the gate once 64-episode screening reaches 0.8 on the gate key, which never
happened for the stages that carry the 2.0 s goal. `glide` has now been gated properly.

**`glide` — `model_10499`, 2 × 256 first episodes, seeds 71301 / 71302 — FAILED**

| Gate condition | Required | seed 71301 | seed 71302 | |
|---|---|---:|---:|---|
| episodes | ≥ 256 | 256 | 256 | ok |
| distinct seeds | 2 | ✓ | ✓ | ok |
| checkpoint SHA256 | match | ✓ | ✓ | ok |
| `initial_assistance == (stage != 'full')` | True | True | True | ok |
| NaNs | 0 | 0 | 0 | ok |
| `discounted_shaping_max_abs` | < 2e-5 | 7.5e-7 | 8.1e-7 | ok |
| `success_200_rate` | ≥ 0.80 | 0.0 % | 0.4 % | **FAIL** |
| `credible_200_rate` | ≥ 0.80 | **0.0 %** | **0.39 %** | **FAIL** |

Everything except the 2.0 s duration passes. On the same 512 first episodes:
credible 0.5 s **99.6 %**, credible 1.0 s **95.3 %**, longest credible 1.96 / 2.00 s.

So the honest statement is not "never attempted" any more: the stage is measured, the
1.0 s skill is essentially solved at 95 %, and the 2.0 s bar is missed by a factor of 200
in rate while being missed by 0.04 s in the single longest episode. That gap — a 95 %
skill at 1.0 s against a 0.2 % skill at 2.0 s — is the shape of the remaining problem.

Earlier gates:

| Stage | Seeds gated | 256-episode credible 0.5 s | Passed |
|---|---|---:|---|
| `near` | 71301, 71302 | 86.3 %, 87.1 % | no |
| `unload` | 71301, 71302 | 59.4 %, 59.4 % | no |
| `unload` (ext-04) | 71401, 71402 | 79.7–83.9 % | no |
| `transfer-near` | 71501, 71502 | 78.1–94.9 % | no |
| `transfer` | 71601, 71602 | 76.6–85.5 % | no |
| **`full`** | — | **never gated** | — |

The assisted transfer stages have a 0.5 s goal (`SUSTAINED_STAGES` excludes them), which is
why they were gated on the 0.5 s metric. `full` has still never been gated, because its
screening never approached the 0.8 trigger.

Raw gate reports, including the two-seed aggregates: [`evidence/gates.json`](evidence/gates.json)
and `records/glide-glide-02-gate.json`.

## 8. The physics of the remaining gap

### 8.1 The entry-speed lever: implemented, measured, and now closed

The `glide` stage injected `(.45,.58)` m/s, a band taken from the handoff the bridge
happened to produce. Sweeping the injected speed over a frozen policy
(`scripts/probe-glide-entry-speed.py`) showed the band sat **entirely below** the speed at
which a 2.0 s glide is reachable, and turned up two separate findings.

**Finding 1 — the recorded wheel speeds were contradicting the injected speed.**
The pose was captured at 0.51 m/s, so its SUPPORT wheels read 36.7 / 36.5 rad/s (v/r ≈ 34).
Injecting any other speed while leaving those wheels there puts a contradiction in the
actor's own observation, and the policy then refused to settle fast:

| Injected | entry with recorded wheels | entry with wheels following |
|---:|---:|---:|
| 0.58 m/s | 0.526 | **0.558** |
| 0.72 m/s | 0.557 | **0.685** |
| 0.78 m/s | 0.580 | **0.742** |

Raising the injection by 0.20 m/s moved the *achieved* entry speed by 0.054 m/s with the
recorded wheels and by 0.184 m/s with the support wheels tracking the injection. **So
widening the band alone would have bought almost nothing** — the wheel fix is what makes the
speed lever work. The two UNLOADED swing wheels keep their recorded near-still values,
because a wheel in mid-air does not roll with the ground.

**Finding 2 — where 2.0 s first becomes possible, and the first 2.0 s glides ever seen.**
With the wheels following:

| Injected | mean entry | longest credible | ≥1.5 s | ≥2.0 s |
|---:|---:|---:|---:|---:|
| 0.45 | 0.433 | 1.14 s | 0 % | 0 % |
| 0.52 | 0.498 | 1.48 s | 0 % | 0 % |
| 0.58 | 0.558 | 1.80 s | 37.5 % | 0 % |
| 0.65 | 0.621 | 1.78 s | 54.7 % | 0 % |
| **0.72** | **0.685** | **2.14 s** | 51.6 % | **1.6 %** |
| **0.78** | **0.742** | **2.10 s** | 39.1 % | **3.1 %** |

At *every* injected speed 51–59 of 64 episodes ran to the 3.4 s horizon with only a handful
of falls, so the glides end **on the 0.10 m/s speed floor, not on a fall**. That is the
cleanest confirmation that the speed budget — not balance — is the binding constraint at
this point, and it is why the earlier "the robot falls sideways" reading had to be revised.

The change (recorded as `task_contract_transition`, `records/actor-transition-glide-speed-band.json`)
widened the band to `(.50,.76)` and made the support wheels follow. Results: §7.1.

**Why the lever is nevertheless closed.** The valid predicate rejects root speed ≥ 0.80 m/s,
so entry speed cannot exceed ~0.79. With the measured mean deceleration of ~0.34 m/s², 80 %
of episodes would need an entry of about **0.94 m/s** to hold 2.0 s. Tightening the band to
the predicate ceiling also does not do it: at entry 0.75 and |a| = 0.34 the glide is 1.9 s.
Entry speed can turn "never" into "occasionally" (it did), and it cannot turn "occasionally"
into "80 %".

### 8.2 Two candidate explanations for the drag, both measured and both rejected

The ~0.34 m/s² glide deceleration — 6× the 0.058 m/s² of a four-wheel coast — had no
identified mechanism. Two plausible ones were tested directly.

**The frozen expert is not the problem, but it is also not doing the work.** Driving the
frozen one-foot expert alone from the same spawn, pinned at 0.72 m/s:

| Policy | mean entry | mean credible | longest | falls |
|---|---:|---:|---:|---:|
| frozen one-foot expert alone | 0.397 m/s | **0.022 s** | 0.18 s | **59 / 64** |
| the trained bridge | 0.676 m/s | **1.50 s** | 2.08 s | 2 / 64 |

The preserved expert cannot glide at all from a fast spawn — so the entire glide is produced
by the trainable correction, while the expert still supplies 100 % of the base action and the
correction is scaled by 0.25. That made "the correction is fighting a base it cannot beat"
the leading explanation.

**Raising the correction's authority destroys the robot.** Recomposing the bridge with the
hold-authority coefficient swept, same spawn and same seed:

| authority A | mean credible | falls |
|---:|---:|---:|
| **0.25 (shipped)** | **1.46 s** | 1 / 64 |
| 0.50 | 0.028 s | 72 |
| 0.75 | 0.005 s | 67 |
| 1.00 | 0.003 s | 64 |

**Rejected.** The correction is not an under-scaled fixer; it is a *trained equilibrium* at
A = 0.25, and its output magnitude is calibrated to that gain, so scaling it up commands
actions roughly 4× too large. Changing `HOLD_AUTHORITY` is therefore not a free improvement
— it would require retraining the correction from scratch at the new gain. It stays at 0.25
and the actor is untouched.

Also tested and rejected as the drag source: rolling resistance (four-wheel coast is 6×
lower), wheel skid (the joint model has `damping=0` and `frictionloss=0`, and the measured
support-wheel spin at the real handoff is 36.7 rad/s at 0.51 m/s, i.e. rolling), lateral
scrub (mean |lateral velocity| is 0.026 m/s, 8 % of forward speed), and blade yaw (median
11°, but *negatively* correlated with deceleration). Speed-controlled frame-level
correlations removed the collapse-driven confound that had produced several misleading
per-glide correlations, and after controlling for speed no single correlate dominates.

**This is the open problem.** The gap is now a 95 % skill at 1.0 s against a 0.2 % skill at
2.0 s, which is a statement about deceleration, not about speed, balance, or authority.



The full derivation is in [`docs/GLIDE-PHYSICS.md`](docs/GLIDE-PHYSICS.md); this is the
summary.

The requirement for a `T`-second credible glide at constant deceleration `a` is
`v₀ ≥ 0.10 + T·|a|`.

The key measurement is that **the deceleration of a glide is not constant and is not caused
by rolling resistance**:

| Window after glide start | Mean deceleration |
|---|---:|
| 0 – 0.25 s | **−0.039 m/s²** |
| 0.25 – 0.50 s | −0.193 m/s² |
| 0.50 – 0.75 s | −0.154 m/s² |
| 0.75 – 1.00 s | −0.166 m/s² |
| > 1.00 s | −0.486 m/s² |

Four-wheel coasting decelerates at about **−0.058 m/s²**, so the first quarter-second is at
rolling-resistance levels and all the later deceleration is **the robot losing balance**.

The glide a given speed buys is `(v₀ − 0.10) / |a|`, and comparing the budget with what was
actually achieved shows that **the bottleneck changed kind between v15 and v16-02**:

| Checkpoint | mean entry | mean \|deceleration\| | speed budget | best glide achieved | budget used |
|---|---:|---:|---:|---:|---:|
| v15 `model_4899` | 0.338 m/s | 0.161 m/s² | 1.48 s | 1.16 s | **78 %** |
| v16-02 `model_7899` | 0.528 m/s | 0.243 m/s² | 1.76 s | 1.72 s | **98 %** |

Under v15 the robot fell well before spending its speed — 42 of 64 rollouts fell, and the
longest glide used only 78 % of the 1.48 s its speed allowed. Under v16-02 the glides use
**98 %** of the budget and end at the 0.10 m/s floor. Balance stopped being the binding
limit. Extrapolating the observed durations from entry speed and deceleration reproduces
them closely (1.76 s predicted vs 1.80 s reported by the tracer), which is what makes this
reading safe.

So exactly one of two things has to move:

| Lever | Needs | Currently |
|---|---:|---:|
| Mean deceleration | ≤ **0.214 m/s²** | 0.243 m/s² (a 12 % reduction) |
| Mean entry speed | ≥ **0.587 m/s** | 0.528 mean, **0.633** max |

The best individual episodes already enter above the requirement (0.633 m/s) and still do
not hold two seconds, because their deceleration is no better than average. Reward term B
did its job on speed; what is left is deceleration, which is a balance problem:

```
t=3.0 s  phase 1.0, entry 0.363 m/s, right foot load 0.38 N (lift complete)
t=3.3 s  com 0.327, clearance 0.085, upright 0.984
t=3.6 s  com 0.272, clearance 0.108, upright 0.958, lateral 0.081
t=3.7 s  clearance 0.114, upright 0.900, lateral 0.152   ← collapse begins
t=4.0 s  clearance 0.094, upright 0.856, lateral 0.244
t=4.2 s  17 / 64 survive, clearance 0.069, upright 0.810, lateral 0.332
```

The support blade's yaw stays between 0.03 and 0.18 rad even on the longest glides. The
support skate can steer; **the policy never steers it to recover balance.** That unused
degree of freedom is what reward term E tries to reach, and it is the most plausible reason
2.0 s is still 0 %.

### A correction that matters

The first version of this diagnosis concluded that the 2.0 s gate was *physically
impossible* and that the acceptance criterion should be lowered. That conclusion was wrong.
It was derived from the mean deceleration of *whole* glides, which includes the collapse.
The first 0.25 s decelerates at −0.039 m/s² — i.e. essentially free rolling — and a 0.2 s
window at glide start has a median deceleration of **+0.016 m/s²** (slightly accelerating),
with 60 of 63 episodes within |a| ≤ 0.1315. Retracting that diagnosis is what motivated
reward v16 rather than lowering the bar.

## 9. Process failure, and what the guard caught

Mid-way through this work the operator edited `scripts/train-credit-bridge.py` **while a
training run was in flight**. The trainer hashes its own sources and verifies them every
chunk; the integrity guard fired correctly and stopped the run at update 700. All seven
recorded reports matched their SHA256, there were zero NaNs, and no completion marker had
been written, so the run was recovered from the u600 checkpoint and continued as
`records/incident-v16b-01.json` records.

The guard worked. The lesson is a procedural one and is written into the affected
documents: **never edit a pinned source while a run is active.** The pinned set is
`retained_skill_bridge.py`, `public_residual_policy.py`, `checkpoint_safety.py`,
`bridge_recovery.py`, `tasks/mdp.py`, `tasks/microduck_onefoot_credit_bridge_env_cfg.py`,
`tasks/microduck_onefoot_curriculum_poses.py`, plus `scripts/train-credit-bridge.py`.

Two audit gaps were closed at the same time: the env cfg and the pose table were added to
`COMPATIBLE_SOURCES` so that continuation records cannot silently change them, and a third
transition kind (`task_contract_transition`) was introduced.

### Continuation policy

A plain continuation refuses any source change. When a change is legitimate it must be
declared as one of three kinds, each with a named owner:

| Kind | Owner (must change) | Forbidden |
|---|---|---|
| `actor_forward_transition` | `retained_skill_bridge.py` | may not touch any other pinned source |
| `reward_forward_transition` | `tasks/mdp.py` | may not touch the actor |
| `task_contract_transition` | the env cfg | may not touch the actor |

The actor is treated most strictly, because parameter identity and the frozen-expert
digest are what make continuation meaningful. Non-actor kinds must move their owner and
must record before/after hashes for any other tracked file they touch. Tests enforce the
cross-prohibitions, and the four records for this work are in `records/`.

## 10. What would have to be true for 2.0 s

**One number.** With the measured mean entry speed `v₀` and mean deceleration `|a|`, a
credible glide lasts `(v₀ − 0.10) / |a|` seconds. Currently:

```
v₀ = 0.592 m/s     |a| = 0.343 m/s²     glide = 1.43 s
needed for 2.0 s at this entry  : |a| <= 0.246 m/s²   (a 28 % reduction)
needed for 2.0 s at this decel. : v₀  >= 0.786 m/s    (already reached by the best episodes)
```

The best episodes already enter at 0.79 m/s and reach 2.14 s, so the entry-speed side is
essentially maxed against the predicate ceiling of 0.80. What is left is the 28 %.

**The routes that are closed, with the measurement that closed them:**

| Route | Status |
|---|---|
| Widen the entry band | done (`glide-02`), and capped: the predicate rejects ≥ 0.80 m/s |
| Fix the wheel-speed contradiction | done; it is what made the band change work at all |
| Raise the correction's authority | **rejected** — A = 0.25 is a trained equilibrium; A = 0.5 gives 0.028 s and 72 falls |
| Blame the frozen expert | **rejected** — it falls out in 0.022 s, so it is not resisting; it is simply not contributing |
| Rolling resistance / skid / lateral scrub / blade yaw as the drag | **all measured and rejected** (§8.2) |

**What is left is a genuinely open problem:** reduce the one-foot glide deceleration from
~0.34 to ~0.25 m/s², i.e. understand and remove a drag that is 6× the four-wheel rolling
resistance and whose mechanism is not yet identified. Candidate directions, none of them
measured yet:

1. **Make the support skate steer.** The blade yaw sits at a steady ~11° and never moves,
   so the steering degree of freedom is unused. Blade yaw correlated *negatively* with
   deceleration, so this is not a simple scrub story — but a steady 11° yaw on two in-line
   wheels is not obviously a free-rolling configuration either, and no experiment has yet
   asked the policy to hold the blade aligned with its travel.
2. **Look for internal-motion loss.** The swing leg's joints move at up to 7.6 rad/s. The
   measured deceleration is spread across all speeds (0.21–0.32 m/s²) with no dominant
   frame-level correlate, which is consistent with a distributed loss rather than a single
   contact effect.
3. **Measure the drag with the correction removed and the expert held at the glide pose.**
   The frozen expert falls over immediately because its *balance* policy is wrong for this
   state, which masks whether its *posture* also drags. Holding the pose open-loop would
   separate the two.
4. **Retrain the correction at a higher authority.** Only as a deliberate architecture
   change with its own actor-forward transition, not as a post-hoc scale factor — §8.2 shows
   post-hoc scaling cannot work.

## 11. Verification status of everything claimed here

* The full test suite is green: **434 passed, 1 skipped**, CPU-only, with
  `CUDA_VISIBLE_DEVICES=''`.
* Every checkpoint shipped in `checkpoints/` is recorded with its SHA256 in
  [`evidence/checkpoints.json`](evidence/checkpoints.json), alongside the pinned-source
  hashes of the run that produced it.
* The valid-glide predicate and the two-seed 80 % gate were **not loosened** at any point,
  including when they failed.
* `discounted_shaping_max_abs < 2e-5` is asserted on every evaluation, so the shaping
  potential is verified to telescope to zero and cannot be farmed by standing still.
* Reward weights and constants were changed only in the two declared, recorded
  transitions (`actor-transition-reward-v16.json`, `actor-transition-reward-v16b.json`).
  The stage change in §8.1 is a third record, `actor-transition-glide-stage.json`'s
  successor `actor-transition-glide-speed-band.json`, of kind `task_contract_transition`,
  with `reward_terms_before == after` and the frozen-expert digest unchanged.
* The `glide` stage has now been put through the **formal two-seed 256-episode gate** and
  is recorded as **failed** (§7.2), rather than only screened. Everything except the 2.0 s
  duration passes.
* The three negative results in §8.2 are kept in the record as negative results. A probe
  that justifies *not* making a change is worth as much as one that justifies making it,
  and `HOLD_AUTHORITY` in particular would otherwise look like an obvious free win.

## 12. The artifact: what was done to make this usable

A report and a checkpoint are not a reusable result. Three things were added, and each was
verified rather than asserted.

### 12.1 The learning curves, without the logs

The training logs are 377 TensorBoard event files and 155 MB, spread over 85 run
directories, and they are not part of upstream's tracked tree. `scripts/export-training-trajectories.py`
reads only the runs that produced shipped checkpoints or recorded negative results and writes
them to `trajectories/`: 15 runs, 6.2 MB of CSV, one row per update, plus `index.json` with
the SHA256 of every source event file and `screening.json` with the evaluation series.

Two kinds of curve had to be kept separate, and conflating them is the easy mistake:

| | training aggregate (CSV) | screening (`screening.json`) |
|---|---|---|
| population | 4096 envs, every update | 64 first episodes, seed 70601, every 100 updates |
| actions | stochastic (PPO noise) | deterministic mean |
| `glide-glide-02` at u1000, credible ≥1.0 s | 0.774 | **0.969** |

The CSV is the signal that was optimized; the screening series is what the gate and §7 judge.
The screening export also retains the `updates: 0` pre-training baseline that
`evidence/milestones.json` omits, which is what makes each run's starting point visible.
Cross-checked against `evidence/milestones.json`: **1288 field comparisons over 13 runs, 0
mismatches**, the only difference being that extra baseline row. Re-running the export
reproduces every file byte-identically.

### 12.2 A surface that runs

Thirteen of the 24 test modules load a tool by file path from `local/`, following upstream's
own convention (`tests/test_onefoot_knee.py` does the same). Upstream's `local/` is **not
tracked by git** — 281 files, `git ls-files local` empty — so those tools had no distribution
at all. They are now vendored here in full at the exact paths the tests expect, together with
the three data artifacts the tests read: the measured glide pose, the pre-v15 digest snapshot
that `test_old_mdp_and_registry_bytes_are_preserved` hashes against, and the pinned public
roller ONNX at its content-addressed path (byte-identical to
`checkpoints/frozen/public-roller-expert.onnx`).

Measured before and after, on the same machine:

| Layout | Result |
|---|---|
| bare clone of this repository | **42 failed, 180 passed, 3 skipped, 1 collection error** |
| upstream `53b8971` + this overlay | **434 passed, 1 skipped, 0 failed** |

The one skip is upstream's `test_aarch64_cuda_torch.py`, conditional on an aarch64 host. The
bare-clone failures were 39 `FileNotFoundError` on `local/*.py` and 3
`ValueError: ParseXML` on the robot model, and the overlay cures both without any test being
weakened.

`src/` remains an overlay rather than a vendored copy of upstream's 24 MB of meshes and ~230
package files. That is the deliberate trade: no duplication, at the cost of a documented setup
step. `docs/REPRODUCING.md` gives the exact recipe.

### 12.3 The ONNX files, checked rather than assumed

The existing export test,
`test_standard_exports_include_both_skills_and_bridge`, proves the *exporter* is faithful
using a synthetic actor. It does not prove that a given shipped `.onnx` matches a given
recorded `.pt`. `scripts/verify-shipped-onnx.py` closes that gap by running both on live
environment observations:

| Shipped policy | Recorded checkpoint | max abs difference |
|---|---|---:|
| `policy-glide-02.onnx` | `model_10499.pt` | **2.15e-06** |
| `policy.onnx` | `model_9499.pt` | **1.07e-06** |

Both pass `atol=3e-6`. `records/onnx-vs-actor.json` holds the result.

Writing this up caught a real documentation bug. The observation slot table had the command
block wrong twice: `command` is **3 wide at 48:51**, not 14 wide at 48:62, and the two experts
read slot 48 differently — the frozen public roller expert ignores it and is always fed
`push_command = 0.6`, while the frozen one-foot expert sees it as given, so at `phase = 1` it
matters and must be the environment's `0.3`, not `0.6`. `joint_pos` also had to be
HOME-relative, while the recorded `median_pose` is an absolute qpos. All three are now correct
in `docs/ONNX-INFERENCE.md` and `scripts/policy_inference.py`, and the 61-dim total is
confirmed against a live `observation_manager.group_obs_dim['actor'] == (61,)`.

Checking the action against the model turned up a fourth thing, and it is the one that
matters most for anyone else driving this policy. The 14 values are in **actuator order**,
which is *not* the robot's joint order — the head block is permuted (`neck_pitch,
head_pitch, head_yaw, head_roll` versus `head_yaw, head_roll, neck_pitch, head_pitch`), and
`robot.data.default_joint_pos` is in the robot's order and looks plausible either way. And
the policy **drives joints into their hard limits**: the ranges are tight (`left_hip_roll`
±0.384 rad, `head_roll` ±0.436), and over an 80-step rollout of `policy-glide-02.onnx`,
**10 of 14 joints reach a limit**, with `left_hip_roll` and `neck_pitch` sitting against
theirs through the glide. The measured entry pose agrees — its `left_hip_roll` is −0.3867
against a limit of −0.3840.

That is not a defect in the policy; using the stops as reference points is how it balances
on one skate. But it does mean this is a simulation result in a strong sense, and a
deployment needs its own answer for holding servos against mechanical stops. The ranges and
the measurement protocol are recorded in `records/joint-limits.json` rather than left in
prose, and `docs/ONNX-INFERENCE.md` states the caveat next to the action convention.

### 12.4 Licence and content check

Before treating this repository as publishable: Apache-2.0, the same licence as upstream;
no credentials, tokens, private keys, e-mail addresses or external IPs in the working tree or
anywhere in history; checkpoints consistent under a fresh clone with LFS pulled. The
absolute filesystem paths in `evidence/` and `records/` are provenance — they name where the
measured file was at the time — so they were left as recorded rather than rewritten.
