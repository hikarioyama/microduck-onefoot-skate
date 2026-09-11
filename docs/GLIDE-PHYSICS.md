# Glide physics — why a one-foot glide stops, and a corrected diagnosis

Two checkpoints were traced, and the comparison between them is where the conclusion
changed. Reproduce with `scripts/trace-credit-bridge-full.py` (glide physics) and
`scripts/probe-credit-bridge-acceleration.py` (the frozen expert's speed ceiling). Raw
output is shipped: `records/full-trace-u1000.json` (v15, `model_4899`),
`records/full-trace-v16-02-u1000.json` (v16-02, `model_7899`) and
`records/accel-probe-long.json`. The derived figures are in
`evidence/coast-down.json`, computed from those files by `scripts/trace-credit-bridge-full.py`
output alone. Both tracing tools are read-only with respect to active training runs.

> **Correction, same day.** The first version of this diagnosis concluded that the 2.0 s
> gate was *physically impossible*. That was wrong: it extrapolated from the **mean
> deceleration of whole glides**, which includes the fall. The deceleration in the first
> 0.25 s of a glide is essentially zero, and the deceleration later on comes from losing
> balance. The correct analysis follows §2 onward. Recording the error rather than quietly
> replacing it matters, because "the bar is unreachable" and "we cannot hold balance for
> two seconds yet" call for completely different responses.

## 1. Measurements

### 1.1 Speed at handoff

| Quantity | Measured |
|---|---|
| COM forward speed at the instant one-foot support is established | mean **0.336 m/s**, p50 0.338, max 0.397 |
| Valid-predicate speed ceiling | 0.80 m/s (never reached) |
| Credible-predicate speed floor | 0.10 m/s |

### 1.2 The frozen public expert saturates at 0.38 m/s

With an 8 s acceleration window, `acceleration_s = 1e6`, and `lift_blend` pinned to 0 — so
the output is *strictly* the public expert, since `bridge_mean` at `blend = 0, gate = 0` is
the public endpoint — and holding the right foot down:

```
t=2.6 s  com 0.373
t=3.0 s  com 0.374   ← peak (plateau)
t=3.4 s  com 0.375
t=5.0 s  com 0.317
t=8.0 s  com 0.096
```

There is a **flat plateau at 0.37–0.38 m/s between t ≈ 2.6 s and 3.4 s**, after which the
robot slows. The limit is the frozen public expert itself, not insufficient acceleration
time. (`FrozenPublicRoller` overwrites observation slot 48 — the forward twist — with
**0.6** and zeroes slots 49+, so the environment's own `target_speed = 0.3` is invisible to
it. Even commanded 0.6, it reaches 0.38.)

### 1.3 The current schedule already releases at top speed

The `full` stage uses `acceleration_s = 2.0`, `lift_s = 1.0`, so the handoff lands at
t = 3.0 s — the **centre of the plateau**, at 95 % of peak (measured handoff 0.363 vs peak
0.382). Delaying is worse: at t = 5.0 s it is 0.317.

> **"Unload at top speed" is already achieved. There is no headroom there.**

### 1.4 The deceleration during a glide is not constant — and not rolling resistance

Binning the credible glide phases of 63 episodes by elapsed time since glide start:

| Elapsed since glide start | Mean deceleration | Mean \|lateral COM offset\| |
|---|---:|---:|
| 0 – 0.25 s | **−0.039 m/s²** | 0.0098 m |
| 0.25 – 0.50 s | −0.193 m/s² | 0.0088 m |
| 0.50 – 0.75 s | −0.154 m/s² | 0.0120 m |
| 0.75 – 1.00 s | −0.166 m/s² | 0.0137 m |
| > 1.00 s | −0.486 m/s² | 0.0149 m |

A 0.2 s window at glide start has **median +0.016 m/s²** — marginally *accelerating* —
with p25 −0.017 and p75 +0.064; **60 of 63 episodes (95.2 %) stay within |a| ≤ 0.1315**.

For reference: a four-wheel coast by the public expert decelerates at **−0.058 m/s²**
(0.382 → 0.096 over 5 s). So passive rolling resistance is 0.04–0.06 m/s², the first
quarter-second of a glide is already at that level, and the **−0.17 m/s² of a whole glide
is balance loss, not friction**.

### 1.5 The collapse comes from the side

64-env mean time evolution:

```
t=3.0 s  phase 1.0, entry speed 0.363 m/s, right foot load 0.38 N (lift complete)
t=3.3 s  com 0.327  ← gliding normally (clearance 0.085 m, upright 0.984)
t=3.6 s  com 0.272  clearance 0.108, upright 0.958, lateral 0.081
t=3.7 s             clearance 0.114, upright 0.900, lateral 0.152   ← collapse begins
t=4.0 s             clearance 0.094, upright 0.856, lateral 0.244
t=4.2 s  17 / 64 survive, clearance 0.069, upright 0.810, lateral 0.332
```

How a glide ends is a direct read of its exit speed, and it changed between the two
checkpoints:

| Checkpoint | Glides ending at the 0.10–0.135 m/s floor | Median exit speed |
|---|---:|---:|
| v15 `model_4899` | 5 / 64 = **8 %** | 0.246 m/s |
| v16-02 `model_7899` | 43 / 64 = **67 %** | 0.112 m/s |

So in v15 glides were ending **while still fast** — losing balance at a median 0.246 m/s —
and in v16-02 they end **on speed**, at the floor. This is the same fact as the budget
comparison in §2 and is the single most useful measurement in this document.

The support blade's yaw stays between **0.03 and 0.18 rad even on the longest glides**.
The support skate *can* steer. The policy never steers it to recover balance — the inward
support-skate steering degree of freedom is unused.

## 2. The condition for a 2.0 s glide

Travelling for `T` seconds from entry speed `v₀` at deceleration `a` while staying above
the 0.10 m/s credible floor requires

```
v₀ ≥ 0.10 + T · |a|          equivalently          T ≤ (v₀ − 0.10) / |a|
```

The right-hand form is the **speed budget**: how many seconds of credible glide the robot's
entry speed can pay for. Comparing budget with outcome is what shows where the robot
actually stands:

| Checkpoint | mean `v₀` | mean \|a\| | speed budget | best glide achieved | budget used | glides ending at the floor |
|---|---:|---:|---:|---:|---:|---:|
| v15 `model_4899` | 0.338 | 0.161 | 1.48 s | 1.16 s | **78 %** | 8 % |
| v16-02 `model_7899` | 0.528 | 0.243 | 1.76 s | 1.72 s | **98 %** | 67 % |

The tracer's own extrapolation (`implied_max_glide_s_from_coast`) agrees: 1.72 s for v15
and 1.80 s for v16-02.

**The binding constraint changed from balance to speed.** Under v15 the robot fell long
before spending its speed budget — only 78 % used, 8 % of glides ever reached the speed
floor, and 42 of 64 rollouts fell. Under v16-02 the glides consume 98 % of the budget and
two thirds of them end precisely at the speed floor. Balance is no longer what stops the
glide; the speed budget is.

So exactly one of the two levers has to move:

| Lever | Needs | Currently (v16-02) |
|---|---:|---:|
| Mean deceleration | ≤ **0.214 m/s²** | 0.243 m/s² — a **12 %** reduction |
| Mean entry speed | ≥ **0.587 m/s** | 0.528 mean, **0.633** max |

Two further facts bound the options:

* The `0.587` requirement is **not** out of reach: the best episodes already enter at
  0.633 m/s. They still fail, because their deceleration is no better than average — so
  raising the mean entry speed alone is not obviously sufficient.
* The frozen public expert's ceiling is **0.382 m/s**, so any entry speed above that must
  come from the bridge learning to push during the 1.0 s lift phase. Reward term B has
  already done part of this (0.338 → 0.528 m/s).
* Only the 1.0 s lift phase has learnable authority. During `phase = 0` (the 2.0 s
  acceleration) the correction gate is exactly zero, so **the acceleration phase cannot be
  improved by training at all** — there is zero headroom there by construction.

## 2b. What was tried and where it converged (2026-09-11, later the same day)

### The entry-speed lever: works, and is capped

Sweeping the injected speed over a frozen policy located where 2.0 s first becomes possible,
and produced the first 2.0 s glides ever seen:

| Injected | mean entry | longest credible | ≥2.0 s |
|---:|---:|---:|---:|
| 0.58 | 0.558 | 1.80 s | 0 % |
| 0.65 | 0.621 | 1.78 s | 0 % |
| **0.72** | **0.685** | **2.14 s** | 1.6 % |
| **0.78** | **0.742** | 2.10 s | 3.1 % |

But the precondition turned out to be a bug-like detail: the pose's *recorded* wheel speeds
only hold at the speed they were recorded at (0.51 m/s → 36.7 rad/s on the support wheels).
Injecting anything else while leaving them there makes the actor refuse to settle: raising
the injection 0.58 → 0.78 m/s moves the achieved entry by 0.054 m/s with the recorded wheels
against 0.184 m/s when the support wheels follow the injection. So the band change only
works because the wheel speeds were made consistent with it.

Cap: the valid predicate rejects root speed ≥ 0.80 m/s, and 80 % of episodes would need
~0.94 m/s at the measured deceleration. Entry speed converts "never" into "occasionally";
it cannot convert "occasionally" into "80 %".

### The authority lever: measured, and rejected

Everything about the composition said the correction had too little room: the hold is
`onefoot + 0.25 · correction`, and driving the frozen expert alone from the same fast spawn
gives a **0.022 s** glide with 59 falls out of 64, while the bridge gives **1.50 s** with 2
falls. The whole glide is the correction, at quarter strength.

Sweeping the coefficient directly:

| authority A | mean credible | falls |
|---:|---:|---:|
| **0.25 (shipped)** | **1.46 s** | 1 / 64 |
| 0.50 | 0.028 s | 72 / 64 |
| 0.75 | 0.005 s | 67 / 64 |
| 1.00 | 0.003 s | 64 / 64 |

**Rejected.** The correction is a trained equilibrium at A = 0.25 and its output magnitude is
calibrated to that gain; at A = 0.50 the commanded actions are roughly twice too large and
the robot is destroyed in a few steps. `HOLD_AUTHORITY` cannot be raised as a post-hoc scale
factor — only as a deliberate retraining. This is why the value stays at 0.25.

### Mechanisms tested for the drag, and rejected

| Hypothesis | Measurement | Verdict |
|---|---|---|
| Rolling resistance | four-wheel coast decelerates at −0.058 m/s² | rejected, 6× too small |
| Wheel skid | wheel joints have `damping=0`, `frictionloss=0`; measured support-wheel spin at the real handoff is 36.7 rad/s at 0.51 m/s = rolling | rejected |
| Lateral scrub | mean \|lateral velocity\| is 0.026 m/s, 8 % of forward speed | rejected, far too small |
| Blade yaw / skate misalignment | median 11°, but *negatively* correlated with deceleration | rejected |
| Speed-confounded correlates | per-glide correlations were dominated by the fact that \|a\| is larger at low speed; controlling for speed removes every | the earlier "correlations" were an artefact |

After controlling for speed, no single frame-level correlate dominates, and \|a\| sits in
0.21–0.32 m/s² across the whole speed range with a shallow minimum around 0.36–0.46 m/s.
That is consistent with a **distributed loss** rather than one contact effect.

### Where it stands

| | value |
|---|---:|
| mean entry speed (best block) | 0.592 m/s |
| mean deceleration | 0.343 m/s² |
| mean credible glide | 1.43 s |
| best credible glide | 2.14 s |
| what 2.0 s needs — deceleration | ≤ 0.246 m/s² (a 28 % cut) |
| what 2.0 s needs — entry speed | ≥ 0.786 m/s (essentially reached) |

So the remaining gap is **a 28 % reduction in glide deceleration**, nothing else.

## 3. Options that were on the table

| Option | Content | Lever |
|---|---|---|
| **B** (adopted) | one-shot bonus on COM speed at the instant of one-foot support | raise `v₀` to ≥ 0.44 |
| **E** (adopted) | shape lateral *and* fore/aft drift during the glide, rewarding support-skate steering | hold \|a\| ≤ 0.13 |
| A | extend `lift_s` 1.0 → 2.0 s and the episode | more authority time, but the handoff speed does not rise |
| C | move the correction gate away from 0 at `phase = 0` | makes the acceleration phase learnable (last resort; it is actor-side and touches the public expert's behaviour) |
| D | lower the certified goal to 1.5 s or 1.0 s and keep 2.0 s as a stretch goal | honest certification |

B and E were adopted **together**, precisely because either one alone can close the gap.
E is deliberately a *shape* rather than a penalty on falling, so that dying early cannot
escape it. Option C was kept as a last resort, and D has not been adopted — the 2.0 s
criterion is unchanged.

## 4. Where it stands today

B succeeded, and by more than the first measurement suggested: **entry speed 0.338 → 0.528
m/s**, above the frozen expert's 0.382 m/s ceiling and above what the v15 deceleration would
have required. E partially succeeded: lateral capture error 0.0203 → 0.0065 m.

Both changes together did something the individual numbers do not show: they **moved the
bottleneck**. Measured against the two traces, glides that under v15 ended *while still
moving* (8 % ever reached the speed floor, median exit 0.246 m/s) now under v16-02 end
*at the floor* (67 %, median exit 0.112 m/s). The robot no longer falls out of its glide
early; it now uses essentially the whole glide its speed pays for, and stops.

That is real progress and it is also a narrowing of the remaining gap to a single number:
**deceleration must fall from 0.243 to 0.214 m/s²**, or the mean entry speed must rise from
0.528 to 0.587 m/s. The lateral collapse of §1.5 — with the support skate's steering never
used — is the mechanism behind the first option, and it is the state this repository is
handed over in.

## 5. Reference values

| Quantity | Value | Source |
|---|---:|---|
| Frozen public expert speed ceiling (8 s window) | 0.382 m/s | `accel-probe-long.json` |
| Plateau interval | t ≈ 2.6–3.4 s at 0.37–0.38 m/s | `accel-probe-long.json` |
| Handoff at t = 3.0 s | 95 % of peak | `accel-probe-long.json` |
| Speed at a delayed handoff (t = 5.0 s) | 0.317 m/s | `accel-probe-long.json` |
| Speed cap with `lift_s = 6.0` | 0.49 m/s | `accel-probe-long.json` |
| Four-wheel coasting deceleration | −0.058 m/s² | `accel-probe-long.json` |
| Glide deceleration, first 0.25 s | −0.039 m/s² | `full-trace-u1000.json` |
| Credible speed floor | 0.10 m/s | predicate |
| Credible speed ceiling | 0.80 m/s | predicate |
| v15 mean entry / mean \|deceleration\| | 0.338 / 0.161 m/s² | `coast-down.json` |
| v16-02 mean entry / mean \|deceleration\| | 0.528 / 0.243 m/s² | `coast-down.json` |
| v15 best glide / budget used | 1.16 s / 78 % | `coast-down.json` |
| v16-02 best glide / budget used | 1.72 s / 98 % | `coast-down.json` |
| Deceleration needed for 2.0 s at v16-02 entry | 0.214 m/s² | derived |
| Entry needed for 2.0 s at v16-02 deceleration | 0.587 m/s | derived |
