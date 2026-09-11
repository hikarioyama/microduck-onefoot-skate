# Method — robot, contracts, architecture

## 1. The robot

**Microduck**: a ~25 cm tall, ~800 g biped built by Pollen Robotics, with 14 Dynamixel
XL330 servos and **four passive roller wheels** (two under each foot). The wheels have no
actuators — they are free joints. There is no linkage joint: the only passive joints in the
model are the four wheels.

Servo joints (14, the action space):

```
left_hip_yaw   left_hip_roll  left_hip_pitch  left_knee  left_ankle
neck_pitch     head_pitch     head_yaw        head_roll
right_hip_yaw  right_hip_roll right_hip_pitch right_knee right_ankle
```

Passive joints (4): `passive_LF_wheel`, `passive_LR_wheel`, `passive_RF_wheel`,
`passive_RR_wheel`, plus the floating base `trunk_base_freejoint`.

Model: `src/mjlab_microduck/robot/microduck/scene_rollers.xml`.

### Wheel geometry, which turned out to matter

The tyre mesh radius is **0.015 m** but its bounding extent is **0.016 m**, and the
clearance required by the valid-glide predicate is **0.010 m**. A pose that is grounded by
nominal radius is not grounded. All spawn grounding in this work is computed from the
**tyre mesh vertices** (`wheel_mesh_min_z_at_root_zero` in the recorded measurements), not
from a nominal radius. The mesh names are `tire`, `tire_2` (left) and `tire_3`, `tire_4`
(right).

## 2. Obs / action / timing contract

These are pinned by tests and were never changed:

| Item | Value |
|---|---|
| Actor observation (raw) | **61D** |
| Critic observation | **78D** |
| Action | **14D**, HOME-relative servo targets |
| Physics timestep | 0.005 s |
| Decimation | 4 |
| Control timestep | **0.02 s** |

Joints and bodies are always resolved **by name**, never by index. Unused command slots are
zero-padded rather than removed, so that the public expert's observation layout stays
valid.

Two observation slots are load-bearing:

* **slot 48** — the forward twist command. The frozen public roller expert is fed a
  constant **0.6** here and slots 49+ are zeroed, so its output is deterministic. Note that
  the environment's own `target_speed` (0.3) is *not* visible to it; commanded 0.6, it
  saturates at ~0.38 m/s.
* **slot 49** — `phase`, the blend variable described below.

## 3. The retained-skill bridge

The policy is not a single network. It is two frozen experts composed with one trainable
correction:

```
                     ┌──────────────────────┐
   raw obs (61D) ────▶│ FrozenPublicRoller   │──▶ public action (14D)
                     ├──────────────────────┤
                     │ FrozenOnefoot  (MLP) │──▶ one-foot action (14D)
                     ├──────────────────────┤
                     │ trainable correction │──▶ residual (14D)
                     └──────────────────────┘
```

```python
phase  = raw[..., 49:50].clamp(0., 1.)
blend  = phase**2 * (3. - 2.*phase)            # smoothstep
gate   = 4.*phase*(1.-phase) + .25*blend
out    = (1.-blend)*public + blend*onefoot + gate*correction
```

Properties that the tests pin:

* **`phase = 0` reproduces the public expert exactly.** `blend = 0` and `gate = 0`, so the
  composed output is bit-for-bit the frozen roller policy. This is what makes the
  from-rest acceleration phase well defined — and it is also why the trainable part has
  *zero authority* there (§4).
* The bridge's output layer is zero-initialised, so at initialisation the correction is
  zero and the composed policy is smooth.
* `phase = 1` gives `blend = 1`, `gate = 0.25`.

### `HOLD_AUTHORITY`

A correction gate of exactly `4·phase·(1−phase)` means the gate is **0.04 at `phase = 0.99`**
and 0 at `phase = 1.0` — i.e. precisely during the sustained hold, which the strict
valid-glide predicate defines as `phase ≥ 0.99`.

That was not a subtlety, it was a hard ceiling. Under the strict predicate, the frozen
one-foot expert capped the task at its own limit: 256 first episodes × 2 seeds from the
balance pose gave credible 0.5 s of 86.3 % / 87.1 % but credible **1.0 s and 2.0 s of 0 %**,
with a maximum credible glide of 0.88 s. The 2.0 s goal was unreachable *by construction*,
not by undertraining.

The fix gives the hold a reduced, explicitly recorded share of the residual:

```python
HOLD_AUTHORITY = .25
# ... inlined as the literal .25 for TorchScript; a test ties the literal to the constant
gate = 4.*phase*(1.-phase) + .25*blend
```

This changed the **actor** and was therefore done as an `actor_forward_transition`, with
the recorded value in `records/actor-transition-hold-authority.json`. The literal is
duplicated inside the TorchScript-visible function (TorchScript cannot close over a
module-level float) and a test asserts the two stay equal — so the duplication cannot
silently drift.

### Immediate consequences

`phase = 0` for the first 2.0 s of a from-rest episode. With a 5.73 s mean episode that is
**2.00 s = 35 % of every rollout with zero learnable authority**, and it is not improvable
by training (see `REPORT.md` §3). This single fact is what justified building the assisted
`glide` stage.

## 4. Curriculum stages

| Stage | Goal (s) | Horizon (s) | Start | Assisted |
|---|---:|---:|---|---|
| `near` | 0.5 | 1.7 | `balance` pose, phase 0.85 | yes |
| `unload` | 0.5 | 2.2 | `unload` pose, phase 0.65, speed 0.25–0.35 | yes |
| `transfer-near` | 0.5 | 2.8 | `transfer_near` pose, phase 0.40, speed 0.25–0.35 | yes |
| `transfer` | 0.5 | 3.6 | `transfer` pose, phase 0.05, speed 0.25–0.35 | yes |
| **`glide`** | **2.0** | **3.4** | **measured glide state, phase 1.0, speed 0.45–0.58** | **yes** |
| `full` | 2.0 | 6.0 | HOME, exactly zero velocity | **no** |

```python
STAGES            = ('full','near','unload','transfer-near','transfer','glide','hold-check')
BRIDGE_CURRICULUM = ('near','unload','transfer-near','transfer','glide','full')
SUSTAINED_STAGES  = ('glide','full')     # judged on the 2.0 s credible metric
```

`SUSTAINED_STAGES` exists so that `glide` and `full` certify against the *same* 2.0 s
metric. Without it, an assisted stage with a shorter horizon would certify a different task
than the from-rest one and the comparison would be meaningless.

`goal_s` is `2.0` for every stage in `SUSTAINED_STAGES` and `0.5` for the rest, except
`hold-check`, which inherits the prior goal.

### Assisted starts

```python
ASSISTED_BRIDGE_STARTS = {
    'near':          ('balance',       .85, 1.7, None        ),
    'unload':        ('unload',        .65, 2.2, (.25, .35)  ),
    'transfer-near': ('transfer_near', .40, 2.8, (.25, .35)  ),
    'transfer':      ('transfer',      .05, 3.6, (.25, .35)  ),
    'glide':         ('glide',         1.0, 3.4, (.45, .58)  ),
}
```

`initial_assistance = (stage != 'full')` is recorded on every evaluation. **The assisted
stages can never certify a from-rest result.** Only `full` — HOME spawn with root, joint
and wheel velocities exactly zero — can.

## 5. Spawning, and the `mdp.py` prefix invariant

`tests/test_old_mdp_and_registry_bytes_are_preserved` requires the **first 351 782 bytes of
`tasks/mdp.py`** — everything before the v15 reward region, which begins around line 8148 —
to remain byte-identical.

The shared spawn helper `reset_curriculum_onefoot` lives inside that pinned prefix. Adding
an optional argument to it would break the invariant *permanently*: once the byte offset
moves, it cannot be restored, and the legacy region is provenance for the earlier
curriculum work.

The faithful spawn therefore lives outside it, in the bridge env cfg:

```python
def reset_credit_bridge_spawn(...):   # in microduck_onefoot_credit_bridge_env_cfg.py
    ...
```

It is an exact copy of the shared helper with `root_pitch` and `joint_vel` added, including
the quaternion composition `[cr*cp, sr*cp, cr*sp, -sr*sp]` and a per-joint velocity write.
The wheel `v/r` fallback is applied **only** to wheels absent from the recorded
`joint_vel`.

It is installed **only** for poses that actually carry the extra measured fields:

```python
if 'joint_vel' in recorded or recorded.get('root_pitch'):
    cfg.events['curriculum_spawn'].func = reset_credit_bridge_spawn
```

Only `POSES['glide']` has them. Every other assisted stage still uses
`mdp.reset_curriculum_onefoot`, byte for byte.

## 6. Training

| Item | Value |
|---|---|
| Algorithm | RSL-RL PPO |
| Environments | 4096 |
| Learning rate | `1e-4`, `algorithm.schedule = 'fixed'` |
| Horizon | **finite and a true terminal** — every termination term has `term.time_out = False`, and `extras['time_outs']` is asserted all-false |
| Reward structure | `task + shaping + failure + handoff` |
| Block size | 1000 updates, screening evaluation every 100 |
| GPU | one RTX 5070 Ti |

The finite-horizon decision is deliberate and is asserted at runtime
(`assert not extras['time_outs'].any(), 'No finite-horizon bootstrap allowed'`). Reaching
the horizon is the *true end of the task attempt*, not an artificial truncation of a
continuing process, so RSL-RL must not bootstrap the value function across it. Combined
with all terminal potentials being zero, this makes the return well defined.

PPO noise applies during training; evaluation uses the deterministic mean action.

## 7. What is deliberately not done here

* No modification of the two frozen experts.
* No change to the 61D / 78D / 14D contract.
* No loosening of the valid-glide predicate or of the acceptance gate, including when they
  failed.
* No promotion of any of these results into the main curriculum state. The upstream main
  branch remains at waist-balance-100 / iteration 16399, untouched.
