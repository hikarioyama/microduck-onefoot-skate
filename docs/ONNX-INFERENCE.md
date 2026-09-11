# Running the shipped policies

The three `.onnx` files in `checkpoints/` are self-contained: one input, one output, no
mjlab and no MuJoCo needed. Everything below was run and verified in this repository, and
the numbers are reproducible from `records/onnx-vs-actor.json`.

## The files

| File | Input | Output | What it is |
|---|---|---|---|
| `checkpoints/glide/policy-glide-02.onnx` | `obs` (1, 61) | `actions` (1, 14) | **The current best.** The full composed bridge at `model_10499`. |
| `checkpoints/glide/policy.onnx` | `obs` (1, 61) | `actions` (1, 14) | The previous glide block, `model_9499`. |
| `checkpoints/frozen/public-roller-expert.onnx` | `obs` (1, 61) | `actions` (1, 14) | The frozen upstream roller-acceleration expert, unchanged. |

`policy-glide-02.onnx` and `policy.onnx` are **composed** policies, not single networks.
Their graphs already contain both frozen experts and the `phase` interpolation described in
[METHOD.md](METHOD.md) §3, so a single call is the whole policy. You do not implement the
blending yourself.

> **What "current best" means.** This policy glides on one foot for ~1.4 s on average and
> 2.14 s at its best, from an assisted mid-glide start. It has **not** passed the
> acceptance gate and it cannot skate from rest. See the status block in the top-level
> README before using it for anything.

## Quickstart

```bash
uv run --with onnxruntime --with numpy --with onnx python scripts/policy_inference.py \
    checkpoints/glide/policy-glide-02.onnx
```

The script feeds the measured glide-entry state by default, so the policy sees a
configuration it was trained on rather than an extrapolation. `--sample upright --phase 0`
shows the other end of the bridge instead.

## The observation, exactly

61 float32 values, in this order. The term names and the 61-dim total come from a live
environment: `observation_manager.active_terms['actor']` is
`base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command, head_command,
body_command` and `group_obs_dim['actor'] == (61,)`.

| Slice | Term | Width | Meaning |
|---|---|---:|---|
| `0:3` | `base_ang_vel` | 3 | Body angular velocity, rad/s. |
| `3:6` | `projected_gravity` | 3 | Gravity direction in the body frame, unit vector. |
| `6:20` | `joint_pos` | 14 | The 14 servo joints, **HOME-relative** (`qpos - HOME`). |
| `20:34` | `joint_vel` | 14 | The same 14 joints, rad/s. |
| `34:48` | `actions` | 14 | The **previous issued action**. Zero on the first step. |
| `48` | `command[0]` | 1 | Forward twist. This task publishes `0.3`. |
| `49` | `command[1]` | 1 | **`phase`**: 0.0 = acceleration half, 1.0 = one-foot glide half. |
| `50` | `command[2]` | 1 | Heading; 0 in the straight-line recipe. |
| `51:55` | `head_command` | 4 | Zero-padded. Feed zeros. |
| `55:61` | `body_command` | 6 | Zero-padded. Feed zeros. |

The passive wheel joints are **not** in the actor observation; they only appear in the
78-dim critic group, which no shipped policy uses.

Two details are easy to get wrong and both cost a lot of accuracy:

1. **`joint_pos` is HOME-relative, not an absolute qpos.** `local/onefoot-bridge-curriculum/glide-pose-measurement.json`
   stores `median_pose` as an absolute qpos vector, so the default pose has to be
   subtracted before it can be fed as `joint_pos`. `scripts/policy_inference.py` does this;
   the subtraction is also where an early version of that script was wrong.
2. **`command[0]` is read differently by the two experts.** The frozen public roller expert
   ignores whatever you put in slot 48: it is always fed `push_command = 0.6` with slots
   49:61 zeroed (`public_roller_policy.py`). The frozen one-foot expert sees slot 48 as
   given, and at `phase = 1` the composition is `onefoot + 0.25 * correction`, so slot 48
   matters there. Feed `0.3`, which is what the environment published during training.
   Feeding `0.6` at `phase = 1` is a different, untrained input.

No normalisation is your job. The graph contains its own observation normalisers, exactly
as they were at the end of training.

## The action, exactly

One output, `actions` (1, 14), the deterministic mean action. The environment applies it as
a HOME-relative position target with scale 1.0 and offset 0.0:

```python
joint_target[i] = default_joint_pos[i] + actions[i]
```

`joint_names` and `default_joint_pos` are embedded in the metadata of
`checkpoints/frozen/public-roller-expert.onnx` — same robot, same 14-actuator order — so
`scripts/policy_inference.py` reads them from there rather than hard-coding a second copy.

The joint order is the actuator order:

```
left_hip_yaw  left_hip_roll  left_hip_pitch  left_knee  left_ankle
neck_pitch    head_pitch     head_yaw        head_roll
right_hip_yaw right_hip_roll right_hip_pitch right_knee right_ankle
```

and `default_joint_pos` is

```
0.000  -0.087  -0.458  -0.005   0.453   0.349   0.349   0.000   0.000
0.000   0.087   0.458   0.005  -0.453
```

**The target is a PD setpoint, not a pose.** It is deliberately offset from the current
joint angles so the servos produce torque, so the numbers are larger than the pose you fed
in and a few radians on the swing knee or the neck is normal. The actor applies **no output
clip** (`JointPositionActionCfg(clip=None)`), so clamp to your own actuator limits if you
drive hardware.

### The action is in actuator order, which is not the joint order

The 14 values are in **actuator order**, and that order is *not* the robot's joint order.
The head block is written differently in the two:

| Index | Actuator order (this is the action) | Robot joint order (from the model) |
|---:|---|---|
| 5 | `neck_pitch` | `head_yaw` |
| 6 | `head_pitch` | `head_roll` |
| 7 | `head_yaw` | `neck_pitch` |
| 8 | `head_roll` | `head_pitch` |

Everything else matches. This is easy to miss: `robot.data.default_joint_pos` is in the
robot's order and looks plausible either way, so cross-check with `joint_names` from the
ONNX metadata rather than by eye. `scripts/policy_inference.py` uses the metadata order
throughout.

### The policy drives joints into their limits

The joint ranges are tight — `left_hip_roll` is only ±0.384 rad, `head_roll` ±0.436 —
and the policy uses those stops as reference points. Measured under
`policy-glide-02.onnx` over 80 control steps: **10 of the 14 joints reach a hard limit**,
`left_hip_roll` and `neck_pitch` spending the glide pressed against theirs.

The full ranges, extracted from the robot XML, are in
[`../records/joint-limits.json`](../records/joint-limits.json) together with the
measurement protocol. That same file records the sim-to-real caveat: `qpos` can exceed the
nominal range by ~0.016 rad because the limit is a soft constraint in the solver, and
holding a real servo against a mechanical stop is not something you want to do for long.
**This policy is a simulation result, and saturating joints against their limits is part of
how it works** — a deployment would need its own answer for that.

## How to check the files yourself

`scripts/verify-shipped-onnx.py` builds the real environment, loads the recorded `.pt`
through the same safe loader the trainer used, and compares

```
onnx(obs)   vs   actor(obs, stochastic_output=False)
```

on the observations the environment actually produces:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --with onnxruntime --with onnx \
    python scripts/verify-shipped-onnx.py
```

It needs the upstream working tree (see [REPRODUCING.md](REPRODUCING.md)). Result, from
`records/onnx-vs-actor.json`:

| Policy | Checkpoint | max abs difference | allclose (atol 3e-6) |
|---|---|---:|---|
| `policy-glide-02.onnx` | `model_10499.pt` | **2.15e-06** | yes |
| `policy.onnx` | `model_9499.pt` | **1.07e-06** | yes |

This is worth stating separately because the existing test,
`tests/test_retained_skill_bridge.py::test_standard_exports_include_both_skills_and_bridge`,
proves the *exporter* is faithful using a synthetic actor. It does not prove that the files
actually shipped here match the recorded checkpoints. `verify-shipped-onnx.py` closes that
gap, and both are correct.

## What is not exported

`checkpoints/glide/model_10499.pt` also contains the PPO critic and the Adam moments. Those
are training state; no policy you can deploy needs them. If you want to continue training
rather than deploy, use the `.pt` and follow `docs/METHOD.md` §6.
