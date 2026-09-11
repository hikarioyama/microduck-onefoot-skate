"""Run one of the shipped policies standalone: 61D raw observation -> 14 joint targets.

This is the minimal way to use the movement without installing mjlab or MuJoCo.
It needs only `numpy` and `onnxruntime`:

    uv run --with onnxruntime --with numpy python scripts/policy_inference.py \
        checkpoints/glide/policy-glide-02.onnx

What the policy expects
-----------------------
ONE input, `obs`, shaped (1, 61) float32, and ONE output, `actions`, shaped (1, 14).
The 61 dimensions are the raw actor observation, in this order. The offsets come from
a live environment (`observation_manager.group_obs_dim['actor'] == (61,)`, terms
`base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command,
head_command, body_command`); slots 34:48, 48 and 49 are additionally pinned by
`tests/test_retained_skill_bridge.py`:

    0:3    base_ang_vel        body angular velocity, rad/s
    3:6    projected_gravity   gravity in the body frame, unit vector
    6:20   joint_pos           the 14 servo joints, HOME-relative (qpos - HOME)
    20:34  joint_vel           the same 14 joints, rad/s
    34:48  actions             the previous issued action (zero on the first step)
    48     command[0]          forward twist. This task publishes target_speed 0.3.
    49     command[1]          phase: 0.0 = acceleration half, 1.0 = one-foot glide
    50     command[2]          heading, 0 in this recipe
    51:55  head_command        zero-padded, feed zeros
    55:61  body_command        zero-padded, feed zeros

Note the two experts read the command block differently, which matters for slot 48:

* the frozen **public roller** expert ignores the caller and is always fed
  `push_command = 0.6` with slots 49:61 zeroed (`public_roller_policy.py`), so at
  `phase = 0` slot 48 has no effect at all;
* the frozen **one-foot** expert sees slot 48 exactly as given, and at `phase = 1`
  the composition is `onefoot + 0.25 * correction`, so slot 48 does matter there.
  Feed 0.3 here, which is what the environment published during training.

This is why `--command` defaults to 0.3 and not to the 0.6 the roller expert is fed.

No normalisation is required from the caller: the network carries its own
observation normalisers internally, exactly as it did in training. The two frozen
experts are baked into the graph, and so is the `phase` interpolation, so a single
call reproduces the composed policy of docs/METHOD.md section 3. That is verified,
not assumed — see docs/ONNX-INFERENCE.md and `records/onnx-vs-actor.json`, where the
shipped `.onnx` reproduces the recorded `.pt` actor to 2.1e-06 on live observations.

Turning the output into joint targets
-------------------------------------
`actions` is a HOME-relative position target with scale and offset 1.0 and 0.0:

    joint_target[i] = default_joint_pos[i] + actions[i]

`joint_names` and `default_joint_pos` for the same robot and the same 14-actuator
order are embedded in the metadata of the frozen roller expert, so this script
reads them from there instead of hard-coding a second copy.

The target is a PD setpoint, not a pose: it is deliberately offset from the current
joint angles so that the servos produce torque. Large values (a few rad on the swing
knee and the neck) are normal for this policy, and the actor applies no output clip,
so clamp to your actuator limits if you drive real hardware.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / 'checkpoints/frozen/public-roller-expert.onnx'
# The measured single-support glide-entry state, captured across 64 real rollouts.
GLIDE_POSE = ROOT / 'local/onefoot-bridge-curriculum/glide-pose-measurement.json'


def metadata(path):
    """Read the string metadata ONNX stores on the model itself."""
    import onnx  # only needed to read metadata; onnxruntime is used for inference
    model = onnx.load(str(path))
    return {entry.key: entry.value for entry in model.metadata_props}


def glide_observation(servo_names, default_joint_pos, command=0.3, phase=1.0,
                      previous_action=None):
    """The MEASURED glide-entry state, so the policy is exercised in-distribution.

    `median_pose` in the recording is an absolute qpos vector, while the `joint_pos`
    observation slot is HOME-RELATIVE, so the default pose is subtracted here.
    """
    measured = json.loads(GLIDE_POSE.read_text())
    obs = np.zeros(61, np.float32)
    gravity = measured['projected_gravity']
    obs[3:6] = [gravity['x'], gravity['y'], gravity['z']]
    obs[6:20] = [measured['median_pose'][name] for name in servo_names] - default_joint_pos
    obs[20:34] = [measured['median_joint_vel'][name] for name in servo_names]
    obs[34:48] = np.zeros(14, np.float32) if previous_action is None else previous_action
    obs[48] = command
    obs[49] = phase
    return obs


def upright_observation(previous_action=None, command=0.3, phase=0.0):
    """A still, upright, HOME-relative robot: the phase-0 end of the bridge."""
    obs = np.zeros(61, np.float32)
    obs[3:6] = (0., 0., -1.)          # upright: gravity points down the body z axis
    obs[34:48] = np.zeros(14, np.float32) if previous_action is None else previous_action
    obs[48] = command
    obs[49] = phase
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('policy', type=Path, help='a shipped .onnx policy')
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA,
                        help='ONNX carrying joint_names / default_joint_pos')
    parser.add_argument('--phase', type=float, default=1.0,
                        help='0.0 acceleration, 1.0 one-foot glide (default 1.0)')
    parser.add_argument('--command', type=float, default=0.3,
                        help='slot 48 forward twist. The env publishes 0.3 for this task; '
                             'the frozen public expert ignores it and is fed 0.6 internally, '
                             'while the frozen one-foot expert sees it as given (default 0.3)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='feed the output back as the previous action N times')
    parser.add_argument('--sample', choices=('glide', 'upright'), default='glide',
                        help="observation to feed: the measured glide-entry state "
                             "(default), or a still upright robot")
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.policy), providers=['CPUExecutionProvider'])
    inputs = {i.name: i.shape for i in session.get_inputs()}
    outputs = {o.name: o.shape for o in session.get_outputs()}
    print(f'policy      {args.policy}')
    print(f'inputs      {inputs}')
    print(f'outputs     {outputs}')

    # These metadata fields are stored as comma-separated strings, not lists.
    names = metadata(args.metadata)['joint_names'].split(',')
    default = np.array(metadata(args.metadata)['default_joint_pos'].split(','), np.float32)
    assert len(names) == 14 and default.shape == (14,)

    if args.phase == 1.0 and args.sample == 'glide':
        # The default: the measured glide-entry state, so the policy sees a
        # configuration it was actually trained on rather than an extrapolation.
        obs = glide_observation(names, default, command=args.command, phase=args.phase)
    else:
        obs = upright_observation(command=args.command, phase=args.phase)
    for step in range(args.repeat):
        action = session.run(['actions'], {'obs': obs[None]})[0][0]
        if step + 1 < args.repeat:
            obs = glide_observation(names, default, command=args.command,
                                    phase=args.phase, previous_action=action)
    target = default + action

    print(f'\nphase {args.phase}, command {args.command}, {args.repeat} step(s), '
          f'observation = {args.sample}')
    print(f'{"joint":16s} {"default":>9s} {"action":>9s} {"target":>9s}')
    for name, d, a, t in zip(names, default, action, target):
        print(f'{name:16s} {d:9.4f} {a:9.4f} {t:9.4f}')
    print(f'\nfinite: {np.isfinite(action).all()}   |action|max: {abs(action).max():.4f} rad')
    print('joint_target[i] = default_joint_pos[i] + actions[i]; clamp to your own '
          'actuator limits, the policy has no output clip.')


if __name__ == '__main__':
    main()
