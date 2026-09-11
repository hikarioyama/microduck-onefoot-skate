"""Frozen official roller ONNX adapter; no training or implicit action filtering.

The shared environment stores HOME-relative, scale=1 issued actions. A roller
policy using a deployment scale must see last_action in its own output units;
the receiving onefoot policy sees the canonical issued action, without a reset.
"""
from pathlib import Path
import hashlib
import math
import numpy as np
import onnxruntime as ort

OBSERVATION_NAMES = (
    'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'actions',
    'command', 'head_command', 'body_command',
)
JOINT_NAMES = (
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
    'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
    'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
)
LAST_ACTION = slice(34, 48)
COMMAND = slice(48, 61)


def roller_observations(observations, *, action_scale, push_command, heading=None):
    """Copy raw61D observations and write ROLLER commands before normalization.

A fixed zero heading is the current roller recipe's straight-line contract.
An explicitly supplied heading is a separately labelled diagnostic, never a
silent reuse of the onefoot line-error/phase command.
"""
    if not math.isfinite(action_scale) or action_scale <= 0:
        raise ValueError('action_scale must be finite and positive')
    if not math.isfinite(push_command) or not -0.5 <= push_command <= 0.6:
        raise ValueError('push_command is outside the documented roller range')
    source = np.asarray(observations, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 61:
        raise ValueError('Expected a batch of raw61D actor observations')
    if not np.isfinite(source).all():
        raise ValueError('Nonfinite policy observation')
    result = source.copy()
    result[:, LAST_ACTION] /= action_scale
    result[:, COMMAND] = 0.0
    result[:, 48] = push_command
    if heading is not None:
        result[:, 50] = heading
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite adapted policy observation')
    return result


class PublicRollerPolicy:
    """Run the unmodified, normalization-baked official ONNX on CPU.

The published graph has a fixed batch of one. Sequential rows preserve its
exact graph, instead of guessing a normalizer or re-exporting it by hand.
"""
    def __init__(self, path, *, expected_sha256, action_scale):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != expected_sha256:
            raise ValueError('Public policy SHA256 mismatch')
        if not math.isfinite(action_scale) or action_scale <= 0:
            raise ValueError('action_scale must be finite and positive')
        self.action_scale = action_scale
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape != [1, 61]
                or outputs[0].shape != [1, 14]
                or inputs[0].type != 'tensor(float)' or outputs[0].type != 'tensor(float)'):
            raise ValueError('Unsupported public policy input/output contract')
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        if tuple(self.metadata.get('joint_names', '').split(',')) != JOINT_NAMES:
            raise ValueError('Public policy joint order mismatch')
        if tuple(self.metadata.get('observation_names', '').split(',')) != OBSERVATION_NAMES:
            raise ValueError('Public policy observation order mismatch')

    def validate_robot(self, joint_names, home):
        if tuple(joint_names) != JOINT_NAMES:
            raise ValueError('Environment servo order mismatch')
        exported_home = np.fromstring(self.metadata['default_joint_pos'], sep=',')
        # Export metadata rounds to three decimals; do not use it as a new HOME.
        if exported_home.shape != (14,) or not np.allclose(exported_home, home, atol=0.000501, rtol=0):
            raise ValueError('Environment HOME does not match rounded policy metadata')

    def __call__(self, observations, *, push_command=0.6, heading=None):
        obs = roller_observations(observations, action_scale=self.action_scale,
                                  push_command=push_command, heading=heading)
        if len(obs) == 0:
            return np.empty((0, 14), dtype=np.float32)
        raw = np.concatenate([self.session.run([self.output_name],
                              {self.input_name: row[None, :]})[0] for row in obs], axis=0)
        actions = raw * self.action_scale
        if actions.shape != (len(obs), 14) or not np.isfinite(actions).all():
            raise ValueError('Invalid public policy action')
        return actions
