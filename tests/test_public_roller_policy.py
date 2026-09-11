import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from mjlab_microduck.public_roller_policy import (
    LAST_ACTION, PublicRollerPolicy, roller_observations,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT/'local/onefoot-public-acceleration/assets/088524a64e2557dc453256b6071dbb9d23888802/roller.onnx'


def diagnostic_module():
    spec = importlib.util.spec_from_file_location('public_roller_diagnostic_test',
                                                ROOT/'local/evaluate-public-roller-connection.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_and_previous_action_are_adapted_without_mutating_source():
    source = np.arange(2*61, dtype=np.float32).reshape(2, 61)/100
    original = source.copy()
    result = roller_observations(source, action_scale=.8, push_command=.6)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(result[:, :34], source[:, :34])
    np.testing.assert_allclose(result[:, LAST_ACTION]*.8, source[:, LAST_ACTION])
    np.testing.assert_allclose(result[:, 48], .6)
    np.testing.assert_array_equal(result[:, 49:], 0.)
    # A switch to the scale=1 onefoot policy receives the original issued action.
    np.testing.assert_array_equal(source[:, LAST_ACTION], original[:, LAST_ACTION])


@pytest.mark.parametrize('scale', [0., -1., float('nan'), float('inf')])
def test_invalid_scales_are_rejected(scale):
    with pytest.raises(ValueError):
        roller_observations(np.zeros((1, 61)), action_scale=scale, push_command=.6)


@pytest.mark.parametrize('obs', [np.zeros(61), np.zeros((2, 51)), np.full((1, 61), np.nan)])
def test_invalid_observations_are_rejected(obs):
    with pytest.raises(ValueError):
        roller_observations(obs, action_scale=1., push_command=.6)


def test_nonfinite_heading_and_out_of_range_push_are_rejected():
    with pytest.raises(ValueError):
        roller_observations(np.zeros((1, 61)), action_scale=1., push_command=.7)
    with pytest.raises(ValueError):
        roller_observations(np.zeros((1, 61)), action_scale=1., push_command=.6, heading=np.nan)


def test_public_file_hash_guard_precedes_onnx_loading(tmp_path):
    path = tmp_path/'bad.onnx'
    path.write_bytes(b'not an ONNX model')
    with pytest.raises(ValueError, match='SHA256'):
        PublicRollerPolicy(path, expected_sha256='wrong', action_scale=1.)


def test_connection_config_preserves_physics_rewards_and_zero_velocity():
    from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import make_waist_onefoot_env_cfg
    from mjlab_microduck.tasks import mdp
    module = diagnostic_module()
    baseline = make_waist_onefoot_env_cfg(stage='self-launch')
    cfg = module.environment_cfg(8, 70101, 6., 2.5)
    assert cfg.actions == baseline.actions
    assert cfg.observations == baseline.observations
    assert cfg.rewards == baseline.rewards
    assert cfg.terminations == baseline.terminations
    assert cfg.scene.entities == baseline.scene.entities
    baseline.sim.nan_guard.enabled = True
    assert cfg.sim == baseline.sim
    assert cfg.events['curriculum_spawn'].func is mdp.reset_curriculum_onefoot
    assert cfg.events['curriculum_spawn'].params['speed_range'] == (0., 0.)
    assert cfg.events['curriculum_spawn'].params['phase_start'] == 0.
    assert all(value == (0., 0.) for value in cfg.events['reset_base'].params['velocity_range'].values())
    assert len(cfg.events['curriculum_spawn'].params['pose']) == 14
    for name in cfg.events:
        if name != 'curriculum_spawn':
            assert cfg.events[name] == baseline.events[name], name
    assert cfg.commands['twist'].acceleration_s == 2.5
    assert module.environment_cfg(8, 70101, 6., None).commands['twist'].acceleration_s > 6.


@pytest.mark.skipif(not PUBLIC_PATH.exists(), reason='Pinned public ONNX not downloaded in this checkout')
def test_pinned_policy_cpu_batch_matches_original_graph_and_home():
    from mjlab_microduck.public_roller_policy import JOINT_NAMES
    import onnxruntime as ort
    module = diagnostic_module()
    policy = PublicRollerPolicy(PUBLIC_PATH, expected_sha256=module.PUBLIC_SHA256, action_scale=.8)
    home = module.grounded_home()['pose']
    policy.validate_robot(JOINT_NAMES, np.array([home[name] for name in JOINT_NAMES]))
    source = np.random.default_rng(1).normal(0, .1, (4, 61)).astype(np.float32)
    adapted = roller_observations(source, action_scale=.8, push_command=.6)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    original = ort.InferenceSession(str(PUBLIC_PATH), sess_options=options, providers=['CPUExecutionProvider'])
    expected = np.concatenate([original.run(None, {'obs': row[None]})[0] for row in adapted])*.8
    np.testing.assert_array_equal(policy(source), expected)
    with pytest.raises(ValueError, match='HOME'):
        policy.validate_robot(JOINT_NAMES, np.ones(14))
    assert hashlib.sha256(PUBLIC_PATH.read_bytes()).hexdigest() == module.PUBLIC_SHA256
