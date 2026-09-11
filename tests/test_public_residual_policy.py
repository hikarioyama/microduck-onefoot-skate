from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from mjlab_microduck.public_residual_policy import (
    PUBLIC_PATH, PUBLIC_SHA256, FrozenPublicRoller, PublicResidualModel, frozen_public_digest,
)
from mjlab_microduck.public_roller_policy import PublicRollerPolicy

pytestmark = pytest.mark.skipif(not PUBLIC_PATH.exists(), reason='Pinned public policy is a local experiment asset')


def observations(n=32):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(n, 61, generator=generator)*.15
    x[:, 5] = -1.
    x[:, 48] = .3
    x[:, 49] = torch.linspace(0., 1., n)
    x[:, 51:] = 0.
    return TensorDict({'actor': x}, batch_size=[n])


def actor(obs):
    return PublicResidualModel(obs=obs, obs_groups={'actor': ['actor']}, obs_set='actor',
        output_dim=14, hidden_dims=[512, 256, 128], activation='elu', obs_normalization=True,
        distribution_cfg={'class_name': 'rsl_rl.modules.distribution:GaussianDistribution',
                          'init_std': .08, 'std_type': 'log'})


def test_torch_public_batch_matches_unmodified_onnx():
    raw = observations(64)['actor']
    frozen = FrozenPublicRoller()
    original = PublicRollerPolicy(PUBLIC_PATH, expected_sha256=PUBLIC_SHA256, action_scale=.8)
    expected = original(raw.numpy())
    np.testing.assert_allclose(frozen(raw).numpy(), expected, atol=2e-6, rtol=2e-5)
    assert all(not p.requires_grad for p in frozen.parameters())


def test_zero_residual_preserves_public_at_all_phases_and_normalizer_updates():
    obs = observations()
    policy = actor(obs)
    with torch.no_grad():
        expected = policy.public(obs['actor']).clone()
        torch.testing.assert_close(policy(obs), expected, atol=0, rtol=0)
        policy.update_normalization(obs)
        torch.testing.assert_close(policy(obs), expected, atol=0, rtol=0)


def test_optimizer_updates_only_connection_and_preserves_prefix():
    obs = observations()
    policy = actor(obs)
    policy.update_normalization(obs)
    digest = frozen_public_digest(policy)
    prefix = obs.clone(); prefix['actor'][:, 49] = 0.
    with torch.no_grad():
        before = policy(prefix).clone()
    optimizer = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=1e-4)
    target = policy.public(obs['actor']).detach()+.1
    for _ in range(3):
        optimizer.zero_grad()
        loss = (policy(obs)-target).square().mean()
        loss.backward()
        optimizer.step()
    assert frozen_public_digest(policy) == digest
    assert policy.mlp[-1].weight.abs().max() > 0
    torch.testing.assert_close(policy(prefix), before, atol=0, rtol=0)
    assert all(p.grad is None for p in policy.public.parameters())


def test_stochastic_actions_are_sampled_from_the_actual_combined_mean():
    obs = observations()
    policy = actor(obs)
    with torch.no_grad():
        policy.mlp[-1].bias.fill_(.2)
    expected = policy(obs).detach()
    action = policy(obs, stochastic_output=True)
    torch.testing.assert_close(policy.output_mean, expected)
    assert torch.isfinite(policy.get_output_log_prob(action)).all()
    assert torch.isfinite(policy.output_entropy).all()
    assert not torch.equal(action, expected)  # Prefix exploration is explicitly NOT frozen.


def test_phase_gate_is_continuous_and_not_a_hidden_lowpass():
    obs = observations(3)
    obs['actor'][:] = obs['actor'][0].clone()
    obs['actor'][:, 49] = torch.tensor([0., .5, 1.])
    policy = actor(obs)
    with torch.no_grad():
        policy.mlp[-1].bias.fill_(.2)
        expected = policy.public(obs['actor']) + torch.tensor([0., .1, .2])[:, None]
    torch.testing.assert_close(policy(obs), expected)


def test_standard_runner_onnx_and_jit_keep_public_normalizer_and_phase_gate(tmp_path):
    import onnxruntime as ort
    from rsl_rl.runners import OnPolicyRunner
    obs = observations(6)
    policy = actor(obs)
    policy.update_normalization(obs)
    with torch.no_grad():
        policy.mlp[-1].bias.fill_(.13)
    policy.eval()
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: policy))
    OnPolicyRunner.export_policy_to_onnx(runner, str(tmp_path), 'policy.onnx')
    OnPolicyRunner.export_policy_to_jit(runner, str(tmp_path), 'policy.pt')
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    exported = ort.InferenceSession(str(tmp_path/'policy.onnx'), sess_options=options,
                                    providers=['CPUExecutionProvider'])
    jit = torch.jit.load(str(tmp_path/'policy.pt'))
    with torch.no_grad():
        expected = policy(obs).numpy()
        rows = [exported.run(None, {'obs': row[None].numpy()})[0] for row in obs['actor']]
        np.testing.assert_allclose(np.concatenate(rows), expected, atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(jit(obs['actor']), torch.from_numpy(expected))
    assert exported.get_inputs()[0].shape == [1, 61]
    assert exported.get_outputs()[0].shape == [1, 14]
