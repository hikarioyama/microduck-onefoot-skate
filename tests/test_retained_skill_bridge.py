from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab_microduck.public_residual_policy import PUBLIC_PATH
from mjlab_microduck.retained_skill_bridge import (
    HOLD_AUTHORITY, ONEFOOT_PATH, FrozenOnefoot, RetainedSkillBridgeModel, retained_skills_digest,
)

pytestmark = pytest.mark.skipif(not PUBLIC_PATH.exists() or not ONEFOOT_PATH.exists(),
                               reason='Retained skill assets are local experiment artifacts')


def observations(n=16):
    raw = torch.randn(n, 61, generator=torch.Generator().manual_seed(7))*.1
    raw[:, 5] = -1.
    raw[:, 48] = .3
    raw[:, 49] = torch.linspace(0., 1., n)
    raw[:, 51:] = 0.
    return TensorDict({'actor': raw}, batch_size=[n])


def make_actor(obs, cls=RetainedSkillBridgeModel, std_type='log'):
    return cls(obs=obs, obs_groups={'actor': ['actor']}, obs_set='actor', output_dim=14,
               hidden_dims=[512, 256, 128], activation='elu', obs_normalization=True,
               distribution_cfg={'class_name': 'rsl_rl.modules.distribution:GaussianDistribution',
                                 'std_type': std_type, 'init_std': .08})


def test_hold_expert_matches_original_evaluated_checkpoint_exactly():
    obs = observations()
    obs['actor'][:, 49] = 1.
    source = make_actor(obs, MLPModel, 'scalar')
    source.load_state_dict(torch.load(ONEFOOT_PATH, weights_only=True, map_location='cpu')['actor_state_dict'], strict=True)
    source.eval()
    frozen = FrozenOnefoot()
    with torch.no_grad():
        torch.testing.assert_close(frozen(obs['actor']), source(obs), atol=0, rtol=0)
    assert frozen.obs_normalizer.eps == source.obs_normalizer.eps
    assert all(not p.requires_grad for p in frozen.parameters())


def test_only_hold_phase_is_reencoded_for_onefoot_expert():
    raw = observations()['actor']
    frozen = FrozenOnefoot()
    with torch.no_grad():
        expected = raw.clone(); expected[:, 49] = 1.
        torch.testing.assert_close(frozen(raw), frozen.mlp(frozen.obs_normalizer(expected)), atol=0, rtol=0)
    # In particular last_action stays in the original scale=1 HOME-relative units.
    assert torch.equal(raw[:, 34:48], expected[:, 34:48])


def test_gradient_steps_preserve_both_expert_states_and_the_roller_endpoint():
    obs = observations()
    model = make_actor(obs)
    model.update_normalization(obs)
    before = retained_skills_digest(model)
    prefix = obs.clone(); prefix['actor'][:, 49] = 0.
    tail = obs.clone(); tail['actor'][:, 49] = 1.
    with torch.no_grad():
        prefix_expected = model.public(prefix['actor']).clone()
        tail_expected = model.onefoot(tail['actor']).clone()
        # The residual starts at exactly zero, so BOTH endpoints are the retained
        # experts bit for bit before the first update. That, and not a permanently
        # zero hold share, is what retaining the existing skill means here.
        torch.testing.assert_close(model(prefix), prefix_expected, atol=0, rtol=0)
        torch.testing.assert_close(model(tail), tail_expected, atol=0, rtol=0)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    target = model(obs).detach()+.2
    for _ in range(3):
        optimizer.zero_grad()
        (model(obs)-target).square().mean().backward()
        optimizer.step()
    assert retained_skills_digest(model) == before
    assert model.mlp[-1].weight.abs().max() > 0
    # phase=0 stays exactly the public roller policy: the gate is still zero there.
    torch.testing.assert_close(model(prefix), prefix_expected, atol=0, rtol=0)
    # The hold keeps exactly HOLD_AUTHORITY of the residual, over the untouched
    # frozen expert mean. A full share would let the residual, not the retained
    # skill, drive the hold.
    with torch.no_grad():
        correction = model.mlp(model.obs_normalizer(tail['actor']))
        torch.testing.assert_close(model(tail), tail_expected+HOLD_AUTHORITY*correction,
                                   atol=0, rtol=0)
    assert 0. < HOLD_AUTHORITY < .5
    assert all(p.grad is None for p in model.public.parameters())
    assert all(p.grad is None for p in model.onefoot.parameters())


def test_ppo_distribution_uses_the_executed_composite_mean():
    obs = observations()
    model = make_actor(obs)
    with torch.no_grad():
        model.mlp[-1].bias.fill_(.2)
        expected = model(obs).clone()
    actions = model(obs, stochastic_output=True)
    torch.testing.assert_close(model.output_mean, expected)
    assert torch.isfinite(model.get_output_log_prob(actions)).all()


def test_standard_exports_include_both_skills_and_bridge(tmp_path):
    import onnxruntime as ort
    from rsl_rl.runners import OnPolicyRunner
    obs = observations(8)
    model = make_actor(obs)
    model.update_normalization(obs)
    with torch.no_grad():
        model.mlp[-1].bias.fill_(.21)
    model.eval()
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: model))
    OnPolicyRunner.export_policy_to_onnx(runner, str(tmp_path), 'policy.onnx')
    OnPolicyRunner.export_policy_to_jit(runner, str(tmp_path), 'policy.pt')
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(tmp_path/'policy.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    jit = torch.jit.load(str(tmp_path/'policy.pt'))
    with torch.no_grad():
        expected = model(obs)
        rows = [session.run(None, {'obs': row[None].numpy()})[0] for row in obs['actor']]
        np.testing.assert_allclose(np.concatenate(rows), expected.numpy(), atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(jit(obs['actor']), expected)
    assert session.get_inputs()[0].shape == [1, 61]
    assert session.get_outputs()[0].shape == [1, 14]
