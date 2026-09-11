from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel,RNNModel
from mjlab.rl import RslRlModelCfg
from mjlab_microduck.checkpoint_safety import safe_runner_load,evaluation_policy,verified_recovery
from mjlab_microduck.tasks.microduck_onefoot_curriculum_env_cfg import STAGES,gate_passes


def setup(recurrent=False,batch=3):
    obs=TensorDict({'actor':torch.randn(batch,61)},batch_size=[batch])
    cfg=RslRlModelCfg(hidden_dims=(32,24),obs_normalization=True,
        distribution_cfg=dict(class_name='GaussianDistribution',init_std=.1,std_type='scalar'))
    kw=dict(obs=obs,obs_groups={'actor':['actor']},obs_set='actor',output_dim=14,
        hidden_dims=cfg.hidden_dims,obs_normalization=True,distribution_cfg=deepcopy(cfg.distribution_cfg))
    if recurrent:
        cfg.rnn_type='gru';cfg.rnn_hidden_dim=64;cfg.class_name='RNNModel'
        model=RNNModel(**kw,rnn_type='gru',rnn_hidden_dim=64)
    else:model=MLPModel(**kw)
    return model,cfg,obs


@pytest.mark.parametrize('recurrent',[False,True])
def test_evaluation_works_after_gradient_distribution_cache(recurrent):
    model,cfg,obs=setup(recurrent)
    # A PPO update leaves a non-leaf Normal.loc tensor cached in the actor.
    out=model(obs,stochastic_output=True)
    loss=model.output_mean.square().mean();loss.backward()
    with pytest.raises(RuntimeError):deepcopy(model)
    original_hidden=model.get_hidden_state()
    if original_hidden is not None:original_hidden=original_hidden.detach().clone()
    random_state=torch.random.get_rng_state().clone()
    copy=evaluation_policy(model,obs,cfg,14)
    assert torch.equal(torch.random.get_rng_state(),random_state)
    assert model.training and not copy.training
    with torch.no_grad():
        result=copy(TensorDict({'actor':torch.randn(2,61)},batch_size=[2]))
    assert result.shape==(2,14)
    if recurrent:
        assert torch.equal(model.get_hidden_state(),original_hidden)
        assert copy.get_hidden_state().shape==(1,2,64)


def test_checkpoint_load_materializes_inference_buffers_not_optimizer_state(tmp_path):
    actor,_,obs=setup();critic,_,_=setup()
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=1e-3)
    actor(obs).sum().backward();optimizer.step();optimizer.zero_grad()
    path=tmp_path/'checkpoint.pt'
    torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict()),path)
    with torch.inference_mode():
        actor.obs_normalizer._std=torch.ones_like(actor.obs_normalizer._std)
        critic.obs_normalizer._std=torch.ones_like(critic.obs_normalizer._std)
    assert actor.obs_normalizer._std.is_inference()
    def load(path,**kwargs):
        state=torch.load(path,weights_only=False)
        actor.load_state_dict(state['actor']);critic.load_state_dict(state['critic'])
        optimizer.load_state_dict(state['optimizer'])
    runner=SimpleNamespace(alg=SimpleNamespace(actor=actor,critic=critic),load=load)
    with pytest.raises(RuntimeError):runner.load(path)
    safe_runner_load(runner,path)
    assert not actor.obs_normalizer._std.is_inference()
    assert not critic.obs_normalizer._std.is_inference()
    for values in optimizer.state.values():
        assert all(not v.is_inference() for v in values.values() if isinstance(v,torch.Tensor))
    actor(obs).square().mean().backward();optimizer.step()


def recovery_state():
    checkpoint='/run/balance-050/model_99.pt'
    reports=[dict(stage='balance-050',goal_s=.5,episodes=256,success_rate=.91,nan_episodes=0,
                  checkpoint=checkpoint,seed=seed) for seed in (10,11)]
    return dict(status='error',stage='balance-100',iteration=3299,best_checkpoint='/run/balance-100/model_2699.pt',
        passed_stages=[dict(stage='balance-050',checkpoint=checkpoint,reports=reports)])


def test_recovery_keeps_prior_gates_and_monotonic_iteration():
    s=recovery_state()
    assert verified_recovery(s,STAGES,gate_passes)==('balance-100','/run/balance-100/model_2699.pt',3300)
    s['passed_stages']=[]
    with pytest.raises(ValueError):verified_recovery(s,STAGES,gate_passes)
    s=recovery_state();s['passed_stages'][0]['reports'][1]['seed']=10
    with pytest.raises(ValueError):verified_recovery(s,STAGES,gate_passes)
    s=recovery_state();s['status']='training'
    with pytest.raises(ValueError):verified_recovery(s,STAGES,gate_passes)


def test_intentional_pause_can_resume_latest_without_rewinding_learning():
    s=recovery_state();s['status']='interrupted';s['checkpoint']='/run/balance-100/model_3299.pt'
    assert verified_recovery(s,STAGES,gate_passes,prefer_latest=True)==('balance-100',s['checkpoint'],3300)
    assert verified_recovery(s,STAGES,gate_passes)[1]==s['best_checkpoint']
    s['passed_stages']=[]
    with pytest.raises(ValueError):verified_recovery(s,STAGES,gate_passes,prefer_latest=True)
    s=recovery_state();s['checkpoint']='/run/balance-100/model_3299.pt'
    with pytest.raises(ValueError):verified_recovery(s,STAGES,gate_passes,prefer_latest=True)
