from copy import deepcopy
from itertools import chain
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab_microduck.history_policy import SplitHistoryGRUModel,warm_split_history_from_mlp
from mjlab_microduck.matched_optimizer import named_policy_parameters,restore_matched_optimizer,sync_optimizer_learning_rate
from mjlab_microduck.checkpoint_safety import safe_runner_load


def algorithm(history=False):
    obs=TensorDict({'actor':torch.zeros(3,61),'critic':torch.zeros(3,78)},batch_size=[3])
    common=dict(obs=obs,obs_groups={'actor':['actor'],'critic':['critic']},hidden_dims=(32,24),obs_normalization=True)
    cls=SplitHistoryGRUModel if history else MLPModel
    extra=dict(rnn_type='gru',rnn_hidden_dim=16) if history else {}
    actor=cls(**common,obs_set='actor',output_dim=14,distribution_cfg=dict(class_name='GaussianDistribution',init_std=.1,std_type='scalar'),**extra)
    critic=MLPModel(**common,obs_set='critic',output_dim=1)
    opt=torch.optim.Adam(chain(actor.parameters(),critic.parameters()),lr=3e-5)
    return SimpleNamespace(actor=actor,critic=critic,optimizer=opt,learning_rate=1e-4)


def transferred():
    torch.manual_seed(532)
    source=algorithm();target=algorithm(True)
    for _ in range(3):
        for _,p in named_policy_parameters(source):p.grad=torch.full_like(p,.002)
        source.optimizer.step()
    target.actor.load_state_dict(warm_split_history_from_mlp(source.actor.state_dict(),target.actor.state_dict()))
    target.critic.load_state_dict(source.critic.state_dict())
    saved=deepcopy(source.optimizer.state_dict())
    proof=restore_matched_optimizer(target,saved,named_policy_parameters(source))
    return source,target,saved,proof


def test_all_existing_moments_steps_and_lr_survive_but_memory_state_is_fresh():
    source,target,saved,proof=transferred()
    assert target.learning_rate==3e-5 and target.optimizer.param_groups[0]['lr']==3e-5
    s=dict(named_policy_parameters(source));t=dict(named_policy_parameters(target))
    assert proof['restored_parameter_states']==len(s) and len(proof['new_parameters'])==5
    for name in s:
        torch.testing.assert_close(s[name],t[name],rtol=0,atol=0)
        for key,value in source.optimizer.state[s[name]].items():
            torch.testing.assert_close(value,target.optimizer.state[t[name]][key],rtol=0,atol=0)
    for name in set(t)-set(s):assert t[name] not in target.optimizer.state
    # Equal next gradients yield equal updates for every pre-existing parameter.
    for name in s:
        g=torch.randn_like(s[name])*.001;s[name].grad=g;t[name].grad=g.clone()
    source.optimizer.step();target.optimizer.step()
    for name in s:torch.testing.assert_close(s[name],t[name],rtol=0,atol=1e-8)
    assert saved['param_groups'][0]['lr']==3e-5


def test_split_projection_preserves_current_policy_and_learns_only_new_influence():
    source,target,_,_=transferred();a=source.actor;b=target.actor;a.eval();b.eval()
    assert b.mlp[0].weight.shape==a.mlp[0].weight.shape and b.mlp[0].history_weight.shape==(32,16)
    with torch.no_grad():
        for _ in range(12):
            obs=TensorDict({'actor':torch.randn(3,61)},batch_size=[3])
            torch.testing.assert_close(a(obs),b(obs),rtol=1e-6,atol=1e-6)
    b.reset();target.optimizer.zero_grad()
    b(TensorDict({'actor':torch.randn(3,61)},batch_size=[3])).square().sum().backward()
    assert b.mlp[0].history_weight.grad.abs().sum()>0
    assert not b.rnn.rnn.weight_ih_l0.grad.any()  # zero initial memory influence
    target.optimizer.step();b.reset();target.optimizer.zero_grad()
    b(TensorDict({'actor':torch.randn(3,61)},batch_size=[3])).square().sum().backward()
    assert b.rnn.rnn.weight_ih_l0.grad.abs().sum()>0


def test_full_checkpoint_load_syncs_ppo_scalar_but_actor_only_load_does_not():
    source,target,saved,_=transferred()
    # Use a same-architecture control to reproduce the upstream restore behavior.
    control=algorithm()
    def load(path,**kwargs):
        cfg=kwargs.get('load_cfg')
        if cfg is None or cfg.get('optimizer'):control.optimizer.load_state_dict(saved)
        return {'loaded':True}
    runner=SimpleNamespace(alg=control,load=load)
    assert safe_runner_load(runner,'unused')['loaded']
    assert control.learning_rate==3e-5
    control.learning_rate=1e-4
    safe_runner_load(runner,'unused',load_cfg={'actor':True})
    assert control.learning_rate==1e-4


def test_optimizer_mapping_rejects_shape_and_group_mismatches():
    source,target,saved,_=transferred()
    broken=deepcopy(saved);key=next(iter(broken['state']))
    broken['state'][key]['exp_avg']=torch.zeros(99)
    with pytest.raises(ValueError):restore_matched_optimizer(target,broken,named_policy_parameters(source))
    broken=deepcopy(saved);broken['param_groups'].append(deepcopy(broken['param_groups'][0]))
    with pytest.raises(ValueError):restore_matched_optimizer(target,broken,named_policy_parameters(source))
    target.optimizer.param_groups[0]['lr']=float('nan')
    with pytest.raises(ValueError):sync_optimizer_learning_rate(target)


def test_canonical_split_export_has_correct_current_and_history_paths(tmp_path):
    import onnx
    import onnxruntime as ort
    _,target,_,_=transferred();model=target.actor;model.eval()
    with torch.no_grad():model.mlp[0].history_weight.normal_(0,.1)
    runner=SimpleNamespace(alg=SimpleNamespace(get_policy=lambda:model))
    MjlabOnPolicyRunner.export_policy_to_onnx(runner,str(tmp_path),'split.onnx')
    p=tmp_path/'split.onnx';onnx.checker.check_model(onnx.load(str(p)))
    session=ort.InferenceSession(str(p),providers=['CPUExecutionProvider'])
    jit=torch.jit.script(model.as_jit());h=np.zeros((1,1,16),dtype=np.float32);model.reset()
    with torch.no_grad():
        for step in range(10):
            if step==4:model.reset();jit.reset();h[:]=0
            x=torch.randn(1,61)
            expected=model(TensorDict({'actor':x},batch_size=[1])).numpy()
            actual,h=session.run(None,{'obs':x.numpy(),'h_in':h})
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
            np.testing.assert_allclose(jit(x).numpy(),expected,rtol=1e-4,atol=1e-5)


def test_matched_history_recipe_changes_only_actor_architecture(monkeypatch):
    import mjlab.tasks
    from rsl_rl.utils import resolve_callable
    from mjlab_microduck.tasks import microduck_onefoot_history_env_cfg as m
    from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import make_waist_onefoot_env_cfg,make_waist_onefoot_rl_cfg
    cfg=make_waist_onefoot_env_cfg(stage='balance-100')
    monkeypatch.setattr(m,'make_waist_onefoot_env_cfg',lambda **kwargs:deepcopy(cfg))
    assert m.make_matched_history_onefoot_env_cfg(stage='balance-100')==cfg
    a=m.make_matched_history_onefoot_rl_cfg();b=make_waist_onefoot_rl_cfg()
    assert resolve_callable(a.actor.class_name) is SplitHistoryGRUModel
    a.actor=b.actor;a.run_name=b.run_name;assert a==b
