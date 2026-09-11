from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.utils import split_and_pad_trajectories,resolve_callable
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab_microduck.history_policy import HistoryGRUModel,warm_history_from_mlp


def models(batch=3):
    torch.manual_seed(411)
    obs=TensorDict({'actor':torch.zeros(batch,61)},batch_size=[batch])
    common=dict(obs=obs,obs_groups={'actor':['actor']},obs_set='actor',output_dim=14,
        hidden_dims=(32,24),activation='elu',obs_normalization=True)
    dist=lambda:dict(class_name='GaussianDistribution',init_std=.1,std_type='scalar')
    mlp=MLPModel(**common,distribution_cfg=dist())
    gru=HistoryGRUModel(**common,distribution_cfg=dist(),rnn_type='gru',rnn_hidden_dim=16)
    mlp.update_normalization(TensorDict({'actor':torch.randn(100,61)*2+.4},batch_size=[100]))
    with torch.no_grad():mlp.distribution.std_param.copy_(torch.linspace(.01,.24,14))
    source={name:value.clone() for name,value in mlp.state_dict().items()}
    target={name:value.clone() for name,value in gru.state_dict().items()}
    gru.load_state_dict(warm_history_from_mlp(source,target))
    assert all(torch.equal(mlp.state_dict()[k],v) for k,v in source.items())
    mlp.eval();gru.eval();return mlp,gru


def test_current_path_preserves_teacher_mean_normalizer_and_quiet_exploration():
    mlp,gru=models()
    assert gru.obs_dim==61 and gru.mlp[0].in_features==77
    assert torch.equal(mlp.distribution.std_param,gru.distribution.std_param)
    with torch.no_grad():
        for step in range(25):
            obs=TensorDict({'actor':torch.randn(3,61)*4},batch_size=[3])
            torch.testing.assert_close(gru(obs),mlp(obs),rtol=1e-6,atol=1e-6)
            if step==8:gru.reset(torch.tensor([False,True,False]))
    assert torch.count_nonzero(gru.mlp[0].weight[:,61:])==0


def test_memory_can_affect_actions_and_resets_per_environment():
    _,gru=models()
    with torch.no_grad():
        gru.mlp[0].weight[:,61:].normal_(0,.1)
        gru(TensorDict({'actor':torch.randn(3,61)},batch_size=[3]))
        saved=gru.get_hidden_state().clone()
        gru.reset(torch.tensor([False,True,False]))
        assert not gru.get_hidden_state()[:,1].any()
        assert torch.equal(gru.get_hidden_state()[:,0],saved[:,0])
        current=TensorDict({'actor':torch.zeros(3,61)},batch_size=[3])
        with_history=gru(current).clone();gru.reset();without_history=gru(current)
        assert not torch.allclose(with_history,without_history,atol=1e-6)


def test_recurrent_padded_training_matches_stepwise_current_and_memory_paths():
    _,gru=models();reference=deepcopy(gru)
    with torch.no_grad():gru.mlp[0].weight[:,61:].normal_(0,.1)
    reference.load_state_dict(gru.state_dict())
    x=torch.randn(6,3,61);dones=torch.zeros(6,3,dtype=torch.bool)
    dones[1,0]=True;dones[3,1]=True;dones[-1]=True
    padded,masks=split_and_pad_trajectories(x,dones)
    h=torch.zeros(1,padded.shape[1],16)
    with torch.no_grad():
        expected=[]
        for t in range(len(x)):
            expected.append(reference(TensorDict({'actor':x[t]},batch_size=[3])))
            reference.reset(dones[t])
        actual=gru(TensorDict({'actor':padded},batch_size=list(padded.shape[:2])),masks=masks,hidden_state=h)
    torch.testing.assert_close(actual,torch.stack(expected),rtol=1e-5,atol=1e-6)


def test_memory_path_has_gradients_after_zero_influence_warmstart():
    _,gru=models()
    opt=torch.optim.Adam(gru.parameters(),lr=1e-3)
    x=TensorDict({'actor':torch.randn(3,61)},batch_size=[3])
    for step in range(2):
        gru.reset();opt.zero_grad();loss=(gru(x)-torch.ones(3,14)).square().mean();loss.backward()
        assert gru.mlp[0].weight.grad[:,61:].abs().sum()>0
        if step==1:assert gru.rnn.rnn.weight_ih_l0.grad.abs().sum()>0
        opt.step()


def test_canonical_export_bakes_normalization_and_carries_learned_history(tmp_path):
    import onnx
    import onnxruntime as ort
    _,gru=models(batch=1)
    with torch.no_grad():gru.mlp[0].weight[:,61:].normal_(0,.1)
    runner=SimpleNamespace(alg=SimpleNamespace(get_policy=lambda:gru))
    MjlabOnPolicyRunner.export_policy_to_onnx(runner,str(tmp_path),'history.onnx')
    path=tmp_path/'history.onnx';onnx.checker.check_model(onnx.load(str(path)))
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    assert [x.name for x in session.get_inputs()]==['obs','h_in']
    assert [x.name for x in session.get_outputs()]==['actions','h_out']
    jit=torch.jit.script(gru.as_jit());h=np.zeros((1,1,16),dtype=np.float32)
    with torch.no_grad():
        for step in range(12):
            if step==6:gru.reset();jit.reset();h[:]=0
            x=torch.randn(1,61)*2+.4
            expected=gru(TensorDict({'actor':x},batch_size=[1])).numpy()
            actual,h=session.run(None,{'obs':x.numpy(),'h_in':h})
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
            np.testing.assert_allclose(h,gru.get_hidden_state().numpy(),rtol=1e-4,atol=1e-5)
            np.testing.assert_allclose(jit(x).numpy(),expected,rtol=1e-4,atol=1e-5)


def test_rejects_incompatible_warmstarts_without_mutating_source():
    mlp,gru=models();source=mlp.state_dict();target=gru.state_dict()
    before=deepcopy(source);warm_history_from_mlp(source,target)
    assert all(torch.equal(source[k],v) for k,v in before.items())
    with pytest.raises(ValueError):warm_history_from_mlp(target,target)
    bad=deepcopy(target);bad['unrelated.weight']=torch.zeros(1)
    with pytest.raises(ValueError):warm_history_from_mlp(source,bad)


def test_history_recipe_changes_only_actor_architecture(monkeypatch):
    import mjlab.tasks
    from mjlab_microduck.tasks import microduck_onefoot_history_env_cfg as history_cfg
    from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import make_waist_onefoot_env_cfg,make_waist_onefoot_rl_cfg
    from mjlab.tasks.registry import load_env_cfg
    template=make_waist_onefoot_env_cfg(stage='balance-100')
    monkeypatch.setattr(history_cfg,'make_waist_onefoot_env_cfg',lambda **kwargs:deepcopy(template))
    assert history_cfg.make_history_onefoot_env_cfg(stage='balance-100')==template
    a=history_cfg.make_history_onefoot_rl_cfg();b=make_waist_onefoot_rl_cfg()
    assert resolve_callable(a.actor.class_name) is HistoryGRUModel
    assert a.actor.rnn_type=='gru' and a.actor.rnn_hidden_dim==128
    a.actor=deepcopy(b.actor);a.run_name=b.run_name
    assert a==b
    cfg=load_env_cfg(history_cfg.TASK)
    assert cfg.rewards['curriculum_hold'].func.__name__=='waist_target_onefoot_signal'
    assert not any('support_knee' in name for name in cfg.metrics)
