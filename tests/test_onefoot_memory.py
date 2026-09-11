from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel,RNNModel
from mjlab.rl.runner import MjlabOnPolicyRunner
import mjlab.tasks
from mjlab_microduck.recurrent_warmstart import warm_gru_from_mlp
from mjlab_microduck.tasks.microduck_onefoot_memory_env_cfg import make_memory_onefoot_env_cfg,make_memory_onefoot_rl_cfg
from mjlab_microduck.tasks.microduck_onefoot_continuous_env_cfg import make_continuous_onefoot_env_cfg


def models(batch=3):
    torch.manual_seed(9)
    obs=TensorDict({'actor':torch.zeros(batch,61)},batch_size=[batch])
    kw=dict(obs=obs,obs_groups={'actor':['actor']},obs_set='actor',output_dim=14,
            hidden_dims=(32,24),activation='elu',obs_normalization=True)
    distribution=lambda:dict(class_name='GaussianDistribution',init_std=.1,std_type='scalar')
    mlp=MLPModel(**kw,distribution_cfg=distribution())
    gru=RNNModel(**kw,distribution_cfg=distribution(),rnn_type='gru',rnn_hidden_dim=64)
    gru.load_state_dict(warm_gru_from_mlp(mlp.state_dict(),gru.state_dict()))
    mlp.eval();gru.eval()
    return mlp,gru


def test_approximately_preserves_the_warm_policy():
    mlp,gru=models()
    with torch.no_grad():
        for _ in range(12):
            obs=TensorDict({'actor':torch.randn(3,61).clamp(-2,2)},batch_size=[3])
            torch.testing.assert_close(gru(obs),mlp(obs),atol=5e-4,rtol=5e-4)


def test_per_environment_hidden_reset_and_separate_evaluation_instance():
    _,gru=models()
    with torch.no_grad():gru(TensorDict({'actor':torch.randn(3,61)},batch_size=[3]))
    before=gru.get_hidden_state().clone()
    assert before.abs().sum()>0
    gru.reset(torch.tensor([False,True,False]))
    assert not gru.get_hidden_state()[:,1].any()
    assert torch.equal(gru.get_hidden_state()[:,0],before[:,0])
    saved=gru.get_hidden_state().clone()
    evaluation=deepcopy(gru);evaluation.reset()
    with torch.no_grad():evaluation(TensorDict({'actor':torch.randn(2,61)},batch_size=[2]))
    assert evaluation.get_hidden_state().shape==(1,2,64)
    assert torch.equal(gru.get_hidden_state(),saved)


def test_canonical_recurrent_onnx_matches_multistep_policy(tmp_path):
    import onnx
    import onnxruntime as ort
    _,gru=models(batch=1)
    # Use the same canonical exporter as train/play, not a hand conversion.
    runner=SimpleNamespace(alg=SimpleNamespace(get_policy=lambda:gru))
    MjlabOnPolicyRunner.export_policy_to_onnx(runner,str(tmp_path),'memory.onnx')
    path=tmp_path/'memory.onnx';onnx.checker.check_model(onnx.load(str(path)))
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    assert [x.name for x in session.get_inputs()]==['obs','h_in']
    assert [x.name for x in session.get_outputs()]==['actions','h_out']
    h=np.zeros((1,1,64),dtype=np.float32)
    with torch.no_grad():
        for step in range(8):
            if step==4:gru.reset();h[:]=0
            x=torch.randn(1,61).clamp(-2,2)
            expected=gru(TensorDict({'actor':x},batch_size=[1])).numpy()
            actual,h=session.run(None,{'obs':x.numpy(),'h_in':h})
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
            np.testing.assert_allclose(h,gru.get_hidden_state().numpy(),rtol=1e-4,atol=1e-4)


def test_memory_changes_neither_sensors_nor_physical_success_criteria():
    a=make_memory_onefoot_env_cfg(stage='balance-100')
    b=make_continuous_onefoot_env_cfg(stage='balance-100')
    assert a.observations==b.observations and a.actions==b.actions
    assert a.events==b.events and a.terminations==b.terminations
    assert a.rewards==b.rewards and a.metrics==b.metrics
    cfg=make_memory_onefoot_rl_cfg()
    assert cfg.actor.class_name=='RNNModel' and cfg.actor.rnn_type=='gru'
    assert cfg.actor.rnn_hidden_dim==128 and cfg.actor.rnn_num_layers==1
    assert cfg.algorithm.symmetry_cfg is None
