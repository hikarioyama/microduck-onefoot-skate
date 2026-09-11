"""Behavior-preserving cache and CPU-playback regressions (CPU-only)."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import torch
from mjlab_microduck.tasks import mdp


def test_name_resolved_index_tensors_preserve_order():
    names=('tire','tire_2','tire_3','tire_4')
    robot=SimpleNamespace(find_bodies=lambda name:([names.index(name)],None),
                          find_joints=lambda _:([3,1],None))
    env=SimpleNamespace(num_envs=2,device='cpu',scene={'robot':robot})
    cmd=mdp.CurriculumOneFootCommandCfg(resampling_time_range=(1000,1000)).build(env)
    assert cmd.wheel_ids==[0,1,2,3]
    assert cmd.wheel_ids_tensor.tolist()==cmd.wheel_ids
    assert cmd.head_ids_tensor.tolist()==cmd.head_ids==[3,1]
    assert cmd.wheel_ids_tensor.dtype==torch.long
    data=torch.randn(2,4,3)
    assert torch.equal(data[:,cmd.wheel_ids],data[:,cmd.wheel_ids_tensor])


def test_target_cache_matches_eager_and_invalidates_on_commands_and_states(monkeypatch):
    values=dict(waist_capture_signed=torch.tensor([0.,-.02,.03]),
                waist_com_height=torch.tensor([.15,.16,.17]),
                waist_roll_radians=torch.tensor([-.25,-.30,-.10]),
                hold=torch.tensor([1.,.8,0.]),transfer=torch.tensor([.2,0.,.4]),
                waist_support=torch.tensor([.5,.2,0.]))
    cmd=SimpleNamespace(command=torch.tensor([[.3,1.,0.]]*3))
    env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:cmd))
    monkeypatch.setattr(mdp,'_waist_onefoot_values',lambda _:values)
    original=mdp.waist_support_roll_target;calls=[]
    def target(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(mdp,'waist_support_roll_target',target)
    def verify():
        t=original(cmd.command[:,1],values['waist_capture_signed'],values['waist_com_height'])
        e=values['waist_roll_radians']-t;e=torch.atan2(torch.sin(e),torch.cos(e))
        match=torch.exp(-(e/.20).square())
        expected={'hold':values['hold']*(.35+.65*match),'transfer':values['transfer']*(.35+.65*match),
                  'waist_support':values['waist_support']*match,'waist_target_roll_deg':t*180/torch.pi,
                  'waist_roll_error_deg':e.abs()*180/torch.pi,'waist_roll_match':match}
        for key,result in expected.items():assert torch.equal(mdp.waist_target_onefoot_signal(env,key),result)
    verify();assert len(calls)==1
    cmd.command[:,1]=.5
    verify();assert len(calls)==2
    cmd.command=cmd.command.clone()
    verify();assert len(calls)==3
    values=dict(values,waist_roll_radians=values['waist_roll_radians']+.1)
    verify();assert len(calls)==4
    with torch.inference_mode():cmd.command=cmd.command.clone()
    verify();assert len(calls)==10  # No unsupported inference-version access.


def test_cpu_policy_loader_does_not_initialize_cuda(monkeypatch,tmp_path):
    local=Path(__file__).resolve().parents[1]/'local'
    monkeypatch.syspath_prepend(str(local))
    spec=importlib.util.spec_from_file_location('performance_viewer_test',local/'play-latest-onefoot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checkpoint=tmp_path/'model_1.pt';checkpoint.write_bytes(b'test checkpoint placeholder')
    cp=SimpleNamespace(path=checkpoint,fingerprint=module.fingerprint(checkpoint))
    seen=[]
    class Runner:
        def __init__(self,env,cfg,device):seen.append(('runner',device))
        def get_inference_policy(self,device):
            seen.append(('actor',device))
            return SimpleNamespace(is_recurrent=False,
                                   state_dict=lambda:{'weight':torch.zeros(2)})
    def load(*args,**kwargs):seen.append(('map',kwargs['map_location']))
    def reject_cuda(*args,**kwargs):raise AssertionError('CPU playback initialized CUDA')
    monkeypatch.setattr(module,'safe_runner_load',load)
    monkeypatch.setattr(torch.cuda,'init',reject_cuda)
    from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import make_waist_onefoot_rl_cfg
    loader=module.PolicyLoader(SimpleNamespace(device='cpu'),make_waist_onefoot_rl_cfg(),Runner)
    loader.load(cp)
    assert seen==[('runner','cpu'),('map','cpu'),('actor','cpu')]
    assert loader.loaded is cp
