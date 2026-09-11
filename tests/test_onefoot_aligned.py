from types import SimpleNamespace
import pytest
import torch
import mjlab.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_aligned_env_cfg import make_aligned_onefoot_env_cfg


def test_alignment_is_active_before_single_support_and_at_rest():
    # No right_contact, clearance, or speed argument is allowed to hide the cost.
    angle=torch.tensor([.44,0.])
    cost,target=mdp.onefoot_v4_alignment_values(angle,torch.tensor([-.3,-.3]),torch.ones(2),torch.ones(2,dtype=torch.bool))
    assert cost[0]>1 and cost[1]==0
    assert torch.all(target==0)


def test_inward_transfer_then_straight_without_conflicting_target():
    phase=torch.tensor([0.,.5,1.])
    _,target=mdp.onefoot_v4_alignment_values(torch.zeros(3),torch.full((3,),-.3),phase,torch.ones(3,dtype=torch.bool))
    assert target[0]==0 and target[1]==pytest.approx(-.3) and target[2]==0


def test_no_alignment_blocker_during_initial_propulsion():
    cost,_=mdp.onefoot_v4_alignment_values(torch.tensor([.5]),torch.tensor([-.3]),torch.tensor([0.]),torch.tensor([True]))
    assert cost==0


def env_with_state(left,right,elapsed=3.,dwell=0.):
    cmd=SimpleNamespace(elapsed=torch.tensor([elapsed]),best_dwell=torch.tensor([dwell]),
                        cfg=SimpleNamespace(acceleration_s=.6,lift_s=1.))
    contact=lambda x:SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[[x],[x]]])))
    return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:cmd),
        scene={'onefoot_left':contact(left),'onefoot_right':contact(right)})


def test_stalled_double_support_is_a_failed_attempt_not_timeout():
    assert mdp.onefoot_v4_missed_transfer(env_with_state(True,True)).item()
    assert not mdp.onefoot_v4_missed_transfer(env_with_state(True,True,elapsed=1.)).item()
    assert not mdp.onefoot_v4_missed_transfer(env_with_state(True,False)).item()
    assert not mdp.onefoot_v4_missed_transfer(env_with_state(True,True,dwell=2.)).item()
    cfg=make_aligned_onefoot_env_cfg()
    assert cfg.terminations['missed_transfer'].time_out is False
    assert cfg.rewards['support_alignment'].weight<0


def test_failed_timeout_cannot_escape_failure_cost():
    env=env_with_state(True,True,dwell=0.)
    env.termination_manager=SimpleNamespace(terminated=torch.tensor([False]),time_outs=torch.tensor([True]))
    assert mdp.onefoot_v4_failure(env)==1
    env.command_manager.get_term('twist').best_dwell[:]=2
    assert mdp.onefoot_v4_failure(env)==0


def test_post_transfer_two_foot_progress_is_zero(monkeypatch):
    values=dict(progress=torch.ones(2),clearance=torch.tensor([0.,.03]))
    monkeypatch.setattr(mdp,'_dynamic_onefoot_values',lambda _:values)
    left=SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[[True],[True]],[[True],[True]]])))
    right=SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[[True],[True]],[[False],[False]]])))
    env=SimpleNamespace(scene={'onefoot_left':left,'onefoot_right':right},
        command_manager=SimpleNamespace(get_command=lambda _:torch.tensor([[.3,1.,0.],[.3,1.,0.]])))
    result=mdp.onefoot_v4_progress(env)
    assert result[0]==0 and result[1]==1
