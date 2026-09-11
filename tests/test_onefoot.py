"""CPU regression tests for fixed-left glide, phase reset and reward exploits."""
from types import SimpleNamespace
import pytest
import torch
import mjlab.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_env_cfg import make_onefoot_env_cfg


def scores(**changes):
    values = dict(speed=.3,lateral=0.,yaw=0.,cross_track=0.,upright=1.,
                  left_contact=True,right_contact=False,clearance=.025,
                  blend=1.,dwell=2.,balance_error=0.,quiet=1.,forbidden=False)
    values.update(changes)
    return mdp.onefoot_scores_from_values(**{k:torch.tensor([v]) for k,v in values.items()})


def test_good_glide_scores_one():
    assert scores()['glide'].item() == 1
    assert scores()['accelerate'].item() == 0


@pytest.mark.parametrize('changes',[
    {'speed':0.},{'speed':-.2},{'left_contact':False},
    {'right_contact':True},{'clearance':.005},{'forbidden':True},
    {'upright':.4},{'blend':0.},{'speed':float('nan')},
])
def test_glide_exploits_do_not_pay(changes):
    assert scores(**changes)['glide'].item() == 0


def test_dwell_preferred_over_flutter():
    assert scores(dwell=2.)['glide'] > scores(dwell=.02)['glide']


def test_veering_and_lateral_skid_reduce_reward():
    for key in ('yaw','lateral','cross_track'):
        assert scores(**{key:.4})['glide'] < scores()['glide']


def test_lift_shaping_requires_motion_and_contact():
    for changes in ({'speed':0.},{'left_contact':False,'right_contact':False},{'forbidden':True}):
        assert scores(**changes)['shape'].item() == 0


def test_task_keeps_physics_but_removes_conflicting_stride_terms():
    cfg = make_onefoot_env_cfg()
    assert cfg.events['reset_base'].params['velocity_range']['x'] == (0,0)
    assert 'push_robot' not in cfg.events
    assert 'expand_bam_friction_fields' in cfg.events
    assert 'nan_state' in cfg.terminations
    assert 'body_ground' in cfg.terminations
    assert not cfg.curriculum
    assert not {'gait_symmetry','skating_air_time','single_support','wheel_speed','pose'} & cfg.rewards.keys()
    assert cfg.metrics['onefoot/success'].reduce == 'last'
    assert cfg.metrics['onefoot/best_dwell'].reduce == 'last'
    assert make_onefoot_env_cfg(assisted=True).events['reset_base'].params['velocity_range']['x'][0] > 0


def test_command_reset_and_monotonic_phase():
    data=SimpleNamespace(root_link_pos_w=torch.zeros(2,3),heading_w=torch.zeros(2))
    env=SimpleNamespace(num_envs=2,device='cpu',scene={'robot':SimpleNamespace(data=data)})
    cfg=mdp.OneFootGlideCommandCfg(resampling_time_range=(1000,1000),acceleration_s=.6,lift_s=.75)
    c=cfg.build(env)
    c.reset(torch.arange(2)); c.compute(0)
    assert torch.all(c.command[:,1] == 0)
    for _ in range(100): c.compute(.02)
    assert torch.all(c.command[:,1] == 1)
    c.dwell[:]=2; c.best_dwell[:]=3
    c.reset(torch.tensor([0])); c.compute(.02)
    assert c.command[0,1] == 0 and c.command[1,1] == 1
    assert c.dwell[0] == 0 and c.best_dwell[0] == 0
    assert c.best_dwell[1] == 3
    # The timer clamps to glide; no cyclic return to accelerate.
    c.compute(20)
    assert c.command[1,1] == 1


def test_command_encodes_cross_track_correction():
    data=SimpleNamespace(root_link_pos_w=torch.zeros(1,3),heading_w=torch.zeros(1))
    env=SimpleNamespace(num_envs=1,device='cpu',scene={'robot':SimpleNamespace(data=data)})
    c=mdp.OneFootGlideCommandCfg(resampling_time_range=(1000,1000)).build(env)
    c.compute(0); data.root_link_pos_w[0,1]=.1; c.compute(.02)
    assert c.command[0,2] < 0
