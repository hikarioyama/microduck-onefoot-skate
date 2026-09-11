from types import SimpleNamespace
import importlib.util
from pathlib import Path
import pytest
import torch
import mjlab.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_dynamic_env_cfg import make_dynamic_onefoot_env_cfg


def t(x): return torch.tensor([x],dtype=torch.float32)

def score(**updates):
    args=dict(speed=.3,lateral=0.,yaw=0.,cross_track=0.,upright=1.,
              left_contact=True,right_contact=False,clearance=.03,blend=1.,
              dwell=2.,capture_error=0.,joint_speed_sq=0.,forbidden=False)
    args.update(updates)
    return mdp.dynamic_onefoot_scores(**{k:torch.tensor([v]) for k,v in args.items()})


def steering(yaw=-.25,error=-.03,blend=.5,clearance=.03,left=True,right=False,motion=1.):
    return mdp.onefoot_steering_scores(t(yaw),t(error),t(blend),t(clearance),
                                      torch.tensor([left]),torch.tensor([right]),t(motion))


def test_capture_point_accounts_for_velocity_direction():
    toward=mdp.onefoot_capture_error(t(0),t(.04),t(.2),t(0),t(.15))
    away=mdp.onefoot_capture_error(t(0),t(.04),t(-.2),t(0),t(.15))
    assert toward.abs()<away.abs()
    a=mdp.onefoot_capture_error(t(0),t(.04),t(.2),t(.1),t(.15))
    b=mdp.onefoot_capture_error(t(0),t(.04),t(.1),t(0),t(.15))
    assert torch.allclose(a,b)


def test_head_and_leg_motion_is_free_during_transfer_but_settles_after():
    assert score(blend=.5,joint_speed_sq=100)['transfer']==score(blend=.5,joint_speed_sq=0)['transfer']
    assert score(joint_speed_sq=100)['settle']<score(joint_speed_sq=0)['settle']
    cfg=make_dynamic_onefoot_env_cfg()
    assert 'neck_joint_pos_l2' not in cfg.rewards
    assert 'angular_momentum' not in cfg.rewards
    assert cfg.rewards['failure'].weight<0
    assert 'nan_state' in cfg.terminations and 'push_robot' not in cfg.events
    assert cfg.events['reset_base'].params['velocity_range']['x']==(0,0)


@pytest.mark.parametrize('updates',[{'speed':0.},{'speed':-.3},
    {'left_contact':False,'right_contact':False},{'forbidden':True},
    {'upright':.5},{'capture_error':float('nan')}])
def test_no_fake_positive_reward(updates):
    assert all(v.item()==0 for v in score(**updates).values())


def test_no_support_switch_or_two_foot_glide():
    for args in ({'right_contact':True},{'left_contact':False,'right_contact':True}):
        for name in ('glide','capture_hold','settle'): assert score(**args)[name]==0


def test_left_skate_steers_inward_then_returns_straight():
    assert steering()['steering_target']<0
    assert steering(yaw=-.25)['steer_inward']>steering(yaw=.25)['steer_inward']
    assert steering(error=0)['steering_target']==0
    assert steering(error=-1)['steering_target']>=-.350001
    assert steering(blend=1)['steer_inward']==0
    assert steering(blend=1,yaw=0)['blade_straight']>steering(blend=1,yaw=-.3)['blade_straight']


def test_steering_in_air_or_at_rest_does_not_pay():
    for args in ({'left':False},{'motion':0.}):
        assert steering(**args)['steer_inward']==0
    assert steering(blend=1,right=True)['blade_straight']==0


def test_late_glide_preferred_to_transition_shaping():
    assert score(blend=.5)['glide']==0
    assert score(blend=1)['transfer']==0
    assert score(dwell=2)['glide']>score(dwell=.02)['glide']


def test_motion_cost_is_relaxed_only_during_transition():
    env=SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _:torch.tensor([[.3,.5,0],[.3,1.,0]])),
        action_manager=SimpleNamespace(action=torch.ones(2,14),prev_action=torch.zeros(2,14)))
    env.scene={'robot':SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=torch.ones(2,3)))}
    for kind in ('angular','action_rate'):
        cost=mdp.dynamic_onefoot_motion_cost(env,kind)
        assert torch.allclose(cost[0]*20,cost[1])


def test_warm_start_resets_exploration_only():
    spec=importlib.util.spec_from_file_location('dynamic_train',Path(__file__).parents[1]/'local/train-dynamic-onefoot.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    previous={'mlp.weight':t(1),'obs_normalizer.mean':t(2),'distribution.std':t(.2)}
    current={k:t(9) for k in previous}
    result=module.warm_actor_state(previous,current)
    assert result['mlp.weight']==1 and result['obs_normalizer.mean']==2 and result['distribution.std']==9


def test_dashboard_no_hardware_telemetry():
    root=Path(__file__).parents[1]/'local'
    html=(root/'onefoot-dashboard.html').read_text()
    backend=(root/'onefoot-dashboard.py').read_text()
    for text in (html,backend):
        for forbidden in ('NVML','nvidia','fan_service','current_pct','temp_c','RTX 5070'):
            assert forbidden not in text
    for forbidden in ('温度','ファン','GPU情報'):
        assert forbidden not in html


def test_reset_seed_probabilities_and_source_separation():
    train=make_dynamic_onefoot_env_cfg(assisted=True)
    play=make_dynamic_onefoot_env_cfg(play=True,assisted=True)
    assert train.events['onefoot_seed'].params['probability']==.70
    assert play.events['onefoot_seed'].params['probability']==0
    for name in ('launch_fraction','launch_success','launch_best_dwell','seeded_start'):
        assert train.metrics['onefoot/'+name].reduce=='last'


def test_seeded_command_phase_and_velocity_history_reset():
    # One rollout starts from launch; the other is an explicitly assisted tail.
    inner=SimpleNamespace(subtree_com=torch.tensor([[[0.,.01,.15]],[[0.,.02,.15]]]))
    data=SimpleNamespace(root_link_pos_w=torch.zeros(2,3),heading_w=torch.zeros(2),
        data=inner,indexing=SimpleNamespace(root_body_id=0),body_link_pos_w=torch.zeros(2,2,3))
    robot=SimpleNamespace(data=data,find_bodies=lambda n:([0 if n=='tire' else 1],None))
    env=SimpleNamespace(num_envs=2,device='cpu',scene={'robot':robot},
                        _onefoot_seeded=torch.tensor([False,True]))
    cmd=mdp.DynamicOneFootCommandCfg(resampling_time_range=(1000,1000),acceleration_s=.6,lift_s=1).build(env)
    cmd.reset(torch.arange(2)); cmd.compute(0)
    assert cmd.command[0,1]==0 and cmd.command[1,1]==1
    assert torch.allclose(cmd.previous_com_y,torch.tensor([.01,.02]))
    cmd.dwell[:]=1; cmd.best_dwell[:]=2; cmd.compute(.1)
    env._onefoot_seeded[1]=False
    inner.subtree_com[1,0,1]=.03
    cmd.reset(torch.tensor([1])); cmd.compute(0)
    assert cmd.command[1,1]==0 and cmd.previous_com_y[1]==.03
    assert cmd.dwell[1]==0 and cmd.best_dwell[1]==0 and cmd.best_dwell[0]==2


def test_dashboard_uses_launch_only_denominator(tmp_path):
    import json
    module_path=Path(__file__).parents[1]/'local/onefoot-dashboard.py'
    spec=importlib.util.spec_from_file_location('dynamic_dashboard',module_path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.BASE=tmp_path
    (tmp_path/'onefoot-active.json').write_text(json.dumps({'log_file':'train.log','label':'test'}))
    (tmp_path/'train.log').write_text('''Learning iteration 10/1500
Episode_Metrics/onefoot/launch_fraction: 0.3
Episode_Metrics/onefoot/launch_success: 0.03
Episode_Metrics/onefoot/launch_best_dwell: 0.06
Episode_Metrics/onefoot/best_dwell: 1.2
Episode_Metrics/onefoot/success: 0.7
''')
    result=module.status()
    assert result['success']==pytest.approx(.1)
    assert result['dwell']==pytest.approx(.2)
    assert result['practice_dwell']==1.2 and result['practice_success']==.7
    assert not {'gpu','fans','fan_service','temperature'} & result.keys()


def test_old_dashboard_url_only_redirects_to_standard_viewer():
    root=Path(__file__).parents[1]/'local'
    html=(root/'onefoot-dashboard.html').read_text()
    assert "viewer.port = '8086'" in html and 'location.replace' in html
    for removed in ('<header','<iframe','class="card"','/status.json'):
        assert removed not in html
    launcher=(root/'play-standard-loopback.py').read_text()
    assert 'play.main()' in launcher and 'viser_server=server' in launcher
    assert "host='127.0.0.1'" in launcher
    for custom_ui in ('add_markdown','add_checkbox','add_icosphere','class LivePolicy'):
        assert custom_ui not in launcher
