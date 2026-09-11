from types import SimpleNamespace
import torch
import mjlab.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_continuous_env_cfg import make_continuous_onefoot_env_cfg
from mjlab_microduck.tasks.microduck_onefoot_centroid_env_cfg import make_centroid_onefoot_env_cfg


def test_foreaft_margin_is_soft_and_finite():
    x=torch.tensor([0.,.018,.0325,.056,-.056,float('nan')])
    factor=mdp.foreaft_onefoot_factor(x)
    assert factor[0]==1 and factor[1]==1
    assert 0<factor[3]<factor[2]<1
    assert factor[3]==factor[4] and factor[5]==0


def test_moving_support_velocity_is_subtracted():
    data=SimpleNamespace(data=SimpleNamespace(subtree_com=torch.tensor([[[0.,0.,.16]],[[.025,0.,.16]]])),
        indexing=SimpleNamespace(root_body_id=0,body_ids=[0,1]),
        model=SimpleNamespace(body_mass=torch.ones(2,2)),
        body_com_lin_vel_w=torch.tensor([[[.3,0.,0.],[.3,0.,0.]]]*2),
        body_link_pos_w=torch.tensor([[[.0325,0.,.015],[-.0325,0.,.015]]]*2),
        body_link_lin_vel_w=torch.tensor([[[.3,0.,0.],[.3,0.,0.]]]*2))
    cmd=SimpleNamespace(wheel_ids=[0,1,2,3])
    env=SimpleNamespace(num_envs=2,common_step_counter=1,
        command_manager=SimpleNamespace(get_term=lambda _:cmd),scene={'robot':SimpleNamespace(data=data),
        'onefoot_left':SimpleNamespace(data=SimpleNamespace(force=torch.tensor([[[0.,0.,-3.],[0.,0.,-3.]]]*2)))})
    result=mdp._centroid_onefoot_values(env)
    assert torch.allclose(result['foreaft_capture'],torch.tensor([0.,.025]))
    assert torch.all(result['front_load']==.5)
    assert mdp._centroid_onefoot_values(env) is result
    env.common_step_counter+=1
    data.body_com_lin_vel_w[:,:,0]+=.1
    assert (mdp._centroid_onefoot_values(env)['foreaft_capture']>result['foreaft_capture']).all()


def test_candidate_keeps_success_criteria_and_physics_unchanged():
    for stage in ('balance-100','unload','self-launch'):
        a=make_continuous_onefoot_env_cfg(stage=stage);b=make_centroid_onefoot_env_cfg(stage=stage)
        assert a.commands==b.commands and a.terminations==b.terminations
        assert a.events==b.events and a.actions==b.actions and a.observations==b.observations
        assert a.scene.entities==b.scene.entities and a.scene.sensors==b.scene.sensors
        assert all(b.metrics[k]==v for k,v in a.metrics.items())
        assert 'onefoot/foreaft_capture' in b.metrics
