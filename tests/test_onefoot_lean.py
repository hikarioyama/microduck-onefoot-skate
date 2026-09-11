from types import SimpleNamespace
import math
import numpy as np
import torch
import pytest
import mjlab.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_lean_env_cfg import make_lean_onefoot_env_cfg
from mjlab_microduck.tasks.microduck_onefoot_continuous_env_cfg import make_continuous_onefoot_env_cfg


def test_lateral_lean_is_allowed_but_forward_pitch_and_falls_are_not_free():
    angles=torch.tensor([0.,15.,30.,35.])*torch.pi/180
    g=torch.stack((torch.zeros_like(angles),-torch.sin(angles),-torch.cos(angles)),1)
    old=((-g[:,2]-.65)/.30).clamp(0,1)
    assert old[2]<.73
    assert torch.allclose(mdp.waist_lean_posture_credit(g),torch.ones(4))
    forward=torch.tensor([[.5,0.,-math.sqrt(.75)]])
    assert mdp.waist_lean_posture_credit(forward)<.73
    assert mdp.waist_lean_posture_credit(torch.tensor([[0.,.94,-.34]]))==0


def test_alignment_not_angle_magnitude_is_rewarded():
    h=mdp.WAIST_COM_REFERENCE_BODY[2];target=torch.tensor([.006])
    helpful=h*math.sin(.28);harmful=-helpful
    scores=mdp.waist_axis_alignment(torch.tensor([0.,helpful,harmful]),target)
    assert scores[1]>scores[0]>scores[2]
    # If the pelvis is already over support, upright is better than gratuitous lean.
    scores=mdp.waist_axis_alignment(torch.tensor([0.,helpful]),torch.zeros(2))
    assert scores[0]==1 and scores[1]<1


def test_foreaft_penalty_uses_position_and_relative_momentum_symmetrically():
    x=torch.tensor([0.,.01,.04,-.04,0.])
    capture=torch.tensor([0.,.01,.06,-.06,.06])
    cost=mdp.waist_foreaft_support_cost(x,capture,torch.full((5,),.0325))
    assert cost[0]==0 and 0<cost[1]<cost[2]
    assert cost[2]==cost[3] and cost[4]>1
    assert torch.all((cost>=0)&(cost<=9))


def test_contact_center_uses_loaded_global_points_and_safe_fallback():
    p=torch.tensor([[[0.,.01,0.],[0.,.03,0.]],[[0.,.1,0.],[0.,.3,0.]]])
    force=torch.tensor([[-3.,-1.],[0.,0.]])
    found=torch.ones(2,2,dtype=torch.bool);fallback=torch.zeros(2,3)
    result=mdp.loaded_contact_center(p,force,found,fallback)
    assert result[0,1]==pytest.approx(.015) and torch.all(result[1]==0)


def test_nominal_pelvis_reference_is_measured_on_the_robot():
    import mujoco
    from pathlib import Path
    from mjlab_microduck.tasks.microduck_onefoot_curriculum_poses import POSES
    path=Path(__file__).parents[1]/'src/mjlab_microduck/robot/microduck/scene_rollers.xml'
    # The robot model and its meshes live in the upstream working tree; see
    # docs/REPRODUCING.md for the overlay layout that provides them.
    if not path.exists():pytest.skip('Robot model not in this checkout')
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,m.key('STAND').id)
    p=POSES['balance'];r=p['root_roll'];d.qpos[2]=p['root_z']
    d.qpos[3:7]=[math.cos(r/2),math.sin(r/2),0,0]
    for name,value in p['pose'].items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    mujoco.mj_forward(m,d);bid=m.body('trunk_base').id
    local=d.xmat[bid].reshape(3,3).T@(d.subtree_com[bid]-d.xpos[bid])
    np.testing.assert_allclose(local,mdp.WAIST_COM_REFERENCE_BODY,atol=1e-8)


def test_skill_gates_and_actual_physics_are_not_relaxed():
    for stage in ('balance-100','transfer','self-launch'):
        before=make_continuous_onefoot_env_cfg(stage=stage);after=make_lean_onefoot_env_cfg(stage=stage)
        assert before.commands==after.commands and before.terminations==after.terminations
        assert before.events==after.events and before.actions==after.actions
        assert before.observations==after.observations and before.scene.entities==after.scene.entities
        assert after.scene.sensors[:-1]==before.scene.sensors
        cop=after.scene.sensors[-1]
        assert cop.name=='onefoot_left_cop' and cop.reduce=='maxforce' and cop.global_frame
        assert {'pos','normal','tangent','force','found'}<=set(cop.fields)
        assert after.rewards['foreaft_balance'].weight<0


def test_no_waist_bonus_for_stationary_wrong_support_flight_or_fallen(monkeypatch):
    n=6;ones=torch.ones(n);zeros=torch.zeros(n)
    base=dict(left_contact=ones.clone(),right_contact=zeros.clone(),left_force=torch.full((n,),6.),
        speed=torch.tensor([.3,0.,.3,.3,-.2,.3]),lateral=zeros,heading_error=zeros,cross_track=zeros,
        clearance=torch.full((n,),.025),left_load=ones,dwell=ones,single_support=ones.clone())
    base['left_contact'][2:4]=0;base['right_contact'][2]=1
    base['single_support'][1:]=0
    monkeypatch.setattr(mdp,'_curriculum_onefoot_values',lambda _:base)
    monkeypatch.setattr(mdp,'onefoot_bad_contact',lambda _:torch.tensor([False]*5+[True]))
    wheels=torch.tensor([[[.0325,0.,.015],[-.0325,0.,.015],[.0325,-.08,.041],[-.0325,-.08,.041]]]*n)
    q=torch.zeros(n,4);q[:,0]=1
    data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.,0.,.145]]*n),root_link_quat_w=q,
        projected_gravity_b=torch.tensor([[0.,0.,-1.]]*n),root_link_ang_vel_b=torch.zeros(n,3),
        body_link_pos_w=wheels,body_link_lin_vel_w=torch.zeros(n,4,3),
        body_com_lin_vel_w=torch.tensor([[[.3,0.,0.]]]*n),model=SimpleNamespace(body_mass=torch.ones(n,1)),
        indexing=SimpleNamespace(root_body_id=0,body_ids=[0]),
        data=SimpleNamespace(subtree_com=torch.tensor([[[0.,0.,.17]]]*n)))
    found=torch.ones(n,2,1,dtype=torch.bool);found[2:4]=False
    force=torch.zeros(n,2,3);force[:,:,2]=3.;force[2:4]=0
    cop=SimpleNamespace(data=SimpleNamespace(pos=torch.zeros(n,2,3),force=force,found=found))
    cmd=SimpleNamespace(wheel_ids=[0,1,2,3],command=torch.tensor([[.3,1.,0.]]*n),cfg=SimpleNamespace(goal_s=1.))
    env=SimpleNamespace(num_envs=n,common_step_counter=1,
        command_manager=SimpleNamespace(get_term=lambda _:cmd),scene={'robot':SimpleNamespace(data=data),
        'onefoot_left':SimpleNamespace(data=SimpleNamespace(found=found,force=force)), 'onefoot_left_cop':cop})
    result=mdp.waist_onefoot_signal(env,'waist_support')
    assert result[0]>0 and torch.all(result[1:]==0)


def test_explicit_roll_target_prefers_support_side_but_allows_braking():
    from mjlab_microduck.tasks.microduck_onefoot_curriculum_poses import POSES
    assert mdp.WAIST_REFERENCE_ROLL==POSES['balance']['root_roll']
    phase=torch.ones(4);height=torch.full((4,),.15)
    target=mdp.waist_support_roll_target(phase,torch.tensor([0.,-.025,.025,.10]),height)
    assert target[1]<target[0]<target[2]<target[3]
    assert target[0]==pytest.approx(-.2771360081)
    assert target[3]>0  # COM escaped too far left: counter-lean is allowed.
    assert torch.all(mdp.waist_support_roll_target(torch.zeros(4),torch.zeros(4),height)==0)


def test_torso_target_does_not_pay_for_lean_without_real_support(monkeypatch):
    values=dict(waist_capture_signed=torch.zeros(3),waist_com_height=torch.full((3,),.15),
        waist_roll_radians=torch.tensor([mdp.WAIST_REFERENCE_ROLL,0.,mdp.WAIST_REFERENCE_ROLL]),
        hold=torch.tensor([1.,1.,0.]),waist_support=torch.tensor([1.,1.,0.]))
    monkeypatch.setattr(mdp,'_waist_onefoot_values',lambda _:values)
    cmd=SimpleNamespace(command=torch.tensor([[.3,1.,0.]]*3))
    env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:cmd))
    reward=mdp.waist_target_onefoot_signal(env,'hold')
    assert reward[0]>2*reward[1] and reward[2]==0
    reward=mdp.waist_target_onefoot_signal(env,'waist_support')
    assert reward[0]>reward[1] and reward[2]==0


def test_explicit_waist_recipe_keeps_foreaft_cost_and_physical_skill_gate():
    from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import make_waist_onefoot_env_cfg
    a=make_waist_onefoot_env_cfg(stage='balance-100');b=make_lean_onefoot_env_cfg(stage='balance-100')
    assert a.commands==b.commands and a.terminations==b.terminations
    assert a.events==b.events and a.actions==b.actions and a.observations==b.observations
    assert a.rewards['foreaft_balance']==b.rewards['foreaft_balance']
    assert a.rewards['curriculum_hold'].func is mdp.waist_target_onefoot_signal
    assert 'onefoot/waist_target_roll_deg' in a.metrics
