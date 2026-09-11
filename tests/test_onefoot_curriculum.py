from copy import deepcopy
from types import SimpleNamespace
import math
import pytest
import torch
import mjlab.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_curriculum_env_cfg import (
    STAGES, POSES, make_curriculum_onefoot_env_cfg, gate_passes,
)


def test_order_and_no_skipped_duration_gates():
    assert [s.goal_s for s in STAGES[:3]]==[.5,1.,2.]
    assert [s.group for s in STAGES]==[1,1,1,2,3,3,3,4,4,4]
    assert STAGES[-1].name=='self-launch' and STAGES[-1].speed_range==(0.,0.)


def test_final_has_no_injected_velocity_or_mid_episode_assistance():
    cfg=make_curriculum_onefoot_env_cfg(stage='self-launch')
    assert cfg.events['curriculum_spawn'].params['speed_range']==(0.,0.)
    assert all(v==(0.,0.) for v in cfg.events['reset_base'].params['velocity_range'].values())
    assert not [e for e in cfg.events.values() if e and e.mode in ('interval','step')]
    assert 'onefoot_seed' not in cfg.events and not cfg.curriculum
    assert cfg.commands['twist'].goal_s==2.


def test_preserve_bam_noise_delay_and_61d_terms():
    from mjlab_microduck.tasks.microduck_onefoot_dynamic_env_cfg import make_dynamic_onefoot_env_cfg
    base=make_dynamic_onefoot_env_cfg(assisted=True)
    cfg=make_curriculum_onefoot_env_cfg()
    assert cfg.scene.entities==base.scene.entities
    assert cfg.actions==base.actions and cfg.observations==base.observations
    for name in ('expand_bam_friction_fields','randomize_com','randomize_head_com',
                 'randomize_mass_inertia','randomize_joint_friction','randomize_armature','encoder_bias'):
        assert cfg.events[name]==base.events[name]
    for sensor in cfg.scene.sensors:
        if sensor.name in ('onefoot_left','onefoot_right'):
            assert sensor.fields==('found','force')


def good_values(n=1):
    return dict(speed=torch.full((n,),.3),lateral=torch.zeros(n),heading=torch.zeros(n),
        cross_track=torch.zeros(n),upright=torch.ones(n),blade_yaw=torch.zeros(n),
        left=torch.ones(n,dtype=torch.bool),right=torch.zeros(n,dtype=torch.bool),
        clearance=torch.full((n,),.025),phase=torch.ones(n),forbidden=torch.zeros(n,dtype=torch.bool))


@pytest.mark.parametrize('key,value',[
    ('speed',0.),('speed',-.2),('speed',1.),('lateral',.2),('heading',.4),
    ('cross_track',.25),('upright',.6),('blade_yaw',.44),('left',False),
    ('right',True),('clearance',.003),('phase',.5),('forbidden',True),('speed',float('nan')),
])
def test_no_fake_success(key,value):
    v=good_values();assert mdp.curriculum_onefoot_valid(**v).all()
    v[key][:]=value
    assert not mdp.curriculum_onefoot_valid(**v).any()


def row(**updates):
    v=dict(episodes=256,goal_s=.5,success_rate=.8,nan_episodes=0)
    return dict(v,**updates)


def test_gate_uses_two_sufficient_batches_not_reward():
    assert gate_passes([row(),row()],.5)
    assert not gate_passes([row()],.5)
    assert not gate_passes([row(),row(success_rate=.799)],.5)
    assert not gate_passes([row(),row(episodes=64)],.5)
    assert not gate_passes([row(),row(nan_episodes=1)],.5)
    assert not gate_passes([row(),row()],2.)


def test_mesh_radius_and_grounded_spawns():
    import json
    import mujoco
    import numpy as np
    from pathlib import Path
    path=Path(__file__).parents[1]/'src/mjlab_microduck/robot/microduck/scene_rollers.xml'
    # The robot model and its meshes live in the upstream working tree; see
    # docs/REPRODUCING.md for the overlay layout that provides them.
    if not path.exists():pytest.skip('Robot model not in this checkout')
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    # `glide` is a MEASURED single-support spawn: its support skate is grounded and its
    # swing skate is deliberately off the ground by whatever the recorded glide-entry
    # frames show. Read that evidence back instead of hardcoding it, so the pose and
    # the measurement cannot drift apart.
    record=Path(__file__).parents[1]/'local/onefoot-bridge-curriculum/glide-pose-measurement.json'
    measured=json.loads(record.read_text()) if record.exists() else None
    if measured is not None:
        assert measured['captured']>=32, 'The pose must come from many real frames'
        assert measured['speed']['p50']>.40, 'The recorded entry is a fast glide'
    for name,pose in POSES.items():
        mujoco.mj_resetDataKeyframe(model,data,model.key('STAND').id)
        data.qpos[2]=pose['root_z'];r=pose['root_roll'];pitch=pose.get('root_pitch',0.)
        # roll about x then pitch about y; pitch defaults to 0 for the legacy poses.
        cr,sr=math.cos(r/2),math.sin(r/2);cp,sp=math.cos(pitch/2),math.sin(pitch/2)
        data.qpos[3:7]=[cr*cp,sr*cp,cr*sp,-sr*sp]
        for joint,q in pose['pose'].items():
            jid=model.joint(joint).id;data.qpos[model.jnt_qposadr[jid]]=q
            assert model.jnt_range[jid,0]<q<model.jnt_range[jid,1]
        mujoco.mj_forward(model,data)
        wheel_min={}
        for body in ('tire','tire_2','tire_3','tire_4'):
            bid=model.body(body).id
            gid=next(g for g in range(model.ngeom) if model.geom_bodyid[g]==bid and model.geom_contype[g])
            mesh=model.geom_dataid[gid];offset=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh]
            vertices=model.mesh_vert[offset:offset+n]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
            assert np.linalg.norm(vertices-data.xpos[bid],axis=1).max()<mdp.CURRICULUM_WHEEL_BOUND
            wheel_min[body]=float(vertices[:,2].min())
        # Every measured pose is mesh-grounded: its lowest tyre vertex sits on the floor.
        assert abs(min(wheel_min.values()))<.003, (name,wheel_min)
        lift={body:wheel_min[body]-min(wheel_min.values()) for body in wheel_min}
        if name=='balance':
            assert abs(min(lift['tire_3'],lift['tire_4'])-.025)<.004
        if name=='glide':
            assert min(lift['tire'],lift['tire_2'])<.012, 'Support skate must be grounded'
            assert min(lift['tire_3'],lift['tire_4'])>.010, 'Swing skate must be clearly up'
            # ... and that is exactly the single-support state the evidence recorded.
            assert pose.get('joint_vel'), 'A mid-motion spawn must restore measured joint velocities'
            assert max(abs(v) for v in pose['joint_vel'].values())>1., 'Real motion, not a standstill'
            assert set(pose['joint_vel'])>set(pose['pose']), 'Wheels must be recorded too'
            assert abs(pose['root_pitch']-measured['pitch']['p50'])<1e-9
            assert abs(pose['root_roll']-measured['roll']['p50'])<1e-9
            assert pose['root_z']==pytest.approx(-measured['lowest_wheel_mesh_min_z'],abs=1e-6)


def test_command_phase_and_integer_dwell_reset_are_per_environment():
    robot=SimpleNamespace(find_bodies=lambda name:([('tire','tire_2','tire_3','tire_4').index(name)],None),
        find_joints=lambda _:([0],None),data=SimpleNamespace(
            root_link_pos_w=torch.zeros(2,3),heading_w=torch.zeros(2),
            body_link_pos_w=torch.zeros(2,4,3),data=SimpleNamespace(subtree_com=torch.zeros(2,1,3)),
            indexing=SimpleNamespace(root_body_id=0)))
    env=SimpleNamespace(num_envs=2,device='cpu',scene={'robot':robot},
                        _curriculum_phase_start=torch.tensor([1.,0.]))
    cmd=mdp.CurriculumOneFootCommandCfg(resampling_time_range=(1000,1000),
                                       acceleration_s=0.,lift_s=1.).build(env)
    cmd.reset(torch.arange(2));cmd.compute(0)
    assert cmd.command[:,1].tolist()==[1.,0.]
    cmd.good_steps[:]=25;cmd.best_steps[:]=50
    cmd.reset(torch.tensor([0]));cmd.compute(0)
    assert cmd.good_steps.tolist()==[0,25] and cmd.best_steps.tolist()==[0,50]


def test_reward_and_termination_share_one_dwell_update_per_step():
    data=SimpleNamespace(root_link_pos_w=torch.zeros(2,3),heading_w=torch.zeros(2),
        root_link_lin_vel_w=torch.tensor([[.3,0.,0.],[.3,0.,0.]]),
        projected_gravity_b=torch.tensor([[0.,0.,-1.],[0.,0.,-1.]]),
        joint_vel=torch.zeros(2,14),body_link_pos_w=torch.tensor([
            [[.032,0.,.015],[-.032,0.,.015],[.032,-.08,.041],[-.032,-.08,.041]],
            [[.032,0.,.015],[-.032,0.,.015],[.032,-.08,.041],[-.032,-.08,.041]]]),
        data=SimpleNamespace(subtree_com=torch.tensor([[[0.,0.,.17]],[[0.,0.,.17]]])),
        indexing=SimpleNamespace(root_body_id=0))
    robot=SimpleNamespace(data=data,
        find_bodies=lambda n:([('tire','tire_2','tire_3','tire_4').index(n)],None),
        find_joints=lambda _:([0,1],None))
    sensor=lambda found,force:SimpleNamespace(data=SimpleNamespace(
        found=torch.full((2,2,1),found),force=torch.tensor([[[0.,0.,force],[0.,0.,force]]]*2)))
    env=SimpleNamespace(num_envs=2,device='cpu',step_dt=.02,common_step_counter=1,
        _curriculum_phase_start=torch.ones(2),_curriculum_injected_speed=torch.full((2,),.3),
        scene={'robot':robot,'onefoot_left':sensor(True,3.),'onefoot_right':sensor(False,0.),
               'onefoot_body_ground':sensor(False,0.)})
    cmd=mdp.CurriculumOneFootCommandCfg(resampling_time_range=(1000,1000),
                                       acceleration_s=0.,lift_s=1.).build(env)
    env.command_manager=SimpleNamespace(get_term=lambda _:cmd)
    cmd.reset(torch.arange(2));cmd.compute(0)
    for _ in range(4):
        mdp.curriculum_onefoot_stalled(env)
        assert (mdp.curriculum_onefoot_signal(env,'hold')>0).all()
        mdp.curriculum_onefoot_signal(env,'success')
    assert cmd.good_steps.tolist()==[1,1]
    env.common_step_counter+=1
    # Detector-only contact without load must break the first world's streak.
    env.scene['onefoot_left'].data.force[0]=0
    mdp.curriculum_onefoot_stalled(env)
    assert cmd.good_steps.tolist()==[0,2]
    assert cmd.best_steps.tolist()==[1,2]
