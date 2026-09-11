from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import math
import numpy as np
import torch
import pytest
import mjlab.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks import microduck_onefoot_knee_env_cfg as knee_cfg
from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import make_waist_onefoot_env_cfg,make_waist_onefoot_rl_cfg


def test_knee_credit_has_flexible_plateau_not_straight_leg_target():
    knee=torch.tensor([-1.,-.5,-.018,.1,.35,.55,.8,1.2,1.5])
    credit=mdp.support_knee_credit(knee,torch.ones_like(knee))
    assert torch.equal(credit[:5],torch.ones(5))
    assert torch.all(credit[4:-1]>credit[5:])
    assert credit[6]>.3 and credit[6]<.5  # useful gradient at the current strategy
    assert credit[-1]>0 and credit[-1]<.01
    assert torch.equal(mdp.support_knee_credit(knee,torch.zeros_like(knee)),torch.ones_like(knee))
    halfway=mdp.support_knee_credit(knee,torch.full_like(knee,.5))
    assert torch.allclose(halfway,.75+.25*credit)


def test_knee_recipe_changes_only_existing_positive_maneuver_credit(monkeypatch):
    changed={'curriculum_hold','curriculum_continuous','curriculum_transfer','waist_support'}
    for stage in ('balance-050','balance-100','balance-200','transfer','self-launch'):
        template=make_waist_onefoot_env_cfg(stage=stage)
        monkeypatch.setattr(knee_cfg,'make_waist_onefoot_env_cfg',lambda **kwargs:deepcopy(template))
        cfg=knee_cfg.make_knee_onefoot_env_cfg(stage=stage)
        for name in ('scene','sim','commands','observations','actions','events','terminations','curriculum','episode_length_s'):
            assert getattr(cfg,name)==getattr(template,name),name
        assert cfg.rewards.keys()==template.rewards.keys()
        for name in cfg.rewards:
            if name in changed:
                assert cfg.rewards[name].weight==template.rewards[name].weight>0
                assert cfg.rewards[name].func is mdp.knee_guard_onefoot_signal
            else:assert cfg.rewards[name]==template.rewards[name]
    a=knee_cfg.make_knee_onefoot_rl_cfg();b=make_waist_onefoot_rl_cfg()
    a.run_name=b.run_name;assert a==b


def test_knee_guard_cannot_pay_for_standing_flight_wrong_support_or_extension(monkeypatch):
    knee=torch.tensor([.1,.8,.1,.8])
    data=SimpleNamespace(joint_pos=torch.stack((torch.full_like(knee,999.),knee),1))
    calls=[]
    def find(name):calls.append(name);return [1],[name]
    robot=SimpleNamespace(data=data,find_joints=find)
    cmd=SimpleNamespace(command=torch.tensor([[.3,1.,0.]]*4))
    env=SimpleNamespace(scene={'robot':robot},command_manager=SimpleNamespace(get_term=lambda _:cmd))
    base=torch.tensor([1.,1.,0.,0.])
    monkeypatch.setattr(mdp,'waist_target_onefoot_signal',lambda env,component:base)
    for component in ('hold','continuous','transfer','waist_support'):
        result=mdp.knee_guard_onefoot_signal(env,component)
        assert result[0]==1 and 0<result[1]<1 and torch.all(result[2:]==0)
    assert calls==['left_knee']
    assert torch.equal(mdp.knee_guard_onefoot_signal(env,'accelerate'),base)


def test_knee_direction_and_nonstraight_plateau_on_actual_roller_model():
    import mujoco
    from mjlab_microduck.tasks.microduck_onefoot_curriculum_poses import POSES
    root=Path(__file__).resolve().parents[1]
    xml=root/'src/mjlab_microduck/robot/microduck/scene_rollers.xml'
    # The robot model and its meshes live in the upstream working tree; see
    # docs/REPRODUCING.md for the overlay layout that provides them.
    if not xml.exists():pytest.skip('Robot model not in this checkout')
    model=mujoco.MjModel.from_xml_path(str(xml))
    data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,model.key('STAND').id)
    for name,value in POSES['balance']['pose'].items():data.qpos[model.jnt_qposadr[model.joint(name).id]]=value
    ids=[model.joint(name).id for name in ('left_hip_pitch','left_knee','left_ankle')]
    angles=[];lengths=[]
    for q in (-.018,knee_cfg.SUPPORT_KNEE_COMFORTABLE_MAX,.8,1.2):
        data.qpos[model.jnt_qposadr[ids[1]]]=q;mujoco.mj_forward(model,data)
        hip,knee,ankle=data.xanchor[ids]
        u=hip-knee;v=ankle-knee
        angles.append(180-np.degrees(np.arccos(np.clip(u@v/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))))
        lengths.append(np.linalg.norm(hip-ankle))
    assert all(a<b for a,b in zip(angles,angles[1:]))
    assert all(a>b for a,b in zip(lengths,lengths[1:]))
    assert 75<angles[1]<90  # explicitly not a locked/straight knee


def test_knee_candidate_is_registered_but_not_silently_active():
    from mjlab.tasks.registry import load_env_cfg
    assert load_env_cfg(knee_cfg.TASK).rewards['curriculum_hold'].func is mdp.knee_guard_onefoot_signal
    cfg=make_waist_onefoot_env_cfg()
    assert cfg.rewards['curriculum_hold'].func is mdp.waist_target_onefoot_signal
    assert not any('support_knee' in name for name in cfg.metrics)


def test_pilot_uses_same_bounded_training_budget_without_live_publication():
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('knee_pilot_test',root/'local/run-onefoot-knee-pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for recipe in ('knee','waist'):
        args=module.train_arguments(recipe,Path('/same/model_6199.pt'),Path('/new/run'))
        assert args[args.index('--mode')+1]=='train'
        assert args[args.index('--recipe')+1]==recipe
        assert args[args.index('--iterations')+1]=='100'
        assert args[args.index('--max-chunks')+1]=='6'
        assert args[args.index('--num-envs')+1]=='4096'
        assert '--resume' in args and '--resume-state' not in args
        assert args[args.index('--exploration-max-std')+1]=='.08'
    assert module.HOLDOUT_SEEDS==(64101,64102)
