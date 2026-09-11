from pathlib import Path
import importlib.util
import numpy as np
import pytest
from mjlab_microduck.onefoot_supervision import FAILURE_CHECK_NAMES,FAILURE_VALUE_NAMES


def module():
    path=Path(__file__).resolve().parents[1]/'local/record-onefoot-timeline.py'
    spec=importlib.util.spec_from_file_location('onefoot_timeline_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_fixed_time_summaries_preserve_success_failure_and_censoring():
    m=module();n=2;steps=60
    values=np.zeros((steps,n,len(FAILURE_VALUE_NAMES)),dtype=np.float32)
    values[:,:,0]=.3
    values[:,:,list(FAILURE_VALUE_NAMES).index('left_front_force_n')]=6.
    values[:,:,list(FAILURE_VALUE_NAMES).index('left_rear_force_n')]=2.
    alive=np.ones((steps,n),dtype=bool);alive[40:,1]=False
    values[40:,1]=np.nan
    checks=np.ones((steps,n,len(FAILURE_CHECK_NAMES)),dtype=bool)
    checks[40:,1]=False
    rows=m.summarize_times(values,checks,alive,np.array([True,False]),.02,times=(.5,.9,2.))
    row=next(r for r in rows if r['time_s']==.9 and r['group']=='all')
    assert row['total_episodes']==2 and row['present_episodes']==1 and row['ended_before_sample']==1
    assert row['front_load_fraction']['mean']==.75
    assert row['values']['speed_m_s']['mean']==pytest.approx(.3)
    row=next(r for r in rows if r['time_s']==.9 and r['group']=='failed')
    assert row['present_episodes']==0 and row['ended_before_sample']==1 and 'values' not in row
    assert all(r['present_episodes']==0 for r in rows if r['time_s']==2.)


def test_empty_and_nonfinite_measurements_are_not_reported_as_zero():
    m=module()
    assert m.statistics([float('nan'),float('inf')])['mean'] is None
    assert m.statistics([float('nan'),1.,3.])['mean']==2.


def test_support_kinematics_resolves_names_and_copies_before_reset():
    from types import SimpleNamespace
    import torch
    from mjlab_microduck.tasks import mdp
    # Passive wheel slots intentionally interleave the servos.
    ids={'left_knee':6,'left_hip_pitch':2,'left_ankle':8}
    q=torch.arange(10,dtype=torch.float)[None,:]
    data=SimpleNamespace(joint_pos=q,joint_vel=q+10,joint_pos_target=q+20,
        root_link_pos_w=torch.tensor([[0.,0.,2.15]]),root_link_lin_vel_w=torch.tensor([[.3,0.,-.1]]),
        body_link_lin_vel_w=torch.tensor([[[.3,0.,0.],[.1,0.,0.]]]),
        data=SimpleNamespace(subtree_com=torch.tensor([[[0.,0.,2.18]]])),indexing=SimpleNamespace(root_body_id=0))
    robot=SimpleNamespace(data=data,find_joints=lambda name:([ids[name]],[name]))
    class Scene(dict):pass
    scene=Scene(robot=robot);scene.env_origins=torch.tensor([[0.,0.,2.]])
    cmd=SimpleNamespace(wheel_ids=[0,1])
    env=SimpleNamespace(scene=scene,command_manager=SimpleNamespace(get_term=lambda _:cmd))
    sample=mdp.curriculum_onefoot_support_kinematics(env)
    q.zero_()  # A following automatic reset cannot overwrite the recorded sample.
    assert sample[0,:5].tolist()==[6.,16.,26.,2.,8.]
    assert sample[0,5:].tolist()==pytest.approx([.15,-.1,.18,.2],abs=1e-6)
    assert len(mdp.SUPPORT_KINEMATIC_NAMES)==sample.shape[1]


def test_support_kinematic_summary_never_mixes_reset_episodes():
    from mjlab_microduck.tasks import mdp
    m=module();values=np.zeros((50,2,len(mdp.SUPPORT_KINEMATIC_NAMES)))
    alive=np.ones((50,2),dtype=bool);alive[20:,1]=False
    values[20:,1]=9999
    rows=m.summarize_kinematics(values,alive,np.array([True,False]),.02,times=(.2,.8,2.))
    row=next(r for r in rows if r['time_s']==.8 and r['group']=='all')
    assert row['present_episodes']==1 and row['ended_before_sample']==1
    assert row['values']['left_knee_rad']['mean']==0
    assert all(r['present_episodes']==0 for r in rows if r['time_s']==2.)


@pytest.mark.parametrize('steps',[2,10,40])
def test_knee_summary_counts_early_death_and_never_uses_reset_poses(tmp_path,steps):
    import json
    from mjlab_microduck.tasks import mdp
    path=Path(__file__).resolve().parents[1]/'local/summarize-onefoot-knee.py'
    spec=importlib.util.spec_from_file_location('summarize_knee_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    k=np.ones((steps,2,len(mdp.SUPPORT_KINEMATIC_NAMES)))
    alive=np.ones((steps,2),dtype=bool);alive[8:,1]=False;k[8:,1]=9999
    npz=tmp_path/'trajectory.npz'
    np.savez_compressed(npz,support_kinematics=k,support_kinematic_names=np.array(mdp.SUPPORT_KINEMATIC_NAMES),
        values=np.ones((steps,2,len(FAILURE_VALUE_NAMES))),value_names=np.array(FAILURE_VALUE_NAMES),
        episode_alive=alive,time_s=np.arange(1,steps+1)*.02)
    report=dict(timeline_npz=str(npz),checkpoint='test',checkpoint_sha256='hash',recipe='knee',seed=1,
                episodes=2,success_rate=0.,mean_best_glide_s=0.,nan_episodes=0)
    p=tmp_path/'report.json';p.write_text(json.dumps(report))
    result=m.summarize(p)
    assert result['complete_window_episodes']==(1 if steps>=30 else 0)
    assert result['incomplete_window_episodes']==(1 if steps>=30 else 2)
    assert result['early_peak_knee_rad']['mean']==(None if steps<5 else 1.)
    json.dumps(result,allow_nan=False)
