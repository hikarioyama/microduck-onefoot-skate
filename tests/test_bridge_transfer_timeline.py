import importlib.util
from pathlib import Path
import numpy as np
import pytest
from mjlab_microduck.onefoot_supervision import FAILURE_VALUE_NAMES
from mjlab_microduck.tasks import mdp


def module():
    path=Path(__file__).resolve().parents[1]/'local/record-retained-bridge-transfer.py'
    spec=importlib.util.spec_from_file_location('bridge_transfer_timeline_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_summary_preserves_censoring_overlap_and_reward_dt():
    steps=160;n=2
    alive=np.ones((steps,n),bool);alive[120:,1]=False
    done=np.zeros_like(alive);done[-1,0]=True;done[119,1]=True
    flags=np.zeros((steps,n,2),bool);flags[119,1,:]=True;flags[-1,0,0]=True
    rates=np.ones((steps,n,2));rates[~alive]=np.nan
    rewards=np.nansum(rates,axis=-1)*.02;rewards[~alive]=np.nan
    arrays=dict(alive=alive,done=done,termination_flags=flags,termination_names=np.array(['fall','contact']),
        reward_rates=rates,rewards=rewards,reward_names=np.array(['first','second']),
        values=np.ones((steps,n,len(FAILURE_VALUE_NAMES))),
        kinematics=np.ones((steps,n,len(mdp.SUPPORT_KINEMATIC_NAMES))),
        correction_max_abs=np.ones((steps,n)))
    arrays['values'][120:,1]=9999  # Must not report the reset episode.
    result=module().summarize(arrays,.02)
    assert result['termination_counts']=={'fall':2,'contact':1}
    assert result['episode_length_s']['mean']==pytest.approx(2.8)
    assert result['episode_return']['mean']==pytest.approx(5.6)
    row=next(row for row in result['fixed_time_samples'] if row['time_s']==2.6)
    assert row['present_episodes']==1 and row['ended_before_sample']==1
    assert row['values']['speed_m_s']['mean']==1.
    assert next(row for row in result['fixed_time_samples'] if row['time_s']==3.5)['present_episodes']==0
    arrays['rewards'][0,0]+=1
    with pytest.raises(AssertionError):module().summarize(arrays,.02)
