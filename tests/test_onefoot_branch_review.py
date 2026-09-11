from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest


def module():
    p=Path(__file__).resolve().parents[1]/'local/preserve-onefoot-branch-review.py'
    spec=importlib.util.spec_from_file_location('branch_boundary_review_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def state(count,status='training'):
    anchor=13399
    return dict(status=status,pid=1,recipe='waist',stage='balance-100',run_root='/run',
        best_checkpoint='/run/model_6199.pt',iteration=anchor+count*100+1,
        adaptations=[dict(stage='balance-100',iteration=anchor)],
        evaluations=[dict(stage='balance-100',checkpoint=f'/run/model_{anchor+i*100}.pt',nan_episodes=0)
                     for i in range(1,count+1)])


def test_stop_request_only_at_start_of_final_allowed_chunk():
    m=module();before=state(12)
    for count in (0,12,16):assert m.decision(before,state(count),13399,18) is None
    assert m.decision(before,state(17,'evaluating'),13399,18) is None
    not_started=state(17);not_started['iteration']-=1
    assert m.decision(before,not_started,13399,18) is None
    assert m.decision(before,state(17),13399,18)=='request_saved_stop_in_final_chunk'
    assert m.decision(before,state(18),13399,18)=='boundary_already_reached_review_needed'


@pytest.mark.parametrize('key,value,reason',[
    ('pid',2,'controller_or_stage_changed'),('stage','balance-200','controller_or_stage_changed'),
    ('recipe','knee','controller_or_stage_changed'),('run_root','/new','controller_or_stage_changed'),
    ('best_checkpoint','/run/model_14999.pt','new_best_candidate'),
    ('status','error','controller_error'),('status','interrupted','controller_paused')])
def test_new_progress_or_intervention_cancels_scheduled_stop(key,value,reason):
    m=module();before=state(12);current=state(17);current[key]=value
    assert m.decision(before,current,13399,18)==reason


def test_never_stops_a_replacement_exploration_branch():
    m=module();before=state(12);current=state(18)
    current['adaptations'].append(dict(stage='balance-100',iteration=15199))
    assert m.decision(before,current,13399,18)=='exploration_branch_changed'
    current=state(17);current['evaluations'][-1]['nan_episodes']=1
    assert m.decision(before,current,13399,18)=='numerical_error'
