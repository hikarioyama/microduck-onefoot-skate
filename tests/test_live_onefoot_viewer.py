import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

LOCAL=Path(__file__).parents[1]/'local'


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,LOCAL/file)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


source=load('onefoot_live_source','onefoot_live_source.py')


def saved(path,data=b'checkpoint',mtime=90):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    os.utime(path,(mtime,mtime));return path


def fixture(tmp_path,stage='balance-100'):
    root=tmp_path/'run';directory=root/stage;directory.mkdir(parents=True)
    state=dict(run_root=str(root),run_dir=str(directory),stage=stage,recipe='continuous')
    file=tmp_path/'state.json';file.write_text(json.dumps(state))
    return file,root,source.read_source(file,log_root=tmp_path)


def test_follows_active_stage_numeric_latest_not_best_or_pilot(tmp_path):
    file,root,spec=fixture(tmp_path)
    saved(spec.directory/'model_2.pt');saved(spec.directory/'model_10.pt')
    saved(root/'balance-050/model_9999.pt');saved(tmp_path/'pilot/model_99999.pt')
    saved(spec.directory/'model_20.pt.tmp')
    state=json.loads(file.read_text());state['best_checkpoint']=str(spec.directory/'model_2.pt')
    file.write_text(json.dumps(state))
    catalog=source.StableCheckpoints(3)
    assert catalog.scan(spec,now=100,monotonic=0)==[]
    assert catalog.scan(spec,now=102,monotonic=2)==[]
    ready=catalog.scan(spec,now=104,monotonic=4)
    assert [p.iteration for p in ready]==[2,10]
    assert ready[-1].label=='balance-100/model_10.pt'
    assert not source.needs_reload(ready[-1],ready[0])


def test_modified_incomplete_or_rejected_file_waits_and_can_recover(tmp_path):
    _,_,spec=fixture(tmp_path);path=saved(spec.directory/'model_5.pt',b'first')
    catalog=source.StableCheckpoints(3);catalog.scan(spec,now=100,monotonic=0)
    first=catalog.scan(spec,now=104,monotonic=4)[0]
    catalog.reject(first)
    assert not catalog.scan(spec,now=105,monotonic=5)
    saved(path,b'new-complete-version',mtime=105)
    assert not catalog.scan(spec,now=105,monotonic=5)
    second=catalog.scan(spec,now=109,monotonic=9)[0]
    assert source.needs_reload(first,second)  # same filename, different data
    assert not source.needs_reload(second,second)


def test_stage_switch_requires_matching_directory_and_stays_inside_run(tmp_path):
    file,root,first=fixture(tmp_path)
    state=json.loads(file.read_text());state['stage']='self-launch'
    file.write_text(json.dumps(state))
    with pytest.raises(ValueError):source.read_source(file,log_root=tmp_path)
    state['run_dir']=str(root/'self-launch');file.write_text(json.dumps(state))
    second=source.read_source(file,log_root=tmp_path)
    assert second!=first and second.stage=='self-launch'
    (root/'self-launch').symlink_to(tmp_path/'unrelated',target_is_directory=True)
    with pytest.raises(ValueError):source.read_source(file,log_root=tmp_path)


def test_uses_official_ui_and_loop_with_reset_aware_policy():
    viewer=load('latest_onefoot_viewer','play-latest-onefoot.py')
    from mjlab.viewer import ViserPlayViewer
    assert issubclass(viewer.FollowingViewer,ViserPlayViewer)
    assert viewer.FollowingViewer.setup is ViserPlayViewer.setup
    assert viewer.FollowingViewer.run is ViserPlayViewer.run
    assert viewer.FollowingViewer._execute_step is ViserPlayViewer._execute_step
    calls=[]
    class Actor:
        def reset(self,dones=None):calls.append(dones)
        def __call__(self,obs):return obs
    policy=viewer.EpisodePolicy(Actor(),SimpleNamespace(episode_length_buf=torch.tensor([0,8])))
    x=torch.ones(2,14);assert policy(x) is x
    assert calls[-1].tolist()==[True,False]
    policy.reset();assert calls[-1] is None
    text=(LOCAL/'play-latest-onefoot.py').read_text()
    for custom in ('add_markdown','add_checkbox','add_html','add_icosphere','add_tab_group'):
        assert custom not in text
    assert "host='127.0.0.1'" in text


def test_every_recipe_preserves_unassisted_self_launch():
    viewer=sys.modules.get('latest_onefoot_viewer') or load('latest_onefoot_viewer','play-latest-onefoot.py')
    for task,make_env,make_rl in viewer.FACTORIES.values():
        cfg=make_env(play=True,stage='self-launch')
        assert cfg.events['curriculum_spawn'].params['speed_range']==(0.,0.)
        assert cfg.events['curriculum_spawn'].params['phase_start']==0.
        assert cfg.commands['twist'].goal_s==2.


def test_initial_camera_is_above_ground_without_changing_the_task(tmp_path):
    viewer=sys.modules.get('latest_onefoot_viewer') or load('latest_onefoot_viewer','play-latest-onefoot.py')
    spec=source.ActiveSource(tmp_path,'continuous','balance-100')
    cfg,rl,task=viewer.viewer_configuration(spec)
    baseline=viewer.FACTORIES['continuous'][1](play=True,stage='balance-100')
    assert cfg.viewer.elevation>0
    assert cfg.rewards==baseline.rewards and cfg.terminations==baseline.terminations
    assert cfg.events==baseline.events and cfg.observations==baseline.observations
    assert cfg.scene.num_envs==1


def bridge_fixture(tmp_path,stage='unload'):
    local=tmp_path/'local';local.mkdir()
    run=tmp_path/'logs'/'run';run.mkdir(parents=True)
    state=local/'training.json'
    data=dict(log_root=str(run),stage=stage,onefoot_source='retained-model.pt',status='running',
              initial_checkpoint=str(run/'model_initial.pt'),source_iteration=299)
    state.write_text(json.dumps(data))
    course=local/'course.json';course.write_text(json.dumps(dict(stage=stage,active_state=str(state))))
    pointer=local/'live.json';pointer.write_text(json.dumps(dict(state_file=str(course))))
    spec=source.read_source(pointer,log_root=tmp_path/'logs',state_root=local)
    return pointer,course,state,run,spec


def test_bridge_follows_pointer_and_flat_latest_not_stale_state_checkpoint(tmp_path):
    pointer,course,state,run,spec=bridge_fixture(tmp_path)
    assert spec.recipe=='credit-bridge' and spec.flat and spec.directory==run
    saved(run/'model_initial.pt');saved(run/'model_300.pt');saved(run/'model_399.pt')
    saved(tmp_path/'logs/smoke/model_9999.pt')
    c=source.StableCheckpoints(3);c.scan(spec,now=100,monotonic=0)
    ready=c.scan(spec,now=104,monotonic=4)
    assert [item.iteration for item in ready]==[299,300,399]
    assert '[assisted]' in ready[-1].label
    data=json.loads(state.read_text());data.update(iteration=499,status='completed')
    state.write_text(json.dumps(data))
    refreshed=source.read_source(pointer,log_root=tmp_path/'logs',state_root=tmp_path/'local')
    assert refreshed==spec
    assert refreshed.training_status=='completed'
    data['stage']='transfer';state.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='consistent'):
        source.read_source(pointer,log_root=tmp_path/'logs',state_root=tmp_path/'local')
    course.write_text(json.dumps(dict(stage='transfer',active_state=str(state))))
    changed=source.read_source(pointer,log_root=tmp_path/'logs',state_root=tmp_path/'local')
    assert changed!=spec and changed.stage=='transfer'


def test_viewer_rejects_pointer_escape_cycles_and_wrong_initial_path(tmp_path):
    pointer,course,state,run,spec=bridge_fixture(tmp_path)
    course.write_text(json.dumps(dict(active_state=str(tmp_path/'outside.json'))))
    with pytest.raises(ValueError,match='leaves'):
        source.read_source(pointer,log_root=tmp_path/'logs',state_root=tmp_path/'local')
    course.write_text(json.dumps(dict(state_file=str(pointer))))
    with pytest.raises(ValueError,match='Cyclic'):
        source.read_source(pointer,log_root=tmp_path/'logs',state_root=tmp_path/'local')
    course.write_text(json.dumps(dict(stage='unload',active_state=str(state))))
    data=json.loads(state.read_text());data['initial_checkpoint']=str(tmp_path/'other.pt');state.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Initial checkpoint'):
        source.read_source(pointer,log_root=tmp_path/'logs',state_root=tmp_path/'local')


def test_trained_checkpoint_wins_initial_at_same_iteration(tmp_path):
    run=tmp_path/'run';run.mkdir()
    spec=source.ActiveSource(run,'credit-bridge','full',flat=True,initial_iteration=0)
    saved(run/'model_initial.pt');saved(run/'model_0.pt')
    c=source.StableCheckpoints(3);c.scan(spec,now=100,monotonic=0)
    ready=c.scan(spec,now=104,monotonic=4)
    assert [item.path.name for item in ready]==['model_initial.pt','model_0.pt']
    assert max(ready,key=source.checkpoint_order).path.name=='model_0.pt'
    assert not source.needs_reload(ready[-1],ready[0])
    assert '[from rest]' in ready[-1].label


def test_credit_viewer_uses_current_training_stage_without_changing_physics(tmp_path):
    viewer=load('latest_credit_viewer_test','play-latest-onefoot.py')
    from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import make_credit_bridge_env_cfg
    for stage in ('unload','transfer-near','full'):
        spec=source.ActiveSource(tmp_path,'credit-bridge',stage,flat=True)
        cfg,rl,task=viewer.viewer_configuration(spec)
        baseline=make_credit_bridge_env_cfg(play=False,stage=stage)
        assert cfg.events==baseline.events and cfg.observations==baseline.observations
        assert cfg.rewards==baseline.rewards and cfg.terminations==baseline.terminations
        assert cfg.actions==baseline.actions and cfg.scene.num_envs==1
        assert rl.actor.class_name.endswith('RetainedSkillBridgeModel')
        assert task==viewer.CREDIT_TASK
