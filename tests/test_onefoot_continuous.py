from types import SimpleNamespace
import torch
import mjlab.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_curriculum_env_cfg import make_curriculum_onefoot_env_cfg
from mjlab_microduck.tasks.microduck_onefoot_continuous_env_cfg import make_continuous_onefoot_env_cfg


def test_contiguous_time_beats_same_total_fragmented_time():
    continuous=torch.arange(1,26)*.02
    fragments=(torch.arange(25)%5+1)*.02
    speed=torch.full((25,),.3)
    sustained=mdp.continuous_onefoot_factor(continuous,.5,speed).sum()
    interrupted=mdp.continuous_onefoot_factor(fragments,.5,speed).sum()
    assert sustained>2*interrupted
    assert mdp.continuous_onefoot_factor(torch.tensor([0.]),.5,torch.tensor([.3]))==.1


def test_rocking_trunk_without_centroid_progress_earns_nothing():
    assert not mdp.continuous_onefoot_factor(torch.ones(3),.5,torch.tensor([0.,-.1,-.5])).any()


def test_replant_is_failure_only_after_glide_established(monkeypatch):
    monkeypatch.setattr(mdp,'_curriculum_onefoot_values',lambda _: {})
    cmd=SimpleNamespace(best_steps=torch.tensor([0,4,5,50,100]))
    contact=torch.tensor([True,True,True,False,True])[:,None,None].expand(-1,2,1)
    env=SimpleNamespace(step_dt=.02,command_manager=SimpleNamespace(get_term=lambda _:cmd),
        scene={'onefoot_right':SimpleNamespace(data=SimpleNamespace(found=contact))})
    assert mdp.continuous_onefoot_replanted(env).tolist()==[False,False,True,False,True]
    # Resetting an episode removes the latch, even after a previous 2s success.
    cmd.best_steps[-1]=0
    assert not mdp.continuous_onefoot_replanted(env)[-1]


def test_no_relaxed_success_criteria_or_changed_physics():
    for stage in ('balance-050','unload','self-launch'):
        before=make_curriculum_onefoot_env_cfg(stage=stage)
        after=make_continuous_onefoot_env_cfg(stage=stage)
        assert before.commands==after.commands
        assert before.events==after.events and before.observations==after.observations
        assert before.actions==after.actions and before.sim==after.sim
        assert before.scene.entities==after.scene.entities and before.scene.sensors==after.scene.sensors
        # The default empty-spec factory is a newly allocated lambda per cfg;
        # compare its output and other terrain fields, not callable identity.
        bt=dict(vars(before.scene.terrain));at=dict(vars(after.scene.terrain))
        bf=bt.pop('spec_fn');af=at.pop('spec_fn')
        assert bt==at and bf().to_xml()==af().to_xml()
        assert before.metrics==after.metrics
        assert after.terminations['replanted_swing'].time_out is False
        assert after.rewards['curriculum_hold'].func is mdp.continuous_onefoot_reward


def test_standard_play_has_explicit_stage_and_unassisted_final_task():
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks.microduck_onefoot_continuous_env_cfg import STAGE_TASKS
    for stage,task in STAGE_TASKS.items():
        cfg=load_env_cfg(task,play=True)
        expected=make_continuous_onefoot_env_cfg(play=True,stage=stage)
        assert cfg.commands==expected.commands and cfg.events==expected.events
    final=load_env_cfg(STAGE_TASKS['self-launch'],play=True)
    assert final.events['curriculum_spawn'].params['speed_range']==(0.,0.)
    assert final.events['curriculum_spawn'].params['phase_start']==0.
    assert final.commands['twist'].goal_s==2.
