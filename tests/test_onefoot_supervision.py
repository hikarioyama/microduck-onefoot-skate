from copy import deepcopy
import math
from pathlib import Path
import pytest
import torch
from mjlab_microduck.onefoot_supervision import (
    ExplorationBudget, exploration_plan, verified_gate_reports, audit_stage_history,
    FirstBreakDiagnostics, FAILURE_CHECK_NAMES, FAILURE_VALUE_NAMES,
)


def report(iteration, success=.1, duration=.9, stage='balance-100', **kw):
    return dict(stage=stage, goal_s=1., episodes=256, success_rate=success,
                mean_best_glide_s=duration, nan_episodes=0, seed=iteration,
                checkpoint=f'/run/{stage}/model_{iteration}.pt', **kw)


def history(probabilities, durations=None):
    durations=durations or [.9]*len(probabilities)
    return [report(99,.707,1.03)]+[report(199+i*100,p,d) for i,(p,d) in enumerate(zip(probabilities,durations))]


def test_exploration_does_not_react_to_one_bad_checkpoint():
    p=exploration_plan(history([.01]*5),'balance-100')
    assert p['action']=='continue' and p['allotted_evaluations']==6


def test_recovering_branch_gets_a_whole_extra_window():
    rows=history([.06,.06,.10,.125,.18,.207])
    p=exploration_plan(rows,'balance-100')
    assert p['action']=='continue' and p['reason']=='recovering_exploration'
    assert p['allotted_evaluations']==12
    # One weak sample after extension cannot immediately revoke the extension.
    p=exploration_plan(rows+[report(799,.01)],'balance-100')
    assert p['action']=='continue' and p['allotted_evaluations']==12


def test_flat_branch_single_outlier_and_reward_are_not_progress():
    for ps in ([.1]*6,[.1]*5+[.19]):
        rows=history(ps)
        for i,row in enumerate(rows):row['training_reward']=i*1000
        assert exploration_plan(rows,'balance-100')['action']=='adapt'


def test_duration_recovery_counts_before_a_longer_goal_is_reached():
    rows=history([0.]*6,[.6,.61,.62,.66,.67,.68])
    assert exploration_plan(rows,'balance-100')['allotted_evaluations']==12
    # Longer best streak must not hide a large success-rate regression.
    rows=history([.5,.5,.5,.1,.1,.1],[.6,.61,.62,.8,.81,.82])
    assert exploration_plan(rows,'balance-100')['action']=='adapt'


def test_recovery_budget_is_bounded_and_flat_second_window_ends_it():
    rows=history([.01,.02,.03,.1,.12,.13]*3)
    p=exploration_plan(rows,'balance-100')
    assert p['action']=='adapt' and p['reason']=='exploration_budget_exhausted'
    rows=history([.01,.02,.03,.1,.12,.13]+[.15]*6)
    assert exploration_plan(rows,'balance-100')['reason']=='no_measured_recovery'


def test_new_best_adaptation_and_confirmation_are_handled_separately():
    rows=history([.1]*6)
    p=exploration_plan(rows,'balance-100',[dict(stage='balance-100',iteration=499)])
    assert p['evaluated_checkpoints']==2 and p['action']=='continue'
    rows.append(report(799,.75,1.1))
    assert exploration_plan(rows,'balance-100')['evaluated_checkpoints']==0
    rows=history([.1]*5)
    rows.append(dict(rows[-1],seed=12345))
    assert exploration_plan(rows,'balance-100')['evaluated_checkpoints']==5


def test_nan_and_invalid_evidence_cannot_trigger_progress():
    rows=history([.1]*6);rows[-1]['nan_episodes']=1
    assert exploration_plan(rows,'balance-100')['action']=='diagnose'
    for field,value in [('success_rate',float('nan')),('mean_best_glide_s',-1),('episodes',64)]:
        rows=history([.1]);rows[-1][field]=value
        with pytest.raises(ValueError):exploration_plan(rows,'balance-100')
    with pytest.raises(ValueError):ExplorationBudget(block_evaluations=2)
    with pytest.raises(ValueError):ExplorationBudget(maximum_evaluations=100)


def gate_rows(stage='balance-100',goal=1.):
    a=report(99,.85,1.1,stage=stage);a['goal_s']=goal
    return [a,dict(a,seed=100)]


def test_gate_requires_same_weights_different_seeds_and_no_fixed_pose():
    rows=gate_rows();assert verified_gate_reports(rows,'balance-100',1.)
    for field,value in [('seed',99),('checkpoint','/run/balance-100/model_100.pt'),
                        ('fixed_pose',True),('action_noise_in_evaluation',True),('episodes',64),('success_rate',.79)]:
        changed=deepcopy(rows);changed[1][field]=value
        assert not verified_gate_reports(changed,'balance-100',1.)
    for row in rows:row['checkpoint_sha256']='a'*64
    assert verified_gate_reports(rows,'balance-100',1.)
    rows[1]['checkpoint_sha256']='b'*64
    assert not verified_gate_reports(rows,'balance-100',1.)


def test_final_gate_rejects_injected_speed_and_audit_rejects_skipped_stage():
    rows=gate_rows('self-launch',2.)
    for row in rows:row.update(injected_speed_min=0.,injected_speed_max=0.)
    assert verified_gate_reports(rows,'self-launch',2.)
    rows[0]['injected_speed_max']=.1
    assert not verified_gate_reports(rows,'self-launch',2.)
    state=dict(stage='balance-200',passed_stages=[],final_self_launch_completed=True)
    issues=audit_stage_history(state,[('balance-050',.5),('balance-100',1.),('balance-200',2.)])
    assert 'skipped_unpassed_stage' in issues and 'unverified_completion' in issues


def test_first_break_is_captured_before_fall_not_overwritten_after_reset():
    n=3;diag=FirstBreakDiagnostics(n,.02,'cpu')
    checks=torch.ones(n,len(FAILURE_CHECK_NAMES),dtype=torch.bool)
    values=torch.zeros(n,len(FAILURE_VALUE_NAMES));alive=torch.ones(n,dtype=torch.bool)
    for step in range(5):diag.observe(checks,values,alive,step)
    broken=checks.clone();broken[0,FAILURE_CHECK_NAMES.index('forward_speed')]=False
    broken[1,FAILURE_CHECK_NAMES.index('left_support')]=False
    values[:,0]=torch.tensor([.09,.2,.3]);diag.observe(broken,values,alive,5)
    # Later collapse and an automatic reset do not rewrite the first event.
    alive[0]=False;broken[:]=False;values[:,0]=99.
    diag.observe(broken,values,alive,6)
    result=diag.report(torch.tensor([False,True,False]))
    assert result['failed_episodes']==2 and result['recorded_break_episodes']==2
    assert result['reason_counts']['forward_speed']==2
    assert result['reason_counts']['left_support']==1  # Successful env 1 is excluded.
    first=next(r for r in result['trials'] if r['env_id']==0)
    assert first['reasons']==['forward_speed']
    assert first['values']['speed_m_s']==pytest.approx(.09)
    assert first['time_s']==pytest.approx(.12) and first['preceding_streak_s']==pytest.approx(.10)


def test_unestablished_failures_remain_in_the_denominator():
    diag=FirstBreakDiagnostics(2,.02,'cpu')
    checks=torch.zeros(2,len(FAILURE_CHECK_NAMES),dtype=torch.bool)
    values=torch.zeros(2,len(FAILURE_VALUE_NAMES))
    diag.observe(checks,values,torch.ones(2,dtype=torch.bool),0)
    result=diag.report(torch.tensor([False,True]))
    assert result['failed_episodes']==1 and result['failed_without_recorded_break']==1
    assert result['failed_without_established_streak']==1
    assert not any(result['reason_counts'].values()) and 'mean_first_break_time_s' not in result


def test_diagnostic_predicates_exactly_match_existing_physical_gate():
    from mjlab_microduck.tasks import mdp
    torch.manual_seed(910)
    n=512
    values=dict(speed=torch.rand(n),lateral=torch.randn(n)*.15,heading=torch.randn(n)*.35,
                cross_track=torch.randn(n)*.2,upright=torch.rand(n),blade_yaw=torch.randn(n)*.35,
                left=torch.rand(n)>.1,right=torch.rand(n)>.8,clearance=torch.rand(n)*.03,
                phase=torch.rand(n)*1.1,forbidden=torch.rand(n)>.9)
    values['speed'][:5]=torch.tensor([.1,.8,float('nan'),float('inf'),-.1])
    old=mdp.curriculum_onefoot_valid(**values)
    checks=mdp.curriculum_onefoot_diagnostic_checks(**values)
    assert checks.shape==(n,len(FAILURE_CHECK_NAMES))
    assert torch.equal(old,checks.all(1))


def test_diagnostics_keep_nonfinite_evidence_json_safe():
    import json
    diag=FirstBreakDiagnostics(1,.02,'cpu')
    checks=torch.ones(1,len(FAILURE_CHECK_NAMES),dtype=torch.bool)
    values=torch.zeros(1,len(FAILURE_VALUE_NAMES));alive=torch.ones(1,dtype=torch.bool)
    for step in range(5):diag.observe(checks,values,alive,step)
    checks[0,0]=False;values[0,0]=float('nan')
    diag.observe(checks,values,alive,5)
    result=diag.report(torch.tensor([False]))
    assert result['mean_values_at_break']['speed_m_s'] is None
    assert result['nonfinite_value_counts']['speed_m_s']==1
    json.dumps(result,allow_nan=False)


def load_local(name):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'local'/name
    spec=importlib.util.spec_from_file_location('supervision_'+path.stem.replace('-','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_evaluation_observer_does_not_change_training_or_physics(monkeypatch):
    from types import SimpleNamespace
    from mjlab_microduck.tasks import mdp
    m=load_local('train-onefoot-curriculum.py');m.RECIPE='waist'
    # Terrain factories contain fresh lambdas; compare changes to the SAME
    # configuration, not unrelated callable identities from two constructions.
    template=m.ENV_FACTORIES['waist'](stage='balance-100')
    monkeypatch.setitem(m.ENV_FACTORIES,'waist',lambda stage:deepcopy(template))
    configs=[]
    def build(cfg,device):
        configs.append(cfg)
        return SimpleNamespace(cfg=cfg,num_actions=14,get_observations=lambda:{'actor':torch.zeros(64,61)})
    monkeypatch.setattr(m,'ManagerBasedRlEnv',build)
    monkeypatch.setattr(m,'RslRlVecEnvWrapper',lambda env,clip_actions:env)
    m.environment('balance-100',64,91)
    m.environment('balance-100',64,91,diagnostics=True)
    m.environment('balance-100',64,91,support_kinematics=True)
    before,after,kinematic=configs
    for name in ('sim','scene','commands','observations','actions','events','rewards','terminations'):
        assert getattr(before,name)==getattr(after,name)==getattr(kinematic,name),name
    assert 'onefoot/diagnostic_valid' not in before.metrics
    assert after.metrics['onefoot/diagnostic_valid'].func is mdp.observe_curriculum_onefoot_diagnostics
    assert kinematic.metrics['onefoot/diagnostic_valid'].func is mdp.observe_curriculum_onefoot_support_kinematics
    source=Path(m.__file__).read_text()
    assert 'passed=verified_gate_reports(reports,stage.name,stage.goal_s)' in source
    assert 'plan=exploration_plan(' in source and 'if stale>=6:' not in source


def test_cpu_monitor_distinguishes_pause_failure_and_fresh_log(monkeypatch,tmp_path):
    import os
    from datetime import datetime,timezone
    m=load_local('watch-onefoot-training.py');monkeypatch.setattr(m,'ROOT',tmp_path)
    root=tmp_path/'logs/rsl_rl/onefoot/run';folder=root/'balance-100';folder.mkdir(parents=True)
    cp=folder/'model_199.pt';cp.write_bytes(b'checkpoint')
    history=gate_rows('balance-050',.5)
    s=dict(run_root=str(root),status='training',pid=123,stage='balance-100',recipe='waist',iteration=200,
           updated=datetime.fromtimestamp(1000,timezone.utc).isoformat(),checkpoint=str(cp),best_checkpoint=str(cp),
           passed_stages=[dict(stage='balance-050',checkpoint=history[0]['checkpoint'],reports=history)],evaluations=[])
    log=tmp_path/'training.log';log.write_text('progress');os.utime(log,(1999,1999));s['training_log']=str(log)
    probe=lambda pid,name:'running'
    r=m.sample(s,None,now=2000,process_probe=probe)
    assert r['errors']==[] and r['heartbeat_age_s']==1
    r=m.sample(s,None,now=4000,process_probe=probe)
    assert 'training_progress_stale' in r['errors']
    r=m.sample(s,None,now=2000,process_probe=lambda *a:'not_running')
    assert 'trainer_not_running' in r['errors']
    s['status']='interrupted'
    assert not m.sample(s,None,now=4000,process_probe=lambda *a:'not_running')['errors']
    source=Path(m.__file__).read_text()
    for forbidden in ('os.kill(', 'subprocess.Popen', 'torch.cuda.', 'onefoot-curriculum.stop'):
        assert forbidden not in source


def test_tensorboard_first_break_denominator_is_failed_trials_not_all_trials():
    m=load_local('tensorboard-evaluations.py')
    row=report(99,.5);row['failure_diagnostics']=dict(failed_episodes=128,recorded_break_episodes=100,
        failed_without_recorded_break=28,failed_without_established_streak=3,
        reason_counts=dict(forward_speed=64,left_support=64),mean_first_break_time_s=.92,
        mean_values_at_break=dict(speed_m_s=.099,waist_pitch_deg=None))
    values=m.scalar_values(row)
    prefix='Evaluation/FirstBreak/'
    assert values[prefix+'reason_pct_of_failed/forward_speed']==50.
    assert values[prefix+'mean_first_break_time_s']==.92
    assert prefix+'mean_waist_pitch_deg' not in values
    row['failure_diagnostics']['reason_counts']['forward_speed']=129
    with pytest.raises(ValueError):m.scalar_values(row)


def test_monitor_reports_nan_evidence_instead_of_crashing_or_hiding_it():
    import json
    m=load_local('watch-onefoot-training.py')
    result=m.safe_monitor_evidence(dict(severity='ok',errors=[],measurement=float('nan')))
    assert result['severity']=='needs_attention' and result['measurement'] is None
    assert 'nonfinite_monitor_evidence' in result['errors']
    json.dumps(result,allow_nan=False)


def test_review_notification_distinguishes_candidates_gates_and_duplicate_confirmation():
    m=load_local('wait-onefoot-review.py')
    before=dict(status='training',stage='balance-100',run_root='/run',passed_stages=[],best_checkpoint='best',
                evaluations=[report(99)],supervision=dict(anchor_iteration=99,allotted_evaluations=6))
    after=deepcopy(before);after['evaluations'] += [report(199),dict(report(199),seed=999)]
    assert m.review_reason(before,after,max_new=2) is None
    after['evaluations'].append(report(299))
    assert m.review_reason(before,after,max_new=2)=='scheduled_evidence_review'
    after=deepcopy(before);after['best_checkpoint']='candidate'
    assert m.review_reason(before,after)=='new_best_candidate_not_necessarily_gate_passed'
    after=deepcopy(before);after['supervision']['allotted_evaluations']=12
    assert m.review_reason(before,after)=='exploration_management_decision'
    after=deepcopy(before);after['stage']='balance-200'
    assert m.review_reason(before,after)=='stage_progress'
    after=deepcopy(before);after['status']='interrupted'
    assert m.review_reason(before,after)=='controller_paused'


def test_consolidation_cap_never_increases_quiet_joints_or_mutates_input():
    from mjlab_microduck.onefoot_supervision import cap_scalar_exploration_std
    before=torch.tensor([.014,.043,.096,.245]);saved=before.clone()
    after=cap_scalar_exploration_std(before,.08)
    assert torch.equal(before,saved)
    assert torch.equal(after,torch.tensor([.014,.043,.08,.08]))
    assert torch.all(after<=before)
    for maximum in (0.,-1.,float('nan'),float('inf'),2.):
        with pytest.raises(ValueError):cap_scalar_exploration_std(before,maximum)
    with pytest.raises(ValueError):cap_scalar_exploration_std(torch.tensor([float('nan')]),.08)


def test_consolidation_preserves_deterministic_actor_output():
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from mjlab_microduck.onefoot_supervision import cap_scalar_exploration_std
    obs=TensorDict({'actor':torch.randn(4,61)},batch_size=[4])
    actor=MLPModel(obs=obs,obs_groups={'actor':['actor']},obs_set='actor',output_dim=14,
                   hidden_dims=(32,24),obs_normalization=True,
                   distribution_cfg=dict(class_name='GaussianDistribution',init_std=.1,std_type='scalar'))
    actor.eval()
    with torch.no_grad():
        actor.distribution.std_param.copy_(torch.linspace(.01,.25,14))
        original=actor(obs).clone()
        saved={k:v.clone() for k,v in actor.state_dict().items()}
        actor.distribution.std_param.copy_(cap_scalar_exploration_std(actor.distribution.std_param,.08))
        assert torch.equal(original,actor(obs))
        for key,value in saved.items():
            if key!='distribution.std_param':assert torch.equal(actor.state_dict()[key],value)


def test_bad_confirmation_cannot_be_promoted_or_recovered_as_best():
    from mjlab_microduck.onefoot_supervision import safe_evaluation_score,eligible_stage_reports
    a=report(99,.85,1.1,max_glide_s=1.2);b=dict(a,seed=100,nan_episodes=1)
    with pytest.raises(ValueError):safe_evaluation_score([a,b])
    good=report(199,.7,1.01,max_glide_s=1.2)
    assert eligible_stage_reports([a,b,good],'balance-100')==[good]
    assert safe_evaluation_score([good])==(.7,1.01)
    source=Path(__file__).resolve().parents[1]/'local/train-onefoot-curriculum.py'
    text=source.read_text()
    assert text.index('score=safe_evaluation_score(reports)')<text.index("atomic_json(run/'best.json'")


def test_acknowledged_stagnation_does_not_loop_or_hide_new_progress_or_errors():
    m=load_local('wait-onefoot-review.py')
    before=dict(status='training',stage='balance-100',run_root='/run',pid=1,
                best_checkpoint='best',passed_stages=[],evaluations=[],
                supervision=dict(anchor_iteration=99,allotted_evaluations=6,action='continue',reason='minimum'))
    health=dict(run_root='/run',trainer_pid=1,severity='warning',warnings=['long_stagnation_review_due'])
    assert m.monitored_review_reason(before,before,health)=='long_stagnation_review'
    assert m.monitored_review_reason(before,before,health,acknowledged_stagnation=True) is None
    after=deepcopy(before);after['best_checkpoint']='candidate'
    assert m.monitored_review_reason(before,after,health,acknowledged_stagnation=True)=='new_best_candidate_not_necessarily_gate_passed'
    after=deepcopy(before);after['supervision']['action']='adapt'
    assert m.monitored_review_reason(before,after,health,acknowledged_stagnation=True)=='exploration_management_decision'
    health['severity']='needs_attention'
    assert m.monitored_review_reason(before,before,health,acknowledged_stagnation=True)=='resident_audit_attention'


def test_bounded_multichunk_train_never_turns_into_autonomous_reset_loop():
    m=load_local('train-onefoot-curriculum.py')
    assert m.bounded_run_finished('train',1,0)
    assert m.bounded_run_finished('smoke',1,1)
    assert not m.bounded_run_finished('auto',10000,0)
    for mode in ('auto','train'):
        assert not m.bounded_run_finished(mode,5,6)
        assert m.bounded_run_finished(mode,6,6)
    source=Path(m.__file__).read_text()
    assert "if a.mode!='auto':\n                        continue" in source
    assert "publish=a.mode=='auto'" in source


def test_best_checkpoint_restore_starts_fresh_window_without_resetting_std():
    m=load_local('train-onefoot-curriculum.py')
    original=[dict(stage='balance-100',iteration=12499,exploration_std=.18)]
    state=dict(adaptations=deepcopy(original))
    change=m.record_best_recovery_boundary(state,'balance-100',12800,'/run/model_6199.pt')
    assert state['adaptations'][:-1]==original
    assert change['restore']=='/run/model_6199.pt' and change['exploration_changed'] is False
    assert 'exploration_std' not in change
    rows=[report(6199,.7,1.02)]+[report(i,.2,.9) for i in (12599,12699,12799)]
    plan=exploration_plan(rows,'balance-100',state['adaptations'])
    assert plan['evaluated_checkpoints']==0 and plan['anchor_iteration']==12800
    plan=exploration_plan(rows+[report(12899,.69,1.01)],'balance-100',state['adaptations'])
    assert plan['evaluated_checkpoints']==1 and plan['action']=='continue'
    assert plan['allotted_evaluations']==6
    source=Path(m.__file__).read_text()
    assert "if not a.resume_latest:\n                    change=record_best_recovery_boundary" in source
    assert "r.get('reason')!='best_checkpoint_recovery'" in source


def reviewed_recovery_history():
    return history([.01]*3+[.10]*3+[.20]*3+[.30]*3+[.40]*3+[.52]*3)


def test_reviewed_extension_requires_all_three_completed_recovery_blocks():
    from mjlab_microduck.onefoot_supervision import reviewed_exploration_extension
    rows=reviewed_recovery_history();saved=deepcopy(rows)
    budget,plan=reviewed_exploration_extension(rows,'balance-100')
    assert budget.maximum_evaluations==24 and plan['evaluated_checkpoints']==18
    assert plan['anchor_iteration']==99 and plan['allotted_evaluations']==24
    assert len(plan['trends'])==3 and all(t['improving'] for t in plan['trends'])
    assert rows==saved and ExplorationBudget().maximum_evaluations==18
    flat=deepcopy(rows)
    for row in flat[-6:]:row['success_rate']=.4
    with pytest.raises(ValueError):reviewed_exploration_extension(flat,'balance-100')
    with pytest.raises(ValueError):reviewed_exploration_extension(rows[:-1],'balance-100')
    with pytest.raises(ValueError):reviewed_exploration_extension(rows+[report(1999,.53)],'balance-100')


@pytest.mark.parametrize('bad', ['nan_episodes','fixed_pose','action_noise_in_evaluation'])
def test_reviewed_extension_rejects_invalid_primary_or_confirmation(bad):
    from mjlab_microduck.onefoot_supervision import reviewed_exploration_extension
    rows=reviewed_recovery_history();confirm=dict(rows[-1],seed=888)
    confirm[bad]=1
    with pytest.raises(ValueError):reviewed_exploration_extension(rows+[confirm],'balance-100')


def test_reviewed_extra_block_is_not_revoked_by_one_weak_batch():
    from mjlab_microduck.onefoot_supervision import reviewed_exploration_extension
    rows=reviewed_recovery_history();budget,_=reviewed_exploration_extension(rows,'balance-100')
    for i in range(1,6):
        p=exploration_plan(rows+[report(1899+100*j,.01) for j in range(1,i+1)],'balance-100',budget=budget)
        assert p['action']=='continue' and p['allotted_evaluations']==24
    p=exploration_plan(rows+[report(1899+100*j,.01) for j in range(1,7)],'balance-100',budget=budget)
    assert p['action']=='adapt' and p['reason']=='exploration_budget_exhausted'


def test_bounded_auto_completion_is_a_resumable_saved_pause_not_a_skill_pass():
    m=load_local('train-onefoot-curriculum.py')
    state=dict(status='evaluating',stage_passed=False,passed_stages=[],checkpoint='/saved/model_99.pt')
    m.finish_bounded_run(state,'auto')
    assert state['status']=='interrupted' and state['pause_reason']=='explicit_chunk_budget_exhausted'
    assert state['checkpoint']=='/saved/model_99.pt' and not state['stage_passed'] and not state['passed_stages']
    for mode in ('train','smoke'):
        s={};m.finish_bounded_run(s,mode);assert s['status']=='bounded_test_completed'
    source=Path(m.__file__).read_text()
    assert "if a.review_extension:" in source and "or not a.resume_latest" in source
    assert "a.max_chunks=budget.block_evaluations" in source
    assert "plan=exploration_plan(state['evaluations'],stage.name,state['adaptations'],budget)" in source
    # Both a failed gate and a passed gate honor the same explicit compute bound.
    assert source.count('finish_bounded_run(state,a.mode);update();return')==2


def test_new_review_must_explicitly_match_the_recorded_budget():
    m=load_local('train-onefoot-curriculum.py')
    assert m.reviewed_base_budget({'supervision_config':{'maximum_evaluations':18}},18).maximum_evaluations==18
    assert m.reviewed_base_budget({'supervision_config':{'maximum_evaluations':24}},24).maximum_evaluations==24
    for recorded,requested in ((24,18),(30,24),(30,30),(18,24),(18,0)):
        with pytest.raises(ValueError):m.reviewed_base_budget({'supervision_config':{'maximum_evaluations':recorded}},requested)


def test_separately_reviewed_24_to_30_requires_recovery_in_fourth_block():
    from mjlab_microduck.onefoot_supervision import reviewed_exploration_extension
    rows=reviewed_recovery_history()+[report(1999+i*100,p) for i,p in enumerate([.53]*3+[.63]*3)]
    budget,plan=reviewed_exploration_extension(rows,'balance-100',budget=ExplorationBudget(maximum_evaluations=24))
    assert budget.maximum_evaluations==30 and plan['evaluated_checkpoints']==24
    assert len(plan['trends'])==4 and all(t['improving'] for t in plan['trends'])
    with pytest.raises(ValueError):reviewed_exploration_extension(rows,'balance-100')
    for row in rows[-6:]:row['success_rate']=.53
    with pytest.raises(ValueError):reviewed_exploration_extension(rows,'balance-100',budget=ExplorationBudget(maximum_evaluations=24))


def test_memory_ablation_is_not_stage_gate_evidence():
    rows=[report(99,.9,1.1),dict(report(99,.9,1.1),seed=900)]
    assert verified_gate_reports(rows,'balance-100',1.)
    rows[1]['memory_reset_each_step']=True
    assert not verified_gate_reports(rows,'balance-100',1.)


def test_history_comparison_initial_gate_and_explicit_training_bound():
    m=load_local('run-history-comparison.py')
    a=report(99,.66,.99);b=report(99,.65,.985)
    assert m.initial_parity(a,b)
    assert not m.initial_parity(a,dict(b,success_rate=.04))
    with pytest.raises(ValueError):m.initial_parity(a,dict(b,nan_episodes=1))
    with pytest.raises(ValueError):m.initial_parity(a,dict(b,memory_reset_each_step=True))
    manifest=dict(arms={'history':dict(recipe='waist-history-matched',checkpoint='/prepared/model_16399.pt',run_root='/new/run')},
        num_envs=4096,iterations_per_chunk=100,chunks_per_arm=3,training_seed=73)
    args=m.training_command('history',manifest)
    assert args[args.index('--mode')+1]=='train' and args[args.index('--max-chunks')+1]=='3'
    assert '--resume' in args and '--resume-state' not in args
    assert not any(arg.startswith('--exploration-') for arg in args)
