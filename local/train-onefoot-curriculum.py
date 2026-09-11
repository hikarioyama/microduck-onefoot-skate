"""Single-policy, evaluated reverse curriculum; runs until gates pass or intervention.

No services, other jobs, or hardware settings are modified. SIGINT/SIGTERM or a
local/onefoot-curriculum.stop file requests a checkpointed stop at the next chunk.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.tasks.microduck_onefoot_curriculum_env_cfg import (
    TASK, STAGES, STAGE_BY_NAME, POSES, GATE_EPISODES, GATE_SUCCESS_RATE,
    make_curriculum_onefoot_env_cfg, make_curriculum_onefoot_rl_cfg, gate_passes,
)

from mjlab_microduck.tasks.microduck_onefoot_continuous_env_cfg import (
    TASK as CONTINUOUS_TASK, make_continuous_onefoot_env_cfg, make_continuous_onefoot_rl_cfg,
)
from mjlab_microduck.tasks.microduck_onefoot_centroid_env_cfg import (
    TASK as CENTROID_TASK,make_centroid_onefoot_env_cfg,make_centroid_onefoot_rl_cfg,
)
ENV_FACTORIES={'curriculum':make_curriculum_onefoot_env_cfg,'continuous':make_continuous_onefoot_env_cfg,'centroid':make_centroid_onefoot_env_cfg}
RL_FACTORIES={'curriculum':make_curriculum_onefoot_rl_cfg,'continuous':make_continuous_onefoot_rl_cfg,'centroid':make_centroid_onefoot_rl_cfg}
RECIPE_TASKS={'curriculum':TASK,'continuous':CONTINUOUS_TASK,'centroid':CENTROID_TASK}
RUN_SUFFIXES={'curriculum':'curriculum-v5','continuous':'continuous-curriculum-v6','centroid':'centroid-curriculum-v7'}
from mjlab_microduck.tasks.microduck_onefoot_memory_env_cfg import (
    TASK as MEMORY_TASK,make_memory_onefoot_env_cfg,make_memory_onefoot_rl_cfg,
)
from mjlab_microduck.recurrent_warmstart import warm_gru_from_mlp
from mjlab_microduck.checkpoint_safety import safe_runner_load,evaluation_policy,verified_recovery
from mjlab_microduck.onefoot_supervision import (
    ExplorationBudget,exploration_plan,verified_gate_reports,FirstBreakDiagnostics,
    cap_scalar_exploration_std,safe_evaluation_score,eligible_stage_reports,
    reviewed_exploration_extension,
)
ENV_FACTORIES['memory']=make_memory_onefoot_env_cfg
RL_FACTORIES['memory']=make_memory_onefoot_rl_cfg
RECIPE_TASKS['memory']=MEMORY_TASK
RUN_SUFFIXES['memory']='memory-curriculum-v8'
from mjlab_microduck.tasks.microduck_onefoot_lean_env_cfg import (
    TASK as LEAN_TASK,make_lean_onefoot_env_cfg,make_lean_onefoot_rl_cfg,
)
ENV_FACTORIES['lean']=make_lean_onefoot_env_cfg
RL_FACTORIES['lean']=make_lean_onefoot_rl_cfg
RECIPE_TASKS['lean']=LEAN_TASK
RUN_SUFFIXES['lean']='lean-curriculum-v9'
from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import (
    TASK as WAIST_TASK,make_waist_onefoot_env_cfg,make_waist_onefoot_rl_cfg,
)
ENV_FACTORIES['waist']=make_waist_onefoot_env_cfg
RL_FACTORIES['waist']=make_waist_onefoot_rl_cfg
RECIPE_TASKS['waist']=WAIST_TASK
RUN_SUFFIXES['waist']='waist-curriculum-v10'
from mjlab_microduck.tasks.microduck_onefoot_knee_env_cfg import (
    TASK as KNEE_TASK,make_knee_onefoot_env_cfg,make_knee_onefoot_rl_cfg,
)
ENV_FACTORIES['knee']=make_knee_onefoot_env_cfg
RL_FACTORIES['knee']=make_knee_onefoot_rl_cfg
RECIPE_TASKS['knee']=KNEE_TASK
RUN_SUFFIXES['knee']='knee-curriculum-v11-pilot'
from mjlab_microduck.tasks.microduck_onefoot_history_env_cfg import (
    TASK as HISTORY_TASK,make_history_onefoot_env_cfg,make_history_onefoot_rl_cfg,
)
from mjlab_microduck.history_policy import (
    HistoryGRUModel,SplitHistoryGRUModel,warm_history_from_mlp,warm_split_history_from_mlp,
)
ENV_FACTORIES['waist-history']=make_history_onefoot_env_cfg
RL_FACTORIES['waist-history']=make_history_onefoot_rl_cfg
RECIPE_TASKS['waist-history']=HISTORY_TASK
RUN_SUFFIXES['waist-history']='waist-history-curriculum-v12-pilot'
from mjlab_microduck.tasks.microduck_onefoot_history_env_cfg import (
    MATCHED_TASK,make_matched_history_onefoot_env_cfg,make_matched_history_onefoot_rl_cfg,
)
ENV_FACTORIES['waist-history-matched']=make_matched_history_onefoot_env_cfg
RL_FACTORIES['waist-history-matched']=make_matched_history_onefoot_rl_cfg
RECIPE_TASKS['waist-history-matched']=MATCHED_TASK
RUN_SUFFIXES['waist-history-matched']='waist-history-matched-v12b-pilot'
RECIPE = 'curriculum'

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'local/onefoot-curriculum-state.json'
STOP = ROOT/'local/onefoot-curriculum.stop'
stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True
    print('STOP_REQUESTED: will save at chunk boundary',flush=True)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def environment(stage, num_envs, seed, diagnostics=False, support_kinematics=False):
    cfg = ENV_FACTORIES[RECIPE](stage=stage)
    cfg.scene.num_envs = num_envs; cfg.seed = seed
    cfg.sim.nan_guard.enabled = True
    if diagnostics or support_kinematics:
        from mjlab.managers.metrics_manager import MetricsTermCfg
        from mjlab_microduck.tasks import mdp
        observer=(mdp.observe_curriculum_onefoot_support_kinematics if support_kinematics
                  else mdp.observe_curriculum_onefoot_diagnostics)
        cfg.metrics['onefoot/diagnostic_valid']=MetricsTermCfg(func=observer,reduce='last')
    rl = RL_FACTORIES[RECIPE]()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg,device='cuda:0'),clip_actions=rl.clip_actions)
    obs = env.get_observations()
    assert obs['actor'].shape == (num_envs,61),obs['actor'].shape
    assert env.num_actions == 14
    return env,rl


def initialize(runner, env, stage, checkpoint=None, resume=False):
    if checkpoint:
        if resume:
            safe_runner_load(runner,checkpoint,strict=True,map_location='cuda:0')
            runner.current_learning_iteration += 1
        else:
            ckpt = torch.load(checkpoint,map_location='cuda:0',weights_only=False)
            current = runner.alg.actor.state_dict()
            prior = ckpt['actor_state_dict']
            if getattr(runner.alg.actor,'is_recurrent',False) and 'rnn.rnn.weight_ih_l0' not in prior:
                history=isinstance(runner.alg.actor,HistoryGRUModel)
                if isinstance(runner.alg.actor,SplitHistoryGRUModel):
                    transfer=warm_split_history_from_mlp(prior,current)
                else:
                    transfer=warm_history_from_mlp(prior,current) if history else warm_gru_from_mlp(prior,current)
                runner.alg.actor.load_state_dict(transfer,strict=True)
                runner.alg.critic.load_state_dict(ckpt['critic_state_dict'],strict=True)
                print('Direct-path history warm start with preserved std' if history else
                      'Approximate MLP-to-GRU warm start; unchanged sensors and fresh optimizer',flush=True)
            else:
                assert current.keys() == prior.keys()
                runner.alg.actor.load_state_dict({k:current[k] if k.startswith('distribution.') else prior[k]
                                                 for k in current},strict=True)
    else:
        robot = env.unwrapped.scene['robot']; ids,names = robot.find_joints(r'^(?!passive_).*')
        target = POSES[STAGE_BY_NAME[stage].spawn]['pose']
        bias = torch.tensor([target[n] for n in names],device='cuda:0')-robot.data.default_joint_pos[0,ids]
        last = [m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        with torch.no_grad():
            last.weight.zero_();last.bias.copy_(bias)


@torch.no_grad()
def evaluate(actor, stage_name, seed, episodes=GATE_EPISODES, fixed_pose=False, trace_path=None):
    """First episode of every world, deterministic actor, independent DR and seed."""
    # Construct from weights, not cached non-leaf tensors left by PPO updates.
    source_actor=actor
    env,rl = environment(stage_name,episodes,seed,diagnostics=True)
    raw=env.unwrapped; cmd=raw.command_manager.get_term('twist')
    try:
        obs=env.get_observations()
        actor=evaluation_policy(source_actor,obs,rl.actor,env.num_actions)
        alive=torch.ones(episodes,dtype=torch.bool,device='cuda:0')
        diagnostic=FirstBreakDiagnostics(episodes,raw.step_dt,env.device)
        best=torch.zeros(episodes,device='cuda:0'); length=best.clone(); distance=best.clone()
        nan=torch.zeros_like(alive); terms={name:0 for name in raw.termination_manager.active_terms}
        totals={k:0. for k in ('speed','com_speed','lateral','support_yaw','left_contact','right_contact','left_load','head_speed')}
        torso_totals={k:0. for k in ('waist_roll_deg','waist_abs_roll_deg','waist_abs_pitch_deg','waist_roll_speed_rad_s')}
        extra_totals={}
        frames=0; trace=[]; initial_speed=raw._curriculum_injected_speed.clone()
        robot=raw.scene['robot']; ids,names=robot.find_joints(r'^(?!passive_).*')
        target=POSES[STAGE_BY_NAME[stage_name].spawn]['pose']
        constant=torch.tensor([target[n] for n in names],device='cuda:0')-robot.data.default_joint_pos[:,ids]
        for step in range(raw.max_episode_length+1):
            # Read the pre-step body pose before an automatic reset can replace it.
            g=robot.data.projected_gravity_b
            roll=torch.atan2(-g[:,1],-g[:,2])*180/torch.pi
            pitch=torch.asin(g[:,0].clamp(-1,1))*180/torch.pi
            torso=dict(waist_roll_deg=roll,waist_abs_roll_deg=roll.abs(),waist_abs_pitch_deg=pitch.abs(),
                       waist_roll_speed_rad_s=robot.data.root_link_ang_vel_b[:,0].abs())
            for key,value in torso.items():torso_totals[key]+=float(value[alive].sum())
            actions=constant if fixed_pose else actor(obs)
            nan |= alive & ~torch.isfinite(actions).all(1)
            obs,rew,done,info=env.step(actions)
            actor.reset(done)
            v=cmd.curriculum_values  # cached BEFORE auto-reset; never recompute here
            captured=cmd.onefoot_diagnostics
            assert captured['step']==raw.common_step_counter
            assert torch.equal(captured['checks'].all(1),v['single_support']>.5), 'Diagnostic observer disagrees with physical success predicate'
            diagnostic.observe(captured['checks'],captured['values'],alive,step)
            best=torch.maximum(best,torch.where(alive,v['best_dwell'],0))
            distance=torch.where(alive,v['distance'],distance)
            length += alive*raw.step_dt
            nan |= alive & raw.termination_manager.get_term('nan_state')
            for name in terms:
                terms[name] += int((alive & raw.termination_manager.get_term(name)).sum())
            count=int(alive.sum());frames+=count
            for key in totals:totals[key]+=float(v[key][alive].sum())
            if hasattr(cmd,'waist_values'):
                for key in ('waist_axis_error','waist_capture_error','waist_foreaft_error','waist_foreaft_capture','waist_front_load'):
                    extra_totals[key]=extra_totals.get(key,0.)+float(cmd.waist_values[key][alive].sum())
            if trace_path:
                trace.append(dict(t=(step+1)*raw.step_dt,alive=int(alive.sum()),
                    **{k:float(v[k][0]) for k in ('speed','com_speed','clearance','upright','support_yaw',
                       'capture_error','left_force','right_force','left_load','dwell','head_speed')},
                    env0_alive=bool(alive[0])))
            alive &= ~done.bool()
            if not bool(alive.any()):break
        stage=STAGE_BY_NAME[stage_name]
        success=(best >= stage.goal_s-1e-5) & ~nan
        report=dict(stage=stage_name,goal_s=stage.goal_s,episodes=episodes,seed=seed,
            fixed_pose=fixed_pose,success_rate=float(success.float().mean()),
            success_050=float(((best>=.5-1e-5)&~nan).float().mean()),
            success_100=float(((best>=1.-1e-5)&~nan).float().mean()),
            success_200=float(((best>=2.-1e-5)&~nan).float().mean()),
            mean_best_glide_s=float(best.mean()),max_glide_s=float(best.max()),
            p10_best_glide_s=float(torch.quantile(best,.1)),mean_survival_s=float(length.mean()),
            mean_distance_m=float(distance.mean()),nan_episodes=int(nan.sum()),
            injected_speed_min=float(initial_speed.min()),injected_speed_max=float(initial_speed.max()),
            termination_counts=terms,recipe=RECIPE,
            **{'mean_'+k:total/max(frames,1) for k,total in (totals|torso_totals|extra_totals).items()})
        details=diagnostic.report(success)
        trials=details.pop('trials',[])
        report['failure_diagnostics']=details
        report['action_noise_in_evaluation']=False
        if trace_path:
            diagnostic_path=trace_path.with_name(trace_path.stem.removesuffix('.trace')+'.diagnostics.json')
            assert diagnostic_path!=trace_path
            atomic_json(diagnostic_path,dict(stage=stage_name,seed=seed,episodes=episodes,**details,trials=trials))
            report['failure_diagnostics_artifact']=str(diagnostic_path)
        distribution=getattr(source_actor,'distribution',None)
        if distribution is not None and hasattr(distribution,'std_param'):
            report['exploration_std_param_mean']=float(distribution.std_param.detach().mean())
        if stage_name=='self-launch':
            assert not bool(initial_speed.any()),'Final stage must start at rest'
        if trace_path:atomic_json(trace_path,trace)
        print('EVALUATION:',json.dumps(report),flush=True)
        return report
    finally:
        env.close();del env;gc.collect();torch.cuda.empty_cache()


def snapshot(run, cfg, rl, metadata):
    dump_yaml(run/'params/env.yaml',asdict(cfg));dump_yaml(run/'params/agent.yaml',asdict(rl))
    sources = [ROOT/'local/train-onefoot-curriculum.py',ROOT/'local/solve-onefoot-curriculum-poses.py',
               ROOT/'local/onefoot-curriculum-poses.json',ROOT/'src/mjlab_microduck/tasks/mdp.py']
    sources += list((ROOT/'src/mjlab_microduck/tasks').glob('microduck_onefoot*_env_cfg.py'))
    sources += [ROOT/'src/mjlab_microduck/tasks/microduck_onefoot_curriculum_poses.py',
                ROOT/'src/mjlab_microduck/recurrent_warmstart.py',
                ROOT/'src/mjlab_microduck/history_policy.py',
                ROOT/'src/mjlab_microduck/matched_optimizer.py',
                ROOT/'src/mjlab_microduck/checkpoint_safety.py',
                ROOT/'src/mjlab_microduck/onefoot_supervision.py']
    hashes={}
    for source in sources:
        dest=run/'source_snapshot'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    atomic_json(run/'provenance.json',dict(metadata,source_sha256=hashes))


def record_best_recovery_boundary(state,stage,iteration,checkpoint):
    """Record an applied best-weight restore without changing any learned std.

    Evaluations from BEFORE that restore belong to a different learning branch;
    they must not consume the restored policy's six-evaluation minimum window.
    Call only after safe_runner_load has succeeded. Latest-weight continuation
    deliberately keeps its existing exploration window.
    """
    change=dict(stage=stage,iteration=iteration,restore=str(checkpoint),
        reason='best_checkpoint_recovery',exploration_changed=False)
    state.setdefault('adaptations',[]).append(change)
    return change


def bounded_run_finished(mode,completed_chunks,max_chunks):
    """Train defaults to one chunk; an explicit bound permits a monitored pilot.

    This does not enable autonomous adaptation, stage promotion or live-state
    publication for a bounded train run.
    """
    return ((max_chunks>0 and completed_chunks>=max_chunks) or
            (mode!='auto' and max_chunks==0))


def finish_bounded_run(state,mode):
    # An explicitly bounded auto run is a saved pause, not a terminal curriculum
    # result; verified latest-checkpoint recovery remains possible afterwards.
    state['status']='interrupted' if mode=='auto' else 'bounded_test_completed'
    if mode=='auto':state['pause_reason']='explicit_chunk_budget_exhausted'


def reviewed_base_budget(previous,limit):
    """An explicit fresh allocation must name the stopped run's actual limit.

    Default 18 cannot silently repeat the earlier extension after 24 trials;
    the separately reviewed 24-to-30 block requires an explicit new argument.
    """
    recorded=previous.get('supervision_config',{}).get('maximum_evaluations',18)
    if limit not in (18,24) or recorded!=limit:
        raise ValueError('Review base limit must be 18 or 24 and match the stopped run; no automatic budget growth')
    return ExplorationBudget(maximum_evaluations=limit)


def main():
    global RECIPE,TASK
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('smoke','train','auto','eval','probe'),default='auto')
    p.add_argument('--stage',choices=list(STAGE_BY_NAME),default='balance-050')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--recipe',choices=tuple(ENV_FACTORIES))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--resume-state',type=Path)
    p.add_argument('--resume-latest',action='store_true',help='After an intentional pause, preserve the latest saved weights instead of restoring the historical best')
    p.add_argument('--review-extension',action='store_true',help='After an explicit saved recovery review, continue the same policy for exactly six more chunks, then pause')
    p.add_argument('--review-base-limit',type=int,choices=(18,24),default=18,help='Name the stopped run limit explicitly; 24 requires a new review rather than repeating the original 18-to-24 extension')
    p.add_argument('--exploration-std',type=float)
    p.add_argument('--exploration-max-std',type=float,help='Cap scalar Gaussian exploration without increasing already-quiet joints')
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--iterations',type=int,default=200,help='Iterations per evaluated chunk')
    p.add_argument('--seed',type=int,default=73)
    p.add_argument('--output',type=Path)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--max-chunks',type=int,default=0,help='Auto: 0 runs until gates/intervention. Train: 0 runs one chunk; positive values bound a multi-chunk pilot.')
    a=p.parse_args()
    if a.max_chunks<0 or a.iterations<1 or a.num_envs<1:
        p.error('max-chunks must be nonnegative; iterations and num-envs must be positive')
    previous=None;recovery_iteration=None
    if a.resume_latest and not a.resume_state:
        raise ValueError('--resume-latest requires --resume-state')
    if a.resume_state:
        if a.mode!='auto' or a.checkpoint or a.run_dir or a.resume:
            raise ValueError('Use --resume-state with auto mode, without checkpoint/run-dir/resume overrides')
        previous=json.loads(a.resume_state.read_text())
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Prior controller is still running; refusing a duplicate trainer')
        a.stage,checkpoint,recovery_iteration=verified_recovery(previous,STAGES,gate_passes,prefer_latest=a.resume_latest)
        if a.recipe and a.recipe!=previous['recipe']:raise ValueError('Recovery cannot change the recipe')
        a.recipe=previous['recipe'];a.checkpoint=Path(checkpoint);a.resume=True
        root_path=Path(previous['run_root']).resolve()
        if not root_path.is_relative_to((ROOT/'logs/rsl_rl/onefoot').resolve()):
            raise ValueError('Recovery run is outside the experiment log directory')
        if not a.checkpoint.resolve().is_relative_to(root_path) or not a.checkpoint.is_file():
            raise ValueError('Recovery checkpoint is missing or outside the run')
    if a.exploration_std is not None and a.exploration_max_std is not None:
        raise ValueError('Choose a fixed exploration std OR a cap, not both')
    for value in (a.exploration_std,a.exploration_max_std):
        if value is not None and not 0<value<=1:
            raise ValueError('Exploration std must be in (0,1]')
    budget=ExplorationBudget();review_evidence=None
    if a.review_base_limit!=18 and not a.review_extension:
        raise ValueError('--review-base-limit requires an explicitly requested reviewed extension')
    if a.review_extension:
        if (previous is None or a.mode!='auto' or not a.resume_latest or
                a.exploration_std is not None or a.exploration_max_std is not None or a.max_chunks not in (0,6)):
            raise ValueError('Reviewed extension requires auto/latest state recovery, no exploration override, and a six-chunk bound')
        if a.recipe!='waist' or a.stage!=previous['stage'] or a.iterations!=100 or a.num_envs!=4096:
            raise ValueError('This reviewed extension preserves the existing waist stage, 4096 environments and 100-update chunks')
        base_budget=reviewed_base_budget(previous,a.review_base_limit)
        budget,review_evidence=reviewed_exploration_extension(previous['evaluations'],a.stage,previous['adaptations'],base_budget)
        a.max_chunks=budget.block_evaluations
    RECIPE=a.recipe or 'curriculum'
    TASK=RECIPE_TASKS[RECIPE]
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert '5070 Ti' in torch.cuda.get_device_name(0)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    assert not a.resume or a.checkpoint
    configure_torch_backends();torch.set_num_threads(8)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
    if a.mode in ('probe','eval'):
        env,rl=environment(a.stage,64,a.seed)
        runner=load_runner_cls(TASK)(env,deepcopy(asdict(rl)),device='cuda:0')
        initialize(runner,env,a.stage,a.checkpoint,resume=a.resume)
        report=evaluate(runner.alg.actor,a.stage,a.seed,episodes=a.num_envs,
                        fixed_pose=a.mode=='probe',trace_path=a.output.with_suffix('.trace.json') if a.output else None)
        if a.output:atomic_json(a.output,report)
        env.close();return
    if a.mode=='smoke':a.num_envs=64;a.iterations=5;a.max_chunks=1
    name=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+('smoke-' if a.mode=='smoke' else '')+RUN_SUFFIXES[RECIPE]
    root=Path(previous['run_root']) if previous else a.run_dir or ROOT/'logs/rsl_rl/onefoot'/name
    root.mkdir(parents=True,exist_ok=previous is not None)
    state=dict(status='running',pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),
               run_root=str(root),recipe=RECIPE,task=TASK,stage=a.stage,passed_stages=[],evaluations=[],adaptations=[],
               final_self_launch_completed=False,mode=a.mode,checkpoint=None,
               stop_file=str(STOP),command_contract='[speed, lift_phase, heading_correction]')
    recovery_stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if previous:
        atomic_json(root/'recoveries'/recovery_stamp/'state-before.json',previous)
        state=deepcopy(previous)
        state.setdefault('recoveries',[]).append(dict(time=recovery_stamp,
            error=state.pop('error',None),checkpoint=str(a.checkpoint),next_iteration=recovery_iteration,
            selection='latest_saved' if a.resume_latest else 'best_checkpoint'))
        state.update(status='recovering',pid=os.getpid(),stage=a.stage,mode='auto')
        pending=[r for r in state.get('adaptations',[]) if r['stage']==a.stage and r['iteration']==previous['iteration']]
        if a.resume_latest and pending:
            state['recoveries'][-1]['in_memory_adaptation_not_restored']=deepcopy(pending[-1])
    state['supervision_version']=1
    state['supervision_config']=asdict(budget)
    if review_evidence is not None:
        state.setdefault('review_extensions',[]).append(dict(checkpoint=str(a.checkpoint),
            next_iteration=recovery_iteration,additional_chunks=a.max_chunks,updates_per_chunk=a.iterations,
            reviewed_base_limit=a.review_base_limit,
            preserve_learned_exploration=True,evidence=deepcopy(review_evidence)))
        state['supervision']=dict(stage=a.stage,iteration=previous['iteration'],**review_evidence)
        print('REVIEWED_FINITE_EXTENSION:',json.dumps(state['review_extensions'][-1]),flush=True)
    state['training_config']=dict(num_envs=a.num_envs,iterations_per_chunk=a.iterations,seed=a.seed)
    try:
        log_path=Path(os.readlink('/proc/self/fd/1')).resolve()
        if log_path.is_file() and log_path.is_relative_to(ROOT):state['training_log']=str(log_path)
    except OSError:pass
    # Smoke/bounded training cannot replace the ongoing autonomous controller state.
    publish=a.mode=='auto'
    def update():
        state['updated']=datetime.now(timezone.utc).isoformat()
        atomic_json(root/'state.json',state)
        if publish:atomic_json(STATE,state)
    checkpoint=a.checkpoint; resume=a.resume; chunk_total=0
    first=next(i for i,s in enumerate(STAGES) if s.name==a.stage)
    if a.mode=='auto' and first!=0 and previous is None:
        raise ValueError('Autonomous curriculum must start at its first gate; use verified state recovery')
    try:
        for stage_index in range(first,len(STAGES)):
            if stop_requested or (a.mode=='auto' and STOP.exists()):
                state['status']='interrupted';update();return
            stage=STAGES[stage_index];state['stage']=stage.name
            recovering=previous is not None and stage_index==first
            run=root/stage.name;run.mkdir(exist_ok=recovering)
            env,rl=environment(stage.name,a.num_envs,a.seed+stage_index)
            rl.logger='tensorboard';rl.max_iterations=a.iterations
            runner=load_runner_cls(TASK)(env,deepcopy(asdict(rl)),str(run),device='cuda:0')
            initialize(runner,env,stage.name,checkpoint,resume)
            if recovering:
                runner.current_learning_iteration=max(runner.current_learning_iteration,recovery_iteration)
                if not a.resume_latest:
                    change=record_best_recovery_boundary(state,stage.name,runner.current_learning_iteration,checkpoint)
                    print('BEST_CHECKPOINT_RECOVERY:',json.dumps(change),flush=True)
            if stage_index==first and (a.exploration_std is not None or a.exploration_max_std is not None):
                parameter=runner.alg.actor.distribution.std_param
                before=parameter.detach().clone()
                with torch.no_grad():
                    if a.exploration_max_std is not None:
                        if rl.actor.distribution_cfg.get('std_type')!='scalar':
                            raise ValueError('The exploration cap currently supports scalar Gaussian policies only')
                        parameter.copy_(cap_scalar_exploration_std(parameter,a.exploration_max_std))
                    else:parameter.fill_(a.exploration_std)
                for item in runner.alg.actor.distribution.parameters():runner.alg.optimizer.state.pop(item,None)
                change=dict(stage=stage.name,iteration=runner.current_learning_iteration,restore=str(checkpoint),
                    exploration_std=float(parameter.detach().mean()),
                    reason='consolidation_noise_cap' if a.exploration_max_std is not None else 'explicit checkpoint recovery',
                    exploration_max_std=a.exploration_max_std,std_before=before.cpu().tolist(),
                    std_after=parameter.detach().cpu().tolist(),
                    joint_names=env.unwrapped.scene['robot'].find_joints(r'^(?!passive_).*')[1])
                state['adaptations'].append(change)
                print('EXPLORATION_ADJUSTMENT:',json.dumps(change),flush=True)
            snapshot(run/'recoveries'/recovery_stamp if recovering else run,env.cfg,rl,
                dict(task=TASK,recipe=RECIPE,stage=asdict(stage),initial_checkpoint=str(checkpoint),resume=resume))
            runner.add_git_repo_to_log(__file__)
            best_score=(-1.,-1.);best_checkpoint=None;branch=0;batch=0
            if recovering:
                all_rows=[r for r in state['evaluations'] if r['stage']==stage.name]
                rows=eligible_stage_reports(all_rows,stage.name)
                if rows:
                    best=max(rows,key=lambda r:(r['success_rate'],r['mean_best_glide_s']))
                    best_score=(best['success_rate'],best['mean_best_glide_s']);best_checkpoint=Path(best['checkpoint'])
                if all_rows:
                    batch=max((r['seed']-(12000+stage_index*1000))//2 for r in all_rows)+1
                branch=sum(r.get('stage')==stage.name and r.get('reason')!='best_checkpoint_recovery'
                           for r in state.get('adaptations',[]))
            try:
                while not stop_requested and (a.mode!='auto' or not STOP.exists()):
                    if shutil.disk_usage(ROOT).free < 20*1024**3:
                        raise RuntimeError('Less than 20 GiB disk space; safe checkpointed stop')
                    state.update(status='training',run_dir=str(run),iteration=runner.current_learning_iteration)
                    update()
                    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
                    if runner.logger.writer is not None:runner.logger.writer.close()
                    iteration=runner.current_learning_iteration
                    checkpoint=run/f'model_{iteration}.pt'
                    assert checkpoint.is_file()
                    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    runner.current_learning_iteration=iteration+1
                    state.update(checkpoint=str(checkpoint),iteration=iteration,status='evaluating');update()
                    if a.mode=='smoke':
                        import onnx
                        exports=list(run.rglob('*.onnx'));assert exports,'Canonical ONNX export did not succeed'
                        for file in exports:onnx.checker.check_model(onnx.load(str(file)))
                    eval_seed=12000+stage_index*1000+batch*2
                    row=evaluate(runner.alg.actor,stage.name,eval_seed,
                                 episodes=64 if a.mode=='smoke' else GATE_EPISODES,
                                 trace_path=run/f'eval_{iteration}.trace.json')
                    row['checkpoint']=str(checkpoint);row['checkpoint_sha256']=checkpoint_hash
                    atomic_json(run/f'eval_{iteration}.json',row)
                    state['evaluations'].append(row)
                    if row['nan_episodes']:
                        raise RuntimeError('Independent evaluation found non-finite states; preserve best and diagnose')
                    reports=[row]
                    if a.mode=='auto' and row['success_rate']>=GATE_SUCCESS_RATE:
                        confirm=evaluate(runner.alg.actor,stage.name,eval_seed+1)
                        confirm['checkpoint']=str(checkpoint);confirm['checkpoint_sha256']=checkpoint_hash
                        state['evaluations'].append(confirm);reports.append(confirm)
                        atomic_json(run/f'eval_{iteration}_confirm.json',confirm)
                    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=checkpoint_hash:
                        raise RuntimeError('Evaluated checkpoint changed during evaluation')
                    score=safe_evaluation_score(reports)
                    if score>best_score:
                        best_score=score;best_checkpoint=checkpoint
                        atomic_json(run/'best.json',dict(checkpoint=str(checkpoint),evaluation=row))
                    passed=verified_gate_reports(reports,stage.name,stage.goal_s)
                    state['best_checkpoint']=str(best_checkpoint);state['stage_passed']=passed;update()
                    chunk_total+=1;batch+=1
                    if passed and a.mode=='auto':
                        state['passed_stages'].append(dict(stage=stage.name,checkpoint=str(checkpoint),reports=reports))
                        resume=True;update()
                        if bounded_run_finished(a.mode,chunk_total,a.max_chunks):
                            finish_bounded_run(state,a.mode);update();return
                        break
                    if bounded_run_finished(a.mode,chunk_total,a.max_chunks):
                        finish_bounded_run(state,a.mode);update();return
                    if a.mode!='auto':
                        continue  # Fixed-budget pilot: no old-best/std reset or stage promotion.
                    # Honor a saved pause before any unsaved weight/std change.
                    if stop_requested or STOP.exists():
                        state['status']='interrupted';update();return
                    plan=exploration_plan(state['evaluations'],stage.name,state['adaptations'],budget)
                    signature=(plan['action'],plan['reason'],plan['anchor_iteration'],plan['allotted_evaluations'])
                    old=state.get('supervision',{})
                    previous_signature=(old.get('action'),old.get('reason'),old.get('anchor_iteration'),old.get('allotted_evaluations'))
                    state['supervision']=dict(stage=stage.name,iteration=iteration,**plan)
                    if signature!=previous_signature:
                        state.setdefault('supervision_history',[]).append(deepcopy(state['supervision']))
                        print('SUPERVISION:',json.dumps(state['supervision']),flush=True)
                    update()
                    if plan['action']=='diagnose':raise RuntimeError('Supervision requested numerical-failure diagnosis')
                    if plan['action']=='adapt':
                        # Keep the best policy, vary exploration only. Never loosen
                        # the success criterion, add stabilizing forces, or skip gates.
                        next_iteration=runner.current_learning_iteration
                        safe_runner_load(runner,best_checkpoint,strict=True,map_location='cuda:0')
                        runner.current_learning_iteration=next_iteration
                        std=(.08,.18,.25)[branch%3]
                        with torch.no_grad():runner.alg.actor.distribution.std_param.fill_(std)
                        for parameter in runner.alg.actor.distribution.parameters():
                            runner.alg.optimizer.state.pop(parameter,None)
                        change=dict(stage=stage.name,iteration=iteration,restore=str(best_checkpoint),exploration_std=std,
                                    reason=plan['reason'],evaluated_checkpoints=plan['evaluated_checkpoints'])
                        state['adaptations'].append(change);print('ADAPTATION:',json.dumps(change),flush=True)
                        # The restored policy starts a fresh physical episode;
                        # reset recurrent tensors in-place so RSL storage sees zeros.
                        with torch.inference_mode():
                            env.reset()
                            done=torch.ones(env.num_envs,dtype=torch.bool,device='cuda:0')
                            runner.alg.actor.reset(done);runner.alg.critic.reset(done)
                        branch+=1;update()
                else:
                    state['status']='interrupted';update();return
            finally:
                env.close();del runner,env;gc.collect();torch.cuda.empty_cache()
            if stage.name=='self-launch':
                state['final_self_launch_completed']=True
                state['status']='simulation_gates_passed_visual_review_pending';update();return
    except BaseException as error:
        state['status']='error';state['error']=str(error);update();raise


if __name__=='__main__':main()
