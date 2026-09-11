"""Frozen-policy timeline comparison; no training/promotion/state-file writes.

Records the existing evaluation-only observer before automatic resets. Includes
successful and failed FIRST episodes with explicit censoring masks. No MDP,
physics, reward, action, or observation changes are made by this diagnostic.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('timeline_trainer',ROOT/'local/train-onefoot-curriculum.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
from mjlab_microduck.onefoot_supervision import FAILURE_CHECK_NAMES,FAILURE_VALUE_NAMES,FirstBreakDiagnostics
from mjlab_microduck.tasks import mdp


def statistics(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not values.size:return dict(n=0,mean=None,median=None,p10=None,p90=None)
    return dict(n=int(values.size),mean=float(values.mean()),median=float(np.median(values)),
                p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))


def summarize_times(values,checks,alive,success,dt,times=(.5,.8,.9,1.,1.1,1.2)):
    output=[]
    names=list(FAILURE_VALUE_NAMES);check_names=list(FAILURE_CHECK_NAMES)
    for t in times:
        index=round(t/dt)-1
        for group,mask in [('all',np.ones_like(success,dtype=bool)),('successful',success),('failed',~success)]:
            present=mask & alive[index] if index<len(alive) else np.zeros_like(mask)
            row=dict(time_s=t,group=group,total_episodes=int(mask.sum()),present_episodes=int(present.sum()),
                     ended_before_sample=int((mask&~present).sum()))
            if present.any():
                sample=values[index,present]
                row['values']={name:statistics(sample[:,j]) for j,name in enumerate(names)}
                row['condition_pass_pct']={name:float(checks[index,present,j].mean()*100) for j,name in enumerate(check_names)}
                front=sample[:,names.index('left_front_force_n')];rear=sample[:,names.index('left_rear_force_n')]
                total=front+rear;loaded=np.isfinite(total)&(total>.1)
                row['loaded_episodes']=int(loaded.sum())
                row['front_load_fraction']=statistics(front[loaded]/total[loaded])
            output.append(row)
    return output


def summarize_kinematics(values,alive,success,dt,times=(.1,.2,.4,.6,.8,1.,1.2)):
    rows=[]
    for t in times:
        index=round(t/dt)-1
        for group,mask in [('all',np.ones_like(success,dtype=bool)),('successful',success),('failed',~success)]:
            present=mask & alive[index] if 0<=index<len(alive) else np.zeros_like(mask)
            row=dict(time_s=t,group=group,total_episodes=int(mask.sum()),present_episodes=int(present.sum()),
                     ended_before_sample=int((mask&~present).sum()))
            if present.any():
                row['values']={name:statistics(values[index,present,j]) for j,name in enumerate(mdp.SUPPORT_KINEMATIC_NAMES)}
            rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--recipe',choices=tuple(trainer.ENV_FACTORIES),default='waist')
    p.add_argument('--stage',choices=tuple(trainer.STAGE_BY_NAME),default='balance-100')
    p.add_argument('--episodes',type=int,default=256)
    p.add_argument('--seed',type=int,default=62101)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--support-kinematics',action='store_true')
    p.add_argument('--reset-memory-each-step',action='store_true',help='Diagnostic ablation only: discard recurrent history before every action; never gate evidence')
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    assert torch.cuda.device_count()==1 and '5070 Ti' in torch.cuda.get_device_name(0)
    checkpoint=a.checkpoint.resolve()
    assert checkpoint.is_relative_to(ROOT/'logs/rsl_rl/onefoot') and checkpoint.is_file()
    if a.output.exists() or a.output.with_suffix('.npz').exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    trainer.RECIPE=a.recipe;trainer.TASK=trainer.RECIPE_TASKS[a.recipe]
    trainer.configure_torch_backends();torch.set_num_threads(4)
    source_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    env,rl=trainer.environment(a.stage,a.episodes,a.seed,diagnostics=True,support_kinematics=a.support_kinematics)
    raw=env.unwrapped;cmd=raw.command_manager.get_term('twist')
    try:
        # Do not let the temporary actor/critic construction consume the rollout RNG.
        with torch.random.fork_rng(devices=[0]):
            runner=trainer.load_runner_cls(trainer.TASK)(env,deepcopy(asdict(rl)),device='cuda:0')
            trainer.safe_runner_load(runner,checkpoint,load_cfg={'actor':True},strict=True,map_location='cuda:0')
            actor=runner.get_inference_policy(device='cuda:0')
        if a.reset_memory_each_step and not actor.is_recurrent:
            raise ValueError('Memory ablation requires a recurrent policy')
        alive=torch.ones(a.episodes,dtype=torch.bool,device='cuda:0')
        best=torch.zeros(a.episodes,device='cuda:0');nan=torch.zeros_like(alive)
        injected=raw._curriculum_injected_speed.clone()
        observer=FirstBreakDiagnostics(a.episodes,raw.step_dt,env.device)
        history_values=[];history_checks=[];history_alive=[];history_done=[];history_kinematics=[]
        initial_kinematics=mdp.curriculum_onefoot_support_kinematics(raw).cpu().numpy() if a.support_kinematics else None
        with torch.no_grad():
            obs=env.get_observations();actor.reset()
            for step in range(raw.max_episode_length+1):
                if a.reset_memory_each_step:actor.reset()
                actions=actor(obs);nan|=alive & ~torch.isfinite(actions).all(1)
                obs,_,done,_=env.step(actions);actor.reset(done)
                snapshot=cmd.onefoot_diagnostics;v=cmd.curriculum_values
                assert snapshot['step']==raw.common_step_counter
                assert torch.equal(snapshot['checks'].all(1),v['single_support']>.5)
                history_alive.append(alive.clone())
                history_values.append(torch.where(alive[:,None],snapshot['values'],float('nan')))
                history_checks.append(snapshot['checks'] & alive[:,None])
                history_done.append(done.bool() & alive)
                if a.support_kinematics:
                    history_kinematics.append(torch.where(alive[:,None],snapshot['support_kinematics'],float('nan')))
                observer.observe(snapshot['checks'],snapshot['values'],alive,step)
                best=torch.maximum(best,torch.where(alive,v['best_dwell'],0))
                nan|=alive & raw.termination_manager.get_term('nan_state')
                alive &= ~done.bool()
                if not bool(alive.any()):break
        goal=trainer.STAGE_BY_NAME[a.stage].goal_s
        success=(best>=goal-1e-5)&~nan
        arrays=dict(values=torch.stack(history_values).cpu().numpy(),checks=torch.stack(history_checks).cpu().numpy(),
                    episode_alive=torch.stack(history_alive).cpu().numpy(),first_episode_done=torch.stack(history_done).cpu().numpy(),
                    success=success.cpu().numpy(),best_glide_s=best.cpu().numpy(),initial_speed_m_s=injected.cpu().numpy(),
                    value_names=np.array(FAILURE_VALUE_NAMES),check_names=np.array(FAILURE_CHECK_NAMES))
        arrays['time_s']=np.arange(1,len(history_values)+1)*raw.step_dt
        if a.support_kinematics:
            arrays.update(support_kinematics=torch.stack(history_kinematics).cpu().numpy(),
                          support_kinematic_names=np.array(mdp.SUPPORT_KINEMATIC_NAMES),initial_kinematics=initial_kinematics)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source_hash
        np.savez_compressed(a.output.with_suffix('.npz'),**arrays)
        diagnostics=observer.report(success);diagnostics.pop('trials',None)
        report=dict(scope='frozen policy diagnostic, not a training/promotion run',checkpoint=str(checkpoint),
                    checkpoint_sha256=source_hash,recipe=a.recipe,stage=a.stage,seed=a.seed,episodes=a.episodes,goal_s=goal,
                    success_rate=float(success.float().mean()),mean_best_glide_s=float(best.mean()),max_glide_s=float(best.max()),
                    nan_episodes=int(nan.sum()),action_noise_in_evaluation=False,memory_reset_each_step=a.reset_memory_each_step,step_dt=raw.step_dt,
                    timeline_npz=str(a.output.with_suffix('.npz')),failure_diagnostics=diagnostics,
                    fixed_time_samples=summarize_times(arrays['values'],arrays['checks'],arrays['episode_alive'],arrays['success'],raw.step_dt),
                    limitations=['Fixed-time means are conditional on the reported surviving episodes.',
                                 'Association is not proof that load asymmetry causes braking.',
                                 'Only initial episodes count; auto-reset trajectories are masked out.'])
        if a.support_kinematics:
            report['support_kinematic_samples']=summarize_kinematics(arrays['support_kinematics'],arrays['episode_alive'],arrays['success'],raw.step_dt)
            report['initial_kinematics']={name:statistics(initial_kinematics[:,j]) for j,name in enumerate(mdp.SUPPORT_KINEMATIC_NAMES)}
            report['limitations'].append('Knee rad values are MJCF joint coordinates, not anatomical bend angles; zero is already flexed.')
        if a.reset_memory_each_step:
            report['scope']='recurrent memory ablation, not gate or adoption evidence'
            report['limitations'].append('Hidden state is reset before every action; this is not normal recurrent inference.')
        trainer.atomic_json(a.output,report)
        print('TIMELINE_RECORDED',json.dumps({k:report[k] for k in ['checkpoint','seed','episodes','success_rate','mean_best_glide_s','nan_episodes','timeline_npz']}),flush=True)
    finally:
        env.close()


if __name__=='__main__':main()
