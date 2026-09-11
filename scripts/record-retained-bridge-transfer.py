"""Frozen first-episode transfer timeline; no training or physics changes."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.checkpoint_safety import safe_runner_load
from mjlab_microduck.onefoot_supervision import FAILURE_CHECK_NAMES, FAILURE_VALUE_NAMES
from mjlab_microduck.retained_skill_bridge import retained_skills_digest
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_waist_env_cfg import TASK

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bridge_verification_timeline_base', ROOT/'local/verify-retained-skill-bridge.py')
verify = importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)


def summarize(arrays, dt):
    alive = arrays['alive']; done = arrays['done']; terms = arrays['termination_flags']
    length = alive.sum(0)*dt
    rate_integrals = np.nansum(arrays['reward_rates'], axis=0)*dt
    episode_return = np.nansum(arrays['rewards'], axis=0)
    np.testing.assert_allclose(rate_integrals.sum(1), episode_return, atol=1e-5, rtol=1e-5)
    result = dict(episode_length_s=verify.diag.statistics(length),
                  episode_return=verify.diag.statistics(episode_return),
                  termination_counts={name:int((terms[:,:,i]&done).any(0).sum())
                                      for i,name in enumerate(arrays['termination_names'])},
                  reward_contributions={name:verify.diag.statistics(rate_integrals[:,i])
                                        for i,name in enumerate(arrays['reward_names'])},
                  fixed_time_samples=[])
    for t in (2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.5):
        index = round(t/dt)-1
        present = alive[index] if index < len(alive) else np.zeros(alive.shape[1],bool)
        row = dict(time_s=t, present_episodes=int(present.sum()), ended_before_sample=int((~present).sum()))
        if present.any():
            row['values']={name:verify.diag.statistics(arrays['values'][index,present,i])
                           for i,name in enumerate(FAILURE_VALUE_NAMES)}
            row['kinematics']={name:verify.diag.statistics(arrays['kinematics'][index,present,i])
                               for i,name in enumerate(mdp.SUPPORT_KINEMATIC_NAMES)}
            row['correction_max_abs_rad']=verify.diag.statistics(arrays['correction_max_abs'][index,present])
        result['fixed_time_samples'].append(row)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--episodes',type=int,default=64)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert 1<=a.episodes<=256
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
    assert torch.cuda.device_count()==1 and '5070 Ti' in torch.cuda.get_device_name(0)
    configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
    if a.output.exists() or a.output.with_suffix('.npz').exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    main_hash=hashlib.sha256(verify.diag.STATE.read_bytes()).hexdigest()
    cfg=verify.diag.environment_cfg(a.episodes,a.seed,6.,2.,'home')
    rl=verify.bridge_rl_cfg()
    env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg,device='cuda:0'),clip_actions=rl.clip_actions)
    try:
        raw=env.unwrapped;obs=env.get_observations()
        assert torch.count_nonzero(raw.sim.data.qvel)==0
        with torch.random.fork_rng(devices=[0]):
            runner=load_runner_cls(TASK)(env,deepcopy(asdict(rl)),device='cuda:0')
            safe_runner_load(runner,a.checkpoint,load_cfg={'actor':True},strict=True,map_location='cuda:0')
            actor=runner.get_inference_policy(device='cuda:0')
        digest=retained_skills_digest(actor)
        alive=torch.ones(a.episodes,dtype=torch.bool,device='cuda:0')
        best=torch.zeros(a.episodes,device='cuda:0');nan=torch.zeros_like(alive)
        history={name:[] for name in ('alive','done','termination_flags','reward_rates','rewards','values',
                                     'kinematics','observations_before','actions','correction_max_abs','qpos_before','qvel_before')}
        reward_names=raw.reward_manager.active_terms
        termination_names=raw.termination_manager.active_terms
        with torch.no_grad():
            for step in range(raw.max_episode_length+1):
                x=obs['actor'];phase=x[:,49:50].clamp(0,1)
                correction=4*phase*(1-phase)*actor.mlp(actor.obs_normalizer(x))
                actions=actor(obs)
                if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite diagnostic action')
                for key,value in [('observations_before',x),('actions',actions),('qpos_before',raw.sim.data.qpos),('qvel_before',raw.sim.data.qvel)]:
                    history[key].append(torch.where(alive[:,None],value,float('nan')).clone())
                history['correction_max_abs'].append(torch.where(alive,correction.abs().amax(1),float('nan')))
                obs,reward,done,_=env.step(actions)
                cmd=raw.command_manager.get_term('twist');snapshot=cmd.onefoot_diagnostics
                assert snapshot['step']==raw.common_step_counter
                history['alive'].append(alive.clone());history['done'].append(done.bool()&alive)
                flags=torch.stack([raw.termination_manager.get_term(name) for name in termination_names],-1)
                history['termination_flags'].append(flags&alive[:,None])
                # Pinned mjlab1.3: reset clears episode sums, NOT these per-step rates.
                # The report validates their dt-scaled sum against returned rewards.
                history['reward_rates'].append(torch.where(alive[:,None],raw.reward_manager._step_reward,float('nan')).clone())
                history['rewards'].append(torch.where(alive,reward,float('nan')).clone())
                history['values'].append(torch.where(alive[:,None],snapshot['values'],float('nan')))
                history['kinematics'].append(torch.where(alive[:,None],snapshot['support_kinematics'],float('nan')))
                nan|=alive&raw.termination_manager.get_term('nan_state')
                best=torch.maximum(best,torch.where(alive,cmd.curriculum_values['best_dwell'],0))
                alive&=~done.bool()
                if not bool(alive.any()):break
        arrays={name:torch.stack(items).cpu().numpy() for name,items in history.items()}
        arrays.update(reward_names=np.array(reward_names),termination_names=np.array(termination_names),
                      value_names=np.array(FAILURE_VALUE_NAMES),kinematic_names=np.array(mdp.SUPPORT_KINEMATIC_NAMES),
                      best_glide_s=best.cpu().numpy(),time_s=np.arange(1,len(history['alive'])+1)*raw.step_dt)
        report=summarize(arrays,raw.step_dt)
        report.update(checkpoint=str(a.checkpoint),checkpoint_sha256=checkpoint_hash,seed=a.seed,episodes=a.episodes,
                      frozen_skills_sha256=digest,initial_velocity_max_abs=0.,nan_episodes=int(nan.sum()),
                      mean_best_glide_s=float(best.mean()),max_glide_s=float(best.max()),
                      scope='Frozen transfer failure/return diagnostic, not gate evidence',
                      limitations=['Terminal causes overlap; counts must not be summed as disjoint categories.',
                                   'Fixed-time means condition on the reported surviving first episodes.',
                                   'Reward association is not causal proof; returned episode reward is undiscounted.'])
        assert retained_skills_digest(actor)==digest
        assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==checkpoint_hash
        assert hashlib.sha256(verify.diag.STATE.read_bytes()).hexdigest()==main_hash
        np.savez_compressed(a.output.with_suffix('.npz'),**arrays)
        a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print('BRIDGE_TRANSFER_RECORDED',json.dumps({k:report[k] for k in ('checkpoint','seed','episodes','episode_length_s','episode_return','termination_counts','nan_episodes')}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
