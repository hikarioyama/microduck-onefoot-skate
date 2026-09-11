"""Bounded v15 bridge training with verified continuation and explicit stage gates.

Use full first to isolate reward changes. An optional near stage is explicitly
assisted and is never reported as a from-rest completion or stage promotion.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.checkpoint_safety import evaluation_policy, safe_runner_load
from mjlab_microduck.retained_skill_bridge import retained_skills_digest, ONEFOOT_PATH, ONEFOOT_SHA256
from mjlab_microduck.bridge_recovery import (verified_bridge_source, verified_bridge_transition,
                                               restore_bridge_continuation, bridge_stage_gate)
from mjlab_microduck.public_residual_policy import PUBLIC_PATH
from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import (
    TASK, BRIDGE_CURRICULUM, SUSTAINED_STAGES, make_credit_bridge_env_cfg, make_credit_bridge_rl_cfg,
)

ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'local/onefoot-curriculum-state.json'
INITIAL=ROOT/'logs/rsl_rl/onefoot/2026-09-10_12-59-15_retained-skill-bridge-pilot/model_initial.pt'
INITIAL_SHA256='9fcbaeebc5889add18285c4dbe06feb33d6223ff22aea94fafd7a36e72e08c9e'
stop_requested=False


def request_stop(signum,frame):
    global stop_requested
    stop_requested=True
    print('STOP_REQUESTED: save after current bounded chunk',flush=True)


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def environment(n,seed,stage='full'):
    cfg=make_credit_bridge_env_cfg(stage=stage);cfg.scene.num_envs=n;cfg.seed=seed
    rl=make_credit_bridge_rl_cfg()
    assert cfg.commands['twist'].reward_gamma==rl.algorithm.gamma
    assert all(not term.time_out for term in cfg.terminations.values())
    env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg,device='cuda:0'),clip_actions=rl.clip_actions)
    obs=env.get_observations()
    assert obs['actor'].shape==(n,61) and obs['critic'].shape==(n,78) and env.num_actions==14
    if stage=='full':assert torch.count_nonzero(env.unwrapped.sim.data.qvel)==0
    return env,rl


@torch.no_grad()
def evaluate(actor,rl,seed,episodes,stage='full'):
    with torch.random.fork_rng(devices=[0]):
        env,_=environment(episodes,seed,stage)
        try:
            raw=env.unwrapped;obs=env.get_observations()
            policy=evaluation_policy(actor,obs,rl.actor,14)
            alive=torch.ones(episodes,dtype=torch.bool,device='cuda:0')
            nan=torch.zeros_like(alive)
            best=torch.zeros(episodes,device='cuda:0');credible=torch.zeros_like(best)
            total=torch.zeros(episodes,dtype=torch.float64,device='cuda:0')
            base=torch.zeros_like(total);shaping=torch.zeros_like(total);undiscounted=torch.zeros_like(total)
            handoff=torch.zeros_like(total);entry=torch.zeros_like(total)
            lengths=torch.zeros(episodes,dtype=torch.long,device='cuda:0')
            term_counts={name:torch.zeros_like(alive) for name in raw.termination_manager.active_terms}
            initial_velocity=float(raw.sim.data.qvel.abs().max())
            for step in range(raw.max_episode_length+1):
                action=policy(obs)
                if not torch.isfinite(action).all():raise RuntimeError('Nonfinite evaluation action')
                obs,reward,done,extras=env.step(action)
                if 'time_outs' in extras:assert not extras['time_outs'].any(), 'No finite-horizon bootstrap allowed'
                cmd=raw.command_manager.get_term('twist');r=cmd.bridge_reward_values
                assert torch.allclose(reward,r['total'],atol=1e-5,rtol=1e-5), 'Reward manager double-scaled events'
                best=torch.maximum(best,torch.where(alive,cmd.curriculum_values['best_dwell'],0.))
                credible=torch.maximum(credible,torch.where(alive,r['credible_best_s'],0.))
                gamma=rl.algorithm.gamma**step
                total+=gamma*torch.where(alive,reward,0.).double()
                base+=gamma*torch.where(alive,r['task']+r['failure']+r['handoff'],0.).double()
                shaping+=gamma*torch.where(alive,r['shaping'],0.).double()
                undiscounted+=torch.where(alive,reward,0.).double()
                handoff+=gamma*torch.where(alive,r['handoff'],0.).double()
                entry=torch.maximum(entry,torch.where(alive,r['entry_speed'],0.).double())
                lengths+=alive.long()
                nan|=alive&raw.termination_manager.get_term('nan_state')
                for name in term_counts:
                    term_counts[name]|=alive&done.bool()&raw.termination_manager.get_term(name)
                alive&=~done.bool()
                if not bool(alive.any()):break
            assert not alive.any()
            assert float(shaping.abs().max())<2e-5, 'Terminal potential leaked across episodes'
            assert torch.allclose(total,base,atol=2e-5,rtol=1e-5)
            return dict(seed=seed,episodes=episodes,stage=stage,initial_assistance=stage!='full',
                initial_velocity_max_abs=initial_velocity,mean_best_glide_s=float(best.mean()),max_glide_s=float(best.max()),
                success_050_rate=float(((best>=.5-1e-5)&~nan).float().mean()),
                success_100_rate=float(((best>=1.-1e-5)&~nan).float().mean()),
                success_200_rate=float(((best>=2.-1e-5)&~nan).float().mean()),
                mean_credible_glide_s=float(credible.mean()),max_credible_glide_s=float(credible.max()),
                credible_050_rate=float(((credible>=.5-1e-5)&~nan).float().mean()),
                credible_100_rate=float(((credible>=1.-1e-5)&~nan).float().mean()),
                credible_200_rate=float(((credible>=2.-1e-5)&~nan).float().mean()),
                goal_s=raw.command_manager.get_term('twist').cfg.goal_s,
                discounted_return=float(total.mean()),discounted_base_return=float(base.mean()),
                discounted_shaping_max_abs=float(shaping.abs().max()),undiscounted_return=float(undiscounted.mean()),
                discounted_handoff_return=float(handoff.mean()),mean_entry_speed=float(entry.mean()),
                max_entry_speed=float(entry.max()),
                mean_episode_s=float(lengths.float().mean()*raw.step_dt),nan_episodes=int(nan.sum()),
                termination_counts={name:int(value.sum()) for name,value in term_counts.items()},
                gate_eligible=False,scope='Reward/skill screening; formal valid predicate and credible motion logged separately')
        finally:env.close()


def check_curriculum_gate(actor,rl,stage,checkpoint,output):
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reports=[];base_seed=71301+100*BRIDGE_CURRICULUM.index(stage)
    for seed in (base_seed,base_seed+1):
        if stop_requested:break
        report=evaluate(actor,rl,seed,256,stage)
        report.update(checkpoint=str(checkpoint),checkpoint_sha256=digest,
                      proof_scope='Retained-bridge curriculum stage only, not main-state promotion')
        atomic_json(output/f'gate-{checkpoint.stem}-{seed}.json',report)
        reports.append(report)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    return bridge_stage_gate(reports,stage,digest),reports


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--updates',type=int,default=300)
    parser.add_argument('--chunk',type=int,default=100)
    parser.add_argument('--eval-episodes',type=int,default=64)
    parser.add_argument('--seed',type=int,default=73)
    parser.add_argument('--stage',choices=BRIDGE_CURRICULUM,default='full')
    parser.add_argument('--resume-state',type=Path)
    parser.add_argument('--actor-transition',type=Path,
        help='Recorded, approved actor-forward transition against this resume source')
    parser.add_argument('--exit-on-gate',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-proof',type=Path,default=ROOT/'local/onefoot-reward-review/smoke/state.json')
    a=parser.parse_args()
    if not 1<=a.updates<=1000 or not 1<=a.chunk<=100:parser.error('Bounded pilot only')
    if a.smoke and (a.num_envs!=64 or a.updates!=5):parser.error('Smoke requires 64 envs x 5 updates')
    if a.actor_transition is not None:
        if a.resume_state is None:parser.error('An actor transition needs the run it was recorded against')
        evidence=verified_bridge_transition(a.resume_state,ROOT,a.actor_transition)
    else:
        evidence=verified_bridge_source(a.resume_state,ROOT) if a.resume_state else None
    if a.stage!='full' and evidence is None:parser.error('Assisted curriculum requires a verified continuation source')
    if a.exit_on_gate and evidence is None:parser.error('Curriculum gates require a continuation source')
    source_checkpoint=Path(evidence['checkpoint']) if evidence else INITIAL
    source_digest=evidence['checkpoint_sha256'] if evidence else INITIAL_SHA256
    if not a.smoke:
        smoke=json.loads(a.smoke_proof.read_text())
        if smoke.get('status')!='completed' or not smoke.get('exports_verified') or smoke['stage']!=a.stage:
            parser.error('Verified same-stage 64x5 smoke required')
        if smoke['source_checkpoint_sha256']!=source_digest:parser.error('Smoke must test the same source checkpoint')
        for path,digest in smoke['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:parser.error('Sources changed after smoke')
    assert os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID' and os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    assert torch.cuda.device_count()==1 and '5070 Ti' in torch.cuda.get_device_name(0)
    assert hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()==source_digest
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
    sources=[Path(__file__).resolve(),ROOT/'src/mjlab_microduck/tasks/mdp.py',
        ROOT/'src/mjlab_microduck/tasks/microduck_onefoot_credit_bridge_env_cfg.py',
        ROOT/'src/mjlab_microduck/tasks/microduck_onefoot_curriculum_poses.py',
        ROOT/'src/mjlab_microduck/retained_skill_bridge.py',ROOT/'src/mjlab_microduck/public_residual_policy.py',
        ROOT/'src/mjlab_microduck/checkpoint_safety.py',ROOT/'src/mjlab_microduck/tasks/__init__.py',
        ROOT/'src/mjlab_microduck/bridge_recovery.py',ONEFOOT_PATH,PUBLIC_PATH,source_checkpoint]
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for path in sources:
        target=a.output/'source'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    main_hash=hashlib.sha256(MAIN.read_bytes()).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    log_root=ROOT/'logs/rsl_rl/onefoot'/f'{stamp}_credit-bridge-{a.stage}-{"smoke" if a.smoke else "pilot"}'
    log_root.mkdir(parents=True,exist_ok=False)
    state=dict(status='initializing',adopted=False,pid=os.getpid(),stage=a.stage,
        updates=0,requested_updates=a.updates,num_envs=a.num_envs,log_root=str(log_root),
        source_sha256=hashes,main_state_sha256=main_hash,source_checkpoint=str(source_checkpoint),source_checkpoint_sha256=source_digest,
        continuation=evidence,stage_gate_passed=False,stage_gate_reports=[],
        actor_transition=(str(a.actor_transition) if a.actor_transition else None),
        onefoot_source=str(ONEFOOT_PATH),onefoot_source_sha256=ONEFOOT_SHA256,
        initial_assistance=a.stage!='full',reports=[],exports_verified=False,passed_stages=[],
        objective='Nonnegative new credible glide credit + terminal-correct potential shaping - failed-attempt event cost',
        trainable_part='Transfer bridge and PPO exploration std; both expert means remain frozen')
    sp=a.output/'state.json';atomic_json(sp,state);env=None;runner=None
    try:
        env,rl=environment(a.num_envs,a.seed,a.stage)
        runner=load_runner_cls(TASK)(env,deepcopy(asdict(rl)),log_dir=str(log_root),device='cuda:0')
        runner.logger.writer=None
        if evidence:
            audit=restore_bridge_continuation(runner,evidence,'cuda:0')
            state['continuation_audit']=audit
            next_iteration=audit['next_iteration']
            atomic_json(a.output/'continuation-audit.json',audit)
        else:
            safe_runner_load(runner,source_checkpoint,strict=True,map_location='cuda:0')
            next_iteration=0
        state['source_iteration']=runner.current_learning_iteration
        state['start_iteration']=next_iteration
        state['next_iteration']=next_iteration
        assert runner.alg.gamma==env.unwrapped.command_manager.get_term('twist').cfg.reward_gamma
        assert runner.alg.schedule=='fixed'
        assert runner.alg.learning_rate==rl.algorithm.learning_rate
        assert all(g['lr']==rl.algorithm.learning_rate for g in runner.alg.optimizer.param_groups)
        state['learning_rate']=runner.alg.learning_rate;state['learning_rate_schedule']='fixed'
        frozen=retained_skills_digest(runner.alg.actor);state['frozen_skills_sha256']=frozen
        initial=log_root/'model_initial.pt';runner.save(str(initial));state['initial_checkpoint']=str(initial);state['checkpoint']=str(initial)
        state['iteration']=runner.current_learning_iteration
        report=evaluate(runner.alg.actor,rl,70601,a.eval_episodes,a.stage)
        report.update(updates=0,checkpoint=str(initial),checkpoint_sha256=hashlib.sha256(initial.read_bytes()).hexdigest())
        state['reports'].append(report);state['status']='running';atomic_json(sp,state)
        print('CREDIT_INITIAL_EVALUATION',json.dumps(report),flush=True)
        gate_key='credible_200_rate' if a.stage in SUSTAINED_STAGES else 'credible_050_rate'
        if a.exit_on_gate and report[gate_key]>=.8 and not stop_requested:
            passed,proofs=check_curriculum_gate(runner.alg.actor,rl,a.stage,initial,a.output)
            state['stage_gate_passed']=passed;state['stage_gate_reports']=proofs;atomic_json(sp,state)
        runner.current_learning_iteration=next_iteration
        while state['updates']<a.updates and not stop_requested and not state['stage_gate_passed']:
            count=min(a.chunk,a.updates-state['updates'])
            runner.learn(num_learning_iterations=count,init_at_random_ep_len=False)
            if runner.logger.writer is not None:
                runner.logger.writer.flush();runner.logger.writer.close();runner.logger.writer=None
            state['updates']+=count
            checkpoint=log_root/f'model_{runner.current_learning_iteration}.pt'
            assert checkpoint.is_file() and retained_skills_digest(runner.alg.actor)==frozen
            assert runner.alg.learning_rate==rl.algorithm.learning_rate
            assert all(g['lr']==rl.algorithm.learning_rate for g in runner.alg.optimizer.param_groups)
            for model in (runner.alg.actor,runner.alg.critic):
                assert all(torch.isfinite(value).all() for value in model.state_dict().values())
            assert hashlib.sha256(MAIN.read_bytes()).hexdigest()==main_hash
            for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
            report=evaluate(runner.alg.actor,rl,70601,a.eval_episodes,a.stage)
            report.update(updates=state['updates'],checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            state['reports'].append(report);state['checkpoint']=str(checkpoint)
            state['iteration']=runner.current_learning_iteration
            state['next_iteration']=runner.current_learning_iteration+1
            atomic_json(sp,state)
            print('CREDIT_MILESTONE',json.dumps(report),flush=True)
            if report['nan_episodes']:raise RuntimeError('NaN in evaluation')
            if a.exit_on_gate and report[gate_key]>=.8 and not stop_requested:
                passed,proofs=check_curriculum_gate(runner.alg.actor,rl,a.stage,checkpoint,a.output)
                state['stage_gate_passed']=passed;state['stage_gate_reports']=proofs;atomic_json(sp,state)
            runner.current_learning_iteration+=1
        export_dir=a.output/'export';runner.export_policy_to_onnx(str(export_dir));runner.export_policy_to_jit(str(export_dir))
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=1
        session=ort.InferenceSession(str(export_dir/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        sample=env.get_observations();actor=runner.get_inference_policy(device='cuda:0')
        with torch.no_grad():expected=actor(sample).cpu().numpy()[:8]
        actual=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in sample['actor'][:8].cpu().numpy()])
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
        state['exports_verified']=True
        state['status']='interrupted' if stop_requested else ('stage_passed' if state['stage_gate_passed'] else 'completed')
        atomic_json(sp,state)
        print('CREDIT_RUN_FINISHED',json.dumps({k:state[k] for k in ('status','updates','checkpoint','exports_verified')}),flush=True)
    except BaseException as error:
        state.update(status='error',error=f'{type(error).__name__}: {error}');atomic_json(sp,state);raise
    finally:
        if runner is not None and getattr(runner.logger,'writer',None) is not None:runner.logger.writer.close()
        if env is not None:env.close()


if __name__=='__main__':main()
