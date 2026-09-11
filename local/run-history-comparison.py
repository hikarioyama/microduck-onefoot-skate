"""Isolated matched MLP/history pilot with initial parity and memory ablation.

Never changes the active curriculum state, promotes stages, resets exploration,
modifies physics/rewards, or starts any GPU other than the assigned 5070 Ti.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'local/onefoot-history-review'
MAIN=ROOT/'local/onefoot-curriculum-state.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_parity(control,history):
    for row in (control,history):
        if (row['episodes']<256 or row['nan_episodes'] or row.get('memory_reset_each_step',False) or
                row.get('fixed_pose',False) or row.get('action_noise_in_evaluation',False)):
            raise ValueError('Initial comparison needs valid normal-inference first episodes')
    return (abs(history['success_rate']-control['success_rate'])<=.05 and
            abs(history['mean_best_glide_s']-control['mean_best_glide_s'])<=.02)


def training_command(arm,manifest):
    item=manifest['arms'][arm]
    return ['uv','run','--locked','python','local/train-onefoot-curriculum.py',
        '--mode','train','--recipe',item['recipe'],'--stage','balance-100','--resume',
        '--checkpoint',item['checkpoint'],'--num-envs',str(manifest['num_envs']),
        '--iterations',str(manifest['iterations_per_chunk']),'--max-chunks',str(manifest['chunks_per_arm']),
        '--seed',str(manifest['training_seed']),'--run-dir',item['run_root']]


def main():
    manifest=json.loads((OUT/'matched-initialization.json').read_text())
    assert manifest['chunks_per_arm']==3 and manifest['iterations_per_chunk']==100 and manifest['num_envs']==4096
    current=json.loads(MAIN.read_text());assert current['status']=='interrupted'
    assert not Path(f"/proc/{current['pid']}").exists() and digest(MAIN)==manifest['main_state_sha256']
    smoke={arm:json.loads((OUT/f'matched-smoke-{arm}.json').read_text()) for arm in ('control','history')}
    assert all(row['nan_episodes']==0 and row['episodes']==64 for row in smoke.values())
    if (smoke['history']['success_rate']<smoke['control']['success_rate']-.20 or
            smoke['history']['mean_best_glide_s']<smoke['control']['mean_best_glide_s']-.10):
        raise ValueError('Large recurrent smoke regression; diagnose before a pilot')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert 'GPU-8e73d1bd-4e6b-4fdf-bd0a-602e0cf7d3f9' not in apps
    path=OUT/'pilot-state.json'
    state=dict(status='starting',pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),
        budget_updates_per_arm=300,initial_checkpoint=manifest['source_checkpoint'],adopted=False,
        initial_reports={},final_reports={},completed_arms=[],main_state_sha256=manifest['main_state_sha256'])
    with path.open('x') as stream:json.dump(state,stream,indent=2)
    env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='8',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
    cpu_env=os.environ.copy();cpu_env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
    source_names=['local/run-history-comparison.py','local/train-onefoot-curriculum.py','local/record-onefoot-timeline.py',
        'src/mjlab_microduck/history_policy.py','src/mjlab_microduck/matched_optimizer.py',
        'src/mjlab_microduck/checkpoint_safety.py','src/mjlab_microduck/onefoot_supervision.py','src/mjlab_microduck/tasks/mdp.py',
        'src/mjlab_microduck/tasks/microduck_onefoot_history_env_cfg.py']
    hashes={name:digest(ROOT/name) for name in source_names}
    state['source_sha256']=hashes
    def update(**changes):
        state.update(changes,updated=datetime.now(timezone.utc).isoformat())
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def execute(args,log,environment=env):
        assert digest(MAIN)==manifest['main_state_sha256'],'Main state changed; stop the isolated comparison'
        for name,sha in hashes.items():assert digest(ROOT/name)==sha,f'Source changed during comparison: {name}'
        with log.open('x') as stream:
            child=subprocess.Popen(args,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            update(child_pid=child.pid,child_command=args,child_log=str(log))
            code=child.wait();update(child_pid=None)
            if code:raise RuntimeError(f'Comparison subprocess failed with {code}; see {log}')
    def evaluate(arm,checkpoint,seed,label,ablate=False):
        file=OUT/f'{label}-{arm}-{seed}.json'
        update(status='evaluating',current_arm=arm,evaluation_label=label,evaluation_seed=seed)
        args=['uv','run','--locked','python','local/record-onefoot-timeline.py','--recipe',manifest['arms'][arm]['recipe'],
              '--stage','balance-100','--checkpoint',checkpoint,'--episodes','256','--seed',str(seed),'--support-kinematics','--output',str(file)]
        if ablate:args.append('--reset-memory-each-step')
        execute(args,file.with_suffix('.log'))
        row=json.loads(file.read_text())
        assert row['nan_episodes']==0 and row['checkpoint_sha256']==digest(checkpoint)
        return file,row
    update()
    try:
        initial={}
        for arm in ('control','history'):
            initial[arm]={};state['initial_reports'][arm]=[]
            item=manifest['arms'][arm];assert digest(item['checkpoint'])==item['checkpoint_sha256']
            for seed in manifest['initial_evaluation_seeds']:
                file,row=evaluate(arm,item['checkpoint'],seed,'initial')
                initial[arm][seed]=row;state['initial_reports'][arm].append(str(file));update()
        if not all(initial_parity(initial['control'][seed],initial['history'][seed]) for seed in manifest['initial_evaluation_seeds']):
            update(status='initial_parity_failed_review_required');print('HISTORY_INITIAL_PARITY_FAILED',flush=True);return
        update(initial_parity_passed=True)
        print('INITIAL_PARITY_PASSED',flush=True)
        final_checkpoints={}
        for arm in ('history','control'):
            update(status='training',current_arm=arm,run_root=manifest['arms'][arm]['run_root'])
            execute(training_command(arm,manifest),OUT/f'training-{arm}.log')
            saved=json.loads((Path(manifest['arms'][arm]['run_root'])/'state.json').read_text())
            if saved['status']=='interrupted':
                update(status='cancelled_saved',checkpoint=saved.get('checkpoint'));return
            assert saved['status']=='bounded_test_completed' and len(saved['evaluations'])==3
            assert not saved['adaptations'] and not saved['passed_stages']
            assert all(row['nan_episodes']==0 for row in saved['evaluations'])
            final_checkpoints[arm]=saved['checkpoint'];state['completed_arms'].append(arm);update()
            print('HISTORY_COMPARISON_ARM_FINISHED',arm,saved['checkpoint'],flush=True)
        files=[]
        for arm in ('history','control'):
            state['final_reports'][arm]=[]
            for seed in manifest['final_evaluation_seeds']:
                file,_=evaluate(arm,final_checkpoints[arm],seed,'final')
                files.append(str(file));state['final_reports'][arm].append(str(file));update()
        state['memory_ablation_reports']=[]
        for seed in manifest['final_evaluation_seeds']:
            file,row=evaluate('history',final_checkpoints['history'],seed,'ablation',ablate=True)
            assert row['memory_reset_each_step']
            files.append(str(file));state['memory_ablation_reports'].append(str(file));update()
        execute(['uv','run','--locked','python','local/summarize-onefoot-knee.py',*files,
                 '--output',str(OUT/'comparison.json')],OUT/'comparison.log',cpu_env)
        update(status='completed_review_required',final_checkpoints=final_checkpoints)
        print('HISTORY_COMPARISON_REVIEW_READY',str(OUT/'comparison.json'),flush=True)
    except BaseException as error:
        update(status='error_review_required',error=str(error));raise


if __name__=='__main__':main()
