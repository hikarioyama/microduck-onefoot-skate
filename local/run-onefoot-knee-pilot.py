"""Bounded assigned-GPU A/B pilot. Never publishes main state or adopts a policy.

CPU orchestration only. Each subprocess independently checks the 5070 Ti.
Both arms start at the same saved actor/critic/optimizer with the same initial
std cap. Six evaluated 100-update chunks; no old-best restoration mid-arm.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'local/knee-review'
STATE=ROOT/'local/onefoot-curriculum-state.json'
HOLDOUT_SEEDS=(64101,64102)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def train_arguments(recipe,checkpoint,run):
    return ['uv','run','--locked','python','local/train-onefoot-curriculum.py',
        '--mode','train','--recipe',recipe,'--stage','balance-100','--resume',
        '--checkpoint',str(checkpoint),'--exploration-max-std','.08','--num-envs','4096',
        '--iterations','100','--max-chunks','6','--seed','73','--run-dir',str(run)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');a=parser.parse_args()
    prior=json.loads((OUT/'resume-state.json').read_text());current=json.loads(STATE.read_text())
    assert current['status']=='interrupted' and current['checkpoint']==prior['checkpoint']
    assert not Path(f"/proc/{prior['pid']}").exists()
    assert (ROOT/'local/onefoot-curriculum.stop').exists()
    cpu=json.loads((OUT/'cpu-validation.json').read_text())
    assert cpu['passed']>=335 and cpu['failed']==0
    for name,sha in cpu['source_sha256'].items():assert digest(ROOT/name)==sha,f'Changed after CPU validation: {name}'
    validation=json.loads((OUT/'multichunk-validation.json').read_text())
    assert validation['main_state_unchanged'] and validation['no_automatic_best_restore'] and validation['nan_episodes']==0
    checkpoint=Path(prior['best_checkpoint']);assert checkpoint.is_file()
    stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    runs={arm:ROOT/'logs/rsl_rl/onefoot'/f'{stamp}_knee-pilot-{arm}' for arm in ('candidate','control')}
    if a.dry_run:
        for arm,recipe in (('candidate','knee'),('control','waist')):print(json.dumps(train_arguments(recipe,checkpoint,runs[arm])))
        return
    path=OUT/'pilot-state.json'
    if path.exists():raise FileExistsError(path)
    applications=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert 'GPU-8e73d1bd-4e6b-4fdf-bd0a-602e0cf7d3f9' not in applications,'Assigned GPU is occupied; leave existing workloads alone'
    state=dict(status='starting',pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),
        initial_checkpoint=str(checkpoint),initial_checkpoint_sha256=digest(checkpoint),
        main_state_sha256=digest(STATE),main_checkpoint=prior['checkpoint'],
        budget=dict(arms=2,updates_per_arm=600,environments=4096,evaluation_interval_updates=100),
        holdout_seeds=list(HOLDOUT_SEEDS),evaluation_recipe='waist',arms=[],adopted=False,review_required=True,
        pilot_script_sha256=digest(__file__),protocol_sha256=digest(OUT/'PROTOCOL.ja.md'))
    with path.open('x') as stream:json.dump(state,stream,indent=2,allow_nan=False)
    gpu_env=os.environ.copy();gpu_env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='2',
        OMP_NUM_THREADS='8',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
    cpu_env=os.environ.copy();cpu_env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
    def update(**changes):
        state.update(changes,updated=datetime.now(timezone.utc).isoformat());atomic_json(path,state)
    def guard():
        assert digest(STATE)==state['main_state_sha256'],'Main training state changed; do not start another pilot workload'
        assert digest(checkpoint)==state['initial_checkpoint_sha256']
        assert not Path(f"/proc/{prior['pid']}").exists()
    def execute(arguments,environment,log):
        guard()
        with log.open('x') as stream:
            process=subprocess.Popen(arguments,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            update(child_pid=process.pid,child_log=str(log),child_command=arguments)
            result=process.wait()
            update(child_pid=None)
            if result:raise RuntimeError(f'Pilot subprocess failed with {result}; see {log}')
    update()
    try:
        reports=[]
        for arm,recipe in (('candidate','knee'),('control','waist')):
            run=runs[arm];update(status='training',current_arm=arm,training_recipe=recipe,run_root=str(run))
            execute(train_arguments(recipe,checkpoint,run),gpu_env,OUT/f'{arm}-training.log')
            saved=json.loads((run/'state.json').read_text())
            assert saved['status']=='bounded_test_completed' and len(saved['evaluations'])==6
            assert len(saved['adaptations'])==1 and saved['adaptations'][0]['reason']=='consolidation_noise_cap'
            assert not saved['passed_stages'] and not saved.get('supervision')
            assert all(row['nan_episodes']==0 and row['episodes']==256 for row in saved['evaluations'])
            candidate=Path(saved['checkpoint']);arm_reports=[]
            # Evaluate both trained policies in the SAME unmodified waist-v10
            # environment, not in their respective shaping-reward environments.
            for seed in HOLDOUT_SEEDS:
                output=OUT/f'{arm}-holdout-{seed}.json'
                update(status='evaluating',evaluation_seed=seed,evaluating_checkpoint=str(candidate))
                execute(['uv','run','--locked','python','local/record-onefoot-timeline.py',
                    '--recipe','waist','--stage','balance-100','--checkpoint',str(candidate),
                    '--episodes','256','--seed',str(seed),'--support-kinematics','--output',str(output)],
                    gpu_env,OUT/f'{arm}-holdout-{seed}.log')
                report=json.loads(output.read_text())
                assert report['nan_episodes']==0 and report['episodes']==256
                assert report['checkpoint_sha256']==digest(candidate)
                reports.append(str(output));arm_reports.append(str(output))
            state['arms'].append(dict(arm=arm,training_recipe=recipe,run_root=str(run),
                checkpoint=str(candidate),checkpoint_sha256=digest(candidate),reports=arm_reports))
            update();print('PILOT_ARM_COMPLETED',arm,str(candidate),flush=True)
        execute(['uv','run','--locked','python','local/summarize-onefoot-knee.py',*reports,
                 '--output',str(OUT/'pilot-comparison.json')],cpu_env,OUT/'pilot-comparison.log')
        guard();update(status='completed_review_required')
        print('KNEE_PILOT_REVIEW_READY',str(OUT/'pilot-comparison.json'),flush=True)
    except BaseException as error:
        update(status='error_review_required',error=str(error));raise


if __name__=='__main__':main()
