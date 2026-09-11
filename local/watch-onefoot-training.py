"""Resident CPU-only auditor for the onefoot controller and its saved evidence.

Read-only with respect to training, policies, GPU processes and the viewer.
It never restarts/kills a process, promotes a stage, or changes a reward.
The training controller performs evaluated progression and exploration changes.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

from mjlab_microduck.onefoot_supervision import audit_stage_history, checkpoint_iteration
from mjlab_microduck.tasks.microduck_onefoot_curriculum_env_cfg import STAGES

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'local/onefoot-curriculum-state.json'
VIEWER=ROOT/'local/onefoot-live-viewer.json'


def safe_monitor_evidence(value):
    bad=False
    def clean(item):
        nonlocal bad
        if isinstance(item,float) and not math.isfinite(item):
            bad=True
            return None
        if isinstance(item,dict):return {k:clean(v) for k,v in item.items()}
        if isinstance(item,list):return [clean(v) for v in item]
        return item
    result=clean(value)
    if bad:
        result['severity']='needs_attention'
        result.setdefault('errors',[]).append('nonfinite_monitor_evidence')
    return result


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def process_identity(pid, expected_script):
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0:
        return 'invalid_pid'
    root=Path(f'/proc/{pid}')
    try:
        stat=root.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return 'not_running'
        argv=root.joinpath('cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return 'not_running'
    except PermissionError:return 'unreadable'
    return 'running' if any(Path(a.decode(errors='replace')).name==expected_script for a in argv if a) else 'different_process'


def sample(state,viewer,*,now=None,process_probe=process_identity,stale_seconds=900):
    now=time.time() if now is None else now
    errors=audit_stage_history(state,[(s.name,s.goal_s) for s in STAGES]);warnings=[]
    root=Path(state['run_root']).resolve()
    if not root.is_relative_to((ROOT/'logs/rsl_rl/onefoot').resolve()):
        errors.append('experiment_outside_project')
    identity=process_probe(state.get('pid'),'train-onefoot-curriculum.py')
    running_status=state.get('status') in ('running','training','evaluating','recovering')
    if running_status and identity!='running':errors.append('trainer_'+identity)
    if state.get('status')=='error':errors.append('controller_reported_error')
    for key in ('checkpoint','best_checkpoint'):
        if state.get(key):
            path=Path(state[key]).resolve()
            if not path.is_relative_to(root) or not path.is_file():errors.append(key+'_missing_or_outside_run')
    updated=datetime.fromisoformat(state['updated']).timestamp()
    heartbeat=updated
    log=state.get('training_log')
    if log:
        path=Path(log).resolve()
        if path.is_relative_to(ROOT) and path.is_file():heartbeat=max(heartbeat,path.stat().st_mtime)
    age=max(0,now-heartbeat)
    if running_status and age>stale_seconds:errors.append('training_progress_stale')
    if viewer:
        viewer_identity=process_probe(viewer.get('pid'),'play-latest-onefoot.py')
        if viewer_identity!='running':warnings.append('viewer_'+viewer_identity)
        elif viewer.get('run_root')!=state['run_root'] or viewer.get('stage')!=state['stage']:
            warnings.append('viewer_source_switch_pending')
        elif state.get('checkpoint') and viewer.get('iteration',-1)<checkpoint_iteration(state['checkpoint']):
            # Normal stable-file discovery needs several seconds after saving.
            if now-updated>30:warnings.append('viewer_checkpoint_lag')
    else:warnings.append('viewer_status_unavailable')
    rows=[r for r in state.get('evaluations',[]) if r['stage']==state['stage']]
    latest=rows[-1] if rows else None
    clean=[r for r in rows if r.get('nan_episodes',0)==0]
    best=max(clean,key=lambda r:(r['success_rate'],r['mean_best_glide_s'])) if clean else None
    if latest and latest.get('nan_episodes',0):errors.append('latest_evaluation_nonfinite')
    stage=next((s for s in STAGES if s.name==state['stage']),None)
    iteration=state.get('iteration',0)
    best_age=iteration-checkpoint_iteration(best['checkpoint']) if best else None
    cfg=state.get('training_config',{});chunk=cfg.get('iterations_per_chunk',100)
    if best_age is not None and best_age>=54*chunk:warnings.append('long_stagnation_review_due')
    severity='needs_attention' if errors else 'warning' if warnings else 'ok'
    quality=lambda r:({k:r.get(k) for k in ('checkpoint','seed','episodes','success_rate','mean_best_glide_s','max_glide_s')} if r else None)
    return dict(severity=severity,errors=errors,warnings=warnings,controller_status=state.get('status'),
                trainer_pid=state.get('pid'),trainer_identity=identity,heartbeat_age_s=age,
                run_root=state['run_root'],recipe=state.get('recipe'),stage=state['stage'],
                goal_s=stage.goal_s if stage else None,iteration=iteration,
                passed_stages=[p['stage'] for p in state.get('passed_stages',[])],
                final_self_launch_completed=state.get('final_self_launch_completed',False),
                latest_evaluation=quality(latest),best_evaluation=quality(best),
                iterations_since_best=best_age,management=state.get('supervision'),
                first_break_diagnostics=latest.get('failure_diagnostics') if latest else None,
                note='Historical gates and runtime health are not proof of current-policy skill improvement.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--once',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=15.)
    p.add_argument('--stale-seconds',type=float,default=900.)
    p.add_argument('--output',type=Path,default=ROOT/'local/onefoot-supervision.json')
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Supervision is CPU/file I/O only'
    if a.poll_seconds<1 or a.stale_seconds<300:raise ValueError('Unsafe monitoring interval')
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    lock_path=ROOT/'local/onefoot-supervision.lock'
    events=ROOT/'local/onefoot-supervision.events.jsonl'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last_signature=None
        while not stopped:
            try:
                state=json.loads(STATE.read_text())
                try:viewer=json.loads(VIEWER.read_text())
                except (OSError,ValueError):viewer=None
                result=sample(state,viewer,stale_seconds=a.stale_seconds)
            except (OSError,ValueError,KeyError,TypeError) as error:
                result=dict(severity='needs_attention',errors=['monitor_read_or_schema_error'],error=str(error))
            result=safe_monitor_evidence(result)
            result.update(monitor_pid=os.getpid(),monitor_status='snapshot' if a.once else 'running',updated=datetime.now(timezone.utc).isoformat())
            atomic_json(a.output,result)
            signature=json.dumps({k:result.get(k) for k in ('severity','errors','warnings','controller_status','stage',
                                 'iteration','passed_stages','management')},sort_keys=True)
            if signature!=last_signature:
                with events.open('a') as f:f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                print('SUPERVISOR',result.get('severity'),result.get('stage'),result.get('iteration'),
                      result.get('errors'),result.get('warnings'),flush=True)
                last_signature=signature
            if a.once:break
            time.sleep(a.poll_seconds)
        if stopped:
            result.update(monitor_status='stopped',updated=datetime.now(timezone.utc).isoformat())
            atomic_json(a.output,result)


if __name__=='__main__':main()
