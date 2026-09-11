"""Wait for a recovering branch's fixed limit, preserving it before best reset.

CPU only. At the START of the last allowed training chunk, request the existing
checkpointed stop file. Never kills processes, changes rewards, or starts GPU
work. Exit without intervention for a new best, stage, controller or branch.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'local/onefoot-curriculum-state.json'
STOP=ROOT/'local/onefoot-curriculum.stop'


def iteration(path):return int(Path(path).stem.removeprefix('model_'))


def decision(before,current,anchor,budget):
    if current.get('status')=='error':return 'controller_error'
    if current.get('status')=='interrupted':return 'controller_paused'
    if any(before.get(k)!=current.get(k) for k in ('pid','recipe','stage','run_root')):return 'controller_or_stage_changed'
    if before.get('best_checkpoint')!=current.get('best_checkpoint'):return 'new_best_candidate'
    adaptations=[a['iteration'] for a in current.get('adaptations',[]) if a['stage']==current['stage']]
    if max(adaptations,default=-1)!=anchor:return 'exploration_branch_changed'
    rows={r['checkpoint']:r for r in current.get('evaluations',[]) if r['stage']==current['stage'] and iteration(r['checkpoint'])>anchor}
    if any(r.get('nan_episodes',0) for r in rows.values()):return 'numerical_error'
    if len(rows)>=budget:return 'boundary_already_reached_review_needed'
    if len(rows)==budget-1 and current['status']=='training':
        last=max(iteration(path) for path in rows)
        if current.get('iteration',-1)>last:return 'request_saved_stop_in_final_chunk'
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--anchor',type=int,required=True)
    p.add_argument('--budget',type=int,default=18);p.add_argument('--max-wait-seconds',type=float,default=3600.)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert a.budget>=6 and a.max_wait_seconds>0
    before=json.loads(STATE.read_text());pid=before['pid']
    assert before['recipe']=='waist' and before['status'] in ('training','evaluating')
    assert before['supervision']['anchor_iteration']==a.anchor
    assert before['supervision']['allotted_evaluations']==a.budget
    assert not STOP.exists()
    started=time.monotonic();requested=False
    record=dict(status='watching',started=datetime.now(timezone.utc).isoformat(),trainer_pid=pid,
        run_root=before['run_root'],recipe=before['recipe'],stage=before['stage'],anchor=a.anchor,budget=a.budget,
        trigger='start_of_last_allowed_chunk',knee_shaping_active=False,gpu_processes_killed=False)
    with a.output.open('x') as stream:json.dump(record,stream,indent=2)
    def save(**changes):
        record.update(changes,updated=datetime.now(timezone.utc).isoformat())
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
    while time.monotonic()-started<a.max_wait_seconds:
        current=json.loads(STATE.read_text());reason=decision(before,current,a.anchor,a.budget)
        if requested:
            if current['pid']!=pid:reason='controller_changed_after_stop_request'
            elif not Path(f'/proc/{pid}').exists():
                reason='saved_branch_review_ready' if current['status']=='interrupted' else 'controller_exited_review_needed'
            else:time.sleep(2);continue
        if reason=='request_saved_stop_in_final_chunk':
            args=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            assert 'local/train-onefoot-curriculum.py' in args
            if STOP.exists():reason='another_stop_already_requested'
            else:
                with STOP.open('x') as stream:stream.write(f'Delegated management: preserve recovering waist branch {a.anchor} at its {a.budget}-evaluation limit before an automatic old-best restore; no reward change.\n')
                requested=True;save(status='checkpointed_stop_requested',request_iteration=current['iteration'])
                print('BRANCH_BOUNDARY_STOP_REQUESTED',current['iteration'],flush=True)
                time.sleep(2);continue
        if reason:
            checkpoint=current.get('checkpoint');sha=None
            if checkpoint and Path(checkpoint).is_file():sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            save(status=reason,iteration=current.get('iteration'),checkpoint=checkpoint,checkpoint_sha256=sha,
                 supervision=current.get('supervision'),latest_evaluation=current.get('evaluations',[])[-1:] )
            print('BRANCH_REVIEW_READY',reason,current.get('iteration'),checkpoint,flush=True);return
        time.sleep(3)
    save(status='review_deadline',checkpointed_stop_requested=requested)
    print('BRANCH_REVIEW_DEADLINE',flush=True)


if __name__=='__main__':main()
