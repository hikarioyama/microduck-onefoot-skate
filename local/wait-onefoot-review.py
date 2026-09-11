"""CPU-only, bounded notification job for the next training review point.

Never controls GPU processes or modifies training state. The background job
finishes when new evidence needs review; the resident auditor keeps running.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'local/onefoot-curriculum-state.json'


def checkpoint_keys(state):
    return {r['checkpoint'] for r in state.get('evaluations',[])}


def review_reason(before,after,max_new=6):
    if after.get('status')=='error':return 'controller_error'
    if after.get('status')=='interrupted':return 'controller_paused'
    if after.get('run_root')!=before.get('run_root'):return 'experiment_changed'
    if after.get('final_self_launch_completed'):return 'simulation_complete_visual_review_required'
    if after.get('stage')!=before.get('stage') or len(after.get('passed_stages',[]))>len(before.get('passed_stages',[])):
        return 'stage_progress'
    if after.get('best_checkpoint')!=before.get('best_checkpoint'):
        return 'new_best_candidate_not_necessarily_gate_passed'
    a=before.get('supervision') or {};b=after.get('supervision') or {}
    if tuple(b.get(k) for k in ('anchor_iteration','allotted_evaluations','action','reason'))!=tuple(a.get(k) for k in ('anchor_iteration','allotted_evaluations','action','reason')):
        return 'exploration_management_decision'
    if len(checkpoint_keys(after)-checkpoint_keys(before))>=max_new:return 'scheduled_evidence_review'
    return None


def monitored_review_reason(before,after,health,max_new=6,acknowledged_stagnation=False):
    aligned=bool(health) and health.get('run_root')==after.get('run_root') and health.get('trainer_pid')==after.get('pid')
    if aligned and health.get('severity')=='needs_attention':return 'resident_audit_attention'
    # Current progress takes priority over a slightly older noncritical warning.
    reason=review_reason(before,after,max_new)
    if reason:return reason
    if aligned and 'long_stagnation_review_due' in health.get('warnings',[]):
        same_frontier=all(before.get(k)==after.get(k) for k in ('run_root','stage','best_checkpoint'))
        if not (acknowledged_stagnation and same_frontier):return 'long_stagnation_review'
    return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--poll-seconds',type=float,default=15.)
    p.add_argument('--max-wait-seconds',type=float,default=3600.)
    p.add_argument('--max-new-evaluations',type=int,default=6)
    p.add_argument('--acknowledge-existing-stagnation',action='store_true',help='After a completed review, wait for fresh evidence rather than repeating the same noncritical alert')
    a=p.parse_args()
    if a.poll_seconds<1 or a.max_wait_seconds<=0 or a.max_new_evaluations<1:raise ValueError('Invalid review window')
    before=json.loads(STATE.read_text());start=time.monotonic();reason=None;state=before
    acknowledged=False
    try:
        initial_health=json.loads((ROOT/'local/onefoot-supervision.json').read_text())
        acknowledged=(a.acknowledge_existing_stagnation and
            initial_health.get('run_root')==before['run_root'] and initial_health.get('stage')==before['stage'] and
            'long_stagnation_review_due' in initial_health.get('warnings',[]) and
            (initial_health.get('best_evaluation') or {}).get('checkpoint')==before.get('best_checkpoint'))
    except (OSError,ValueError):pass
    print('WAITING_FOR_REVIEW',before['stage'],before.get('iteration'),'acknowledged_stagnation',acknowledged,flush=True)
    while time.monotonic()-start<a.max_wait_seconds:
        time.sleep(a.poll_seconds)
        state=json.loads(STATE.read_text())
        if (ROOT/'local/onefoot-curriculum.stop').exists():reason='stop_requested';break
        if state.get('status') in ('training','evaluating','recovering') and not Path(f"/proc/{state['pid']}").exists():
            reason='trainer_no_longer_running';break
        health_path=ROOT/'local/onefoot-supervision.json';health=None
        if health_path.exists() and time.time()-health_path.stat().st_mtime<45:
            health=json.loads(health_path.read_text())
        reason=monitored_review_reason(before,state,health,a.max_new_evaluations,acknowledged)
        if reason:break
    result=dict(reason=reason or 'review_deadline',acknowledged_existing_stagnation=acknowledged,updated=datetime.now(timezone.utc).isoformat(),
                start_iteration=before.get('iteration'),iteration=state.get('iteration'),
                stage=state['stage'],run_root=state['run_root'],status=state.get('status'),
                checkpoint=state.get('checkpoint'),best_checkpoint=state.get('best_checkpoint'),
                management=state.get('supervision'),passed_stages=[s['stage'] for s in state.get('passed_stages',[])],
                recent_evaluations=[{k:r.get(k) for k in ('checkpoint','seed','episodes','success_rate','mean_best_glide_s',
                                     'failure_diagnostics')} for r in state.get('evaluations',[])[-3:]])
    path=ROOT/'local/onefoot-review-ready.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
    with (ROOT/'local/onefoot-review-history.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    print('TRAINING_REVIEW_READY',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
