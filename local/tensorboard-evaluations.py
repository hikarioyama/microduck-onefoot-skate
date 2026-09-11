"""Publish independent evaluations to official TensorBoard; CPU/file I/O only.

Never touches policies, training event files, GPU jobs, viewers or services.
Each stage gets an 'evaluation' child run. Primary and confirmation batches
have separate tags, so two independent batches never collapse into one point.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
LOGS=(ROOT/'logs/rsl_rl/onefoot').resolve()
STATE=ROOT/'local/onefoot-curriculum-state.json'


def scalar_values(row, confirmation=False):
    prefix='EvaluationConfirm' if confirmation else 'Evaluation'
    mappings={
        'success_rate_pct':('success_rate',100.),
        'success_050_pct':('success_050',100.),
        'success_100_pct':('success_100',100.),
        'success_200_pct':('success_200',100.),
        'distance_m':('mean_distance_m',1.),
        'head_speed_rad_s':('mean_head_speed',1.),
        'waist_roll_deg':('mean_waist_roll_deg',1.),
        'waist_abs_roll_deg':('mean_waist_abs_roll_deg',1.),
        'waist_abs_pitch_deg':('mean_waist_abs_pitch_deg',1.),
        'waist_roll_speed_rad_s':('mean_waist_roll_speed_rad_s',1.),
        'waist_axis_error_m':('mean_waist_axis_error',1.),
        'waist_capture_error_m':('mean_waist_capture_error',1.),
        'foreaft_com_offset_m':('mean_waist_foreaft_error',1.),
        'foreaft_capture_offset_m':('mean_waist_foreaft_capture',1.),
        'front_wheel_load_pct':('mean_waist_front_load',100.),
        'mean_best_glide_s':('mean_best_glide_s',1.),
        'max_glide_s':('max_glide_s',1.),
        'p10_best_glide_s':('p10_best_glide_s',1.),
        'mean_survival_s':('mean_survival_s',1.),
        'forward_speed_m_s':('mean_speed',1.),
        'centroid_forward_speed_m_s':('mean_com_speed',1.),
        'support_yaw_deg':('mean_support_yaw',180./math.pi),
        'left_wheels_contact_pct':('mean_left_contact',100.),
        'right_foot_contact_pct':('mean_right_contact',100.),
        'left_load_pct':('mean_left_load',100.),
        'goal_s':('goal_s',1.),
        'exploration_std_param_mean':('exploration_std_param_mean',1.),
        'episodes':('episodes',1.),
        'nan_episodes':('nan_episodes',1.),
        'injected_speed_min_m_s':('injected_speed_min',1.),
        'injected_speed_max_m_s':('injected_speed_max',1.),
    }
    result={}
    for tag,(key,scale) in mappings.items():
        if key in row:
            value=float(row[key])*scale
            if not math.isfinite(value):
                raise ValueError(f'Non-finite evaluation value: {key}')
            result[prefix+'/'+tag]=value
    # These are the published, unchanged progression gates.
    result[prefix+'/gate_threshold_pct']=80.
    episodes=int(row['episodes'])
    if episodes<=0:raise ValueError('Evaluation requires positive episode count')
    for name,count in row.get('termination_counts',{}).items():
        # Different termination conditions may overlap; these are NOT a pie chart.
        result[prefix+'/termination_pct/'+name]=100.*float(count)/episodes
    diagnostic=row.get('failure_diagnostics')
    if diagnostic:
        failed=int(diagnostic['failed_episodes']);observed=int(diagnostic['recorded_break_episodes'])
        missing=int(diagnostic['failed_without_recorded_break'])
        if not 0<=observed<=failed<=episodes or observed+missing!=failed:
            raise ValueError('Invalid first-break diagnostic denominator')
        for key in ('failed_episodes','recorded_break_episodes','failed_without_recorded_break',
                    'failed_without_established_streak','mean_first_break_time_s','mean_streak_before_break_s'):
            if key in diagnostic:
                value=float(diagnostic[key])
                if not math.isfinite(value):raise ValueError('Non-finite diagnostic scalar')
                result[prefix+'/FirstBreak/'+key]=value
        for name,count in diagnostic['reason_counts'].items():
            if not 0<=count<=observed:raise ValueError('Invalid first-break reason count')
            # Denominator is FAILED first episodes, and reasons can overlap.
            result[prefix+'/FirstBreak/reason_pct_of_failed/'+name]=100.*count/max(failed,1)
        for name,value in diagnostic.get('mean_values_at_break',{}).items():
            if value is not None:
                if not math.isfinite(value):raise ValueError('Non-finite first-break measurement')
                result[prefix+'/FirstBreak/mean_'+name]=float(value)
    return result


def resolve_artifact(row, run_root):
    checkpoint=Path(row['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run_root) or not run_root.is_relative_to(LOGS):
        raise ValueError('Evaluation is outside its declared experiment directory')
    step=int(checkpoint.stem.removeprefix('model_'))
    for confirmation,suffix in ((False,''),(True,'_confirm')):
        path=checkpoint.parent/f'eval_{step}{suffix}.json'
        if path.is_file():
            saved=json.loads(path.read_text())
            if saved.get('seed')==row['seed'] and saved.get('stage')==row['stage']:
                for key in ('goal_s','episodes','success_rate','mean_best_glide_s','max_glide_s'):
                    if saved[key]!=row[key]:raise ValueError(f'State/artifact disagreement: {path}: {key}')
                return path,checkpoint.parent/'evaluation',step,confirmation
    raise ValueError(f'Matching evaluation artifact not found: {checkpoint}, seed={row["seed"]}')


class Publisher:
    def __init__(self):
        self.writers={}
        self.ledgers={}

    def publish(self, state):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        root=Path(state['run_root']).resolve()
        count=0
        for row in state.get('evaluations',[]):
            artifact,directory,step,confirmation=resolve_artifact(row,root)
            digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
            key=str(artifact)
            ledger_path=directory/'published-evaluations.json'
            if directory not in self.ledgers:
                self.ledgers[directory]=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
            ledger=self.ledgers[directory]
            if key in ledger:
                if ledger[key]!=digest:raise ValueError(f'Published evaluation changed: {artifact}')
                continue
            values=scalar_values(row,confirmation)
            if directory not in self.writers:
                self.writers[directory]=EventFileWriter(str(directory),flush_secs=5)
            writer=self.writers[directory]
            # JSON is written immediately after evaluation; use its mtime rather
            # than presenting historical results as if they occurred at import time.
            writer.add_event(Event(wall_time=artifact.stat().st_mtime,step=step,
                summary=Summary(value=[Summary.Value(tag=k,simple_value=v) for k,v in values.items()])))
            writer.flush()
            ledger[key]=digest
            temp=ledger_path.with_suffix('.tmp')
            temp.write_text(json.dumps(ledger,indent=2)+'\n');temp.replace(ledger_path)
            print(f'Published {row["stage"]} step={step} seed={row["seed"]} confirmation={confirmation}',flush=True)
            count+=1
        return count

    def close(self):
        for writer in self.writers.values():writer.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=5.)
    a=p.parse_args()
    if a.poll_seconds<1:raise ValueError('Poll interval must be at least 1 second')
    with (ROOT/'local/tensorboard-evaluations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publisher=Publisher();mtime=None
        try:
            while True:
                current=STATE.stat().st_mtime_ns
                if current!=mtime:
                    state=json.loads(STATE.read_text());publisher.publish(state);mtime=current
                if a.once:break
                time.sleep(a.poll_seconds)
        finally:publisher.close()


if __name__=='__main__':main()
