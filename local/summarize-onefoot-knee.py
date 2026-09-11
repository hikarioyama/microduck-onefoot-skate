"""Read-only initial-episode knee/pelvis/motion comparison for frozen rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np


def stats(values):
    a=np.asarray(values,dtype=float).reshape(-1);finite=np.isfinite(a)
    return dict(n=int(finite.sum()),nonfinite=int((~finite).sum()),
                mean=float(a[finite].mean()) if finite.any() else None,
                p90=float(np.quantile(a[finite],.9)) if finite.any() else None)


def summarize(path):
    report=json.loads(Path(path).read_text())
    with np.load(report['timeline_npz'],allow_pickle=False) as arrays:
        k=arrays['support_kinematics'];kn=list(arrays['support_kinematic_names'])
        v=arrays['values'];vn=list(arrays['value_names']);alive=arrays['episode_alive'];times=arrays['time_s']
        # Fix the window before candidate training. Do not replace terminated
        # trials with reset poses or silently discard early failures.
        window=(times>=.1-1e-6)&(times<=.6+1e-6)
        q=k[window,:,kn.index('left_knee_rad')]
        z=k[window,:,kn.index('pelvis_height_m')]
        present=alive[window]
        if len(q):
            peak=np.where(present,q,-np.inf).max(0)
            minimum=np.where(present,z,np.inf).min(0)
            peak[~present.any(0)]=np.nan;minimum[~present.any(0)]=np.nan
        else:
            peak=np.full(alive.shape[1],np.nan);minimum=peak.copy()
        complete=present.all(0)&present.any(0)&bool(np.any(times>=.6-1e-6))
        output={name:report[name] for name in ('checkpoint','checkpoint_sha256','recipe','seed','episodes','success_rate','mean_best_glide_s','nan_episodes')}
        output.update(window_s=[.1,.6],complete_window_episodes=int(complete.sum()),
            incomplete_window_episodes=int((~complete).sum()),
            early_peak_knee_rad=stats(peak),early_min_pelvis_height_m=stats(minimum),fixed_times=[])
        for t in (.4,.6,.8,1.):
            indices=np.flatnonzero(np.isclose(times,t,atol=1e-6))
            mask=alive[indices[0]] if len(indices) else np.zeros(report['episodes'],dtype=bool)
            row=dict(time_s=t,present_episodes=int(mask.sum()),ended_before_sample=int((~mask).sum()))
            if mask.any():
                i=indices[0]
                for name in ('left_knee_rad','pelvis_height_m','support_speed_m_s'):
                    row[name]=stats(k[i,mask,kn.index(name)])
                for name in ('speed_m_s','com_speed_m_s','left_front_force_n','left_rear_force_n'):
                    row[name]=stats(v[i,mask,vn.index(name)])
            output['fixed_times'].append(row)
        return output


def main():
    p=argparse.ArgumentParser();p.add_argument('reports',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=dict(scope='diagnostic comparison, not automatic adoption or gate promotion',reports=[summarize(path) for path in a.reports],
        limitations=['Window extrema are censored if an episode ends early; counts are explicit.',
                     'Association alone does not prove knee flexion causes braking.',
                     'Independent seeds retain domain randomization and observation noise.',
                     'GPU simulation is not guaranteed bitwise reproducible; matched seeds do not imply identical contacts.'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for row in result['reports']:
        print(row['recipe'],row['seed'],row['success_rate'],row['early_peak_knee_rad'],row['early_min_pelvis_height_m'])


if __name__=='__main__':main()
