"""Re-score recorded failed trajectories with the exact new reward state machine.

No rollout/retraining claim: these are frozen old observations. The old records
have best_glide_s=0, so the common valid predicate never held on any frame.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onefoot_credit_bridge_env_cfg import REWARD_GAMMA

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'local/onefoot-reward-review/replay-ranking.json')
    args=parser.parse_args()
    rows=[]
    for seed in (70501,70502):
        pair={}
        for name in ('model_initial','model_299'):
            source=ROOT/f'local/onefoot-retained-bridge/transfer-{name}-{seed}.npz'
            with np.load(source,allow_pickle=False) as archive:
                a={key:archive[key] for key in archive.files}
            assert np.all(a['best_glide_s']==0), 'Cannot infer all-false valid mask from a successful record'
            values=torch.from_numpy(a['values']);kin=torch.from_numpy(a['kinematics'])
            names=list(a['value_names']);knames=list(a['kinematic_names']);tnames=list(a['termination_names'])
            n=a['alive'].shape[1]
            state=mdp.CreditBridgeRewardState(n,'cpu',gamma=REWARD_GAMMA,goal_s=2.)
            new_return=np.zeros(n);base_return=np.zeros(n);shaping_return=np.zeros(n)
            undiscounted=np.zeros(n)
            for t in range(len(values)):
                v=lambda key:values[t,:,names.index(key)]
                skate=kin[t,:,knames.index('support_speed_m_s')]
                forbidden=torch.from_numpy(a['termination_flags'][t,:,tnames.index('body_ground')])
                # v16 replay: these frozen records predate the fore/aft axis and
                # carried no valid glide frame, so the new terms contribute zero.
                phi=mdp.credit_bridge_potential(v('com_speed_m_s'),skate,v('capture_error_m'),
                    torch.zeros(n),v('left_front_force_n'),v('left_rear_force_n'),v('right_force_n'),
                    v('clearance_m'),v('upright_cos'),v('support_yaw_rad'),forbidden)
                flags=torch.from_numpy(a['termination_flags'][t])
                physical=flags[:,[i for i,key in enumerate(tnames) if key!='time_out']].any(1)
                r=state.advance(phi=phi,valid=torch.zeros(n,dtype=torch.bool),
                    com_speed=v('com_speed_m_s'),skate_speed=skate,smooth_cost=torch.zeros(n),
                    done=torch.from_numpy(a['done'][t]),physical_failure=physical,
                    balance_cost=torch.zeros(n))
                live=a['alive'][t]
                factor=REWARD_GAMMA**t
                new_return+=factor*np.where(live,r['total'].numpy(),0)
                base_return+=factor*np.where(live,(r['task']+r['failure']+r['handoff']).numpy(),0)
                shaping_return+=factor*np.where(live,r['shaping'].numpy(),0)
                undiscounted+=np.where(live,r['total'].numpy(),0)
            length=a['alive'].sum(0)
            expected=-2*np.power(REWARD_GAMMA,length-1)
            np.testing.assert_allclose(new_return,expected,atol=2e-6,rtol=1e-5)
            np.testing.assert_allclose(shaping_return,0,atol=2e-6)
            old_discounted=np.nansum(a['rewards']*np.power(REWARD_GAMMA,np.arange(len(values)))[:,None],axis=0)
            pair[name]=dict(mean_episode_s=float(length.mean()*.02),
                old_undiscounted_return=float(np.nansum(a['rewards'],axis=0).mean()),
                old_discounted_return=float(old_discounted.mean()),
                new_discounted_return=float(new_return.mean()),new_base_discounted_return=float(base_return.mean()),
                new_undiscounted_return=float(undiscounted.mean()),
                shaping_discounted_max_abs=float(np.abs(shaping_return).max()))
        assert pair['model_299']['old_discounted_return']>pair['model_initial']['old_discounted_return']
        assert pair['model_299']['new_discounted_return']<pair['model_initial']['new_discounted_return']
        rows.append(dict(seed=seed,policies=pair,early_failure_ranking_fixed=True))
    output=args.output
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(dict(gamma=REWARD_GAMMA,rows=rows,
        scope='Fixed-trajectory accounting and ranking test; not evidence of successful new training',
        assumptions=['Original recorded common valid glide was zero on all frames.',
                     'The actual new finite-horizon task disables timeout value bootstrapping.',
                     'Compare discounted PPO return separately from raw logged reward and skill success.']),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
