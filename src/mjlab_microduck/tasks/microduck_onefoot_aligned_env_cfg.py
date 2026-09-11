"""v4 experiment: remove pre-single-support alignment and two-foot reward loopholes."""
from mjlab.managers import RewardTermCfg,TerminationTermCfg
from . import mdp
from .microduck_onefoot_dynamic_env_cfg import make_dynamic_onefoot_env_cfg,make_dynamic_onefoot_rl_cfg

ALIGNED_TASK='Mjlab-OneFoot-Aligned-Flat-MicroDuck-Rollers'
ALIGNED_ASSISTED_TASK='Mjlab-OneFoot-Aligned-Assisted-Flat-MicroDuck-Rollers'


def make_aligned_onefoot_env_cfg(play=False,assisted=False):
    cfg=make_dynamic_onefoot_env_cfg(play=play,assisted=assisted)
    cfg.rewards['dynamic_progress'].func=mdp.onefoot_v4_progress
    cfg.rewards['dynamic_progress'].params={}
    cfg.rewards['dynamic_steer_inward'].func=mdp.onefoot_v4_steer_reward
    cfg.rewards['dynamic_steer_inward'].params={}
    cfg.rewards['support_alignment']=RewardTermCfg(func=mdp.onefoot_v4_alignment_cost,weight=-2.)
    cfg.rewards['failure'].func=mdp.onefoot_v4_failure
    cfg.terminations['missed_transfer']=TerminationTermCfg(func=mdp.onefoot_v4_missed_transfer,
        time_out=False,params={'grace_s':.75})
    return cfg


def make_aligned_onefoot_rl_cfg(assisted=False):
    cfg=make_dynamic_onefoot_rl_cfg(assisted)
    cfg.run_name='aligned-assisted-v4' if assisted else 'aligned-full-v4'
    return cfg
