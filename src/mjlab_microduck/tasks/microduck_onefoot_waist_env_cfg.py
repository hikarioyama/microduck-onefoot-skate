"""v10: explicit support-side pelvis/torso lean plus fore-aft balance.

User-directed style/skill reference. It is not a claim that an upright torso
can never balance; the goal is to stop this policy keeping its pelvis level
while compensating only with large head/free-leg motion.
"""
from mjlab.managers.metrics_manager import MetricsTermCfg
from . import mdp
from .microduck_onefoot_lean_env_cfg import make_lean_onefoot_env_cfg,make_lean_onefoot_rl_cfg

TASK='Mjlab-OneFoot-Waist-Curriculum-Flat-MicroDuck-Rollers'


def make_waist_onefoot_env_cfg(play=False,stage='balance-050'):
    cfg=make_lean_onefoot_env_cfg(play=play,stage=stage)
    for name in ('curriculum_hold','curriculum_transfer','waist_support'):
        cfg.rewards[name].func=mdp.waist_target_onefoot_signal
    for name in ('waist_target_roll_deg','waist_roll_error_deg','waist_roll_match'):
        cfg.metrics['onefoot/'+name]=MetricsTermCfg(func=mdp.waist_target_onefoot_signal,
            params={'component':name},reduce='mean')
    return cfg


def make_waist_onefoot_rl_cfg():
    cfg=make_lean_onefoot_rl_cfg();cfg.run_name='waist-curriculum-v10'
    return cfg
