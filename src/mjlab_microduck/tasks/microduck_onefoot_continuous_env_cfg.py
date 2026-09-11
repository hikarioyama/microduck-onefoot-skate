"""v6: retain curriculum/physics/gates; close the swing-foot replant loophole."""
from mjlab.managers import TerminationTermCfg
from . import mdp
from .microduck_onefoot_curriculum_env_cfg import (
    STAGES,make_curriculum_onefoot_env_cfg,make_curriculum_onefoot_rl_cfg,
)

TASK = 'Mjlab-OneFoot-Continuous-Curriculum-Flat-MicroDuck-Rollers'
# Explicit stock-play tasks prevent accidentally showing an assisted spawn as self-launch.
STAGE_TASKS={stage.name:'Mjlab-OneFoot-Continuous-'+''.join(part.title() for part in stage.name.split('-'))+'-Flat-MicroDuck-Rollers' for stage in STAGES}


def make_continuous_onefoot_env_cfg(play=False,stage='balance-050'):
    cfg=make_curriculum_onefoot_env_cfg(play=play,stage=stage)
    for component in ('hold','continuous','accelerate','transfer','steering'):
        cfg.rewards['curriculum_'+component].func=mdp.continuous_onefoot_reward
    cfg.terminations['replanted_swing']=TerminationTermCfg(
        func=mdp.continuous_onefoot_replanted,time_out=False)
    return cfg


def make_continuous_onefoot_rl_cfg():
    cfg=make_curriculum_onefoot_rl_cfg()
    cfg.run_name='continuous-curriculum-v6'
    cfg.actor.distribution_cfg['init_std']=.10
    cfg.algorithm.entropy_coef=.003
    return cfg
