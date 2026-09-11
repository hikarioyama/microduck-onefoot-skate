"""v11 bounded candidate: suppress deep support-knee collapse, not knee motion.

Derived from waist v10. Only multiply existing maneuver rewards; unchanged
physics, initial states, observations, actions, terminations and skill gates.
This recipe is NOT adopted merely by registering it or passing a smoke test.
"""
from mjlab.managers.metrics_manager import MetricsTermCfg
from . import mdp
from .microduck_onefoot_waist_env_cfg import make_waist_onefoot_env_cfg,make_waist_onefoot_rl_cfg

TASK='Mjlab-OneFoot-Knee-Curriculum-Flat-MicroDuck-Rollers'
# Verified MJCF positive left-knee qpos increases flexion. Neutral qpos is
# already bent; the plateau explicitly permits useful compliance and lean.
SUPPORT_KNEE_COMFORTABLE_MAX=.35
SUPPORT_KNEE_CREDIT_WIDTH=.45


def make_knee_onefoot_env_cfg(play=False,stage='balance-050'):
    cfg=make_waist_onefoot_env_cfg(play=play,stage=stage)
    params=dict(comfortable_max=SUPPORT_KNEE_COMFORTABLE_MAX,width=SUPPORT_KNEE_CREDIT_WIDTH)
    for name in ('curriculum_hold','curriculum_continuous','curriculum_transfer','waist_support'):
        cfg.rewards[name].func=mdp.knee_guard_onefoot_signal
        cfg.rewards[name].params.update(params)
    for component in ('support_knee_rad','support_knee_credit','support_knee_excess_rad'):
        cfg.metrics['onefoot/'+component]=MetricsTermCfg(func=mdp.knee_guard_onefoot_signal,
            params=dict(component=component,**params),reduce='mean')
    return cfg


def make_knee_onefoot_rl_cfg():
    cfg=make_waist_onefoot_rl_cfg();cfg.run_name='knee-curriculum-v11-pilot'
    return cfg
