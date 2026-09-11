"""v9 candidate: use the pelvis/torso, without paying for tilt itself.

Permit necessary lateral lean, retain pitch/fall safety, and price a COM or
relative forward momentum escaping the support wheelbase. No new actor inputs,
external forces, altered actuators, or relaxed skill-success criteria.
"""
from copy import deepcopy
from mjlab.managers import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from . import mdp
from .microduck_onefoot_continuous_env_cfg import make_continuous_onefoot_env_cfg,make_continuous_onefoot_rl_cfg

TASK='Mjlab-OneFoot-Lean-Curriculum-Flat-MicroDuck-Rollers'


def make_lean_onefoot_env_cfg(play=False,stage='balance-050'):
    cfg=make_continuous_onefoot_env_cfg(play=play,stage=stage)
    # Keep the original net-load sensors and all success gates unchanged.
    # An additional reward-only sensor measures actual global contact positions.
    cop=deepcopy(next(s for s in cfg.scene.sensors if s.name=='onefoot_left'))
    cop.name='onefoot_left_cop';cop.reduce='maxforce';cop.global_frame=True
    cop.fields=('found','force','pos','normal','tangent')
    cfg.scene.sensors=(*cfg.scene.sensors,cop)
    for component in ('hold','continuous','transfer','accelerate','steering'):
        cfg.rewards['curriculum_'+component].func=mdp.waist_onefoot_signal
    cfg.rewards['waist_support']=RewardTermCfg(func=mdp.waist_onefoot_signal,
        weight=2.,params={'component':'waist_support'})
    cfg.rewards['foreaft_balance']=RewardTermCfg(func=mdp.waist_onefoot_signal,
        weight=-2.,params={'component':'foreaft_balance'})
    for name in ('waist_roll_deg','waist_pitch_deg','waist_roll_speed','waist_axis_error',
                 'waist_axis_alignment','waist_pelvis_offset','waist_capture_error',
                 'waist_foreaft_error','waist_foreaft_capture','waist_front_load','support_cop_y_offset'):
        cfg.metrics['onefoot/'+name]=MetricsTermCfg(func=mdp.waist_onefoot_signal,
            params={'component':name},reduce='mean')
    return cfg


def make_lean_onefoot_rl_cfg():
    cfg=make_continuous_onefoot_rl_cfg();cfg.run_name='lean-curriculum-v9'
    # Retain the learned skill while adapting to the new body-balance objective.
    cfg.algorithm.learning_rate=1e-4
    return cfg
