"""v7 candidate: shape fore-aft balance inside the measured 65 mm wheelbase.

Keep v6's actual success criteria, contact rules, BAM and sensor contract. This
adds an approximate capture-margin reward, not a physical stability guarantee.
"""
from mjlab.managers.metrics_manager import MetricsTermCfg
from . import mdp
from .microduck_onefoot_continuous_env_cfg import make_continuous_onefoot_env_cfg,make_continuous_onefoot_rl_cfg

TASK='Mjlab-OneFoot-Centroid-Curriculum-Flat-MicroDuck-Rollers'


def make_centroid_onefoot_env_cfg(play=False,stage='balance-050'):
    cfg=make_continuous_onefoot_env_cfg(play=play,stage=stage)
    for component in ('hold','continuous'):
        cfg.rewards['curriculum_'+component].func=mdp.centroid_onefoot_reward
    for name in ('foreaft_error','foreaft_capture','front_load'):
        cfg.metrics['onefoot/'+name]=MetricsTermCfg(func=mdp.centroid_onefoot_metric,
            params={'component':name},reduce='mean')
    return cfg


def make_centroid_onefoot_rl_cfg():
    cfg=make_continuous_onefoot_rl_cfg();cfg.run_name='centroid-curriculum-v7'
    return cfg
