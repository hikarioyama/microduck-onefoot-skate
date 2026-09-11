"""Dynamic head/contralateral-leg transfer, then quiet straight left-skate glide.

v1 remains available for reproducibility. No prescribed head-swing direction:
whole-body COM position + velocity determine whether the transfer actually helps.
"""
from mjlab.managers import RewardTermCfg, EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from . import mdp
from .microduck_onefoot_env_cfg import make_onefoot_env_cfg, make_onefoot_rl_cfg

DYNAMIC_TASK = 'Mjlab-OneFoot-Dynamic-Flat-MicroDuck-Rollers'
DYNAMIC_ASSISTED_TASK = 'Mjlab-OneFoot-Dynamic-Assisted-Flat-MicroDuck-Rollers'

# Measured kinematic seed, dynamically tested (not an equilibrium guarantee).
SEED_POSE = {'left_hip_yaw': -2.5001359159235572e-05, 'left_hip_roll': 0.002181287604467587, 'left_hip_pitch': -0.4059343185390501, 'left_knee': -0.02646085060033334, 'left_ankle': 0.37954301399804935, 'neck_pitch': 0.2905586233773542, 'head_pitch': 0.3743566905937197, 'head_yaw': -0.002778998524603555, 'head_roll': -0.0008677986361256484, 'right_hip_yaw': -0.018223186921735267, 'right_hip_roll': -0.029174736249190543, 'right_hip_pitch': 0.4529290539000027, 'right_knee': -0.01756358993742159, 'right_ankle': -0.46410750778694493}
SEED_ROOT_Z = 0.1503596863055633
SEED_ROOT_ROLL = -0.33237112280539494


def make_dynamic_onefoot_env_cfg(play=False,assisted=False):
    cfg=make_onefoot_env_cfg(play=play,assisted=assisted)
    cfg.commands['twist']=mdp.DynamicOneFootCommandCfg(
        resampling_time_range=(1000,1000),acceleration_s=.6 if assisted else 2.5,lift_s=1.0)
    for name in ('neck_joint_pos_l2','body_ang_vel','angular_momentum','action_rate_l2',
                 'onefoot_accelerate','onefoot_shape','onefoot_glide','onefoot_quiet_glide'):
        cfg.rewards.pop(name,None)
    for component,weight in [('progress',2.),('transfer',6.),('capture_hold',4.),
                             ('glide',16.),('settle',1.),('steer_inward',2.),('blade_straight',2.)]:
        cfg.rewards['dynamic_'+component]=RewardTermCfg(
            func=mdp.dynamic_onefoot_reward,weight=weight,params={'component':component})
    cfg.rewards['dynamic_action_rate']=RewardTermCfg(func=mdp.dynamic_onefoot_motion_cost,
        weight=-.02,params={'kind':'action_rate'})
    cfg.rewards['dynamic_angular']=RewardTermCfg(func=mdp.dynamic_onefoot_motion_cost,
        weight=-.01,params={'kind':'angular'})
    # RewardManager multiplies weights by dt=.02 => -8 on failure, not on timeout.
    cfg.rewards['failure']=RewardTermCfg(func=mdp.dynamic_onefoot_failure,weight=-400.)
    cfg.events['onefoot_seed']=EventTermCfg(func=mdp.reset_dynamic_onefoot_seed,mode='reset',
        params={'probability':0.0 if play else .70,'pose':SEED_POSE,
                'root_z':SEED_ROOT_Z,'root_roll':SEED_ROOT_ROLL})
    cfg.metrics={f'onefoot/{name}':MetricsTermCfg(func=mdp.dynamic_onefoot_metric,
        params={'component':name},reduce='last' if name in ('best_dwell','success','seeded_start','launch_fraction','launch_success','launch_best_dwell') else 'mean')
        for name in ('speed','clearance','cross_track','heading_error','single_support',
                     'dwell','best_dwell','success','capture_error','com_offset',
                     'com_lateral_speed','head_speed','transition','support_yaw','steering_target','seeded_start','launch_fraction','launch_success','launch_best_dwell')}
    cfg.viewer.distance=.7
    cfg.viewer.elevation=-18
    cfg.viewer.azimuth=140
    cfg.viewer.lookat=(0,0,.06)
    return cfg


def make_dynamic_onefoot_rl_cfg(assisted=False):
    cfg=make_onefoot_rl_cfg(assisted)
    cfg.experiment_name='onefoot'
    cfg.run_name='dynamic-assisted-v3' if assisted else 'dynamic-full-v3'
    cfg.actor.distribution_cfg['init_std']=.20
    cfg.algorithm.entropy_coef=.015
    return cfg
