"""Left-foot straight roller glide. Separate assisted and self-launch tasks.

Assisted starts with forward momentum to discover balance; the main task starts
at REST. No interval pushes, hidden velocity kicks or automatic milestone switch.
Phase lives in the 3D twist slot; actor stays 61D but command semantics are custom.
"""
from copy import deepcopy
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from . import mdp
from .microduck_velocity_rollers_env_cfg import make_microduck_velocity_rollers_env_cfg, MicroduckRollersRlCfg

TASK = 'Mjlab-OneFoot-Flat-MicroDuck-Rollers'
ASSISTED_TASK = 'Mjlab-OneFoot-Assisted-Flat-MicroDuck-Rollers'


def make_onefoot_env_cfg(play=False, assisted=False):
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)
    cfg.episode_length_s = 8.0
    cfg.commands['twist'] = mdp.OneFootGlideCommandCfg(
        resampling_time_range=(1000,1000), acceleration_s=.6 if assisted else 2.5)
    # Fixed world +X line. Initial y may be randomized; command captures that line.
    cfg.events['reset_base'].params['pose_range'].update(
        roll=(0,0), pitch=(0,0), yaw=(0,0), z=(.144,.146))
    cfg.events['reset_base'].params['velocity_range'] = {
        'x': (.25,.35) if assisted else (0,0), 'y': (0,0), 'z': (0,0),
        'roll': (0,0), 'pitch': (0,0), 'yaw': (0,0)}
    cfg.events.pop('push_robot',None)
    # Retain the base BAM + noise/delay/DR implementation, freeze its severity.
    # No inherited stride/action-rate curriculum may silently change this task.
    cfg.curriculum = {}
    for name in ('randomize_com','randomize_head_com'):
        if name in cfg.events:
            cfg.events[name].params['ranges'] = (-.002,.002)
    keep = {'body_ang_vel','angular_momentum','action_rate_l2','self_collisions',
            'feet_flat','neck_joint_pos_l2','joint_torques_l2','action_over_limit'}
    cfg.rewards = {k:v for k,v in cfg.rewards.items() if k in keep}
    cfg.rewards['body_ang_vel'].weight = -.02
    cfg.rewards['angular_momentum'].weight = -.01
    cfg.rewards['action_rate_l2'].weight = -.05
    cfg.rewards['neck_joint_pos_l2'].weight = -.05
    cfg.rewards['feet_flat'].weight = -.2
    cfg.rewards['action_over_limit'].weight = -.1
    for component,weight in [('accelerate',8.0),('shape',2.0),('glide',12.0),('quiet_glide',.5)]:
        cfg.rewards['onefoot_'+component] = RewardTermCfg(
            func=mdp.onefoot_reward,weight=weight,params={'component':component})
    extra = []
    for name,pattern in [('onefoot_left',r'^(tire|tire_2)$'),
                         ('onefoot_right',r'^(tire_3|tire_4)$'),
                         ('onefoot_body_ground',r'^(?!tire(?:_[234])?$).*$')]:
        extra.append(ContactSensorCfg(name=name,
            primary=ContactMatch(mode='body',pattern=pattern,entity='robot'),
            secondary=ContactMatch(mode='body',pattern='terrain'),
            fields=('found',),reduce='netforce',num_slots=1))
    cfg.scene.sensors = (*cfg.scene.sensors,*extra)
    cfg.terminations['body_ground'] = TerminationTermCfg(func=mdp.onefoot_bad_contact,time_out=False)
    # Kept from the base: falls, timeouts, NaN termination, BAM expansion event.
    cfg.metrics = {f'onefoot/{name}':MetricsTermCfg(func=mdp.onefoot_metric,
        params={'component':name},reduce='last' if name in ('best_dwell','success') else 'mean')
        for name in ('speed','clearance','cross_track','heading_error','single_support',
                     'dwell','best_dwell','success')}
    return cfg


def make_onefoot_rl_cfg(assisted=False):
    cfg = deepcopy(MicroduckRollersRlCfg)
    cfg.experiment_name = 'onefoot_assisted' if assisted else 'onefoot'
    cfg.run_name = 'left_glide_v1'
    cfg.save_interval = 100
    cfg.max_iterations = 3000
    cfg.algorithm.entropy_coef = .01
    return cfg
